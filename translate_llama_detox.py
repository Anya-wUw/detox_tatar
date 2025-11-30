import argparse
import csv
import logging
import os
import re
import zipfile
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Restrict to GPU 1 unless overridden
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


class NLLBTranslator:
    def __init__(self, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        max_new_tokens: int = 256,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> List[str]:
        out: List[str] = []
        self.tok.src_lang = src_lang
        forced_bos = None
        # Robust mapping across tokenizer versions
        try:
            if hasattr(self.tok, "lang_code_to_id") and tgt_lang in getattr(self.tok, "lang_code_to_id"):
                forced_bos = self.tok.lang_code_to_id[tgt_lang]
        except Exception:
            forced_bos = None
        if forced_bos is None:
            for cand in (tgt_lang, f"__{tgt_lang}__"):
                try:
                    tid = self.tok.convert_tokens_to_ids(cand)
                    if tid is not None and tid != getattr(self.tok, "unk_token_id", None):
                        forced_bos = tid
                        break
                except Exception:
                    continue
        if forced_bos is None:
            raise RuntimeError(f"Could not resolve language id for {tgt_lang}")

        total = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total, desc=f"Translate {src_lang}->{tgt_lang}", unit="batch"):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            gen = self.model.generate(**enc, forced_bos_token_id=forced_bos, max_new_tokens=max_new_tokens)
            out.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return out


class LlamaDetox:
    SYSTEM_PROMPT = (
        """
        Син татар телендәге тәкъдимнәрне детоксификацияләү буенча белгеч.

        Бирем: Әгәр кертелгән җөмләдә ачык (эксплицит) рәвештә сүгенү сүзләре, кимсетү, мыскыллау, дорфа лексика булса — җөмләне яңадан яз, һәм:
        1) Дорфа сүзләрне алып ташла яки нейтраль вариантка алыштыр.
        2) Калган мәгънә, фактлар һәм контекст мөмкин кадәр саклансын.
        3) Яңа фикер өстәмә, аңлатма бирмә, бәяләү өстәмә.
        4) Әгәр эксплицит дорфалык юк икән — җөмләне үзгәртмичә кире кайтар.

        Мисаллар:

        Toxic:
        Сиңа аңлату өчен әйтәм, өстәге җөмлә, әйткәнчә, бөтенләй ахмакча.
        Detoxed:
        Сиңа аңлату өчен әйтәм, өстәге җөмлә дөрес түгел.

        Toxic:
        Бәйләнчек кешеләр моны сугыштан игътибарны читкә алу өчен язган.
        Detoxed:
        Кемдер моны сугыштан игътибарны читкә алу өчен язган.

        Toxic:
        Трампны хурлау өчен бар дөнья чүп-чарны җыйдыңмыни?
        Detoxed:
        Син бары шуны гына әйтә аласыңмы?

        Toxic:
        Бу очракларның нинди көлке дәрәҗәгә баруын күрмисеңме?
        Detoxed:
        Бу очракларның ничек үзгәрүен күрәсеңме?

        Toxic:
        Ахмаклар кораблары тулы тизлектә йөзеп килә.
        Detoxed:
        Кораб тулы тизлектә йөзеп килә.

        Toxic:
        Мәкаләне укыдыңмы соң, әллә тагын бер “гукта” ялчысымы?
        Detoxed:
        Мәкаләне укыдыңмы, әллә юкмы?

        Хәзер синең бирем:
        Toxic: {text}
        Detoxed:

        """
    )

    def __init__(self, model_id: str, device: torch.device, fp16: bool = True):
        if device.type == "cuda" and fp16:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # Decoder-only: left padding
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        self.device = device
        self.model.eval()

    def _build_messages(self, text: str) -> List[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: {text}\nOutput:"},
        ]

    @torch.no_grad()
    def detox(self, texts: List[str], batch_size: int = 2, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.95, single_line: bool = True) -> List[str]:
        outs: List[str] = []
        total = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total, desc="Detox EN (LLaMA)", unit="batch"):
            batch = texts[i : i + batch_size]
            messages_batch = [self._build_messages(t) for t in batch]
            rendered = self.tok.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
            enc = self.tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            # Keep only newly generated tokens per sample
            input_lengths = enc["attention_mask"].sum(dim=1).tolist()
            for j, in_len in enumerate(input_lengths):
                new_txt = self.tok.decode(gen[j, in_len:], skip_special_tokens=True)
                if single_line:
                    new_txt = re.sub(r"\s+", " ", new_txt).strip()
                outs.append(new_txt)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_llama_xlat")
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--llama_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--translate_max_new_tokens", type=int, default=256)
    ap.add_argument("--detox_max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--need_transl_eng", type=lambda x: str(x).lower() == "true", default=True,
                    help="If true, compute tt->en and (optionally) save; else load from --transl_eng_path")
    ap.add_argument("--transl_eng_path", type=str, default=None,
                    help="Path to .txt with one English translation per input (saved/loaded)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = get_device()
    logging.info("Device: %s", device)

    # Read inputs
    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    if not {"ID", "tat_toxic"}.issubset(df.columns):
        raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")
    ids = df["ID"].tolist()
    tt_texts = df["tat_toxic"].astype(str).tolist()

    # Init models
    translator = NLLBTranslator(device)
    llama = LlamaDetox(args.llama_id, device, fp16=args.fp16)

    # 1) tt -> en (precompute/load if requested)
    if args.need_transl_eng:
        logging.info("Computing tt->en translations for all inputs")
        en_texts = translator.translate(
            tt_texts,
            src_lang="tat_Cyrl",
            tgt_lang="eng_Latn",
            max_new_tokens=args.translate_max_new_tokens,
            batch_size=max(1, args.batch_size // 2),
        )
        if args.transl_eng_path:
            try:
                os.makedirs(os.path.dirname(args.transl_eng_path), exist_ok=True)
            except Exception:
                pass
            with open(args.transl_eng_path, "w", encoding="utf-8") as f:
                f.write("\n".join(en_texts))
            logging.info("Saved precomputed English translations to %s", args.transl_eng_path)
    else:
        if args.transl_eng_path and os.path.exists(args.transl_eng_path):
            logging.info("Loading precomputed English translations from %s", args.transl_eng_path)
            with open(args.transl_eng_path, "r", encoding="utf-8") as f:
                en_texts = [line.rstrip("\n") for line in f]
            if len(en_texts) != len(tt_texts):
                logging.warning("Loaded %d translations but have %d inputs; aligning by truncation/padding", len(en_texts), len(tt_texts))
                if len(en_texts) < len(tt_texts):
                    en_texts = en_texts + [""] * (len(tt_texts) - len(en_texts))
                else:
                    en_texts = en_texts[: len(tt_texts)]
        else:
            logging.warning("No transl_eng_path provided; computing translations on the fly")
            en_texts = translator.translate(tt_texts, src_lang="tat_Cyrl", tgt_lang="eng_Latn", max_new_tokens=args.translate_max_new_tokens, batch_size=max(1, args.batch_size // 2))
    # 2) detox in English with LLaMA (new tokens only)
    en_detox = llama.detox(en_texts, batch_size=args.batch_size, max_new_tokens=args.detox_max_new_tokens, temperature=args.temperature, top_p=args.top_p, single_line=True)
    # 3) en -> tt
    tt_back = translator.translate(en_detox, src_lang="eng_Latn", tgt_lang="tat_Cyrl", max_new_tokens=args.translate_max_new_tokens, batch_size=max(1, args.batch_size // 2))

    # Ensure no empties
    out_texts = [o if (o and o.strip()) else t for o, t in zip(tt_back, tt_texts)]

    # Save submission-format TSV and optional ZIP
    tsv_path = os.path.join(args.output_dir, "submission_llama_xlat.tsv")
    pd.DataFrame({"ID": ids, "tat_toxic": tt_texts, "tat_detox1": out_texts}).to_csv(tsv_path, sep="\t", index=False)
    logging.info("Saved TSV: %s", tsv_path)
    if args.make_zip:
        zip_path = os.path.join(args.output_dir, "submission_llama_xlat.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, arcname=os.path.basename(tsv_path))
        logging.info("Saved ZIP: %s", zip_path)


if __name__ == "__main__":
    main()
