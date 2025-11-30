import argparse
import csv
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
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
        batch_size: int = 8,
        max_new_tokens: int = 256,
        token_max_length: int = 512,
    ) -> List[str]:
        outputs: List[str] = []
        self.tok.src_lang = src_lang
        # Resolve target language token id robustly across tokenizer versions
        forced_bos = None
        try:
            if hasattr(self.tok, "lang_code_to_id") and isinstance(getattr(self.tok, "lang_code_to_id"), dict):
                mapping = getattr(self.tok, "lang_code_to_id")
                if tgt_lang in mapping:
                    forced_bos = mapping[tgt_lang]
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
            raise RuntimeError(f"Could not resolve target language token id for '{tgt_lang}'")

        total = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total, desc=f"Translate {src_lang}->{tgt_lang}", unit="batch"):
            batch = texts[i : i + batch_size]
            enc = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=token_max_length,
            ).to(self.device)
            gen = self.model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
            )
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


def load_tt_split(mode: str = "full", test_size: float = 0.1, seed: int = 42) -> List[str]:
    logging.info("Loading dataset: textdetox/multilingual_toxicity_dataset (default config)")
    ds_any = load_dataset("textdetox/multilingual_toxicity_dataset")
    if "tt" in ds_any:
        ds_tt = ds_any["tt"]
    else:
        first_key = next(iter(ds_any.keys()))
        base = ds_any[first_key]
        if "lang" in base.column_names:
            ds_tt = base.filter(lambda ex: ex.get("lang") == "tt")
        else:
            ds_tt = base

    if mode in ("train", "validation"):
        split_dict = ds_tt.train_test_split(test_size=test_size, seed=seed)
        part = split_dict["test"] if mode == "validation" else split_dict["train"]
    else:
        part = ds_tt

    col_text = "text" if "text" in part.column_names else ("sentence" if "sentence" in part.column_names else None)
    if col_text is None:
        raise ValueError("Could not find text column in dataset. Expected 'text' or 'sentence'.")
    return part[col_text]


def main():
    ap = argparse.ArgumentParser()
    # Source selection: dataset split or TSV
    ap.add_argument("--input_tsv", type=str, default=None, help="If set, read texts from TSV (columns: ID, tat_toxic)")
    ap.add_argument("--split", type=str, default="full", choices=["full", "train", "validation"], help="Dataset split when not using TSV")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_roundtrip")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--token_max_length", type=int, default=512)
    ap.add_argument("--translate_max_new_tokens", type=int, default=256)
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--max_samples", type=int, default=None)

    # Precompute/load English translations
    ap.add_argument("--need_transl_eng", type=lambda x: str(x).lower() == "true", default=True,
                    help="If true, compute tt->en and optionally save; else load from --transl_eng_path")
    ap.add_argument("--transl_eng_path", type=str, default=None, help="Path to .txt with one English translation per input (saved/loaded)")

    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    device = get_device()
    logging.info("Device: %s", device)

    # Load input texts
    ids: Optional[List] = None
    if args.input_tsv:
        df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
        if not {"ID", "tat_toxic"}.issubset(df.columns):
            raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")
        ids = df["ID"].tolist()
        texts = df["tat_toxic"].astype(str).tolist()
        logging.info("Loaded %d rows from TSV", len(texts))
    else:
        texts = list(load_tt_split(args.split, test_size=args.test_size, seed=args.seed))
        logging.info("Loaded %d texts from dataset split '%s'", len(texts), args.split)

    if args.max_samples is not None:
        texts = texts[: args.max_samples]
        if ids is not None:
            ids = ids[: args.max_samples]
        logging.warning("Limiting to first %d samples", args.max_samples)

    # Translator
    translator = NLLBTranslator(device)

    # 1) tt -> en (precompute/load if requested)
    if args.need_transl_eng:
        logging.info("Computing tt->en translations for all inputs")
        en_all = translator.translate(
            texts,
            src_lang="tat_Cyrl",
            tgt_lang="eng_Latn",
            batch_size=max(1, args.batch_size // 2),
            max_new_tokens=args.translate_max_new_tokens,
            token_max_length=args.token_max_length,
        )
        if args.transl_eng_path:
            try:
                os.makedirs(os.path.dirname(args.transl_eng_path), exist_ok=True)
            except Exception:
                pass
            with open(args.transl_eng_path, "w", encoding="utf-8") as f:
                f.write("\n".join(en_all))
            logging.info("Saved precomputed English translations to %s", args.transl_eng_path)
    else:
        if args.transl_eng_path and os.path.exists(args.transl_eng_path):
            logging.info("Loading precomputed English translations from %s", args.transl_eng_path)
            with open(args.transl_eng_path, "r", encoding="utf-8") as f:
                en_all = [line.rstrip("\n") for line in f]
            if len(en_all) != len(texts):
                logging.warning(
                    "Loaded %d translations but have %d inputs; aligning by truncation/padding",
                    len(en_all), len(texts)
                )
                if len(en_all) < len(texts):
                    en_all = en_all + [""] * (len(texts) - len(en_all))
                else:
                    en_all = en_all[: len(texts)]
        else:
            logging.warning("No transl_eng_path provided; computing translations on the fly")
            en_all = translator.translate(
                texts,
                src_lang="tat_Cyrl",
                tgt_lang="eng_Latn",
                batch_size=max(1, args.batch_size // 2),
                max_new_tokens=args.translate_max_new_tokens,
                token_max_length=args.token_max_length,
            )

    # 2) en -> tt back-translation
    logging.info("Back-translating en->tt for all inputs")
    tt_back = translator.translate(
        en_all,
        src_lang="eng_Latn",
        tgt_lang="tat_Cyrl",
        batch_size=max(1, args.batch_size // 2),
        max_new_tokens=args.translate_max_new_tokens,
        token_max_length=args.token_max_length,
    )

    # Save outputs
    if ids is not None:
        # Submission-like TSV with original and roundtrip
        tsv_path = os.path.join(args.output_dir, "submission_roundtrip.tsv")
        pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": tt_back}).to_csv(tsv_path, sep="\t", index=False)
        logging.info("Saved TSV: %s", tsv_path)
        # Optional ZIP for submission convenience
        zip_path = os.path.join(args.output_dir, "submission_roundtrip.zip")
        with open(tsv_path, "rb") as f:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(os.path.basename(tsv_path), f.read())
        logging.info("Saved ZIP: %s", zip_path)
    else:
        csv_path = os.path.join(args.output_dir, "roundtrip.csv")
        pd.DataFrame({"text": texts, "roundtrip": tt_back}).to_csv(csv_path, index=False)
        logging.info("Saved CSV: %s", csv_path)


if __name__ == "__main__":
    main()

