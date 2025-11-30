#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import re
import zipfile
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)
try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# Bind to GPU 1 by default
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


SAFE_REMOVE_TOKENS = [
    "бля", "бляя", "блят", "блять", "блядь", "блэт", "блэ", "блт",
    "нах", "нахуй", "нахрен", "нахер", "нахуя", "нихуя", "еба",
    "пизда", "пезда", "пездос", "пиздос", "пиздюк", "пездюк",
    "пиздец", "пездэц", "пездес", "пиздосик", "пиздобратия",
    "пиздобол", "пиздабол",
    "хуеплет", "хуеплёт", "хуесос", "хуесослар", "хуепачмак",
    "хуебет", "хуебоз", "хуйлоп", "хуйло", "хуйня",
    "хуита", "хуёв", "хуевина",
    "пидор", "пидарас", "пидорас", "пидар", "пидрилла",
    "пидарок", "пидр", "пидрия", "педик", "гомик",
    "долбаеб", "долбоеб", "долбоёб", "дебил", "идиот",
    "чмо", "чмошник",
    "срандель", "сратый", "говнюк", "гавнюк",
    "жопа", "жопоротый", "жополиза", "жопашник",
    "әттәгенә", "әттәгенәһе", "әпәт", "әпәәт",
    "сука", "сучара", "мразь", "тварь",
    "😡", "🤬", "👿", "😠",
    "😏", "😒", "🙄",
    "💩", "🍑", "👉👌", "👈👉", "🔥🍑", "🍆", "💦🍑", "💦🍆",
    "🤡", "🖕", "🖕🏻", "🖕🏽", "🖕🏿", "🤦", "🤦‍♂️", "🤦‍♀️",
    "🤷", "🤷‍♂️", "🤷‍♀️",
    "😂💩", "💩😂", "🤡😂", "😂🤡", "🤬🤬", "😡🤬", "🙃",
]


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


class ToxicityClassifier:
    def __init__(self, device: torch.device, max_len: int = 512):
        self.tok = AutoTokenizer.from_pretrained(
            "textdetox/xlmr-large-toxicity-classifier-v2"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "textdetox/xlmr-large-toxicity-classifier-v2"
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs) if probs else np.array([])


class LlamaDetoxTatar:
    SYSTEM_PROMPT = (
        "Син татар текстын детоксификацияләү буенча белгеч. "
        "Бурыч токсик сүзләрне алып кына җөмләнең бөтен мәгънәсен структурасын фактларын һәм нейтраль эчтәлеген саклау. "
        "Бернинди яңа фикер аңлатма өстәмә сүз кертмә. "
        "Әгәр җөмләдә ачыктан ачык сүгенү яки мыскыллау юк икән текстны үзгәртмә. "
        "Җавап татар телендә генә булырга тиеш. "
        "Өстәмә җөмләләр язма. "
        "Фактны адресатны эмоциональ тоны сакла. "
        "Максат ким үзгәртеп токсик сүзне генә юкка чыгару.\n\n"

        "Мисаллар\n\n"

        "Токсик:\nну чапай чапай эйтер идем инде ахмак\n"
        "Детокс:\nну чапай чапай эйтер идем инде\n\n"

        "Токсик:\nсимереп чучка буласыз бит\n"
        "Детокс:\nсимереp барасыз бит\n\n"

        "Токсик:\nандый хайваннарны кабер генә төзәтә\n"
        "Детокс:\nандый кешеләрне бернәрсә дә үзгәртми\n\n"

        "Токсик:\nсине анда барыбер көтеп ятмыйлар так что сиди ровно на жопе\n"
        "Детокс:\nсине анда барыбер көтеп ятмыйлар так что тыныч утыр\n\n"

        "Хәзер синең бурыч\n\n"
        "Токсик:\n{text}\n\n"
        "Детокс:"
    )

    def __init__(self, model_id: str, device: torch.device, fp16: bool = True, lora_adapter: str | None = None):
        if device.type == "cuda" and fp16:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        if lora_adapter:
            if PeftModel is None:
                raise RuntimeError("peft not installed; run `pip install peft` to load LoRA adapter")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter).to(device)
        self.device = device
        self.model.eval()

    def _messages(self, text: str):
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Текст: {text}\nЯңа вариант:"},
        ]

    @torch.no_grad()
    def detox(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 0.01,
        top_p: float = 0.9,
        single_line: bool = True,
    ) -> List[str]:

        outs: List[str] = []
        total = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(texts), batch_size), total=total, desc="LLaMA detox"):
            batch = texts[i : i + batch_size]
            conv = [self._messages(t) for t in batch]
            rendered = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
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

            input_lengths = enc["attention_mask"].sum(dim=1).tolist()
            for j, in_len in enumerate(input_lengths):
                new_txt = self.tok.decode(gen[j, in_len:], skip_special_tokens=True)
                if single_line:
                    new_txt = re.sub(r"\s+", " ", new_txt).strip()
                outs.append(new_txt)

        return outs


def build_regexes() -> List[re.Pattern]:
    return [
        re.compile(r"б\s*л\s*[еe*#]?\s*[яya@4]+\s*(?:т|ть|\*+|#+)", re.IGNORECASE),
        re.compile(r"х\s*у\s*[йиi\*#]+[а-яё]*", re.IGNORECASE),
        re.compile(r"п\s*и\s*з\s*д[а-яё\*#]+", re.IGNORECASE),
        re.compile(r"н\s*а\s*х[а-яё\*#]+", re.IGNORECASE),
    ]


REGEXES = build_regexes()


def cleanse_text(text: str) -> str:
    t = str(text)
    for tok in SAFE_REMOVE_TOKENS:
        t = re.sub(re.escape(tok), "", t, flags=re.IGNORECASE)
    for rx in REGEXES:
        t = rx.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_mgpt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--llama_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_adapter", type=str, default=None, help="Path to LoRA adapter dir (e.g., DETOX_TATAR/llama_tatar_lora/adapter)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    device = get_device()
    logging.info("Device: %s", device)

    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    if not {"ID", "tat_toxic"}.issubset(df.columns):
        raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")
    ids = df["ID"].tolist()
    texts = df["tat_toxic"].astype(str).tolist()
    logging.info("Loaded %d rows", len(texts))

    logging.info("Applying word removal and regex cleaning")
    cleaned = [cleanse_text(t) for t in tqdm(texts, desc="Clean", unit="row")]

    clf = ToxicityClassifier(device)
    logging.info("Scoring toxicity after cleaning (gating)")
    probs = clf.score(cleaned, batch_size=args.batch_size)
    mask = (probs >= args.threshold).tolist()

    llama = LlamaDetoxTatar(args.llama_id, device, fp16=args.fp16, lora_adapter=args.lora_adapter)
    outputs: List[str] = []
    bs = args.batch_size
    total_batches = (len(cleaned) + bs - 1) // bs

    for i in tqdm(range(0, len(cleaned), bs), total=total_batches, desc="Detox gate"):
        batch = cleaned[i : i + bs]
        batch_mask = mask[i : i + bs]
        to_edit = [t for t, m in zip(batch, batch_mask) if m]
        if not to_edit:
            outputs.extend(batch)
            continue
        edited = llama.detox(
            to_edit,
            batch_size=max(1, bs // 4),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            single_line=True,
        )
        it = iter(edited)
        merged = [next(it) if m else orig for orig, m in zip(batch, batch_mask)]
        outputs.extend(merged)

    final_out = [o if (o and str(o).strip()) else t for o, t in zip(outputs, texts)]

    tsv_path = os.path.join(args.output_dir, "submission_wordrm_mgpt.tsv")
    pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": final_out}).to_csv(
        tsv_path, sep="\t", index=False
    )
    logging.info("Saved TSV: %s", tsv_path)

    if args.make_zip:
        zip_path = os.path.join(args.output_dir, "submission_wordrm_mgpt.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, arcname=os.path.basename(tsv_path))
        logging.info("Saved ZIP: %s", zip_path)


if __name__ == "__main__":
    main()
