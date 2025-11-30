import argparse
import csv
import logging
import os
import re
import zipfile
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# Bind to GPU 1 by default
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


SAFE_REMOVE_TOKENS = [
    # русские матные вставки, часто встречающиеся в татарской речи
    "бля", "бляя", "блят", "блять", "блядь", "блэт", "блэ", "блт",

    "нах", "нахуй", "нахрен", "нахер", "нахуя", "нихуя", "еба",

    # пизд-корень (эмоциональные междометия, без смысла)
    "пизда", "пезда", "пездос", "пиздос", "пиздюк", "пездюк",
    "пиздец", "пездэц", "пездес", "пиздосик", "пиздобратия",
    "пиздобол", "пиздабол",

    # хуе-корень
    "хуеплет", "хуеплёт", "хуесос", "хуесослар", "хуепачмак",
    "хуебет", "хуебоз", "хуйлоп", "хуйло", "хуйня",
    "хуита", "хуёв", "хуевина",

    # оскорбления, не несущие фактического смысла
    "пидор", "пидарас", "пидорас", "пидар", "пидрилла",
    "пидарок", "пидр", "пидрия", "педик", "гомик",

    # мягкая ругань
    "долбаеб", "долбоеб", "долбоёб", "дебил", "идиот",
    "чмо", "чмошник",

    # скатологические ругательства
    "срандель", "сратый", "говнюк", "гавнюк",

    # «жопа»-ругательства
    "жопа", "жопоротый", "жополиза", "жопашник",

    # татарские эмоциональные междометия
    "әттәгенә", "әттәгенәһе", "әпәт", "әпәәт",

    # длинные пустые ругательные обороты
    "сука", "сучара", "мразь", "тварь",

    # токсичные эмодзи и их комбинации
    "😡", "🤬", "👿", "😠",
    "😏", "😒", "🙄",
    "💩", "🍑", "👉👌", "👈👉", "🔥🍑", "🍆", "💦🍑", "💦🍆",
    "🤡", "🖕", "🖕🏻", "🖕🏽", "🖕🏿", "🤦", "🤦‍♂️", "🤦‍♀️",
    "🤷", "🤷‍♂️", "🤷‍♀️",
    "😂💩", "💩😂", "🤡😂", "😂🤡", "🤬🤬", "😡🤬", "🙃", "🫄",

    # заеб-корень, как чистый матный маркер
    "заебал", "заебали", "заебались", "заебись", "заипали",

    # еб-корень, употребляется как матное междометие и жёсткое оскорбление
    "ебать", "ебаный", "ебаная", "ебанутый", "ебанутая",
    "ебанулся", "еблан",

    # оху-корень в явной матной функции
    "охуел", "охуеть", "охуенно", "охуенный", "прихуел",

    # сильные однословные оскорбления
    "мудак", "мудаки",
    "шалава", "шалавы",
    "сволочь", "сволочи",
    "еблан", 
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


class SDMDetox:
    def __init__(self, device: torch.device, fp16: bool = True, max_new_tokens: int = 128):
        # s-nlp/mt0-xl-detox-sdm-full
        self.tok = AutoTokenizer.from_pretrained("s-nlp/mt0-xl-detox-mpd")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/mt0-xl-detox-mpd").to(device)
        if fp16 and device.type == "cuda":
            try:
                self.model.half()
            except Exception:
                pass
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def detox(self, texts: List[str], batch_size: int = 8) -> List[str]:
        outputs: List[str] = []
        prompts = [f"Detoxify: {t}" for t in texts]
        total = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=total, desc="SDM detox", unit="batch"):
            batch = prompts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


# =============================
# Cleaning: word removal + regex
# =============================


def build_regexes() -> List[re.Pattern]:
    patterns = []
    # Obfuscations around "блять": бл*ть, бл##ть, блеать, блят
    patterns.append(re.compile(r"б\s*л\s*[еe*#]?\s*[яya@4]+\s*(?:т|ть|\*+|#+)", re.IGNORECASE))
    # General roots with star/hash in the middle (ху*, пи*д*, на*уй etc.)
    patterns.append(re.compile(r"х\s*у\s*[йиi\*#]+[а-яё]*", re.IGNORECASE))
    patterns.append(re.compile(r"п\s*и\s*з\s*д[а-яё\*#]+", re.IGNORECASE))
    patterns.append(re.compile(r"н\s*а\s*х[а-яё\*#]+", re.IGNORECASE))
    return patterns


REGEXES = build_regexes()


def cleanse_text(text: str) -> str:
    t = str(text)
    # Remove explicit tokens first (case-insensitive, as standalone or embedded)
    for tok in SAFE_REMOVE_TOKENS:
        t = re.sub(re.escape(tok), "", t, flags=re.IGNORECASE)
    # Regex-based removals for obfuscations
    for rx in REGEXES:
        t = rx.sub("", t)
    # Collapse spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_wordrm_sdm")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    device = get_device()
    logging.info("Device: %s", device)

    # Load data
    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    if not {"ID", "tat_toxic"}.issubset(df.columns):
        raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")
    ids = df["ID"].tolist()
    texts = df["tat_toxic"].astype(str).tolist()
    logging.info("Loaded %d rows", len(texts))

    # Cleaning
    logging.info("Applying word removal + regex cleaning")
    cleaned = [cleanse_text(t) for t in tqdm(texts, desc="Clean", unit="row")]

    # Classifier gating
    clf = ToxicityClassifier(device)
    logging.info("Scoring toxicity after cleaning (gating)")
    probs = clf.score(cleaned, batch_size=args.batch_size)
    mask = (probs >= args.threshold).tolist()

    # Detox only masked examples using SDM
    sdm = SDMDetox(device, fp16=args.fp16)
    outputs: List[str] = []
    batch_size = args.batch_size
    for i in tqdm(range(0, len(cleaned), batch_size), total=(len(cleaned)+batch_size-1)//batch_size, desc="Detox gate", unit="batch"):
        batch = cleaned[i : i + batch_size]
        batch_mask = mask[i : i + batch_size]
        to_edit = [t for t, m in zip(batch, batch_mask) if m]
        if not to_edit:
            outputs.extend(batch)
            continue
        edited = sdm.detox(to_edit, batch_size=max(1, batch_size // 2))
        it = iter(edited)
        merged = [next(it) if m else orig for orig, m in zip(batch, batch_mask)]
        outputs.extend(merged)

    # Safety: fallback to original if any empty
    final_out = [o if (o and str(o).strip()) else t for o, t in zip(outputs, texts)]

    # Save submission TSV (only required columns) and optional ZIP
    tsv_path = os.path.join(args.output_dir, "submission_wordrm_sdm.tsv")
    pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": final_out}).to_csv(tsv_path, sep="\t", index=False)
    logging.info("Saved TSV: %s", tsv_path)
    if args.make_zip:
        zip_path = os.path.join(args.output_dir, "submission_wordrm_sdm.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, arcname=os.path.basename(tsv_path))
        logging.info("Saved ZIP: %s", zip_path)


if __name__ == "__main__":
    main()
