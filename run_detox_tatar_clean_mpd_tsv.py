"""
Pipline 1 updated: 
    0. Clean text
    1. Translate tt-> en
    2. Detox model
    3. Translate en-> tt
    Saving in tsv, zip
"""

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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# Force GPU 1 by default
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


# Cleaning rules (same style as SDM pipeline)
SAFE_REMOVE_TOKENS = [
    "бля", "бляя", "блят", "блять", "блядь", "блэт", "блэ", "блт", "еба",
    "нах", "нахуй", "нахрен", "нахер", "нахуя", "нихуя",
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
    "🤡", "🖕", "🖕🏻", "🖕🏽", "🖕🏿", "🤦", "🤦‍♂️", "🤦‍♀️", "🤷", "🤷‍♂️", "🤷‍♀️",
    "😂💩", "💩😂", "🤡😂", "😂🤡", "🤬🤬", "😡🤬", "🙃",
]


def build_regexes():
    import re as _re
    return [
        _re.compile(r"б\s*л\s*[еe*#]?\s*[яya@4]+\s*(?:т|ть|\*+|#+)", _re.IGNORECASE),
        _re.compile(r"х\s*у\s*[йиi\*#]+[а-яё]*", _re.IGNORECASE),
        _re.compile(r"п\s*и\s*з\s*д[а-яё\*#]+", _re.IGNORECASE),
        _re.compile(r"н\s*а\s*х[а-яё\*#]+", _re.IGNORECASE),
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


class ToxicityClassifier:
    def __init__(self, device: torch.device, max_len: int = 512):
        self.tok = AutoTokenizer.from_pretrained("textdetox/xlmr-large-toxicity-classifier-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("textdetox/xlmr-large-toxicity-classifier-v2").to(device)
        self.model.eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs) if probs else np.array([])


class BetterTranslator:
    """Prefer Marian OPUS models specialized for Tatar↔English; fallback to NLLB.

    tt->en: Helsinki-NLP/opus-mt-tt-en
    en->tt: Helsinki-NLP/opus-mt-en-tt
    """

    def __init__(self, device: torch.device, strategy: str = "auto"):
        self.device = device
        self.strategy = strategy  # 'auto' | 'marian' | 'nllb'
        self._init_backends()

    def _init_backends(self):
        self.tt_en_tok = self.tt_en_model = None
        self.en_tt_tok = self.en_tt_model = None
        self.nllb_tok = self.nllb_model = None
        if self.strategy in ("auto", "marian"):
            try:
                self.tt_en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tt-en")
                self.tt_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tt-en").to(self.device)
                self.tt_en_model.eval()
            except Exception:
                self.tt_en_tok = self.tt_en_model = None
            try:
                self.en_tt_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-tt")
                self.en_tt_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-tt").to(self.device)
                self.en_tt_model.eval()
            except Exception:
                self.en_tt_tok = self.en_tt_model = None
        if self.strategy in ("auto", "nllb"):
            try:
                self.nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
                self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(self.device)
                self.nllb_model.eval()
            except Exception:
                self.nllb_tok = self.nllb_model = None

    @torch.no_grad()
    def translate(self, texts: List[str], src: str, tgt: str, batch_size: int = 8, max_new_tokens: int = 256, token_max_length: int = 512) -> List[str]:
        if src == "tat_Cyrl" and tgt == "eng_Latn" and self.tt_en_model is not None:
            tok, model = self.tt_en_tok, self.tt_en_model
            desc = "TT→EN (Marian)"
            forced_bos = None
        elif src == "eng_Latn" and tgt == "tat_Cyrl" and self.en_tt_model is not None:
            tok, model = self.en_tt_tok, self.en_tt_model
            desc = "EN→TT (Marian)"
            forced_bos = None
        elif self.nllb_model is not None:
            tok, model = self.nllb_tok, self.nllb_model
            desc = f"{src}→{tgt} (NLLB)"
            tok.src_lang = src
            forced_bos = None
            try:
                if hasattr(tok, "lang_code_to_id") and tgt in getattr(tok, "lang_code_to_id"):
                    forced_bos = tok.lang_code_to_id[tgt]
            except Exception:
                forced_bos = None
            if forced_bos is None:
                for cand in (tgt, f"__{tgt}__"):
                    try:
                        tid = tok.convert_tokens_to_ids(cand)
                        if tid is not None and tid != getattr(tok, "unk_token_id", None):
                            forced_bos = tid
                            break
                    except Exception:
                        continue
        else:
            raise RuntimeError("No available translation backend (Marian or NLLB) could be loaded")

        outputs: List[str] = []
        total = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total, desc=desc, unit="batch"):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=token_max_length).to(self.device)
            if forced_bos is not None:
                gen = model.generate(**enc, forced_bos_token_id=forced_bos, max_new_tokens=max_new_tokens)
            else:
                gen = model.generate(**enc, max_new_tokens=max_new_tokens)
            outputs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


class DetoxModelMPD:
    def __init__(self, device: torch.device, max_new_tokens: int = 128):
        self.tok = AutoTokenizer.from_pretrained("s-nlp/mt0-xl-detox-mpd")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/mt0-xl-detox-mpd").to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def detoxify(self, texts: List[str], batch_size: int = 8) -> List[str]:
        outputs: List[str] = []
        prompts = [f"Detoxify: {t}" for t in texts]
        total = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=total, desc="Detox MPD", unit="batch"):
            batch = prompts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, default="/mnt/extremessd10tb/borisiuk/LCM/DETOX_TATAR/data/test_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_clean_mpd_tsv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--translate_max_new_tokens", type=int, default=256)
    ap.add_argument("--detox_max_new_tokens", type=int, default=128)
    ap.add_argument("--token_max_length", type=int, default=512)
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--translator", type=str, default="auto", choices=["auto", "marian", "nllb"], help="Prefer Marian OPUS (tt-en/en-tt), NLLB, or auto")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    logging.info("Device: %s", device)

    # Load TSV with IDs and tat_toxic
    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    required_cols = {"ID", "tat_toxic"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input TSV must contain columns: {sorted(required_cols)}")
    ids = df["ID"].tolist()
    texts = df["tat_toxic"].astype(str).tolist()
    logging.info("Loaded %d inputs from TSV", len(texts))

    # Clean all inputs first
    logging.info("Cleaning all inputs")
    cleaned = [cleanse_text(t) for t in tqdm(texts, desc="Clean", unit="row")]

    # Models
    clf = ToxicityClassifier(device, max_len=args.token_max_length)
    translator = BetterTranslator(device, strategy=args.translator)
    detox = DetoxModelMPD(device, max_new_tokens=args.detox_max_new_tokens)

    # Gate on cleaned
    logging.info("Scoring toxicity for gating on cleaned inputs")
    probs = clf.score(cleaned, batch_size=args.batch_size)
    mask = (probs >= args.threshold).tolist()

    outputs: List[str] = []
    bs = args.batch_size
    for i in tqdm(range(0, len(cleaned), bs), total=(len(cleaned)+bs-1)//bs, desc="Pipeline (clean->xlat->detox)", unit="batch"):
        batch = cleaned[i : i + bs]
        batch_mask = mask[i : i + bs]
        to_edit = [t for t, m in zip(batch, batch_mask) if m]
        if not to_edit:
            outputs.extend(batch)
            continue
        # tt->en
        en = translator.translate(to_edit, src="tat_Cyrl", tgt="eng_Latn", batch_size=max(1, bs // 2), max_new_tokens=args.translate_max_new_tokens, token_max_length=args.token_max_length)
        # detox EN (MPD)
        en_detox = detox.detoxify(en, batch_size=max(1, bs // 2))
        # en->tt
        tt_back = translator.translate(en_detox, src="eng_Latn", tgt="tat_Cyrl", batch_size=max(1, bs // 2), max_new_tokens=args.translate_max_new_tokens, token_max_length=args.token_max_length)
        it = iter(tt_back)
        merged = [next(it) if m else orig for orig, m in zip(batch, batch_mask)]
        outputs.extend(merged)

    # Fallback to original if any empty
    final_out = [o if (o and str(o).strip()) else t for o, t in zip(outputs, texts)]

    # Save submission files
    tsv_path = os.path.join(args.output_dir, "submission_clean_mpd.tsv")
    pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": final_out}).to_csv(tsv_path, sep="\t", index=False)
    logging.info("Saved TSV: %s", tsv_path)
    zip_path = os.path.join(args.output_dir, "submission_clean_mpd.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_path, arcname=os.path.basename(tsv_path))
    logging.info("Saved ZIP: %s", zip_path)


if __name__ == "__main__":
    main()
