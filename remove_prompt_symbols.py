import argparse
import csv
import logging
import os
import re
import zipfile
from typing import List

import pandas as pd


ARTIFACT_PATTERNS = [
    r"\bТекст\b\s*:?",      # 'Текст' or 'Текст:'
    r"\bДетокс\b\s*:?",     # 'Детокс' or 'Детокс:'
    r"\bТоксик\b\s*:?",     # 'Токсик' or 'Токсик:'
    r"\{\s*text\s*\}",     # '{text}'
    r"\buser\b\s*:?",       # 'user' or 'user:'
    r"\bassistant\b\s*:?",  # 'assistant' or ':assistant'
]


def clean_text(s: str) -> str:
    if s is None:
        return ""
    txt = str(s)
    # Remove artifacts case-insensitively
    for pat in ARTIFACT_PATTERNS:
        txt = re.sub(pat, "", txt, flags=re.IGNORECASE)
    # Collapse leftover multiple spaces and stray quotes
    txt = re.sub(r"\s+", " ", txt).strip()
    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        txt = txt[1:-1].strip()
    # Remove any repeated labels at line starts if present
    txt = re.sub(r"(?im)^(Текст|Детокс|Токсик)\s*:\s*", "", txt).strip()
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, required=True, help="Path to TSV to clean (expects ID, tat_toxic, tat_detox1)")
    ap.add_argument("--output_dir", type=str, default=None, help="Where to save cleaned files; defaults to input dir")
    ap.add_argument("--columns", type=str, default="tat_detox1", help="Comma-separated columns to clean")
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    args = ap.parse_args()

    in_path = args.input_tsv
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    out_dir = args.output_dir or os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t", quoting=csv.QUOTE_NONE)
    # Keep only required columns if they exist
    required = ["ID", "tat_toxic", "tat_detox1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input must contain columns: {required} (missing: {missing})")

    cols: List[str] = [c.strip() for c in args.columns.split(",") if c.strip()]
    for col in cols:
        if col not in df.columns:
            logging.warning("Column '%s' not in file; skipping", col)
            continue
        df[col] = df[col].astype(str).map(clean_text)

    # Ensure required order and save
    df = df[required]
    out_tsv = os.path.join(out_dir, os.path.basename(in_path).replace(".tsv", ".cleaned.tsv"))
    df.to_csv(out_tsv, sep="\t", index=False)
    logging.info("Saved cleaned TSV: %s", out_tsv)

    if args.make_zip:
        out_zip = out_tsv.replace(".tsv", ".zip")
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_tsv, arcname=os.path.basename(out_tsv))
        logging.info("Saved ZIP: %s", out_zip)


if __name__ == "__main__":
    main()

