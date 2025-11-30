import argparse
import csv
import logging
import os
import re
import zipfile
from typing import List

import pandas as pd


def remove_english_letters(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    # Drop any token that contains at least one ASCII English letter.
    # This also removes any digits/symbols attached to that token ("with it").
    tokens = re.findall(r"\S+", s)
    kept = [t for t in tokens if not re.search(r"[A-Za-z]", t)]
    return " ".join(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_tsv",
        type=str,
        default="/mnt/extremessd10tb/borisiuk/LCM/DETOX_TATAR/outputs_wordrm_gemma9b/submission_wordrm_gemma.tsv",
    )
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_wordrm_gemma9b")
    ap.add_argument(
        "--columns",
        type=str,
        default="tat_detox1",
        help="Comma-separated list of columns to strip English letters from",
    )
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not os.path.exists(args.input_tsv):
        raise FileNotFoundError(args.input_tsv)

    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    required = {"ID", "tat_toxic", "tat_detox1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain columns: {sorted(required)} (missing: {sorted(missing)})")

    cols: List[str] = [c.strip() for c in args.columns.split(",") if c.strip()]
    for col in cols:
        if col not in df.columns:
            logging.warning("Column '%s' not in file; skipping", col)
            continue
        df[col] = df[col].astype(str).map(remove_english_letters)

    # Keep only required columns and preserve order
    df = df[["ID", "tat_toxic", "tat_detox1"]]

    os.makedirs(args.output_dir, exist_ok=True)
    out_tsv = os.path.join(args.output_dir, "submission_wordrm_gemma.cleaned.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    logging.info("Saved cleaned TSV: %s", out_tsv)

    if args.make_zip:
        out_zip = os.path.join(args.output_dir, "submission_wordrm_gemma.cleaned.zip")
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_tsv, arcname=os.path.basename(out_tsv))
        logging.info("Saved ZIP: %s", out_zip)


if __name__ == "__main__":
    main()
