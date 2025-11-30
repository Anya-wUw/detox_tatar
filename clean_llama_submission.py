#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import zipfile


INPUT_PATH = "/mnt/extremessd10tb/borisiuk/LCM/DETOX_TATAR/outputs_llama/submission_llama.tsv"
OUTPUT_TSV = "/mnt/extremessd10tb/borisiuk/LCM/DETOX_TATAR/outputs_llama/submission_llama_clean.tsv"
OUTPUT_ZIP = "/mnt/extremessd10tb/borisiuk/LCM/DETOX_TATAR/outputs_llama/submission_llama_clean.zip"


def clean_detox(text: str) -> str:
    if text is None:
        return ""

    # убираем внешние кавычки, если они остались от TSV
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # разрезаем по строкам и чистим пробелы
    lines = [l.strip() for l in text.splitlines()]

    # убираем совсем пустые строки
    lines = [l for l in lines if l]

    if not lines:
        return ""

    # ищем последнюю строку вида "Detoxed:..."
    last_detox_idx = -1
    for i, l in enumerate(lines):
        if l.startswith("Detoxed:"):
            last_detox_idx = i

    candidate = ""

    if last_detox_idx != -1:
        # берем первую непустую строку после "Detoxed:" которая не выглядит как перевод в скобках
        for j in range(last_detox_idx + 1, len(lines)):
            l = lines[j]
            if not l:
                continue
            if l.startswith("(") and "Translation" in l:
                continue
            candidate = l
            break

        # если ничего не нашли, пробуем саму строку Detoxed
        if not candidate:
            candidate = lines[last_detox_idx]
    else:
        # запасной вариант берем последнюю непустую строку
        candidate = lines[-1]

    # убираем префикс "Detoxed:" если он остался
    if candidate.startswith("Detoxed:assistant"):
        candidate = candidate[len("Detoxed:assistant"):].strip()
    if candidate.startswith("Detoxed:"):
        candidate = candidate[len("Detoxed:"):].strip()

    # убираем финальную кавычку если вдруг осталась
    if candidate.endswith('"'):
        candidate = candidate[:-1].rstrip()

    return candidate


def main():
    fieldnames = ["ID", "tat_toxic", "tat_detox1"]

    cleaned_rows = []

    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            # пропускаем мусорные строки без нужных колонок
            if not all(col in row for col in fieldnames):
                continue

            detox_raw = row.get("tat_detox1", "")
            detox_clean = clean_detox(detox_raw)

            cleaned_rows.append(
                {
                    "ID": row.get("ID", "").strip(),
                    "tat_toxic": row.get("tat_toxic", "").strip(),
                    "tat_detox1": detox_clean,
                }
            )

    # сохраняем очищенный tsv
    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=fieldnames,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(cleaned_rows)

    # упаковываем в zip
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_TSV, arcname=os.path.basename(OUTPUT_TSV))

    print(f"Saved cleaned TSV to: {OUTPUT_TSV}")
    print(f"Saved zip archive to: {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
