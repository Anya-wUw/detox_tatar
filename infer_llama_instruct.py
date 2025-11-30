# import argparse
# import csv
# import logging
# import os
# import zipfile
# from typing import List

# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer


# # Bind to GPU 1 by default; can be overridden from shell
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


# INSTRUCT_PROMPT = (
#     "You are an expert in Tatar text detoxification. Rewrite toxic or offensive text so that:\n"
#     "1) The main meaning, intent, and factual content are preserved.\n"
#     "2) All explicit insults, slurs, profanities, degrading expressions, hate speech, and passive-aggressive tone are removed.\n"
#     "3) The rewritten sentence must remain natural, fluent, and stylistically close to the original.\n"
#     "4) Do NOT change the topic, do NOT invent facts, do NOT distort events. Only soften toxic phrasing.\n"
#     "5) Preserve as much semantic information as possible.\n\n"
#     "Below are examples of different types of toxicity and how to rewrite them correctly.\n\n"
#     "### Examples\n\n"
#     "#### 1. Direct insult (explicit slur)\n"
#     "Toxic:\nСин нинди тинтәк кеше!\n"
#     "Detoxed:\nСин ялгышасың!\n\n"
#     "#### 2. Profanity + strong negative emotion\n"
#     "Toxic:\nУтырасың да күңел ачасың, ахмак!!!!! сүзләр юк!\n"
#     "Detoxed:\nУтырасың да күңел ачасың!!!!! сүзләр юк!\n\n"
#     "#### 3. Dehumanization / humiliation\n"
#     "Toxic:\nАндый хайваннарны кабер генә төзәтә!\n"
#     "Detoxed:\nАндый кешеләрне бернәрсә дә үзгәртми!\n\n"
#     "#### 4. Passive-aggressive, condescending tone\n"
#     "Toxic:\nСинең белән сөйләшеп торуның мәгънәсе юк инде.\n"
#     "Detoxed:\nБу теманы дәвам итү файдалы түгел.\n\n"
#     "#### 5. Group insult / bias\n"
#     "Toxic:\nАлар барысы да пидораслар!\n"
#     "Detoxed:\nАлар турында ямьсез сүзләр әйтү дөрес түгел.\n\n"
#     "#### 6. Aggressive profanity embedded into a meaningful sentence\n"
#     "Toxic:\nБу хуеплетны стадионнан куып чыгарырга кирәк иде.\n"
#     "Detoxed:\nАны стадионнан чыгару кирәк иде.\n\n"
#     "#### 7. Character attack\n"
#     "Toxic:\nСин бит бөтенләй башсыз!\n"
#     "Detoxed:\nСин моны дөрес аңламагансың.\n\n"
#     "#### 8. Sarcasm / contempt\n"
#     "Toxic:\nАфәрин, тагын бер зирәк фикер таптың инде.\n"
#     "Detoxed:\nСин тагын бер фикер әйттең.\n\n"
#     "---\n\n"
#     "### Task\n\n"
#     "Toxic: {text}\n"
#     "Detoxed:"
# )


# def get_device() -> torch.device:
#     if torch.cuda.is_available():
#         try:
#             torch.cuda.set_device(0)
#         except Exception:
#             pass
#         return torch.device("cuda:0")
#     return torch.device("cpu")


# def build_inputs(tokenizer: AutoTokenizer, text_batch: List[str]) -> dict:
#     """Use LLaMA chat template with a system + user message."""
#     messages = []
#     for t in text_batch:
#         sys_msg = {"role": "system", "content": "Follow the instructions and examples carefully."}
#         usr_msg = {"role": "user", "content": INSTRUCT_PROMPT.format(text=t)}
#         messages.append([sys_msg, usr_msg])
#     rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     enc = tokenizer(
#         rendered,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=2048,
#         padding_side='left',
#     )
#     return enc


# def extract_detoxed(prompted_text: str, generated_text: str) -> str:
#     """Try to extract only the detoxed continuation after the last 'Detoxed:' marker."""
#     # The generated_text contains the full assistant response. We try to cut at the prompt end.
#     marker = "Detoxed:"
#     if marker in generated_text:
#         out = generated_text.split(marker, 1)[-1].strip()
#         return out
#     return generated_text.strip()


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
#     ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
#     ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_llama")
#     ap.add_argument("--zip", dest="make_zip", action="store_true")
#     ap.add_argument("--batch_size", type=int, default=2)
#     ap.add_argument("--max_new_tokens", type=int, default=128)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_p", type=float, default=0.95)
#     ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
#     args = ap.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
#     logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
#     device = get_device()
#     logging.info("Device: %s", device)

#     # Load model/tokenizer
#     if device.type == "cuda" and args.fp16:
#         model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
#     tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#     if getattr(model.config, "pad_token_id", None) is None:
#         model.config.pad_token_id = tok.pad_token_id

#     # Read input TSV
#     df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
#     if not {"ID", "tat_toxic"}.issubset(df.columns):
#         raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")
#     ids = df["ID"].tolist()
#     texts = df["tat_toxic"].astype(str).tolist()

#     generations: List[str] = []
#     total = (len(texts) + args.batch_size - 1) // args.batch_size
#     for i in tqdm(range(0, len(texts), args.batch_size), total=total, desc="LLaMA detox", unit="batch"):
#         batch = texts[i : i + args.batch_size]
#         enc = build_inputs(tok, batch)
#         enc = {k: v.to(device) for k, v in enc.items()}
#         with torch.no_grad():
#             gen = model.generate(
#                 **enc,
#                 do_sample=True,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 max_new_tokens=args.max_new_tokens,
#                 pad_token_id=tok.pad_token_id,
#                 eos_token_id=tok.eos_token_id,
#             )
#         # Decode only newly generated tokens, trimming the prompt per-sample
#         input_lengths = enc["attention_mask"].sum(dim=1).tolist()
#         new_texts: List[str] = []
#         for j, in_len in enumerate(input_lengths):
#             new_tokens = gen[j, in_len:]
#             new_texts.append(tok.decode(new_tokens, skip_special_tokens=True))
#         # Extract only the answer after 'Detoxed:' (if echoed)
#         cleaned = [extract_detoxed("", t) for t in new_texts]
#         generations.extend(cleaned)

#     # Fallback to original if any empty
#     final_out = [g if (g and str(g).strip()) else t for g, t in zip(generations, texts)]

#     # Save TSV
#     tsv_path = os.path.join(args.output_dir, "submission_llama.tsv")
#     # Retain only required columns
#     pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": final_out}).to_csv(tsv_path, sep="\t", index=False)
#     logging.info("Saved TSV: %s", tsv_path)

#     # Optional ZIP
#     if args.make_zip:
#         zip_path = os.path.join(args.output_dir, "submission_llama.zip")
#         with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
#             zf.write(tsv_path, arcname=os.path.basename(tsv_path))
#         logging.info("Saved ZIP: %s", zip_path)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import zipfile
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# По умолчанию используем GPU 1, можно переопределить снаружи
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


INSTRUCT_PROMPT = ("""
Син татар текстларын детоксикацияләү буенча эксперт. Түбәндәге таләпләр буенча токсик яки мыскыллы текстны йомшартылган, әдәпле итеп яңадан яз:
1) Төп мәгънә, ният һәм фактик мәгълүмат сакланырга тиеш.
2) Барлык ачык мыскыллау, сүгенү сүзләре, дошманлык белдерүләре, кимсетү, нәфрәт тудыру, пассив-агрессив интонация тулысынча бетерелергә тиеш.
3) Яңадан язылган җөмлә табигый, йөгерек һәм оригиналга стиль ягыннан якын булырга тиеш.
4) Тема үзгәртмә, фактлар уйлап чыгарма, вакыйгаларны бозма. Бурыч — бары тик токсик өлешне йомшарту.
5) Мәгънәви мәгълүматны мөмкин кадәр күбрәк сакларга кирәк.

Түбәндә төрле токсиклык төрләренә мисаллар һәм аларны дөрес итеп үзгәртү үрнәкләре бирелә.

### Мисаллар

#### 1. Турыдан-туры мыскыллау (ачык сүгенү)
Toxic:
Син нинди тинтәк кеше!
Detoxed:
Син ялгышасың!

#### 2. Сүгенү + көчле тискәре эмоция
Toxic:
Утырасың да күңел ачасың, ахмак!!!!! сүзләр юк!
Detoxed:
Утырасың да күңел ачасың!!!!! сүзләр юк!

#### 3. Кешене хайванга тиңләү / кимсетү
Toxic:
Андый хайваннарны кабер генә төзәтә!
Detoxed:
Андый кешеләрне бернәрсә дә үзгәртми!

#### 4. Пассив-агрессив, өстән карап сөйләү
Toxic:
Синең белән сөйләшеп торуның мәгънәсе юк инде.
Detoxed:
Бу теманы дәвам итү файдалы түгел.

#### 5. Группага карата мыскыллау / нәфрәт белдерү
Toxic:
Алар барысы да пидораслар!
Detoxed:
Алар турында ямьсез сүзләр әйтү дөрес түгел.

#### 6. Мәгънәле җөмлә эчендә агрессив сүгенү
Toxic:
Бу хуеплетны стадионнан куып чыгарырга кирәк иде.
Detoxed:
Аны стадионнан чыгару кирәк иде.

#### 7. Шәхескә һөҗүм
Toxic:
Син бит бөтенләй башсыз!
Detoxed:
Син моны дөрес аңламагансың.

#### 8. Сарказм / кимсетеп мыскыллау
Toxic:
Афәрин, тагын бер зирәк фикер таптың инде.
Detoxed:
Син тагын бер фикер әйттең.

---

### Бирем

Toxic: {text}

Бу токсик текстны йомшартылган, әдәпле татар җөмләсенә үзгәрт.

Чыгыш таләпләре:
1) Җавапта бары тик детоксикацияләнгән татар җөмләсе генә булсын.
2) Английча текст, ярлыклар, аңлатмалар, башлыклар булмаска тиеш.
3) «Toxic», «Detoxed», «assistant» кебек сүзләр кулланма.
4) Җавап нәкъ бер юлдан торсын.

Җавап:\n"""

    # "You are an expert in Tatar text detoxification. Rewrite toxic or offensive text so that:\n"
    # "1) The main meaning, intent, and factual content are preserved.\n"
    # "2) All explicit insults, slurs, profanities, degrading expressions, hate speech, and passive-aggressive tone are removed.\n"
    # "3) The rewritten sentence must remain natural, fluent, and stylistically close to the original.\n"
    # "4) Do NOT change the topic, do NOT invent facts, do NOT distort events. Only soften toxic phrasing.\n"
    # "5) Preserve as much semantic information as possible.\n\n"
    # "Below are examples of different types of toxicity and how to rewrite them correctly.\n\n"
    # "### Examples\n\n"
    # "#### 1. Direct insult (explicit slur)\n"
    # "Toxic:\nСин нинди тинтәк кеше!\n"
    # "Detoxed:\nСин ялгышасың!\n\n"
    # "#### 2. Profanity + strong negative emotion\n"
    # "Toxic:\nУтырасың да күңел ачасың, ахмак!!!!! сүзләр юк!\n"
    # "Detoxed:\nУтырасың да күңел ачасың!!!!! сүзләр юк!\n\n"
    # "#### 3. Dehumanization / humiliation\n"
    # "Toxic:\nАндый хайваннарны кабер генә төзәтә!\n"
    # "Detoxed:\nАндый кешеләрне бернәрсә дә үзгәртми!\n\n"
    # "#### 4. Passive-aggressive, condescending tone\n"
    # "Toxic:\nСинең белән сөйләшеп торуның мәгънәсе юк инде.\n"
    # "Detoxed:\nБу теманы дәвам итү файдалы түгел.\n\n"
    # "#### 5. Group insult / bias\n"
    # "Toxic:\nАлар барысы да пидораслар!\n"
    # "Detoxed:\nАлар турында ямьсез сүзләр әйтү дөрес түгел.\n\n"
    # "#### 6. Aggressive profanity embedded into a meaningful sentence\n"
    # "Toxic:\nБу хуеплетны стадионнан куып чыгарырга кирәк иде.\n"
    # "Detoxed:\nАны стадионнан чыгару кирәк иде.\n\n"
    # "#### 7. Character attack\n"
    # "Toxic:\nСин бит бөтенләй башсыз!\n"
    # "Detoxed:\nСин моны дөрес аңламагансың.\n\n"
    # "#### 8. Sarcasm / contempt\n"
    # "Toxic:\nАфәрин, тагын бер зирәк фикер таптың инде.\n"
    # "Detoxed:\nСин тагын бер фикер әйттең.\n\n"
    # "---\n\n"
    # "### Task\n\n"
    # "Toxic: {text}\n\n"
    # "Rewrite the toxic text into a non toxic Tatar sentence.\n\n"
    # "Output requirements:\n"
    # "1) Answer only with the detoxified Tatar sentence.\n"
    # "2) Do not include English text, labels, headings, or explanations.\n"
    # "3) Do not include words like 'Toxic', 'Detoxed', 'assistant'.\n"
    # "4) Output exactly one line of Tatar text.\n\n"
    # "Answer:\n"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_inputs(
    tokenizer: AutoTokenizer,
    text_batch: List[str],
    max_input_tokens: int,
) -> dict:
    """
    Собирает батч сообщений, рендерит через chat template в строки,
    а потом токенизирует, чтобы получить dict с input_ids и attention_mask.
    """
    messages_batch = []
    for t in text_batch:
        sys_msg = {
            "role": "system",
            "content": "Follow the instructions and examples carefully.",
        }
        usr_msg = {
            "role": "user",
            "content": INSTRUCT_PROMPT.format(text=t),
        }
        messages_batch.append([sys_msg, usr_msg])

    tokenizer.padding_side = "left"

    # получаем список строк по батчу
    rendered = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    # токенизируем в dict
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    return enc


def extract_detoxed(generated_text: str) -> str:
    """
    Оставляет ровно одну строку с детоксифицированным текстом.
    Убирает префиксы и кавычки, режет по первой строке.
    """
    if not generated_text:
        return ""

    text = generated_text.strip()

    # берем только первую строку, остальное игнорируем
    first_line = text.splitlines()[0].strip()

    # убираем возможные префиксы
    prefixes = [
        "assistant:",
        "assistant",
        "Assistant:",
        "Assistant",
        "Detoxed:",
        "Detoxed",
        "Answer:",
        "Answer",
        "Җавап",
    ]
    lowered = first_line.lower()
    for pref in prefixes:
        p_low = pref.lower()
        if lowered.startswith(p_low):
            first_line = first_line[len(pref):].lstrip(":").strip()
            break

    # убираем внешние кавычки
    if first_line.startswith('"') and first_line.endswith('"') and len(first_line) > 1:
        first_line = first_line[1:-1].strip()

    return first_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_llama")
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.set_defaults(make_zip=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    device = get_device()
    logging.info("Device: %s", device)

    # Загружаем модель и токенизатор
    if device.type == "cuda" and args.fp16:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    # Читаем входной TSV
    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    required_cols = {"ID", "tat_toxic"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input TSV must contain columns: {required_cols}")
    ids = df["ID"].tolist()
    texts = df["tat_toxic"].astype(str).tolist()

    generations: List[str] = []

    total = (len(texts) + args.batch_size - 1) // args.batch_size
    for i in tqdm(
        range(0, len(texts), args.batch_size),
        total=total,
        desc="LLaMA detox",
        unit="batch",
    ):
        batch = texts[i : i + args.batch_size]
        enc = build_inputs(tok, batch, max_input_tokens=args.max_input_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature if args.temperature > 0.0 else 1.0,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # длины входов по attention_mask
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        new_texts: List[str] = []
        for j, in_len in enumerate(input_lengths):
            new_tokens = gen[j, in_len:]
            decoded = tok.decode(new_tokens, skip_special_tokens=True)
            new_texts.append(decoded)

        cleaned_batch = [extract_detoxed(t) for t in new_texts]
        generations.extend(cleaned_batch)

    # Фолбэк: если модель вернула пустую строку, оставляем исходный текст
    final_out = [
        g if (g and str(g).strip()) else t
        for g, t in zip(generations, texts)
    ]

    # Сохраняем TSV в формате сабмишна
    tsv_path = os.path.join(args.output_dir, "submission_llama.tsv")
    out_df = pd.DataFrame(
        {
            "ID": ids,
            "tat_toxic": texts,
            "tat_detox1": final_out,
        }
    )
    out_df.to_csv(tsv_path, sep="\t", index=False)
    logging.info("Saved TSV: %s", tsv_path)

    # Упаковываем в zip
    if args.make_zip:
        zip_path = os.path.join(args.output_dir, "submission_llama.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, arcname=os.path.basename(tsv_path))
        logging.info("Saved ZIP: %s", zip_path)


if __name__ == "__main__":
    main()
