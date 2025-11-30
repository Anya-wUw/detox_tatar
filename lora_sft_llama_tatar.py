import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Optional: PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

# Bind to GPU 1 by default
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_tt_non_toxic(limit: Optional[int] = 2500, seed: int = 42) -> List[str]:
    ds = load_dataset("textdetox/multilingual_toxicity_dataset")
    d = ds["tt"] if "tt" in ds else ds[next(iter(ds.keys()))]
    text_col = "text" if "text" in d.column_names else ("sentence" if "sentence" in d.column_names else None)
    label_col = "toxic" if "toxic" in d.column_names else ("label" if "label" in d.column_names else None)
    if text_col is None or label_col is None:
        raise ValueError("Dataset must have 'text'/'sentence' and 'toxic'/'label'")
    d = d.filter(lambda ex: int(ex[label_col]) == 0)
    d = d.shuffle(seed=seed)
    texts = d[text_col]
    if limit is not None:
        texts = texts[:limit]
    return [str(t) for t in texts]


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        # Return Python lists; collator will pad and create labels for causal LM
        enc = self.tok(
            txt,
            truncation=True,
            max_length=self.max_length,
        )
        return enc


def build_model_and_tokenizer(model_id: str, dtype: Optional[torch.dtype], device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    return model, tok


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is not installed. Please: pip install peft")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


@torch.no_grad()
def perplexity(model, tok, texts: List[str], max_length: int = 1024, batch_size: int = 4, device: Optional[torch.device] = None) -> float:
    if device is None:
        device = next(model.parameters()).device
    total_loglik = 0.0
    total_tokens = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="PPL", unit="batch"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        # Mask padding tokens
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        # Sum token-level losses: loss is mean over non-masked tokens -> recover sum
        # Use logits to compute per-token losses to be precise
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # Flatten
        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        mask = (shift_labels.view(-1) != -100).float()
        token_loss_sum = (loss * mask).sum().item()
        token_count = mask.sum().item()
        total_loglik += token_loss_sum
        total_tokens += token_count
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loglik / total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/llama_tatar_lora")
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_4bit", type=lambda x: str(x).lower()=="true", default=False)
    ap.add_argument("--eval_tsv", type=str, default="DETOX_TATAR/data/test_inputs.tsv")
    ap.add_argument("--log_level", type=str, default="INFO")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    device = get_device()
    logging.info("Device: %s", device)

    # Load model+tokenizer
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    model, tok = build_model_and_tokenizer(args.model_id, dtype, device)

    # Attach LoRA
    model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.print_trainable_parameters()

    # Data
    texts = load_tt_non_toxic(limit=args.limit)
    train_ds = TextDataset(texts, tok, max_length=args.max_length)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    # Save adapter
    try:
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        tok.save_pretrained(args.output_dir)
    except Exception:
        pass

    # Perplexity on test set (tat_toxic column)
    df = pd.read_csv(args.eval_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    eval_texts = df["tat_toxic"].astype(str).tolist()
    ppl = perplexity(model, tok, eval_texts, max_length=args.max_length, batch_size=max(1, args.per_device_train_batch_size))
    with open(os.path.join(args.output_dir, "eval_ppl.json"), "w", encoding="utf-8") as f:
        json.dump({"perplexity": ppl}, f, ensure_ascii=False, indent=2)
    logging.info("Perplexity on test_inputs.tsv (tat_toxic): %.3f", ppl)


if __name__ == "__main__":
    main()
