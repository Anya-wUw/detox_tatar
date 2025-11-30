import argparse
import json
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Force single GPU 1 unless overridden by shell
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_tt_texts(n_toxic: int, n_nontoxic: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    ds = load_dataset("textdetox/multilingual_toxicity_dataset")
    if "tt" in ds:
        d = ds["tt"]
    else:
        # Fallback: filter by language column
        first_key = next(iter(ds.keys()))
        base = ds[first_key]
        if "lang" in base.column_names:
            d = base.filter(lambda ex: ex.get("lang") == "tt")
        else:
            d = base
    text_col = "text" if "text" in d.column_names else ("sentence" if "sentence" in d.column_names else None)
    if text_col is None:
        raise ValueError("Expected 'text' or 'sentence' column in dataset")
    label_col = "toxic" if "toxic" in d.column_names else ("label" if "label" in d.column_names else None)
    if label_col is None:
        raise ValueError("Expected 'toxic' or 'label' column in dataset")

    d = d.shuffle(seed=seed)
    toxic = d.filter(lambda ex: int(ex[label_col]) == 1)
    nontoxic = d.filter(lambda ex: int(ex[label_col]) == 0)
    toxic_texts = toxic[text_col][:n_toxic]
    nontoxic_texts = nontoxic[text_col][:n_nontoxic]
    return toxic_texts, nontoxic_texts


def last_token_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, T, H], attention_mask: [B, T]
    idxs = attention_mask.sum(dim=1) - 1
    idxs = idxs.clamp(min=0)
    gathered = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), idxs]
    return gathered  # [B, H]


def compute_steering_vector(
    model, tokenizer, texts: List[str], layer_idx: int, device: torch.device, max_length: int = 128, batch_size: int = 4
) -> torch.Tensor:
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Hidden @layer{layer_idx}", unit="batch"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx]  # [B, T, H]
            emb = last_token_embeddings(hs, enc["attention_mask"])  # [B, H]
            embs.append(emb.detach().float().cpu())
    return torch.cat(embs, dim=0).mean(dim=0)  # [H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--layer_idx", type=int, default=16, help="Which hidden layer to extract (0=input, -1=last)")
    ap.add_argument("--n_toxic", type=int, default=500)
    ap.add_argument("--n_nontoxic", type=int, default=500)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="DETOX_TATAR/steering")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_in_4bit", type=lambda x: str(x).lower() == "true", default=False)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    logging.info("Device: %s", device)

    logging.info("Loading model: %s", args.model_id)
    load_kwargs = {}
    if args.load_in_4bit:
        try:
            load_kwargs = dict(load_in_4bit=True, device_map="auto")
        except Exception:
            load_kwargs = {}
    if args.load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            load_in_4bit=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16 if device.type == "cuda" else None,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    logging.info("Loading dataset and sampling toxic/non-toxic")
    toxic_texts, nontoxic_texts = load_tt_texts(args.n_toxic, args.n_nontoxic, seed=args.seed)
    logging.info("Sampled %d toxic and %d non-toxic", len(toxic_texts), len(nontoxic_texts))

    logging.info("Computing mean embeddings for toxic")
    mean_toxic = compute_steering_vector(
        model, tokenizer, toxic_texts, args.layer_idx, device, max_length=args.max_length, batch_size=args.batch_size
    )
    logging.info("Computing mean embeddings for non-toxic")
    mean_nontoxic = compute_steering_vector(
        model, tokenizer, nontoxic_texts, args.layer_idx, device, max_length=args.max_length, batch_size=args.batch_size
    )

    steering = (mean_nontoxic - mean_toxic).to(torch.float32)
    meta = {
        "model_id": args.model_id,
        "layer_idx": args.layer_idx,
        "n_toxic": len(toxic_texts),
        "n_nontoxic": len(nontoxic_texts),
        "max_length": args.max_length,
    }
    torch.save({"vector": steering, "meta": meta}, os.path.join(args.out_dir, "steering_vector.pt"))
    with open(os.path.join(args.out_dir, "steering_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logging.info("Saved steering vector to %s", args.out_dir)


if __name__ == "__main__":
    main()
