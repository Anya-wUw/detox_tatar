import argparse
import csv
import json
import logging
import os
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Force single GPU 1 unless overridden
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


@contextmanager
def steering_hook(model, layer_idx: int, vector: torch.Tensor, alpha: float):
    """Register a forward hook on a transformer layer to subtract alpha*vector."""
    handles = []

    def _hook(module, inputs, output):
        hs = output
        # LlamaDecoderLayer returns hidden_states tensor [B,T,H]
        if isinstance(hs, torch.Tensor):
            v = (-alpha) * vector.to(hs.device).to(hs.dtype)
            return hs + v.view(1, 1, -1)
        # Some variants may return tuple (hidden_states, ...)
        if isinstance(hs, (tuple, list)) and isinstance(hs[0], torch.Tensor):
            t = hs[0]
            v = (-alpha) * vector.to(t.device).to(t.dtype)
            t = t + v.view(1, 1, -1)
            return (t,) + tuple(hs[1:])
        return output

    # Find layer list
    layers = getattr(model, "model", model).layers if hasattr(getattr(model, "model", model), "layers") else None
    if layers is None:
        raise RuntimeError("Could not locate model.layers for hook registration")
    layer = layers[layer_idx]
    handles.append(layer.register_forward_hook(_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--steering_path", type=str, default="DETOX_TATAR/steering/steering_vector.pt")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_steer")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--load_in_4bit", type=lambda x: str(x).lower() == "true", default=False)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    logging.info("Device: %s", device)

    bundle = torch.load(args.steering_path, map_location="cpu")
    vec = bundle["vector"].float()
    layer_idx = int(bundle["meta"]["layer_idx"]) if "meta" in bundle and "layer_idx" in bundle["meta"] else 16
    logging.info("Loaded steering vector dim=%d at layer %d", vec.numel(), layer_idx)

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
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    if not {"ID", "tat_toxic"}.issubset(df.columns):
        raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")

    ids = df["ID"].tolist()
    texts = df["tat_toxic"].astype(str).tolist()

    outputs = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        # Simple Tatar detox instruction (keeps original language)
        prompts = [
            f"Яңача яз: мәгънәсен сакла, ләкин җөмләнең агрессиясен бетер (татарча):\n{text}\nЯңа вариант:"
            for text in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with steering_hook(model, layer_idx, vec, args.alpha):
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
        out = tok.batch_decode(gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)
        outputs.extend(out)

    # Fallback to original if any empty
    out_texts = [o if (o is not None and str(o).strip() != "") else t for o, t in zip(outputs, texts)]

    out_path = os.path.join(args.output_dir, "submission_steer.tsv")
    pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": out_texts}).to_csv(out_path, sep="\t", index=False)
    logging.info("Saved steered submission TSV: %s", out_path)


if __name__ == "__main__":
    main()
