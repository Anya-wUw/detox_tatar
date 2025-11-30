import argparse
import csv
import logging
import os
import re
import zipfile
from contextlib import contextmanager
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

# Bind to GPU 1 unless overridden
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
    handles = []
    layers = getattr(model, "model", model).layers if hasattr(getattr(model, "model", model), "layers") else None
    if layers is None:
        raise RuntimeError("Could not locate model.layers for hook registration")
    v = (-alpha) * vector.to(next(model.parameters()).device).to(model.dtype).view(1, 1, -1)

    def _hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            return output + v
        if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            return (output[0] + v,) + tuple(output[1:])
        return output

    handles.append(layers[layer_idx].register_forward_hook(_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def normalize_single_line(text: str) -> str:
    # Replace newlines with single spaces and collapse repeated spaces
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--steering_path", type=str, default="DETOX_TATAR/steering/l21/steering_vector.pt")
    ap.add_argument("--alpha", type=float, default=0.4, help="Steering strength (can be < 1.0)")
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs_steer_l21")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--zip", dest="make_zip", action="store_true")
    # Auto-build steering vector if missing
    ap.add_argument("--auto_build", type=lambda x: str(x).lower() == "true", default=False,
                    help="If true and steering file missing, compute layer21 vector from tt split")
    ap.add_argument("--n_toxic", type=int, default=500)
    ap.add_argument("--n_nontoxic", type=int, default=500)
    ap.add_argument("--build_max_length", type=int, default=128)
    ap.add_argument("--build_batch_size", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = get_device()
    logging.info("Device: %s", device)

    # Load model
    if device.type == "cuda" and args.fp16:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    # Load or build steering vector (expects key 'vector')
    if os.path.exists(args.steering_path):
        bundle = torch.load(args.steering_path, map_location="cpu")
        vec = bundle["vector"].float()
        logging.info("Loaded steering vector from %s", args.steering_path)
    elif args.auto_build:
        logging.info("Steering file missing. Auto-building vector from textdetox tt split: toxic=%d, non-toxic=%d",
                     args.n_toxic, args.n_nontoxic)

        def _load_tt(n_tox: int, n_non: int):
            ds = load_dataset("textdetox/multilingual_toxicity_dataset")
            d = ds["tt"] if "tt" in ds else ds[next(iter(ds.keys()))]
            text_col = "text" if "text" in d.column_names else ("sentence" if "sentence" in d.column_names else None)
            label_col = "toxic" if "toxic" in d.column_names else ("label" if "label" in d.column_names else None)
            if text_col is None or label_col is None:
                raise ValueError("Dataset missing required columns")
            d = d.shuffle(seed=42)
            tox = d.filter(lambda ex: int(ex[label_col]) == 1)[text_col][:n_tox]
            non = d.filter(lambda ex: int(ex[label_col]) == 0)[text_col][:n_non]
            return tox, non

        def _last_token(hs: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
            idx = am.sum(dim=1) - 1
            idx = idx.clamp(min=0)
            return hs[torch.arange(hs.size(0), device=hs.device), idx]

        def _mean_vec(samples: List[str], lyr: int) -> torch.Tensor:
            acc = []
            total_b = (len(samples) + args.build_batch_size - 1) // args.build_batch_size
            for i in tqdm(range(0, len(samples), args.build_batch_size), total=total_b, desc=f"Build L{lyr}", unit="batch"):
                batch = samples[i:i+args.build_batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.build_max_length).to(device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                    hs = out.hidden_states[lyr]  # [B,T,H]
                    emb = _last_token(hs, enc["attention_mask"]).float().cpu()
                    acc.append(emb)
            return torch.cat(acc, dim=0).mean(dim=0)

        layer_idx = 21
        tox_texts, non_texts = _load_tt(args.n_toxic, args.n_nontoxic)
        mean_tox = _mean_vec(tox_texts, layer_idx)
        mean_non = _mean_vec(non_texts, layer_idx)
        vec = (mean_non - mean_tox).to(torch.float32)
        # Save for reuse
        os.makedirs(os.path.dirname(args.steering_path), exist_ok=True)
        torch.save({"vector": vec, "meta": {"layer_idx": layer_idx}}, args.steering_path)
        logging.info("Built and saved steering vector to %s", args.steering_path)
    else:
        raise FileNotFoundError(args.steering_path)

    layer_idx = 21  # enforce layer 21
    logging.info("Using layer %d with alpha=%.3f (vector dim=%d)", layer_idx, args.alpha, vec.numel())

    # Load input TSV
    df = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    if not {"ID", "tat_toxic"}.issubset(df.columns):
        raise ValueError("Input TSV must contain 'ID' and 'tat_toxic'")
    ids = df["ID"].tolist()
    texts: List[str] = df["tat_toxic"].astype(str).tolist()

    outputs: List[str] = []
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(0, len(texts), args.batch_size), total=total_batches, desc="Gen L21", unit="batch"):
        batch = texts[i : i + args.batch_size]
        # Keep prompt compact to avoid newlines in output; rely on instruction-only style
        prompts = [
            (
                "Токсик түгел итеп, мәгънәсен саклап яңадан яз (татарча, яңа юлларсыз): "
                + t
                + "\nЯңа вариант:"
            )
            for t in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with steering_hook(model, layer_idx, vec, args.alpha):
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
        # Take only continuation per sample
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        for j, in_len in enumerate(input_lengths):
            new_text = tok.decode(gen[j, in_len:], skip_special_tokens=True)
            outputs.append(normalize_single_line(new_text))

    # Ensure no empties
    out_texts = [o if (o and o.strip()) else t for o, t in zip(outputs, texts)]

    # Save TSV (submission format)
    os.makedirs(args.output_dir, exist_ok=True)
    tsv_path = os.path.join(args.output_dir, "submission_steer_l21.tsv")
    pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": out_texts}).to_csv(tsv_path, sep="\t", index=False)
    logging.info("Saved TSV: %s", tsv_path)

    if args.make_zip:
        zip_path = os.path.join(args.output_dir, "submission_steer_l21.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, arcname=os.path.basename(tsv_path))
        logging.info("Saved ZIP: %s", zip_path)


if __name__ == "__main__":
    main()
