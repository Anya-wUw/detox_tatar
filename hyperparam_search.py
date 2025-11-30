import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_submission import ToxicityClassifier, SimilarityLaBSE, CometFluency

# Force single GPU 1 unless overridden
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_tt_splits(n_toxic: int, n_nontoxic: int, n_val: int, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    ds = load_dataset("textdetox/multilingual_toxicity_dataset")
    if "tt" in ds:
        d = ds["tt"]
    else:
        first_key = next(iter(ds.keys()))
        base = ds[first_key]
        d = base.filter(lambda ex: ex.get("lang") == "tt") if "lang" in base.column_names else base
    text_col = "text" if "text" in d.column_names else ("sentence" if "sentence" in d.column_names else None)
    label_col = "toxic" if "toxic" in d.column_names else ("label" if "label" in d.column_names else None)
    if text_col is None or label_col is None:
        raise ValueError("Dataset must have 'text'/'sentence' and 'toxic'/'label' columns")
    d = d.shuffle(seed=seed)
    toxic = d.filter(lambda ex: int(ex[label_col]) == 1)
    nontoxic = d.filter(lambda ex: int(ex[label_col]) == 0)
    train_toxic = toxic[text_col][:n_toxic]
    train_nontoxic = nontoxic[text_col][:n_nontoxic]
    # Validation: next n_val from remaining pool (mix)
    remaining_texts = d[text_col][n_toxic + n_nontoxic : n_toxic + n_nontoxic + n_val]
    return train_toxic, train_nontoxic, remaining_texts


def last_token_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    idxs = attention_mask.sum(dim=1) - 1
    idxs = idxs.clamp(min=0)
    gathered = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), idxs]
    return gathered


def layer_means(
    model, tokenizer, texts: List[str], device: torch.device, max_length: int = 128, batch_size: int = 4
) -> torch.Tensor:
    """Return mean last-token embedding per layer: [num_layers, hidden]."""
    num_layers = model.config.num_hidden_layers
    acc = None
    count = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode for means", unit="batch"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            # hidden_states includes embeddings + each layer; take 1..num_layers inclusive
            hs_list = out.hidden_states[1 : num_layers + 1]
            # For each layer, take last-token embedding and stack
            last_list = [last_token_embeddings(hs, enc["attention_mask"]) for hs in hs_list]
            # [num_layers, B, H] -> [num_layers, H] sum
            stacked = torch.stack(last_list, dim=0).sum(dim=1)  # sum over batch
            acc = stacked if acc is None else acc + stacked
            count += enc["input_ids"].size(0)
    means = acc / max(count, 1)
    return means  # [num_layers, H]


@dataclass
class Candidate:
    layer: int
    alpha: float
    sta: float
    sim: float
    score: float
    out_dir: str


def generate_with_steering(
    model, tokenizer, layer: int, vector: torch.Tensor, alpha: float, texts: List[str], device: torch.device,
    batch_size: int = 4, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.95
) -> List[str]:
    # Register simple forward hook on decoder layer
    handles = []
    layers = getattr(model, "model", model).layers if hasattr(getattr(model, "model", model), "layers") else None
    if layers is None:
        raise RuntimeError("Could not locate model.layers for hook registration")
    v = (-alpha) * vector.to(device).to(model.dtype).view(1, 1, -1)

    def _hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            return output + v
        if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            return (output[0] + v,) + tuple(output[1:])
        return output

    handles.append(layers[layer].register_forward_hook(_hook))

    outputs = []
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Gen L{layer} a{alpha}", unit="batch"):
            batch = texts[i : i + batch_size]
            prompts = [
                f"Яңача яз: мәгънәсен сакла, ләкин җөмләнең агрессиясен бетер (татарча):\n{t}\nЯңа вариант:"
                for t in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            out = tokenizer.batch_decode(gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)
            outputs.extend(out)
    finally:
        for h in handles:
            h.remove()
    return [o if (o and str(o).strip()) else t for o, t in zip(outputs, texts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--layers", type=str, default="21-31", help="Layer range inclusive, e.g., 21-31")
    ap.add_argument("--alphas", type=str, default="1.0,2.0,3.0")
    ap.add_argument("--n_toxic", type=int, default=500)
    ap.add_argument("--n_nontoxic", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out_dir", type=str, default="DETOX_TATAR/search")
    ap.add_argument("--dev_inputs", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--compute_comet_topk", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--topk", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = get_device()
    logging.info("Device: %s", device)

    # Parse layers / alphas
    lstart, lend = [int(x) for x in args.layers.split("-")]
    layers = list(range(lstart, lend + 1))
    alphas = [float(x) for x in args.alphas.split(",")]

    # Model
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

    # Data
    train_toxic, train_nontoxic, val_texts = load_tt_splits(args.n_toxic, args.n_nontoxic, args.n_val)
    logging.info("Train toxic=%d, non-toxic=%d, val=%d", len(train_toxic), len(train_nontoxic), len(val_texts))

    # Compute per-layer means in one pass per class
    logging.info("Encoding toxic means per layer...")
    means_toxic = layer_means(model, tok, train_toxic, device, batch_size=args.batch_size)
    logging.info("Encoding non-toxic means per layer...")
    means_non = layer_means(model, tok, train_nontoxic, device, batch_size=args.batch_size)
    # Steering vectors per layer
    steering = (means_non - means_toxic).to(torch.float32)  # [num_layers, H]

    # Metrics helpers
    tox_clf = ToxicityClassifier(device)
    sim_labse = SimilarityLaBSE(device)

    candidates: List[Candidate] = []
    for layer in layers:
        vec = steering[layer].detach()
        for alpha in alphas:
            out_dir = os.path.join(args.out_dir, f"L{layer}_a{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            # Generate on validation texts
            gens = generate_with_steering(
                model, tok, layer, vec, alpha, val_texts, device,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            # Metrics on validation
            p_after = tox_clf.score(gens, batch_size=32)
            sta = float((p_after < 0.5).mean())
            sim = float(sim_labse.cosine(val_texts, gens).mean().item())
            score = sta + 0.5 * sim
            candidates.append(Candidate(layer=layer, alpha=alpha, sta=sta, sim=sim, score=score, out_dir=out_dir))
            # Save quick preview CSV
            pd.DataFrame({"text": val_texts, "gen": gens, "tox_after": p_after}).to_csv(
                os.path.join(out_dir, "val_preview.csv"), index=False
            )
            with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
                json.dump({"sta": sta, "sim": sim, "score": score}, f, indent=2)

    # Pick top-k
    candidates.sort(key=lambda c: c.score, reverse=True)
    topk = candidates[: args.topk]
    logging.info("Top-%d candidates: %s", args.topk, [(c.layer, c.alpha, round(c.score, 4)) for c in topk])
    with open(os.path.join(args.out_dir, "summary_topk.json"), "w") as f:
        json.dump([c.__dict__ for c in topk], f, indent=2)

    # Validate top-k on dev_inputs.tsv and write TSVs
    df_dev = pd.read_csv(args.dev_inputs, sep="\t", quoting=csv.QUOTE_NONE)
    ids = df_dev["ID"].tolist()
    texts = df_dev["tat_toxic"].astype(str).tolist()

    comet = None
    if args.compute_comet_topk:
        try:
            comet = CometFluency(device)
        except Exception as e:
            logging.warning("COMET unavailable: %s", e)
            comet = None

    results = []
    for c in topk:
        vec = steering[c.layer].detach()
        gens = generate_with_steering(
            model, tok, c.layer, vec, c.alpha, texts, device,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )
        # Save submission TSV
        sub_dir = os.path.join(args.out_dir, f"dev_L{c.layer}_a{c.alpha}")
        os.makedirs(sub_dir, exist_ok=True)
        tsv_path = os.path.join(sub_dir, "submission_steer.tsv")
        pd.DataFrame({"ID": ids, "tat_toxic": texts, "tat_detox1": gens}).to_csv(tsv_path, sep="\t", index=False)

        # Metrics
        p_after = tox_clf.score(gens, batch_size=32)
        sta = float((p_after < 0.5).mean())
        sim = float(sim_labse.cosine(texts, gens).mean().item())
        flu = None
        if comet is not None:
            try:
                scores = comet.score(src=texts, mt=gens, ref=None, batch_size=16)
                flu = float(np.mean(scores))
            except Exception as e:
                logging.warning("COMET scoring failed: %s", e)
        with open(os.path.join(sub_dir, "dev_metrics.json"), "w") as f:
            json.dump({"sta": sta, "sim": sim, "fluency_comet": flu}, f, indent=2)
        results.append({"layer": c.layer, "alpha": c.alpha, "sta": sta, "sim": sim, "flu": flu, "tsv": tsv_path})

    with open(os.path.join(args.out_dir, "dev_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Search complete. Results saved under %s", args.out_dir)


if __name__ == "__main__":
    main()
