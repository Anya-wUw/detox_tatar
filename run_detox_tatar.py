"""
Pipline 1: 
    1. Translate tt-> en
    2. Detox model
    3. Translate en-> tt
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Force using only GPU device 1 (maps to cuda:0 inside the process)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from tqdm import tqdm
import logging


# ------------------------
# Utilities
# ------------------------


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


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


@dataclass
class Metrics:
    sta: float  # style transfer accuracy (non-toxic rate after)
    sim: float  # cosine similarity
    fl: float   # pseudo-perplexity (lower is better)
    clf_acc: Optional[float] = None  # classifier acc vs dataset label


# ------------------------
# Toxicity classifier (textdetox/bert-multilingual-toxicity-classifier)
# ------------------------


class ToxicityClassifier:
    def __init__(self, device: torch.device, token_max_length: int = 512):
        self.tok = AutoTokenizer.from_pretrained(
            "textdetox/bert-multilingual-toxicity-classifier"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "textdetox/bert-multilingual-toxicity-classifier"
        ).to(device)
        self.model.eval()
        self.device = device
        self.token_max_length = token_max_length
        logging.info("Classifier device: %s", next(self.model.parameters()).device)

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Return probability of toxic class (index 1)."""
        probs = []
        for batch in batched(texts, batch_size):
            enc = self.tok(batch, padding=True, truncation=True, return_tensors="pt").to(
                self.device
            )
            # ensure explicit truncation length to avoid warnings
            if enc["input_ids"].size(1) > self.token_max_length:
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.token_max_length,
                    return_tensors="pt",
                ).to(self.device)
            out = self.model(**enc).logits
            p = torch.softmax(out, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs) if probs else np.array([])


# ------------------------
# Translation using NLLB-200 (facebook/nllb-200-distilled-600M)
# ------------------------


class NLLBTranslator:
    def __init__(self, device: torch.device, fp16: bool = False):
        self.tok = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        ).to(device)
        if fp16 and device.type == "cuda":
            self.model.half()
        self.model.eval()
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        logging.info("Translator device: %s (fp16=%s)", next(self.model.parameters()).device, self.fp16)

    @torch.no_grad()
    def translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        token_max_length: int = 512,
    ) -> List[str]:
        outputs: List[str] = []
        self.tok.src_lang = src_lang
        # Use tokenizer's language code mapping for forced BOS
        if hasattr(self.tok, "lang_code_to_id") and tgt_lang in self.tok.lang_code_to_id:
            forced_bos = self.tok.lang_code_to_id[tgt_lang]
        else:
            # Fallback if mapping not found
            forced_bos = self.tok.convert_tokens_to_ids(tgt_lang)
        total_batches = math.ceil(len(texts) / batch_size) if texts else 0
        for batch in tqdm(batched(texts, batch_size), total=total_batches, desc=f"Translate {src_lang}->{tgt_lang}", unit="batch"):
            enc = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=token_max_length,
            ).to(self.device)
            from contextlib import nullcontext
            ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self.fp16 else nullcontext()
            with ctx:
                gen = self.model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=max_new_tokens,
                )
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


# ------------------------
# Detox model (s-nlp/mt0-xl-detox-orpo)
# ------------------------


class DetoxModel:
    def __init__(self, device: torch.device, max_new_tokens: int = 128, fp16: bool = False, token_max_length: int = 512):
        self.tok = AutoTokenizer.from_pretrained("s-nlp/mt0-xl-detox-orpo")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "s-nlp/mt0-xl-detox-orpo"
        ).to(device)
        if fp16 and device.type == "cuda":
            self.model.half()
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.token_max_length = token_max_length
        self.fp16 = fp16 and device.type == "cuda"
        logging.info("Detox model device: %s (fp16=%s)", next(self.model.parameters()).device, self.fp16)

    @torch.no_grad()
    def detoxify(self, texts: List[str], batch_size: int = 8) -> List[str]:
        outputs: List[str] = []
        prompts = [f"Detoxify: {t}" for t in texts]
        total_batches = math.ceil(len(prompts) / batch_size) if prompts else 0
        for batch in tqdm(batched(prompts, batch_size), total=total_batches, desc="Detox EN", unit="batch"):
            enc = self.tok(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.token_max_length
            ).to(self.device)
            from contextlib import nullcontext
            ctx = torch.cuda.amp.autocast(dtype=torch.float16) if self.fp16 else nullcontext()
            with ctx:
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


# ------------------------
# SIM: multilingual sentence similarity
# ------------------------


class SimilarityModel:
    def __init__(self, device: torch.device):
        # Using a lightweight multilingual model
        from sentence_transformers import SentenceTransformer

        devstr = "cuda:0" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=devstr,
        )
        logging.info("Similarity model device: %s", devstr)

    def cosine(self, a_texts: List[str], b_texts: List[str], batch_size: int = 32) -> np.ndarray:
        a = self.model.encode(a_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        b = self.model.encode(b_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return (a * b).sum(axis=1)


# ------------------------
# FL: pseudo-perplexity using mBERT
# ------------------------


class FluencyModel:
    def __init__(self, device: torch.device):
        # Pseudo-perplexity via masked-LM with multilingual BERT
        self.tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.mlm = None
        # Load as masked LM
        from transformers import AutoModelForMaskedLM

        self.mlm = AutoModelForMaskedLM.from_pretrained(
            "bert-base-multilingual-cased"
        ).to(device)
        self.mlm.eval()
        self.device = device

    @torch.no_grad()
    def pseudo_perplexity(self, texts: List[str], max_length: int = 256) -> List[float]:
        # Implementation inspired by Salazar et al. 2020 (masked LM pseudo-perplexity)
        ppl_list: List[float] = []
        for text in tqdm(texts, desc="Fluency (pseudo-ppl)", unit="sent"):
            enc = self.tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            n_tokens = int(attn.sum().item())
            # Skip very short texts
            if n_tokens <= 2:
                ppl_list.append(float("nan"))
                continue
            loss_tokens = []
            # Mask each token once (could be optimized by batching)
            for i in range(1, n_tokens - 1):  # avoid [CLS]/[SEP]
                tmp_ids = input_ids.clone()
                tmp_ids[0, i] = self.tok.mask_token_id
                outputs = self.mlm(tmp_ids, attention_mask=attn)
                logits = outputs.logits[0, i]
                target_id = input_ids[0, i]
                log_prob = torch.log_softmax(logits, dim=-1)[target_id]
                loss_tokens.append(-log_prob.item())
            ppl = math.exp(np.mean(loss_tokens)) if loss_tokens else float("nan")
            ppl_list.append(ppl)
        return ppl_list


# ------------------------
# Main pipeline
# ------------------------


def load_tt_split(
    mode: str = "full", test_size: float = 0.1, seed: int = 42
) -> Tuple[List[str], Optional[List[int]]]:
    """Load the Tatar split and return (texts, labels).

    The dataset has only a default config. It exposes language splits (e.g., 'tt').
    If you request 'train' or 'validation', we create a local split from the full tt set.
    """
    logging.info("Loading dataset: textdetox/multilingual_toxicity_dataset (default config)")
    ds_any = load_dataset("textdetox/multilingual_toxicity_dataset")
    # Prefer language key 'tt' if present (DatasetDict); else filter by lang column
    if "tt" in ds_any:
        logging.info("Found dedicated 'tt' split in dataset")
        ds_tt = ds_any["tt"]
    else:
        # Pick any available split then filter by language
        first_key = next(iter(ds_any.keys()))
        base = ds_any[first_key]
        if "lang" in base.column_names:
            logging.info("Filtering by lang == 'tt' from base split")
            ds_tt = base.filter(lambda ex: ex.get("lang") == "tt")
        else:
            # As a last resort, use the base as is (may already be tt)
            ds_tt = base

    # Local train/validation split if requested
    if mode in ("train", "validation"):
        logging.info("Creating local train/validation split from full Tatar set: test_size=%s seed=%s", test_size, seed)
        split_dict = ds_tt.train_test_split(test_size=test_size, seed=seed)
        part = split_dict["test"] if mode == "validation" else split_dict["train"]
    else:
        part = ds_tt

    # Column mapping
    col_text = "text" if "text" in part.column_names else ("sentence" if "sentence" in part.column_names else None)
    if col_text is None:
        raise ValueError("Could not find text column in dataset. Expected 'text' or 'sentence'.")
    texts = part[col_text]

    labels = None
    if "toxic" in part.column_names:
        labels = part["toxic"]
    elif "label" in part.column_names:
        labels = part["label"]
    return texts, labels


def main():
    # Note: per request, we only record toxicity scores (tox_score_before/after)
    # at the END of the pipeline. We may still use the classifier during the
    # pipeline for gating/editing decisions, but results are computed once at end.
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--do_translation", type=lambda x: str(x).lower() != "false", default=True)
    ap.add_argument("--do_detox", type=lambda x: str(x).lower() != "false", default=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--split",
        type=str,
        default="full",
        choices=["full", "train", "validation"],
        help="Evaluate full tt set or create a local train/validation split",
    )
    ap.add_argument("--test_size", type=float, default=0.1, help="Validation size when splitting")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    ap.add_argument("--max_samples", type=int, default=None, help="Debug: limit number of examples")
    ap.add_argument("--debug_samples", type=int, default=None, help="Alias for --max_samples")
    ap.add_argument("--token_max_length", type=int, default=512, help="Max tokenized input length for classifiers/seq2seq encoders")
    ap.add_argument("--translate_max_new_tokens", type=int, default=256, help="Max new tokens for translation")
    ap.add_argument("--detox_max_new_tokens", type=int, default=128, help="Max new tokens for detox generation")
    ap.add_argument("--skip_sim", type=lambda x: str(x).lower() == "true", default=False, help="Skip similarity metric")
    ap.add_argument("--skip_fluency", type=lambda x: str(x).lower() == "true", default=False, help="Skip fluency metric (pseudo-ppl)")
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False, help="Use FP16 for seq2seq models on CUDA")
    # Precomputed English translations controls
    ap.add_argument("--need_transl_eng", type=lambda x: str(x).lower() == "true", default=True,
                    help="If true, compute tt->en translations; if false, read from --transl_eng_path")
    ap.add_argument("--transl_eng_path", type=str, default=None,
                    help="Path to .txt with one English translation per input (saved/loaded)")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    # Log CUDA visibility and availability before selecting device
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    logging.info("torch.__file__=%s", getattr(torch, "__file__", "<unknown>"))
    logging.info("torch.cuda.is_available()=%s, device_count=%d", torch.cuda.is_available(), torch.cuda.device_count())
    device = get_device()
    logging.info("Device: %s (torch.version.cuda=%s)", device, torch.version.cuda)
    if device.type != "cuda":
        logging.warning("CUDA not available in this environment. Ensure a GPU build of PyTorch is installed.")

    # Load data
    texts, labels = load_tt_split(args.split, test_size=args.test_size, seed=args.seed)
    logging.info("Loaded %d Tatar examples (split='%s')", len(texts), args.split)
    # Debug sample limiting
    limit = args.debug_samples if args.debug_samples is not None else args.max_samples
    if limit is not None:
        texts = texts[: limit]
        if labels is not None:
            labels = labels[: limit]
        logging.warning("Limiting to first %d samples for debug", limit)

    # Filter to only toxic==1 as requested (if labels available)
    if labels is not None:
        idx = [i for i, y in enumerate(labels) if int(y) == 1]
        if idx:
            texts = [texts[i] for i in idx]
            labels = [labels[i] for i in idx]
        logging.info("Filtered to toxic==1: %d examples remain", len(texts))
    else:
        logging.warning("Dataset labels not found; skipping toxic==1 filtering")

    # Models
    logging.info("Loading models: classifier, similarity, fluency, translator=%s, detox=%s", args.do_translation, args.do_detox)
    clf = ToxicityClassifier(device, token_max_length=args.token_max_length)
    sim_model = SimilarityModel(device)
    fl_model = FluencyModel(device)
    translator = NLLBTranslator(device, fp16=args.fp16) if args.do_translation else None
    detox = DetoxModel(device, fp16=args.fp16, max_new_tokens=args.detox_max_new_tokens, token_max_length=args.token_max_length) if args.do_detox else None

    # Optional precompute or load English translations for all inputs
    en_all = None
    if args.do_translation:
        if args.need_transl_eng:
            logging.info("Computing tt->en translations for all inputs (for reuse)")
            en_all = translator.translate(
                texts,
                src_lang="tat_Cyrl",
                tgt_lang="eng_Latn",
                batch_size=max(1, args.batch_size // 2),
                max_new_tokens=args.translate_max_new_tokens,
                token_max_length=args.token_max_length,
            )
            if args.transl_eng_path:
                try:
                    os.makedirs(os.path.dirname(args.transl_eng_path), exist_ok=True)
                except Exception:
                    pass
                with open(args.transl_eng_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(en_all))
                logging.info("Saved precomputed English translations to %s", args.transl_eng_path)
        else:
            if args.transl_eng_path and os.path.exists(args.transl_eng_path):
                logging.info("Loading precomputed English translations from %s", args.transl_eng_path)
                with open(args.transl_eng_path, "r", encoding="utf-8") as f:
                    en_all = [line.rstrip("\n") for line in f]
                if len(en_all) != len(texts):
                    logging.warning(
                        "Loaded %d translations but have %d inputs; aligning by truncation/padding",
                        len(en_all),
                        len(texts),
                    )
                    if len(en_all) < len(texts):
                        en_all = en_all + [""] * (len(texts) - len(en_all))
                    else:
                        en_all = en_all[: len(texts)]
            else:
                logging.warning("No transl_eng_path to load; computing translations on the fly")

    # Inference loop with gating; do not compute/stash results yet
    logging.info("Starting gated inference: batch_size=%d, do_translation=%s, do_detox=%s", args.batch_size, args.do_translation, args.do_detox)

    # Editing
    edited_flags: List[bool] = []
    generated: List[str] = []
    gating_probs: List[float] = []

    # Process in batches with progress bar
    pbar = tqdm(total=len(texts), desc="Detox pipeline", unit="ex")
    for batch in batched(list(range(len(texts))), args.batch_size):
        batch_texts = [texts[i] for i in batch]
        # Compute toxicity probs for gating only
        batch_probs = clf.score(batch_texts, batch_size=len(batch))
        gating_probs.extend(batch_probs.tolist())
        batch_edit_mask = [p >= args.threshold for p in batch_probs]
        edited_flags.extend(batch_edit_mask)

        if not any(batch_edit_mask):
            generated.extend(batch_texts)
            pbar.update(len(batch))
            continue

        # Build list to edit and keep mapping
        to_edit = [t for t, m in zip(batch_texts, batch_edit_mask) if m]
        edited_out = to_edit

        if args.do_translation:
            if en_all is not None:
                logging.debug("Selecting precomputed en for %d toxic examples in batch", len(to_edit))
                en = [en_all[idx] for idx, m in zip(batch, batch_edit_mask) if m]
            else:
                logging.debug("Translating tt->en on the fly for %d examples", len(to_edit))
                en = translator.translate(
                    to_edit,
                    src_lang="tat_Cyrl",
                    tgt_lang="eng_Latn",
                    batch_size=max(1, args.batch_size // 2),
                    max_new_tokens=args.translate_max_new_tokens,
                    token_max_length=args.token_max_length,
                )
        else:
            en = to_edit

        if args.do_detox and detox is not None:
            logging.debug("Detoxifying %d English sentences", len(en))
            en_detox = detox.detoxify(en, batch_size=max(1, args.batch_size // 2))
        else:
            en_detox = en

        if args.do_translation:
            logging.debug("Back-translating en->tt for %d sentences", len(en_detox))
            # en -> tt
            tt_back = translator.translate(en_detox, src_lang="eng_Latn", tgt_lang="tat_Cyrl", batch_size=max(1, args.batch_size // 2), max_new_tokens=args.translate_max_new_tokens, token_max_length=args.token_max_length)
            edited_out = tt_back
        else:
            edited_out = en_detox

        # Merge back with unchanged items
        it = iter(edited_out)
        merged = [next(it) if m else orig for orig, m in zip(batch_texts, batch_edit_mask)]
        generated.extend(merged)
        pbar.update(len(batch))
    pbar.close()

    # Compute toxicity scores only at the end (both before and after)
    logging.info("Scoring toxicity (before and after) on full evaluated set")
    tox_before = clf.score(texts, batch_size=args.batch_size)
    tox_after = clf.score(generated, batch_size=args.batch_size)

    # Metrics
    # STA: share of outputs classified non-toxic (lower toxic prob than threshold)
    sta = float(np.mean(tox_after < args.threshold)) if len(tox_after) else float("nan")
    # SIM: cosine similarity original vs generated
    sim = float("nan")
    if not args.skip_sim:
        logging.info("Computing SIM (cosine similarity)")
        sim = sim_model.cosine(texts, generated).mean().item()
    else:
        logging.info("Skipping SIM as requested")
    # FL: avg pseudo-perplexity of generated
    fl = float("nan")
    if not args.skip_fluency:
        logging.info("Computing FL (pseudo-perplexity) — can be slow")
        fl = float(np.nanmean(fl_model.pseudo_perplexity(generated)))
    else:
        logging.info("Skipping FL as requested")

    # Classifier accuracy vs dataset label after filtering only measures how many remain toxic=1
    clf_acc = None
    if labels is not None and len(labels) == len(texts):
        pred_before = (tox_before >= args.threshold).astype(int)
        clf_acc = float(accuracy_score(labels, pred_before))
        logging.info("Classifier accuracy vs dataset toxic label (before): %.4f", clf_acc)

    metrics = Metrics(sta=sta, sim=sim, fl=fl, clf_acc=clf_acc)
    logging.info("Metrics: %s", json.dumps(metrics.__dict__, ensure_ascii=False))

    # Save dataframe-like CSV
    import pandas as pd

    df = pd.DataFrame(
        {
            "text": texts,
            "generated": generated,
            "edited": edited_flags,
            "tox_score_before": tox_before,
            "tox_score_after": tox_after,
        }
    )
    if labels is not None:
        df["toxic"] = labels

    out_csv = os.path.join(args.output_dir, "results.csv")
    out_json = os.path.join(args.output_dir, "metrics.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, ensure_ascii=False, indent=2)
    logging.info("Saved %s and %s", out_csv, out_json)


if __name__ == "__main__":
    main()
