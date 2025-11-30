import argparse
import csv
import io
import json
import logging
import os
import zipfile
from typing import List, Optional, Dict

# Force single GPU 1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


class ToxicityClassifier:
    def __init__(self, device: torch.device, max_len: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "textdetox/xlmr-large-toxicity-classifier-v2"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "textdetox/xlmr-large-toxicity-classifier-v2"
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs) if probs else np.array([])


class Translator:
    def __init__(self, device: torch.device, fp16: bool = False):
        self.tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        ).to(device)
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        if self.fp16:
            self.model.half()
        self.model.eval()

    @torch.no_grad()
    def translate(self, texts: List[str], src_lang: str, tgt_lang: str, max_new_tokens: int = 256, max_len: int = 512, batch_size: int = 8) -> List[str]:
        outputs: List[str] = []
        self.tok.src_lang = src_lang
        # Resolve target language token id robustly across tokenizer versions
        forced_bos = None
        try:
            if hasattr(self.tok, "lang_code_to_id") and isinstance(getattr(self.tok, "lang_code_to_id"), dict):
                mapping = getattr(self.tok, "lang_code_to_id")
                if tgt_lang in mapping:
                    forced_bos = mapping[tgt_lang]
        except Exception:
            forced_bos = None
        if forced_bos is None:
            candidates = [tgt_lang, f"__{tgt_lang}__"]
            for tok in candidates:
                try:
                    tid = self.tok.convert_tokens_to_ids(tok)
                    if tid is not None and tid != getattr(self.tok, "unk_token_id", None):
                        forced_bos = tid
                        break
                except Exception:
                    continue
        if forced_bos is None:
            raise RuntimeError(f"Could not resolve target language token id for '{tgt_lang}' in NLLB tokenizer")
        total = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total, desc=f"Translate {src_lang}->{tgt_lang}", unit="batch"):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)
            with (torch.cuda.amp.autocast(dtype=torch.float16) if self.fp16 else torch.autocast(device_type=self.device.type, enabled=False)):
                gen = self.model.generate(**enc, forced_bos_token_id=forced_bos, max_new_tokens=max_new_tokens)
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


class DetoxModel:
    def __init__(self, device: torch.device, fp16: bool = False, max_new_tokens: int = 128):
        self.tok = AutoTokenizer.from_pretrained("s-nlp/mt0-xl-detox-orpo")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/mt0-xl-detox-orpo").to(device)
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.max_new_tokens = max_new_tokens
        if self.fp16:
            self.model.half()
        self.model.eval()

    @torch.no_grad()
    def detox(self, texts: List[str], batch_size: int = 8) -> List[str]:
        outputs: List[str] = []
        prompts = [f"Detoxify: {t}" for t in texts]
        total = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=total, desc="Detox EN", unit="batch"):
            batch = prompts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with (torch.cuda.amp.autocast(dtype=torch.float16) if self.fp16 else torch.autocast(device_type=self.device.type, enabled=False)):
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
            outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs


class SimilarityLaBSE:
    def __init__(self, device: torch.device):
        from sentence_transformers import SentenceTransformer

        dev = "cuda:0" if device.type == "cuda" else device.type
        self.model = SentenceTransformer("sentence-transformers/LaBSE", device=dev)

    def cosine(self, a: List[str], b: List[str], batch_size: int = 32) -> np.ndarray:
        ea = self.model.encode(a, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        eb = self.model.encode(b, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return (ea * eb).sum(axis=1)


class CometFluency:
    def __init__(self, device: torch.device, model_name: str = "myyycroft/XCOMET-lite"):
        try:
            from comet import download_model, load_from_checkpoint  # from unbabel-comet
        except Exception as e:
            raise ImportError(
                "Failed to import COMET from 'unbabel-comet'. Please run: pip uninstall -y comet && pip install -U unbabel-comet"
            ) from e

        self.device = device
        last_err = None
        self.model = None
        for name in (model_name, "Unbabel/wmt22-comet-da"):
            try:
                logging.info("Loading COMET model: %s", name)
                ckpt = download_model(name)
                self.model = load_from_checkpoint(ckpt)
                break
            except Exception as e:
                logging.warning("COMET load failed for %s: %s", name, e)
                last_err = e
        if self.model is None:
            raise RuntimeError(
                f"Failed to load COMET model (tried: {model_name}, Unbabel/wmt22-comet-da): {last_err}"
            )
        self.model.to(str(device))
        self.model.eval()

    @torch.no_grad()
    def score(self, src: List[str], mt: List[str], ref: Optional[List[str]] = None, batch_size: int = 16) -> List[float]:
        data = []
        for i in range(len(mt)):
            item: Dict[str, str] = {"src": src[i], "mt": mt[i]}
            if ref is not None:
                item["ref"] = ref[i]
            data.append(item)
        seg_scores, _ = self.model.predict(data, batch_size=batch_size, gpus=1 if self.device.type == "cuda" else 0)
        return [float(s) for s in seg_scores]


def build_submission(df_in: pd.DataFrame, args, device: torch.device) -> pd.DataFrame:
    # Expect columns ID and tat_toxic
    if not {"ID", "tat_toxic"}.issubset(set(df_in.columns)):
        raise ValueError("Input TSV must contain 'ID' and 'tat_toxic' columns")

    tox_clf = ToxicityClassifier(device)
    translator = Translator(device, fp16=args.fp16)
    detoxer = DetoxModel(device, fp16=args.fp16)
    sim_model = None if args.skip_similarity else SimilarityLaBSE(device)
    comet = None if args.skip_fluency else CometFluency(device, model_name=args.comet_model)

    texts = df_in["tat_toxic"].astype(str).tolist()

    # Gate: only detoxify if toxic >= threshold
    logging.info("Scoring toxicity for gating...")
    p_toxic = tox_clf.score(texts, batch_size=args.batch_size)
    edit_mask = p_toxic >= args.threshold

    generated = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Submission pipeline", unit="batch"):
        batch = texts[i : i + args.batch_size]
        mask = edit_mask[i : i + args.batch_size]
        to_edit = [t for t, m in zip(batch, mask) if m]
        if len(to_edit) == 0:
            generated.extend(batch)
            continue
        # tt -> en
        en = translator.translate(to_edit, src_lang="tat_Cyrl", tgt_lang="eng_Latn", batch_size=max(1, args.batch_size // 2), max_new_tokens=args.translate_max_new_tokens)
        # detox in English
        en_detox = detoxer.detox(en, batch_size=max(1, args.batch_size // 2))
        # en -> tt
        tt_back = translator.translate(en_detox, src_lang="eng_Latn", tgt_lang="tat_Cyrl", batch_size=max(1, args.batch_size // 2), max_new_tokens=args.translate_max_new_tokens)
        it = iter(tt_back)
        merged = [next(it) if m else orig for orig, m in zip(batch, mask)]
        generated.extend(merged)

    # Ensure no empty values
    out_texts = [g if (g is not None and str(g).strip() != "") else orig for g, orig in zip(generated, texts)]

    # Metrics (optional, saved to JSON)
    logging.info("Computing metrics: LaBSE SIM, XLM-R toxicity before/after, COMET fluency")
    sim_orig = float("nan")
    if sim_model is not None:
        sim_orig = sim_model.cosine(texts, out_texts).mean().item()

    p_after = tox_clf.score(out_texts, batch_size=args.batch_size)
    sta = float((p_after < args.threshold).mean())

    fluency_avg = float("nan")
    if comet is not None:
        try:
            comet_scores = comet.score(src=texts, mt=out_texts, ref=None, batch_size=max(1, args.batch_size // 2))
            fluency_avg = float(np.mean(comet_scores)) if len(comet_scores) else float("nan")
        except Exception as e:
            logging.error("COMET scoring failed: %s", e)

    metrics = {
        "SIM_LaBSE_mean": sim_orig,
        "Toxicity_STA": sta,
        "Fluency_COMET_mean": fluency_avg,
    }
    with open(os.path.join(args.output_dir, "submission_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Build TSV
    out_df = pd.DataFrame({
        "ID": df_in["ID"].tolist(),
        "tat_toxic": texts,
        "tat_detox1": out_texts,
    })
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, default="DETOX_TATAR/data/dev_inputs.tsv")
    ap.add_argument("--output_dir", type=str, default="DETOX_TATAR/outputs")
    ap.add_argument("--zip_name", type=str, default="submission.zip")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--translate_max_new_tokens", type=int, default=256)
    ap.add_argument("--skip_fluency", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--skip_similarity", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--comet_model", type=str, default="myyycroft/XCOMET-lite")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    device = get_device()
    logging.info("Device: %s", device)

    df_in = pd.read_csv(args.input_tsv, sep="\t", quoting=csv.QUOTE_NONE)
    out_df = build_submission(df_in, args, device)

    # Save TSV
    tsv_path = os.path.join(args.output_dir, "submission.tsv")
    out_df.to_csv(tsv_path, sep="\t", index=False)

    # Zip it
    zip_path = os.path.join(args.output_dir, args.zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_path, arcname=os.path.basename(tsv_path))
    logging.info("Saved TSV: %s and ZIP: %s", tsv_path, zip_path)


if __name__ == "__main__":
    main()
