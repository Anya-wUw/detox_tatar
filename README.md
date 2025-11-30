# DETOX_TATAR: Tatar Detoxification

Detoxification pipelines and helpers for Tatar (tt). Core idea:
- Clean noisy inputs (explicit+obfuscated profanity removal).
- Classifier gate (toxic ≥ threshold → edit; else keep cleaned text).
- Cross‑lingual editing: tt→en → detox → en→tt, or direct LLaMA/Gemma/GPT editing.
- Submission utilities to normalize and package results.

## Environment

Option A — Conda (recommended, with GPU):

```bash
conda env create -f DETOX_TATAR/environment.yml
conda activate detox_tatar
# If created previously without CUDA, ensure GPU packages are installed:
# conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

Option B — Pip:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only by default; for GPU wheels (CUDA 12.1):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r DETOX_TATAR/requirements.txt
```

## Algorithms

All methods produce a submission TSV (ID, tat_toxic, tat_detox1) and a ZIP.

### A) Cleaner + Classifier Gate + SDM (seq2seq)
- Theory: strip explicit/obfuscated profanity with rules; score toxicity; if score ≥ τ, rewrite with a compact seq2seq detox model; else keep the cleaned text.
- Run:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/pipeline_wordremoval_classifier_sdm.py \
    --input_tsv DETOX_TATAR/data/dev_inputs.tsv --output_dir DETOX_TATAR/outputs_wordrm_sdm \
    --threshold 0.5 --batch_size 16 --fp16 true --zip

### B) Cleaner + Classifier Gate + LLaMA Instruct (optionally LoRA)
- Theory: same gate as (A), but detox is a chat LLM prompted to “rewrite without toxicity”; only new tokens are kept; output normalized to one line. LoRA adapter improves Tatar.
- Run:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/pipeline_wordremoval_classifier_mgpt.py \
    --input_tsv DETOX_TATAR/data/dev_inputs.tsv --output_dir DETOX_TATAR/outputs_wordrm_llama \
    --threshold 0.5 --batch_size 16 --fp16 true \
    --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_adapter DETOX_TATAR/llama_tatar_lora/adapter \
    --max_new_tokens 128 --temperature 0.1 --top_p 0.9 --zip

### C) Clean → Translate → Detox (EN) → Back‑translate (TT)
- Theory: after cleaning+gating, project TT→EN; detox in English (MPD seq2seq), then back to TT. Translation prefers Marian OPUS (tt↔en) and falls back to NLLB.
- Run:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/run_detox_tatar_clean_mpd_tsv.py \
    --input_tsv DETOX_TATAR/data/test_inputs.tsv --output_dir DETOX_TATAR/outputs_clean_mpd_tsv \
    --translator auto --threshold 0.5 --batch_size 16 --zip

### D) Translate → LLaMA Detox → Back 
- Theory: direct cross‑lingual editing: TT→EN, detox by LLaMA with an English prompt (keep continuation only), EN→TT back.
- Run:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/translate_llama_detox.py \
    --input_tsv DETOX_TATAR/data/dev_inputs.tsv --output_dir DETOX_TATAR/outputs_llama_xlat \
    --batch_size 2 --fp16 true --zip

### E) Diff‑in‑Means Steering
- Theory: compute Δ = mean(non‑toxic) − mean(toxic) of hidden states at a layer; during generation add −α·Δ at that layer to bias away from toxic directions.
- Train vector:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/diff_steering_train.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --layer_idx 27 \
    --n_toxic 500 --n_nontoxic 500 --out_dir DETOX_TATAR/steering/l27
- Generate with vector:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/diff_steering_generate.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --steering_path DETOX_TATAR/steering/l27/steering_vector.pt --alpha 2.0 \
    --input_tsv DETOX_TATAR/data/dev_inputs.tsv --output_dir DETOX_TATAR/outputs_steer/l27_a2
- Quick L21 submit (auto‑build if missing):
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/steer_layer21_submit.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --steering_path DETOX_TATAR/steering/l21/steering_vector.pt --alpha 0.4 \
    --input_tsv DETOX_TATAR/data/dev_inputs.tsv --output_dir DETOX_TATAR/outputs_steer_l21 --zip --auto_build true
- Grid search (layers 21–31; α in {1.0,2.0,3.0}):
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/hyperparam_search.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --layers 21-31 --alphas 1.0,2.0,3.0 \
    --n_toxic 500 --n_nontoxic 500 --n_val 50 --topk 2 --out_dir DETOX_TATAR/search \
    --dev_inputs DETOX_TATAR/data/dev_inputs.tsv

## Training (optional): LoRA SFT for better Tatar

- Train LLaMA with LoRA on non‑toxic `tt` comments (up to 2.5k), then compute PPL on test inputs:
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/lora_sft_llama_tatar.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_dir DETOX_TATAR/llama_tatar_lora --limit 2500 --epochs 1

- Use the trained adapter via `--lora_adapter DETOX_TATAR/llama_tatar_lora/adapter` in the LLaMA pipeline above.

## Utilities

- Precompute/reuse English translations to save time:
  - Many scripts accept `--need_transl_eng` and `--transl_eng_path` to cache tt→en lines.
- Clean prompt artifacts from TSV:
  - python DETOX_TATAR/remove_prompt_symbols.py --input_tsv path/to/submission.tsv --zip
- Strip English‑letter tokens from outputs:
  - python DETOX_TATAR/clean_english_letters.py --input_tsv path/to/submission.tsv --zip
- Round‑trip only (tt→en→tt):
  - CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/translate_dev_inputs_roundtrip.py --zip

Models download from Hugging Face on first run. Ensure HF auth for LLaMA. Prefer CUDA 12.4 wheels or conda packages for PyTorch on A100.
