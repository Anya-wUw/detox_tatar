# Tatar Text Detoxification

Detoxification pipelines and helper scripts for Tatar (tt). This repo focuses on practical systems for Tatar toxicity removal under shared task style constraints.

**Core idea**

1. Clean noisy inputs (explicit and obfuscated profanity).
2. Score toxicity with a classifier and use a gate  
   toxic score ≥ threshold → edit, otherwise keep the cleaned text.
3. Perform cross-lingual or direct editing  
   tt → en → detox → en → tt, or direct editing with LLaMA/Gemma/GPT.
4. Normalize and package results into submission TSV + ZIP.

---

## Environment setup

### Option A: Conda (recommended, with GPU)

```bash
conda env create -f DETOX_TATAR/environment.yml
conda activate detox_tatar

# If the environment was created without CUDA initially, install GPU packages:
# conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
````

### Option B: Pip

```bash
python -m venv .venv
source .venv/bin/activate

# CPU only by default. For CUDA 12.1 GPU wheels:
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

pip install -r DETOX_TATAR/requirements.txt
```

---

## Algorithms and pipelines

All methods output a submission TSV with columns

* `id`
* `tat_toxic` (original input)
* `tat_detox1` (detoxified output)

and also create a corresponding ZIP archive.

### A) Cleaner + Classifier Gate + SDM (seq2seq)

**Idea**
Rule based cleaner removes explicit and obfuscated profanity, then a toxicity classifier scores the cleaned text.
If score ≥ τ, a compact seq2seq detox model rewrites the sentence, otherwise the cleaned text is kept.

**Run**

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/pipeline_wordremoval_classifier_sdm.py \
  --input_tsv DETOX_TATAR/data/dev_inputs.tsv \
  --output_dir DETOX_TATAR/outputs_wordrm_sdm \
  --threshold 0.5 \
  --batch_size 16 \
  --fp16 true \
  --zip
```

---

### B) Cleaner + Classifier Gate + LLaMA Instruct (optionally LoRA)

**Idea**
Same cleaning and gating as in (A). The detox step uses a chat style LLM with a prompt like
“rewrite the sentence without toxic content, keep the original meaning”.
Only new tokens are kept and the output is normalized to a single line.
Optional LoRA adapter improves Tatar support.

**Run**

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/pipeline_wordremoval_classifier_mgpt.py \
  --input_tsv DETOX_TATAR/data/dev_inputs.tsv \
  --output_dir DETOX_TATAR/outputs_wordrm_llama \
  --threshold 0.5 \
  --batch_size 16 \
  --fp16 true \
  --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora_adapter DETOX_TATAR/llama_tatar_lora/adapter \
  --max_new_tokens 128 \
  --temperature 0.1 \
  --top_p 0.9 \
  --zip
```

LoRA is optional, remove `--lora_adapter` if you want to use the base model only.

---

### C) Clean → Translate → Detox (EN) → Back-translate (TT)

**Idea**
After cleaning and gating, Tatar text is translated to English, detoxified in English with an MPD style seq2seq model, then translated back to Tatar.
Translation prefers Marian OPUS models for `tt ↔ en` and falls back to NLLB if needed.

**Run**

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/run_detox_tatar_clean_mpd_tsv.py \
  --input_tsv DETOX_TATAR/data/test_inputs.tsv \
  --output_dir DETOX_TATAR/outputs_clean_mpd_tsv \
  --translator auto \
  --threshold 0.5 \
  --batch_size 16 \
  --zip
```

---

### D) Translate → LLaMA Detox → Back

**Idea**
Cross-lingual editing with a large English model.
Input tt is translated to en, LLaMA performs English detoxification (we keep only the continuation), then the result is translated back to tt.

**Run**

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/translate_llama_detox.py \
  --input_tsv DETOX_TATAR/data/dev_inputs.tsv \
  --output_dir DETOX_TATAR/outputs_llama_xlat \
  --batch_size 2 \
  --fp16 true \
  --zip
```

---

### E) Diff in means steering

**Idea**
We compute a steering vector at a chosen hidden layer of a causal LLM.
Given hidden states for toxic and non toxic samples at that layer, we compute

Δ = mean(non_toxic) − mean(toxic)

During generation, at the same layer, we add `−α · Δ` to the hidden activations, which biases the model away from toxic directions without changing the decoding objective.

#### Train steering vector

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/diff_steering_train.py \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --layer_idx 27 \
  --n_toxic 500 \
  --n_nontoxic 500 \
  --out_dir DETOX_TATAR/steering/l27
```

This produces `steering_vector.pt` in the output directory.

#### Generate with a trained vector

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/diff_steering_generate.py \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --steering_path DETOX_TATAR/steering/l27/steering_vector.pt \
  --alpha 2.0 \
  --input_tsv DETOX_TATAR/data/dev_inputs.tsv \
  --output_dir DETOX_TATAR/outputs_steer/l27_a2
```

#### Quick layer 21 submission (auto build steering vector if missing)

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/steer_layer21_submit.py \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --steering_path DETOX_TATAR/steering/l21/steering_vector.pt \
  --alpha 0.4 \
  --input_tsv DETOX_TATAR/data/dev_inputs.tsv \
  --output_dir DETOX_TATAR/outputs_steer_l21 \
  --zip \
  --auto_build true
```

#### Grid search (layers 21–31, several α values)

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/hyperparam_search.py \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --layers 21-31 \
  --alphas 1.0,2.0,3.0 \
  --n_toxic 500 \
  --n_nontoxic 500 \
  --n_val 50 \
  --topk 2 \
  --out_dir DETOX_TATAR/search \
  --dev_inputs DETOX_TATAR/data/dev_inputs.tsv
```

---

## Optional: LoRA SFT for better Tatar support

We provide a simple script to fine tune LLaMA on non toxic Tatar comments and then reuse the adapter.

### Train LoRA adapter

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/lora_sft_llama_tatar.py \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir DETOX_TATAR/llama_tatar_lora \
  --limit 2500 \
  --epochs 1
```

This trains a LoRA adapter on up to 2.5k non toxic `tt` comments and reports perplexity on test inputs.

### Use LoRA in pipelines

Pass the adapter path to the LLaMA based pipeline, for example

```bash
--lora_adapter DETOX_TATAR/llama_tatar_lora/adapter
```

as shown in pipeline (B).

---

## Utilities

### Translation caching

To avoid recomputing Tatar to English translations, most scripts support

* `--need_transl_eng`
* `--transl_eng_path path/to/cache.tsv`

Use these flags to precompute and reuse `tt → en` lines.

### Cleaning helpers

Remove prompt artifacts or technical symbols from a TSV:

```bash
python DETOX_TATAR/remove_prompt_symbols.py \
  --input_tsv path/to/submission.tsv \
  --zip
```

Strip English letter tokens from Tatar outputs:

```bash
python DETOX_TATAR/clean_english_letters.py \
  --input_tsv path/to/submission.tsv \
  --zip
```

Round trip only (tt → en → tt) for analysis and debugging:

```bash
CUDA_VISIBLE_DEVICES=1 python DETOX_TATAR/translate_dev_inputs_roundtrip.py \
  --zip
```

---

## Notes and requirements

* Models are downloaded from Hugging Face on first run.
  Make sure your environment has valid Hugging Face credentials for LLaMA and other gated models.
* For A100 and similar GPUs, prefer CUDA 12.4 compatible PyTorch packages
  either via conda (pytorch and pytorch-cuda) or official CUDA wheels.
* All main scripts accept `--input_tsv` and `--output_dir`, and most support `--zip` for submission ready archives.
