# Solution for Detoxification of Tatar Texts

## Description

The solution uses a combined approach:

1. **Supervised Fine-Tuning (SFT)** of the Gemma-2b model with LoRA
2. **Rule-based detoxification** for final processing

## Solution Structure

### Main files:

* `test_outputs.tsv` — final file for submission (inside submission.zip)
* `rule_based_detox.py` — rule-based detoxification
* `finalize_submission.py` — final processing and archive creation
* `inference_gemma_sft.py` — inference of the trained model
* `train_gemma_sft.py` — model training (reference)
* `prepare_sft_data.py` — data preparation for training (reference)

### Execution scripts:

* `run_full_pipeline.sh` — full pipeline (inference + rule-based detox)
* `run_inference_parallel.py` — parallel inference on multiple GPUs

## Usage

### Quick start (for reproducing results):

```bash
# 1. Apply rule-based detox to input data
python3 finalize_submission.py test_inputs.tsv test_outputs_final.tsv

# 2. Or use the full pipeline (if the trained model is available)
./run_full_pipeline.sh checkpoint-1594 test_inputs.tsv 3 4
```

### Requirements:

* Python 3.8+
* PyTorch with CUDA
* transformers, peft, pandas, loguru, tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model

Model used: `google/gemma-2b` with LoRA fine-tuning

* LoRA parameters: r=32, alpha=32, dropout=0.1
* Trained on multilingual data (ParaDetox, mt0_detox) + translated Tatar examples
* Checkpoint: `checkpoint-1594` (after 2 epochs of training)

## Rule-based Detox

Aggressive toxicity removal while preserving structure:

* Removal of Russian profanity (e.g., блэт, нахуй, хуй, пиздец, заебал, etc.)
* Removal of Tatar obscene expressions (e.g., кутак*, сег*, etc.)
* Removal of severe threats (e.g., суеп утерэм ...)
* Fluency-oriented replacements (тинтәк → ялгышасың, хайван → кеше)
* Cleaning punctuation and extra spaces

## Submission Format

`test_outputs.tsv` includes:

* Columns: ID, tat_toxic, tat_detox1
* 701 data rows
* No empty values
* All texts detoxified

## Metrics

The solution is optimized for three metrics:

1. **Toxicity** — aggressive removal of toxic words
2. **Similarity** — preserving structure and meaning (minimal changes)
3. **Fluency** — improving naturalness of the text

## Author

The solution was prepared for a competition on detoxification of Tatar texts.
