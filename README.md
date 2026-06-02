# PaddleOCR-VL Handwritten Formula Recognition

Fine-tuning and evaluation pipeline for handwritten mathematical expression recognition with PaddleOCR-VL. The task maps a handwritten formula image to a LaTeX string.

## Competition Direction

- Scenario: handwritten formula recognition.
- Base model: `PaddlePaddle/PaddleOCR-VL` with LoRA SFT as the first training target.
- Output format: LaTeX wrapped as display math, for example `\[\frac{x}{2}\]`.
- Main metric: normalized edit distance, exact match, and symbol-level F1 after LaTeX normalization.

The repository is structured to satisfy the challenge deliverables: public training/evaluation code, dataset construction report, annotation guideline, reproducible training scripts, local demo, and model-card template.

## Repository Layout

```text
configs/                 PaddleFormers LoRA configs
demo/                    Local Gradio demo
docs/                    Dataset, annotation, training, and submission docs
hf_model_card/           Hugging Face model-card draft
scripts/                 Data preparation, analysis, training, and evaluation utilities
tests/fixtures/          Tiny smoke-test fixtures
```

Large data and model artifacts are intentionally excluded from Git. Use `data/`, `outputs/`, and `models/` locally or on the training server.

## Quick Start

Install local utility dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a tiny fixture and run the local checks:

```bash
python scripts/create_tiny_fixture.py --output-dir tests/fixtures/tiny_formula
python scripts/analyze_dataset.py --input tests/fixtures/tiny_formula/test.jsonl --output-dir outputs/tiny_report
python scripts/evaluate_formula.py --predictions tests/fixtures/tiny_formula/predictions.jsonl
```

Prepare a public-data baseline in PaddleOCR-VL message format:

```bash
python scripts/prepare_hf_formula_dataset.py \
  --dataset SoyVitou/Latex-Math-HME100K \
  --output-dir data/hme100k_paddleocr_vl \
  --train-size 50000 \
  --val-size 2000 \
  --test-size 2000 \
  --prompt "Formula Recognition:" \
  --seed 2026
```

For MathWriting:

```bash
python scripts/prepare_hf_formula_dataset.py \
  --dataset deepcopy/MathWriting-human \
  --output-dir data/mathwriting_paddleocr_vl \
  --train-size 50000 \
  --val-size 2000 \
  --test-size 2000 \
  --prompt "Formula Recognition:" \
  --seed 2026
```

Generate a dataset quality report:

```bash
python scripts/analyze_dataset.py \
  --input data/hme100k_paddleocr_vl/eval.jsonl \
  --output-dir outputs/hme100k_eval_report
```

## Training

The first server target is a single RTX 3090 24 GB, so the default config uses LoRA, `max_seq_len=4096`, and conservative batch settings.

```bash
bash scripts/train_lora.sh \
  data/hme100k_paddleocr_vl/train.jsonl \
  data/hme100k_paddleocr_vl/val.jsonl \
  outputs/paddleocr_vl_formula_lora
```

The training script expects PaddleFormers and PaddlePaddle GPU dependencies to already be installed. See [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md) for the environment plan and server notes.

## Evaluation

Metric-only evaluation from saved predictions:

```bash
python scripts/evaluate_formula.py --predictions outputs/eval_predictions.jsonl
```

Model inference evaluation:

```bash
python scripts/eval_paddleocr_vl.py \
  --model-name-or-path outputs/paddleocr_vl_formula_lora/export \
  --data-path data/hme100k_paddleocr_vl/eval.jsonl \
  --output-path outputs/eval_predictions.jsonl \
  --device gpu
```

## Demo

```bash
MODEL_PATH=outputs/paddleocr_vl_formula_lora/export gradio demo/app.py
```

Open the printed local URL, upload a formula image, and run greedy LaTeX generation.

## Submission Checklist

Before each leaderboard email:

1. Validate the GitHub repository is public and reproducible.
2. Upload the fine-tuned model and model card to Hugging Face.
3. Upload the evaluation set package and dataset report to the selected storage platform.
4. Send the required email to all official addresses with the correct subject format.
5. Keep provenance, licensing, and quality-control evidence for authenticity verification.

See [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md).
