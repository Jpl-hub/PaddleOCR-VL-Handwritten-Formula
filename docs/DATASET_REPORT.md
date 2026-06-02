# Dataset Construction Report

## Task Definition

Input: an image containing a handwritten mathematical expression.

Output: a LaTeX string representing the expression. The canonical format is display math:

```text
\[ ... \]
```

The model is trained with the prompt:

```text
Formula Recognition:
```

## Data Sources

### Public Baseline Data

The repository supports two public handwritten formula sources:

1. `SoyVitou/Latex-Math-HME100K`
   - Hugging Face page: https://huggingface.co/datasets/SoyVitou/Latex-Math-HME100K
   - Described as a Hugging Face version of HME100K with handwritten expression images and LaTeX annotations.
   - Dataset card currently advertises `apache-2.0`; confirm upstream Kaggle terms before final public redistribution.
2. `deepcopy/MathWriting-human`
   - Hugging Face page: https://huggingface.co/datasets/deepcopy/MathWriting-human
   - 230k human-written expressions rendered from online handwriting.
   - Licensed CC BY-NC-SA 4.0; use with care for any cash-prize or commercial follow-up.

Public data is used to establish the first training baseline. A prize-level evaluation set should include additional real samples collected through the protocol below.

### Real Evaluation Set Collection Protocol

Target: at least 500 pages/images and at least 1000 formula instances.

Collection channels:

- University math exercise sheets and notebooks contributed by volunteers.
- Teacher-authored whiteboard or tablet notes exported as images.
- Phone photos of handwritten derivations under varied lighting and perspective.
- Scanned handwritten homework or contest practice pages with contributor consent.

Requirements:

- Contributor permission must be recorded.
- Personally identifying information must be removed or cropped.
- Each image must have a stable sample id, source type, capture device if known, and collection date.
- The evaluation split must not overlap with training samples.

## Split Plan

| Split | Purpose | Target Size |
|---|---|---:|
| train | Fine-tuning | 50k to 200k formula instances |
| val | Hyperparameter selection | 2k to 5k instances |
| eval | Official held-out evaluation | 1k+ instances |

The public baseline converter writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and `eval.jsonl`.

## Difficulty Definition

The scripts assign a first-pass difficulty bucket using LaTeX length, command count, nesting depth, and structural operators.

- easy: short linear expressions.
- medium: fractions, roots, exponents, or moderate nesting.
- hard: integrals, sums, matrices, multi-line structures, high nesting, or long derivations.

Manual review should correct buckets before final submission.

## Quality-Control Workflow

1. Automatic validation.
   - JSONL schema is valid.
   - All image paths exist.
   - Labels are non-empty.
   - Duplicate image hashes are listed.
   - Duplicate normalized labels are listed for visual review.
2. Manual label audit.
   - Two-pass review for the evaluation split.
   - Disagreements are resolved against the source image.
   - Ambiguous symbols are marked in the audit sheet.
3. Distribution review.
   - Confirm easy/medium/hard coverage.
   - Confirm varied capture conditions: clean scans, photos, shadows, blur, tilt, page texture, and stroke styles.
4. Leakage review.
   - Hash image files and normalized labels.
   - Remove exact image duplicates across train/val/eval.
   - Review suspicious near-duplicates manually.

## Reproducible Commands

Convert a baseline dataset:

```bash
python scripts/prepare_hf_formula_dataset.py \
  --dataset SoyVitou/Latex-Math-HME100K \
  --output-dir data/hme100k_paddleocr_vl \
  --train-size 50000 \
  --val-size 2000 \
  --test-size 2000 \
  --seed 2026
```

Analyze an evaluation split:

```bash
python scripts/analyze_dataset.py \
  --input data/hme100k_paddleocr_vl/eval.jsonl \
  --output-dir outputs/hme100k_eval_report
```

## Current Status

- Public baseline pipeline: implemented.
- Real evaluation-set collection: pending sample acquisition.
- Manual label audit: pending real evaluation split.
- Final hosted dataset link: pending.
