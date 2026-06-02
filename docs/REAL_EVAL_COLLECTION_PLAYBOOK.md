# Real Evaluation Set Collection Playbook

The challenge rubric gives high weight to evaluation-set authenticity, diversity, and quality. Public datasets are useful for baseline training, but the final evaluation set should include real collected samples with provenance and quality control.

## Target

- Minimum viable target: 500 formula images.
- Strong target: 1000+ formula images.
- Difficulty mix: about 30% easy, 50% medium, 20% hard.
- Visual mix: clean scans, phone photos, tilt, shadows, blur, ruled paper, tablet handwriting, pencil/pen variation.

## Collection Sources

- Volunteer handwritten math exercises.
- Teacher notes or whiteboard/tablet exports.
- Scanned homework with personal information removed.
- Phone photos of notebook pages, cropped to formula regions.

Do not include personal information, exam answers under restriction, copyrighted textbook pages as the main content, or samples without permission.

## Local Annotation App

Run:

```bash
python -m pip install -r requirements.txt
ANNOTATION_DATASET_DIR=data/real_formula_collection gradio demo/annotate_formula_dataset.py
```

The app appends records to:

```text
data/real_formula_collection/eval.jsonl
data/real_formula_collection/images/eval/
```

Validate and analyze:

```bash
python scripts/validate_formula_dataset.py \
  --input data/real_formula_collection/eval.jsonl \
  --require-meta

python scripts/analyze_dataset.py \
  --input data/real_formula_collection/eval.jsonl \
  --output-dir outputs/real_formula_eval_report
```

Package for submission:

```bash
python scripts/package_evaluation_set.py \
  --annotations data/real_formula_collection/eval.jsonl \
  --output-dir outputs/submission \
  --name handwritten_formula_eval \
  --analysis-report outputs/real_formula_eval_report/dataset_analysis.md
```

## Quality Control

1. First annotation pass writes LaTeX and difficulty.
2. Second review checks rendering and symbol ambiguity.
3. Automatic validation checks paths, labels, prompt format, and metadata.
4. Dataset analysis checks duplicates and difficulty distribution.
5. Final package includes `SHA256SUMS.json` for reproducibility.

## Provenance Sheet

Keep a private spreadsheet with:

| sample_id | source | contributor permission | collection date | capture type | reviewer | notes |
|---|---|---|---|---|---|---|

This sheet does not need to be public, but it should be available for authenticity verification.
