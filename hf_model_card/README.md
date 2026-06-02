---
license: apache-2.0
base_model: PaddlePaddle/PaddleOCR-VL
tags:
  - paddleocr-vl
  - handwritten-formula-recognition
  - latex-ocr
  - image-to-text
pipeline_tag: image-to-text
---

# PaddleOCR-VL Handwritten Formula Recognition

This model fine-tunes `PaddlePaddle/PaddleOCR-VL` for handwritten mathematical expression recognition. Given a formula image, it predicts a LaTeX string.

## Task

Prompt:

```text
Formula Recognition:
```

Output:

```text
\[ ... \]
```

## Training Data

Baseline training uses public handwritten formula data converted into PaddleOCR-VL message JSONL format. The final training mix and evaluation set should be reported with sample counts, provenance, difficulty distribution, and quality-control results.

## Metrics

- Exact match after LaTeX normalization.
- Normalized edit distance.
- Symbol-level precision, recall, and F1.

## Training Configuration

- Base model: `PaddlePaddle/PaddleOCR-VL`
- Fine-tuning: LoRA SFT
- LoRA rank: 16
- Max sequence length: 4096
- Learning rate: 3e-4
- Epochs: 2

## Inference

Use the repository demo or evaluation script:

```bash
MODEL_PATH=/path/to/exported/model gradio demo/app.py
```

## Limitations

- Very long multi-line derivations may require a larger sequence length.
- Ambiguous handwriting can produce mathematically plausible but visually incorrect symbols.
- Public dataset licenses and downstream use constraints must be checked before redistribution.

## Citation

Please cite the upstream PaddleOCR-VL and dataset sources used for training and evaluation.
