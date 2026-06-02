# Training Report

## Hardware Target

Initial rented server check:

- GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM.
- System memory: high enough for preprocessing.
- Root disk: about 27 GB free at initial check.
- Python entry point: `/root/miniconda3/bin/python` in the base Conda installation.

This hardware is appropriate for PaddleOCR-VL 0.9B LoRA with conservative sequence length and batch settings. PaddleOCR-VL-1.5 examples advertise higher memory use, so the first run targets `PaddlePaddle/PaddleOCR-VL`.

## Environment Plan

Preferred approach:

1. Create a separate Conda environment under the existing Miniconda installation.
2. Install PaddlePaddle GPU and PaddleFormers dependencies following the official PaddleOCR-VL best-practice document.
3. Keep the project in `/root/PaddleOCR-VL-Handwritten-Formula`.
4. Keep model cache and dataset cache inside the project or a documented cache directory.

Verified server environment:

- Conda env: `/root/miniconda3/envs/formula_vl`
- PaddlePaddle GPU: `3.2.1`, CUDA-enabled.
- PaddleFormers: `1.0.0`; `1.1.1` was not compatible with Paddle `3.2.1` because it imports `paddle.distributed.fsdp`.
- Required shell setup before running PaddleFormers CLI:

```bash
export PATH=/root/miniconda3/envs/formula_vl/bin:$PATH
```

## Baseline Training

Config:

- `configs/paddleocr_vl_formula_lora.yaml`
- LoRA rank: 16
- max sequence length: 4096
- per-device train batch size: 2
- gradient accumulation: 32
- learning rate: `3e-4`
- epochs: 2

Command:

```bash
bash scripts/train_lora.sh \
  data/hme100k_paddleocr_vl/train.jsonl \
  data/hme100k_paddleocr_vl/val.jsonl \
  outputs/paddleocr_vl_formula_lora
```

Export:

```bash
bash scripts/export_lora.sh outputs/paddleocr_vl_formula_lora
```

Evaluate:

```bash
python scripts/eval_paddleocr_vl.py \
  --model-name-or-path outputs/paddleocr_vl_formula_lora/export \
  --data-path data/hme100k_paddleocr_vl/eval.jsonl \
  --output-path outputs/eval_predictions.jsonl \
  --device gpu

python scripts/evaluate_formula.py \
  --predictions outputs/eval_predictions.jsonl \
  --output-json outputs/eval_metrics.json
```

## Ablation Plan

| Run | Dataset | Rank | LR | Max Seq Len | Notes |
|---|---|---:|---:|---:|---|
| A | HME100K subset | 8 | 5e-4 | 4096 | Official-style LoRA rank |
| B | HME100K subset | 16 | 3e-4 | 4096 | Default baseline |
| C | HME100K + real eval-adjacent train | 16 | 3e-4 | 4096 | Add real collected samples to training only |
| D | Best data mix | 16 | 2e-4 | 8192 | If VRAM allows |

## Result Table

| Run | Train Samples | Eval Samples | Exact Match | NED | Token F1 | Status |
|---|---:|---:|---:|---:|---:|---|
| A | pending | pending | pending | pending | pending | not run |
| B | pending | pending | pending | pending | pending | not run |
| C | pending | pending | pending | pending | pending | waiting for real samples |
| D | pending | pending | pending | pending | pending | not run |
