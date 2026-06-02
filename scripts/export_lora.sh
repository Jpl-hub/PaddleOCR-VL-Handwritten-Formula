#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-outputs/paddleocr_vl_formula_lora}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-PaddlePaddle/PaddleOCR-VL}
CONFIG=${EXPORT_CONFIG:-configs/paddleocr_vl_formula_lora_export.yaml}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES

paddleformers-cli export "$CONFIG" \
  model_name_or_path="$MODEL_NAME_OR_PATH" \
  output_dir="$OUTPUT_DIR"
