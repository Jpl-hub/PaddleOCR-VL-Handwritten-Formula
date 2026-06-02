#!/usr/bin/env bash
set -euo pipefail

TRAIN_JSONL=${1:?train jsonl path is required}
VAL_JSONL=${2:?validation jsonl path is required}
OUTPUT_DIR=${3:-outputs/paddleocr_vl_formula_lora}
CONFIG=${CONFIG:-configs/paddleocr_vl_formula_lora.yaml}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-PaddlePaddle/PaddleOCR-VL}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES

paddleformers-cli train "$CONFIG" \
  model_name_or_path="$MODEL_NAME_OR_PATH" \
  train_dataset_path="$TRAIN_JSONL" \
  eval_dataset_path="$VAL_JSONL" \
  output_dir="$OUTPUT_DIR" \
  logging_dir="$OUTPUT_DIR/visualdl_logs"
