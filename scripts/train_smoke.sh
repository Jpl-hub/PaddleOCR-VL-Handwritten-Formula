#!/usr/bin/env bash
set -euo pipefail

TRAIN_JSONL=${1:-data/mathwriting_rows_smoke/train.jsonl}
VAL_JSONL=${2:-data/mathwriting_rows_smoke/val.jsonl}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-data/model_cache/PaddleOCR-VL}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/paddleocr_vl_formula_lora_smoke}
CONFIG=${CONFIG:-configs/paddleocr_vl_formula_lora.yaml}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES

paddleformers-cli train "$CONFIG" \
  model_name_or_path="$MODEL_NAME_OR_PATH" \
  train_dataset_path="$TRAIN_JSONL" \
  eval_dataset_path="$VAL_JSONL" \
  output_dir="$OUTPUT_DIR" \
  logging_dir="$OUTPUT_DIR/visualdl_logs" \
  max_steps=2 \
  save_steps=2 \
  eval_steps=2 \
  evaluation_strategy=no \
  do_eval=false \
  per_device_train_batch_size=1 \
  per_device_eval_batch_size=1 \
  gradient_accumulation_steps=1 \
  max_seq_len=1024 \
  pre_alloc_memory=8
