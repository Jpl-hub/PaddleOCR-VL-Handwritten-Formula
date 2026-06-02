#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from latex_utils import normalize_latex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PaddleOCR-VL formula JSONL data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--require-meta", action="store_true")
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args()


def label_from_record(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def user_prompt_ok(record: dict) -> bool:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return "<image>" in str(message.get("content", "")) and "Formula" in str(message.get("content", ""))
    return False


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    root = Path(args.dataset_root) if args.dataset_root else input_path.parent
    errors: list[str] = []
    samples = 0

    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            samples += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid json: {exc}")
                continue

            images = record.get("images")
            if not isinstance(images, list) or len(images) != 1:
                errors.append(f"line {line_no}: expected exactly one image path")
            else:
                image_path = Path(images[0])
                if not image_path.is_absolute():
                    image_path = root / image_path
                if not image_path.exists():
                    errors.append(f"line {line_no}: image missing: {images[0]}")

            label = label_from_record(record)
            if not label:
                errors.append(f"line {line_no}: missing assistant label")
            elif not normalize_latex(label):
                errors.append(f"line {line_no}: empty normalized label")

            if not user_prompt_ok(record):
                errors.append(f"line {line_no}: missing user prompt with image placeholder and Formula keyword")

            if args.require_meta and not isinstance(record.get("meta"), dict):
                errors.append(f"line {line_no}: missing meta object")

            if len(errors) >= args.max_errors:
                break

    summary = {"samples": samples, "errors": len(errors), "ok": not errors}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
