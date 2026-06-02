#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm

from latex_utils import difficulty_bucket, wrap_display_math


API_BASE = "https://datasets-server.huggingface.co/rows"
FIELD_CANDIDATES = {
    "image": ("image", "img", "png", "jpg"),
    "latex": ("latex", "text", "formula", "label"),
    "id": ("sample_id", "id", "uid", "filename"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a formula dataset from Hugging Face Dataset Viewer rows.")
    parser.add_argument("--dataset", default="deepcopy/MathWriting-human")
    parser.add_argument("--config", default="default")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="Formula Recognition:")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--no-wrap-display", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def urlopen_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "formula-dataset-prep"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def download_bytes(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "formula-dataset-prep"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{API_BASE}?{query}"


def safe_name(value: Any, fallback: str) -> str:
    text = str(value) if value not in (None, "") else fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:120] or fallback


def find_field(row: dict[str, Any], kind: str) -> str:
    for name in FIELD_CANDIDATES[kind]:
        if name in row:
            return name
    lowered = {name.lower(): name for name in row}
    for name in FIELD_CANDIDATES[kind]:
        if name in lowered:
            return lowered[name]
    raise ValueError(f"Could not find {kind} field in row keys: {sorted(row)}")


def fetch_rows(dataset: str, config: str, split: str, offset: int, size: int, page_size: int, timeout: int, sleep: float):
    remaining = size
    current_offset = offset
    while remaining > 0:
        length = min(page_size, remaining)
        payload = urlopen_json(rows_url(dataset, config, split, current_offset, length), timeout)
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item
        fetched = len(rows)
        remaining -= fetched
        current_offset += fetched
        if fetched < length:
            break
        if sleep > 0:
            time.sleep(sleep)


def convert_split(
    dataset: str,
    config: str,
    source_split: str,
    output_split: str,
    offset: int,
    size: int,
    output_dir: Path,
    prompt: str,
    image_format: str,
    wrap_labels: bool,
    page_size: int,
    timeout: int,
    sleep: float,
) -> dict[str, Any]:
    images_dir = output_dir / "images" / output_split
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{output_split}.jsonl"
    manifest_path = output_dir / f"{output_split}_manifest.jsonl"
    difficulty = {"easy": 0, "medium": 0, "hard": 0}
    written = 0

    iterator = fetch_rows(dataset, config, source_split, offset, size, page_size, timeout, sleep)
    with jsonl_path.open("w", encoding="utf-8") as data_out, manifest_path.open("w", encoding="utf-8") as manifest_out:
        for row_item in tqdm(iterator, total=size, desc=f"download {output_split}"):
            row = row_item["row"]
            image_field = find_field(row, "image")
            latex_field = find_field(row, "latex")
            try:
                id_field = find_field(row, "id")
            except ValueError:
                id_field = ""

            latex = str(row[latex_field]).strip()
            if not latex:
                continue
            image_value = row[image_field]
            if not isinstance(image_value, dict) or not image_value.get("src"):
                raise ValueError(f"Image field does not contain a src URL: {image_value!r}")
            sample_id = safe_name(row.get(id_field) if id_field else row_item.get("row_idx"), f"{output_split}_{written:08d}")
            image_rel = Path("images") / output_split / f"{written + 1:08d}_{sample_id}.{image_format}"
            image_path = output_dir / image_rel
            image_path.write_bytes(download_bytes(image_value["src"], timeout))

            label = wrap_display_math(latex) if wrap_labels else latex
            bucket = difficulty_bucket(label)
            difficulty[bucket] += 1
            record = {
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": label},
                ],
                "images": [str(image_rel)],
                "meta": {
                    "source_dataset": dataset,
                    "source_config": config,
                    "source_split": source_split,
                    "source_row_idx": row_item.get("row_idx"),
                    "sample_id": sample_id,
                    "difficulty": bucket,
                    "downloaded_from": "huggingface_dataset_viewer",
                },
            }
            manifest = {
                "sample_id": sample_id,
                "image": str(image_rel),
                "latex": label,
                "source_split": source_split,
                "source_row_idx": row_item.get("row_idx"),
                "difficulty": bucket,
            }
            data_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_out.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            written += 1

    return {
        "split": output_split,
        "source_split": source_split,
        "offset": offset,
        "samples": written,
        "jsonl": str(jsonl_path),
        "manifest": str(manifest_path),
        "difficulty": difficulty,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_jobs = [
        ("train", args.train_split, args.train_offset, args.train_size),
        ("val", args.val_split, args.val_offset, args.val_size),
        ("test", args.test_split, args.test_offset, args.test_size),
    ]
    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "prompt": args.prompt,
        "source": "huggingface_dataset_viewer",
        "splits": [],
    }
    for output_split, source_split, offset, size in split_jobs:
        summary["splits"].append(
            convert_split(
                args.dataset,
                args.config,
                source_split,
                output_split,
                offset,
                size,
                output_dir,
                args.prompt,
                args.image_format,
                not args.no_wrap_display,
                args.page_size,
                args.timeout,
                args.sleep,
            )
        )
    test_path = output_dir / "test.jsonl"
    eval_path = output_dir / "eval.jsonl"
    if test_path.exists():
        eval_path.write_text(test_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
