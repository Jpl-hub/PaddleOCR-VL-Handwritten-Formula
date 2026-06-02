#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm

from latex_utils import difficulty_bucket, wrap_display_math


FIELD_CANDIDATES = {
    "image": ("image", "img", "png", "jpg"),
    "latex": ("latex", "text", "formula", "label"),
    "id": ("sample_id", "id", "uid", "filename"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert handwritten formula HF datasets to PaddleOCR-VL JSONL.")
    parser.add_argument("--dataset", default="SoyVitou/Latex-Math-HME100K", help="Hugging Face dataset name.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--prompt", default="Formula Recognition:")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--no-wrap-display", action="store_true", help="Keep labels without \\[...\\] wrapping.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def find_field(dataset: Dataset, kind: str) -> str:
    names = list(dataset.features.keys())
    for candidate in FIELD_CANDIDATES[kind]:
        if candidate in names:
            return candidate
    lowered = {name.lower(): name for name in names}
    for candidate in FIELD_CANDIDATES[kind]:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Could not find {kind} field in dataset columns: {names}")


def safe_name(value: Any, fallback: str) -> str:
    text = str(value) if value not in (None, "") else fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:120] or fallback


def split_dataset(ds: Dataset | DatasetDict, seed: int) -> dict[str, Dataset]:
    if isinstance(ds, DatasetDict):
        if {"train", "val", "test"}.issubset(ds.keys()):
            return {"train": ds["train"], "val": ds["val"], "test": ds["test"]}
        if {"train", "validation", "test"}.issubset(ds.keys()):
            return {"train": ds["train"], "val": ds["validation"], "test": ds["test"]}
        first_key = next(iter(ds.keys()))
        base = ds[first_key]
    else:
        base = ds
    shuffled = base.shuffle(seed=seed)
    total = len(shuffled)
    test_size = min(2000, max(1, int(total * 0.02)))
    val_size = min(2000, max(1, int(total * 0.02)))
    test = shuffled.select(range(test_size))
    val = shuffled.select(range(test_size, test_size + val_size))
    train = shuffled.select(range(test_size + val_size, total))
    return {"train": train, "val": val, "test": test}


def pick_indices(total: int, size: int, seed: int) -> list[int]:
    if size <= 0 or size >= total:
        return list(range(total))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:size])


def save_image(image: Any, path: Path, image_format: str) -> None:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, dict) and image.get("bytes") is not None:
        from io import BytesIO

        pil = Image.open(BytesIO(image["bytes"])).convert("RGB")
    elif isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image field type: {type(image)!r}")
    if image_format == "jpg":
        pil.save(path, quality=95)
    else:
        pil.save(path)


def convert_split(
    split_name: str,
    split: Dataset,
    output_dir: Path,
    size: int,
    prompt: str,
    image_format: str,
    wrap_labels: bool,
    seed: int,
    source_dataset: str,
) -> dict[str, Any]:
    image_field = find_field(split, "image")
    latex_field = find_field(split, "latex")
    id_field = None
    try:
        id_field = find_field(split, "id")
    except ValueError:
        pass

    selected = pick_indices(len(split), size, seed)
    images_dir = output_dir / "images" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.jsonl"
    manifest_path = output_dir / f"{split_name}_manifest.jsonl"

    counts = {"easy": 0, "medium": 0, "hard": 0}
    written = 0
    with output_path.open("w", encoding="utf-8") as data_out, manifest_path.open("w", encoding="utf-8") as manifest_out:
        for row_no, idx in enumerate(tqdm(selected, desc=f"convert {split_name}"), start=1):
            row = split[int(idx)]
            latex = str(row[latex_field]).strip()
            if not latex:
                continue
            label = wrap_display_math(latex) if wrap_labels else latex
            sample_id = safe_name(row.get(id_field) if id_field else None, f"{split_name}_{idx:08d}")
            image_rel = Path("images") / split_name / f"{row_no:08d}_{sample_id}.{image_format}"
            image_path = output_dir / image_rel
            save_image(row[image_field], image_path, image_format)
            bucket = difficulty_bucket(label)
            counts[bucket] += 1
            record = {
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": label},
                ],
                "images": [str(image_rel)],
                "meta": {
                    "source_dataset": source_dataset,
                    "source_split": split_name,
                    "source_index": int(idx),
                    "sample_id": sample_id,
                    "difficulty": bucket,
                },
            }
            manifest = {
                "sample_id": sample_id,
                "image": str(image_rel),
                "latex": label,
                "source_index": int(idx),
                "difficulty": bucket,
            }
            data_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_out.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            written += 1
    return {
        "split": split_name,
        "samples": written,
        "output": str(output_path),
        "manifest": str(manifest_path),
        "difficulty": counts,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, local_files_only=args.local_files_only)
    splits = split_dataset(ds, args.seed)
    split_sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    summary = {
        "dataset": args.dataset,
        "prompt": args.prompt,
        "seed": args.seed,
        "splits": [],
    }
    split_seed_offsets = {"train": 11, "val": 23, "test": 37}
    for split_name in ("train", "val", "test"):
        summary["splits"].append(
            convert_split(
                split_name,
                splits[split_name],
                output_dir,
                split_sizes[split_name],
                args.prompt,
                args.image_format,
                not args.no_wrap_display,
                args.seed + split_seed_offsets[split_name],
                args.dataset,
            )
        )

    eval_path = output_dir / "eval.jsonl"
    test_path = output_dir / "test.jsonl"
    if test_path.exists():
        eval_path.write_text(test_path.read_text(encoding="utf-8"), encoding="utf-8")

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
