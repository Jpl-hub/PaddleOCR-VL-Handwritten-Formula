#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from latex_utils import difficulty_bucket, latex_complexity, normalize_latex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a PaddleOCR-VL formula JSONL dataset.")
    parser.add_argument("--input", required=True, help="PaddleOCR-VL JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Report output directory.")
    parser.add_argument("--dataset-root", default=None, help="Root used to resolve relative image paths.")
    return parser.parse_args()


def assistant_label(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def image_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    root = Path(args.dataset_root) if args.dataset_root else input_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = assistant_label(record)
            image_rel = record.get("images", [""])[0]
            image_path = Path(image_rel)
            if not image_path.is_absolute():
                image_path = root / image_path
            width, height = image_size(image_path)
            stats = latex_complexity(label)
            rows.append(
                {
                    "line": line_no,
                    "image": image_rel,
                    "image_exists": image_path.exists(),
                    "image_sha256": image_hash(image_path),
                    "width": width,
                    "height": height,
                    "label": label,
                    "normalized_label": normalize_latex(label),
                    "difficulty": difficulty_bucket(label),
                    **stats,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "dataset_analysis.csv"
    df.to_csv(csv_path, index=False)

    duplicate_images = int(df["image_sha256"].duplicated().sum()) if not df.empty else 0
    duplicate_labels = int(df["normalized_label"].duplicated().sum()) if not df.empty else 0
    missing_images = int((~df["image_exists"]).sum()) if not df.empty else 0
    difficulty = Counter(df["difficulty"]) if not df.empty else Counter()
    quantiles = df[["chars", "tokens", "commands", "max_nesting"]].quantile([0.25, 0.5, 0.75, 0.9]).round(2) if not df.empty else pd.DataFrame()

    markdown = [
        "# Dataset Analysis Report",
        "",
        f"- Input: `{input_path}`",
        f"- Samples: {len(df)}",
        f"- Missing images: {missing_images}",
        f"- Duplicate image hashes: {duplicate_images}",
        f"- Duplicate normalized labels: {duplicate_labels}",
        "",
        "## Difficulty Distribution",
        "",
        "| Difficulty | Count |",
        "|---|---:|",
    ]
    for name in ("easy", "medium", "hard"):
        markdown.append(f"| {name} | {difficulty.get(name, 0)} |")
    markdown.extend(["", "## LaTeX Complexity Quantiles", ""])
    if quantiles.empty:
        markdown.append("No samples.")
    else:
        markdown.append("| Quantile | chars | tokens | commands | max_nesting |")
        markdown.append("|---:|---:|---:|---:|---:|")
        for quantile, row in quantiles.iterrows():
            markdown.append(
                f"| {quantile} | {row['chars']} | {row['tokens']} | {row['commands']} | {row['max_nesting']} |"
            )
    markdown.extend(
        [
            "",
            "## Quality-Control Notes",
            "",
            "- Missing images must be fixed before submission.",
            "- Duplicate hashes indicate repeated visual samples and should be reviewed.",
            "- Duplicate labels are acceptable only when visual handwriting differs.",
            "- Use the CSV output for manual spot checks and label audit sampling.",
        ]
    )

    report_path = output_dir / "dataset_analysis.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
