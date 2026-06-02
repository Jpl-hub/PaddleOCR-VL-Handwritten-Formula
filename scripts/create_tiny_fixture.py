#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny formula fixture for smoke tests.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def make_image(path: Path, text: str) -> None:
    image = Image.new("RGB", (320, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 32), text, fill="black")
    image.save(path)


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    images = root / "images" / "test"
    images.mkdir(parents=True, exist_ok=True)
    samples = [
        ("sample_1.png", r"\[x+1=2\]", "x + 1 = 2"),
        ("sample_2.png", r"\[\frac{a}{b}\]", "a / b"),
    ]
    records = []
    predictions = []
    for filename, latex, draw_text in samples:
        image_rel = Path("images") / "test" / filename
        make_image(root / image_rel, draw_text)
        records.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>Formula Recognition:"},
                    {"role": "assistant", "content": latex},
                ],
                "images": [str(image_rel)],
                "meta": {"difficulty": "easy"},
            }
        )
        predictions.append({"prediction": latex, "ground_truth": latex, "image": str(image_rel)})

    for name, rows in [("test.jsonl", records), ("predictions.jsonl", predictions)]:
        with (root / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(root)


if __name__ == "__main__":
    main()
