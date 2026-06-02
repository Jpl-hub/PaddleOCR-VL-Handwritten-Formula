#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package an evaluation set for challenge submission.")
    parser.add_argument("--annotations", required=True, help="Evaluation JSONL in PaddleOCR-VL message format.")
    parser.add_argument("--dataset-root", default=None, help="Root used to resolve image paths.")
    parser.add_argument("--analysis-report", default=None, help="Optional dataset_analysis.md path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="handwritten_formula_eval")
    parser.add_argument("--task-description", default=None, help="Optional Markdown file to include.")
    parser.add_argument("--license-note", default="Evaluation data may be used for challenge review only unless separately licensed.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_image(src: Path, image_dir: Path, used_names: set[str]) -> str:
    base = src.name
    stem = src.stem
    suffix = src.suffix or ".png"
    candidate = base
    counter = 1
    while candidate in used_names:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    used_names.add(candidate)
    dst = image_dir / candidate
    shutil.copy2(src, dst)
    return f"images/{candidate}"


def main() -> None:
    args = parse_args()
    annotations_path = Path(args.annotations)
    root = Path(args.dataset_root) if args.dataset_root else annotations_path.parent
    output_dir = Path(args.output_dir)
    package_dir = output_dir / args.name
    image_dir = package_dir / "images"
    package_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    packaged_records = []
    source_manifest = []
    with annotations_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_rel = record["images"][0]
            src = Path(image_rel)
            if not src.is_absolute():
                src = root / src
            if not src.exists():
                raise FileNotFoundError(f"Missing image for row {idx}: {image_rel}")
            packaged_image = copy_image(src, image_dir, used_names)
            new_record = dict(record)
            new_record["images"] = [packaged_image]
            packaged_records.append(new_record)
            source_manifest.append(
                {
                    "row": idx,
                    "source_image": image_rel,
                    "packaged_image": packaged_image,
                    "sha256": sha256(package_dir / packaged_image),
                    "meta": record.get("meta", {}),
                }
            )

    annotations_out = package_dir / "annotations.jsonl"
    with annotations_out.open("w", encoding="utf-8") as out:
        for record in packaged_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_out = package_dir / "manifest.jsonl"
    with manifest_out.open("w", encoding="utf-8") as out:
        for record in source_manifest:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    task_description = package_dir / "task_description.md"
    if args.task_description:
        shutil.copy2(args.task_description, task_description)
    else:
        task_description.write_text(
            "\n".join(
                [
                    "# Handwritten Formula Recognition Evaluation Set",
                    "",
                    "Task: given one image containing a handwritten mathematical expression, predict the corresponding LaTeX string.",
                    "",
                    "Input format: PaddleOCR-VL message JSONL with one image per sample.",
                    "Prompt: `Formula Recognition:`",
                    "Output format: display math LaTeX, for example `\\[x+1=2\\]`.",
                    "",
                    f"Samples: {len(packaged_records)}",
                    f"License/use note: {args.license_note}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    if args.analysis_report:
        shutil.copy2(args.analysis_report, package_dir / "dataset_analysis.md")

    shutil.copy2(Path("scripts/evaluate_formula.py"), package_dir / "evaluate_formula.py")
    shutil.copy2(Path("scripts/validate_formula_dataset.py"), package_dir / "validate_formula_dataset.py")

    checksums = []
    for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
        checksums.append({"path": str(path.relative_to(package_dir)), "sha256": sha256(path)})
    (package_dir / "SHA256SUMS.json").write_text(json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tar_path = output_dir / f"{args.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_dir, arcname=args.name)

    print(json.dumps({"package_dir": str(package_dir), "tar": str(tar_path), "samples": len(packaged_records)}, indent=2))


if __name__ == "__main__":
    main()
