#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from latex_utils import normalize_latex, normalized_edit_distance, token_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formula recognition predictions.")
    parser.add_argument("--predictions", required=True, help="JSONL with prediction and ground_truth/reference fields.")
    parser.add_argument("--output-json", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def get_field(record: dict, names: tuple[str, ...]) -> str:
    for name in names:
        if name in record:
            return str(record[name])
    raise KeyError(f"Missing one of {names} in record: {record.keys()}")


def main() -> None:
    args = parse_args()
    path = Path(args.predictions)
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prediction = get_field(record, ("prediction", "pred", "output"))
            reference = get_field(record, ("ground_truth", "reference", "target", "label"))
            ned = normalized_edit_distance(prediction, reference)
            precision, recall, f1 = token_f1(prediction, reference)
            exact = normalize_latex(prediction) == normalize_latex(reference)
            rows.append(
                {
                    "line": line_no,
                    "normalized_edit_distance": ned,
                    "exact_match": exact,
                    "token_precision": precision,
                    "token_recall": recall,
                    "token_f1": f1,
                }
            )

    if not rows:
        raise SystemExit("No prediction rows found.")

    metrics = {
        "samples": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "normalized_edit_distance": sum(row["normalized_edit_distance"] for row in rows) / len(rows),
        "token_precision": sum(row["token_precision"] for row in rows) / len(rows),
        "token_recall": sum(row["token_recall"] for row in rows) / len(rows),
        "token_f1": sum(row["token_f1"] for row in rows) / len(rows),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
