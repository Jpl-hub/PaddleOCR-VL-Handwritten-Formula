#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

FIXTURE_DIR=${FIXTURE_DIR:-tests/fixtures/tiny_formula}
REPORT_DIR=${REPORT_DIR:-outputs/tiny_report}

python3 -m compileall scripts demo
rm -rf "$FIXTURE_DIR" "$REPORT_DIR"
python3 scripts/create_tiny_fixture.py --output-dir "$FIXTURE_DIR"
python3 scripts/analyze_dataset.py --input "$FIXTURE_DIR/test.jsonl" --output-dir "$REPORT_DIR"
python3 scripts/evaluate_formula.py --predictions "$FIXTURE_DIR/predictions.jsonl"
