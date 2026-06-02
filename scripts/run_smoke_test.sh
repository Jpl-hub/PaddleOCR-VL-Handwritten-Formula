#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

FIXTURE_DIR=${FIXTURE_DIR:-outputs/tiny_formula_fixture}
REPORT_DIR=${REPORT_DIR:-outputs/tiny_report}
if [ -z "${PYTHON:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif [ -x /root/miniconda3/bin/python ]; then
    PYTHON=/root/miniconda3/bin/python
  else
    echo "No Python executable found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

"$PYTHON" -m compileall scripts demo
rm -rf "$FIXTURE_DIR" "$REPORT_DIR"
"$PYTHON" scripts/create_tiny_fixture.py --output-dir "$FIXTURE_DIR"
"$PYTHON" scripts/analyze_dataset.py --input "$FIXTURE_DIR/test.jsonl" --output-dir "$REPORT_DIR"
"$PYTHON" scripts/evaluate_formula.py --predictions "$FIXTURE_DIR/predictions.jsonl"
