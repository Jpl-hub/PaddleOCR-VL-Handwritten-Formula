from __future__ import annotations

import re
from collections import Counter


DISPLAY_RE = re.compile(r"^\\\[(.*)\\\]$", re.DOTALL)
INLINE_RE = re.compile(r"^\\\((.*)\\\)$", re.DOTALL)
SPACE_RE = re.compile(r"\s+")
COMMAND_RE = re.compile(r"\\[a-zA-Z]+|\\.")
TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|\\.|[A-Za-z]+|\d+(?:\.\d+)?|[{}_^=+\-*/(),;:\[\]<>|]"
)


def unwrap_display_math(text: str) -> str:
    value = text.strip()
    for pattern in (DISPLAY_RE, INLINE_RE):
        match = pattern.match(value)
        if match:
            return match.group(1).strip()
    if value.startswith("$$") and value.endswith("$$"):
        return value[2:-2].strip()
    return value


def normalize_latex(text: str) -> str:
    value = unwrap_display_math(text)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\;", "").replace("\\:", "")
    value = value.replace("\\!", "")
    value = SPACE_RE.sub("", value)
    return value


def wrap_display_math(text: str) -> str:
    value = unwrap_display_math(text)
    return f"\\[{value}\\]"


def latex_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_latex(text))


def latex_complexity(text: str) -> dict[str, int]:
    normalized = normalize_latex(text)
    commands = COMMAND_RE.findall(normalized)
    tokens = latex_tokens(normalized)
    nesting = 0
    max_nesting = 0
    for char in normalized:
        if char == "{":
            nesting += 1
            max_nesting = max(max_nesting, nesting)
        elif char == "}":
            nesting = max(0, nesting - 1)
    structural = sum(normalized.count(s) for s in ["\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\matrix", "\\begin"])
    return {
        "chars": len(normalized),
        "tokens": len(tokens),
        "commands": len(commands),
        "max_nesting": max_nesting,
        "structural_ops": structural,
    }


def difficulty_bucket(text: str) -> str:
    stats = latex_complexity(text)
    score = (
        stats["chars"] / 40.0
        + stats["commands"] / 4.0
        + stats["max_nesting"] / 2.0
        + stats["structural_ops"]
    )
    if score < 2.0:
        return "easy"
    if score < 4.5:
        return "medium"
    return "hard"


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_edit_distance(prediction: str, reference: str) -> float:
    pred = normalize_latex(prediction)
    ref = normalize_latex(reference)
    denom = max(len(pred), len(ref), 1)
    return edit_distance(pred, ref) / denom


def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_counts = Counter(latex_tokens(prediction))
    ref_counts = Counter(latex_tokens(reference))
    overlap = sum((pred_counts & ref_counts).values())
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
