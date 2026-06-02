from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import gradio as gr


DATASET_DIR = Path(os.environ.get("ANNOTATION_DATASET_DIR", "data/real_formula_collection"))
IMAGE_DIR = DATASET_DIR / "images" / "eval"
ANNOTATIONS_PATH = DATASET_DIR / "eval.jsonl"
PROMPT = "Formula Recognition:"


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(label: str) -> str:
    value = (label or "").strip()
    if not value:
        return ""
    if value.startswith("\\[") and value.endswith("\\]"):
        return value
    return f"\\[{value}\\]"


def save_annotation(image_path: str, latex: str, source: str, difficulty: str, reviewer: str) -> str:
    ensure_dirs()
    if not image_path:
        return "Upload an image first."
    label = normalize_label(latex)
    if not label:
        return "LaTeX label is required."
    sample_id = uuid.uuid4().hex[:16]
    suffix = Path(image_path).suffix or ".png"
    dst_rel = Path("images") / "eval" / f"{sample_id}{suffix}"
    dst = DATASET_DIR / dst_rel
    shutil.copy2(image_path, dst)
    record = {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": label},
        ],
        "images": [str(dst_rel)],
        "meta": {
            "sample_id": sample_id,
            "source": source.strip() or "manual_collection",
            "difficulty": difficulty,
            "reviewer": reviewer.strip(),
            "collection_tool": "local_annotation_app",
        },
    }
    with ANNOTATIONS_PATH.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"Saved {sample_id}. Total samples: {count_samples()}"


def count_samples() -> int:
    if not ANNOTATIONS_PATH.exists():
        return 0
    return sum(1 for line in ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip())


def preview_last() -> str:
    if not ANNOTATIONS_PATH.exists():
        return "No annotations yet."
    lines = [line for line in ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "No annotations yet."
    return json.dumps(json.loads(lines[-1]), ensure_ascii=False, indent=2)


with gr.Blocks(title="Formula Dataset Annotation") as demo:
    gr.Markdown("# Formula Dataset Annotation")
    with gr.Row():
        image = gr.Image(type="filepath", label="Formula image")
        with gr.Column():
            latex = gr.Textbox(label="LaTeX", lines=4, placeholder=r"\frac{x+1}{2}")
            source = gr.Textbox(label="Source/provenance", placeholder="volunteer_phone_photo")
            difficulty = gr.Radio(["easy", "medium", "hard"], value="medium", label="Difficulty")
            reviewer = gr.Textbox(label="Reviewer")
            save = gr.Button("Save annotation", variant="primary")
    status = gr.Textbox(label="Status")
    last = gr.Textbox(label="Last saved JSON", lines=12)
    save.click(save_annotation, [image, latex, source, difficulty, reviewer], status).then(preview_last, None, last)


if __name__ == "__main__":
    ensure_dirs()
    demo.launch(server_name="0.0.0.0")
