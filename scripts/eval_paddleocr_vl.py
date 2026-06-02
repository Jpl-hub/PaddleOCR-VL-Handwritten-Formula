#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PROMPT = "Formula Recognition:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL formula inference on a JSONL dataset.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_backend(model_path: str, device: str):
    import paddle
    from paddleformers.generation import GenerationConfig
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

    paddle.set_device(device)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(model_path, convert_from_hf=True)
    model.config._attn_implementation = "flashmask"
    if hasattr(model, "visual"):
        model.visual.config._attn_implementation = "flashmask"
    model.eval()
    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    return paddle, model, processor, generation_config


def assistant_label(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def user_prompt(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            content = str(message.get("content", ""))
            return content.replace("<image>", "") or PROMPT
    return PROMPT


def generate(paddle, model, processor, generation_config, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pd",
    )
    with paddle.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config, max_new_tokens=max_new_tokens)
        output_ids = outputs[0].tolist()[0]
    return processor.decode(output_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    root = Path(args.dataset_root) if args.dataset_root else data_path.parent
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paddle, model, processor, generation_config = load_backend(args.model_name_or_path, args.device)

    start = time.time()
    count = 0
    with data_path.open("r", encoding="utf-8") as fh, output_path.open("w", encoding="utf-8") as out:
        for line in tqdm(fh, desc="infer"):
            if not line.strip():
                continue
            if args.limit and count >= args.limit:
                break
            record = json.loads(line)
            image_rel = record.get("images", [""])[0]
            image_path = Path(image_rel)
            if not image_path.is_absolute():
                image_path = root / image_path
            image = Image.open(image_path).convert("RGB")
            prediction = generate(
                paddle,
                model,
                processor,
                generation_config,
                image,
                user_prompt(record),
                args.max_new_tokens,
            )
            out.write(
                json.dumps(
                    {
                        "image": image_rel,
                        "prediction": prediction,
                        "ground_truth": assistant_label(record),
                        "meta": record.get("meta", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    elapsed = time.time() - start
    print(json.dumps({"samples": count, "seconds": elapsed, "output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
