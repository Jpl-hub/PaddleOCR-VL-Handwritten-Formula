from __future__ import annotations

import os
from functools import lru_cache

import gradio as gr
from PIL import Image


DEFAULT_PROMPT = "Formula Recognition:"


@lru_cache(maxsize=1)
def load_model(model_path: str, device: str):
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


def recognize(image: Image.Image, prompt: str, model_path: str, device: str, max_new_tokens: int) -> str:
    if image is None:
        return "Please upload an image."
    if not model_path:
        return "Set MODEL_PATH or fill the model path field."
    paddle, model, processor, generation_config = load_model(model_path, device)
    image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt or DEFAULT_PROMPT},
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


with gr.Blocks(title="Handwritten Formula Recognition") as demo:
    gr.Markdown("# Handwritten Formula Recognition")
    with gr.Row():
        image = gr.Image(type="pil", label="Formula image")
        with gr.Column():
            prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt")
            model_path = gr.Textbox(value=os.environ.get("MODEL_PATH", ""), label="Model path")
            device = gr.Radio(["gpu", "cpu"], value=os.environ.get("PADDLE_DEVICE", "gpu"), label="Device")
            max_new_tokens = gr.Slider(64, 1024, value=512, step=32, label="Max new tokens")
            run = gr.Button("Recognize", variant="primary")
    output = gr.Textbox(label="LaTeX prediction", lines=6)
    run.click(recognize, [image, prompt, model_path, device, max_new_tokens], output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
