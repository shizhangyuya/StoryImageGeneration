#!/usr/bin/env python3
"""
Gradio UI: OpenAI generates a refined-YAML-style entry → image sequence via gen_benchmark →
horizontal merge with merge_images.merge_pil_horizontally (same layout as merge_images.py).

Dependencies: pip install gradio openai pyyaml

Run from the 1Prompt1Story project root:
  python resource/gradio_app.py

Environment: OPENAI_API_KEY (optional if set in the UI).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _RESOURCE)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

import gradio as gr
import yaml

_SD_STATE: dict[str, Any] = {}


def _load_sd_stack(device: str, model_path: str, precision: str) -> tuple[str, str]:
    """Load pipe + unet_controller like gen_benchmark; skip reload if unchanged."""
    import torch
    from main import load_unet_controller
    from unet import utils

    key = (device, model_path, precision)
    if _SD_STATE.get("key") == key and _SD_STATE.get("pipe") is not None:
        return "Model already in memory (no reload).", device

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    dtype = torch.float16 if precision == "fp16" else torch.float32
    pipe, _, _ = utils.load_pipe_from_path(
        model_path, device, dtype, precision
    )
    unet_controller = load_unet_controller(pipe, device)
    unet_controller.Save_story_image = False
    unet_controller.Prompt_embeds_mode = "single"
    unet_controller.save_self_attention = False
    unet_controller.use_half_precision = True
    unet_controller.max_heads_to_save = 4

    _SD_STATE.clear()
    _SD_STATE["key"] = key
    _SD_STATE["pipe"] = pipe
    _SD_STATE["unet_controller"] = unet_controller
    _SD_STATE["device"] = device

    return f"Loaded model: {model_path} @ {device}", device


def generate_benchmark_entry_openai(
    api_key: str,
    base_url: str,
    model_name: str,
    seq_len: int,
    theme: str,
) -> tuple[str, str]:
    """Call OpenAI Chat Completions; return YAML text and status."""
    try:
        from openai import OpenAI
    except ImportError:
        return "", "Install OpenAI: pip install openai"

    key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return "", "Provide an API key or set OPENAI_API_KEY."

    n = int(seq_len)
    if n < 1 or n > 32:
        return "", "Sequence length must be between 1 and 32."

    client_kwargs: dict[str, Any] = {"api_key": key}
    bu = (base_url or "").strip()
    if bu:
        client_kwargs["base_url"] = bu
    client = OpenAI(**client_kwargs)

    system = (
        "You write structured benchmark entries for a story-image dataset. "
        "Reply with ONLY valid JSON, no markdown."
    )
    user = f"""Create ONE JSON object with exactly these keys:
- concept_token: short English noun phrase for the subject type (lowercase ok).
- settings: array of EXACTLY {n} strings. Each is a concise English scene/action phrase
  (similar to: "grazing alongside a river", gerund-style where natural).
- style: string starting like "A ... portrait of" or "A ... illustration of" (article + adjectives + genre).
- subject: string starting with "A " describing the subject briefly.
- new_subject: richer appearance description WITHOUT a leading "A " (colors, textures, eyes, etc.).

Theme / user hint (may freely interpret): {theme or "any interesting subject"}

Rules: settings length MUST be {n}. All strings in English."""

    try:
        resp = client.chat.completions.create(
            model=model_name.strip() or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as e:
        return "", f"OpenAI request failed: {e}"

    for k in ("concept_token", "settings", "style", "subject", "new_subject"):
        if k not in data:
            return "", f"JSON missing key: {k}"

    if not isinstance(data["settings"], list) or len(data["settings"]) != n:
        return "", f"settings must be a list of length {n}; got: {data.get('settings')}"

    yaml_body = yaml.safe_dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    return yaml_body, "Generated entry from OpenAI (you can edit the YAML below)."


def _parse_instance_yaml(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML must be a single mapping (concept_token, settings, ...)")
    need = ("settings", "subject")
    for k in need:
        if k not in data:
            raise ValueError(f"Missing required field: {k}")
    if not isinstance(data["settings"], list) or not data["settings"]:
        raise ValueError("settings must be a non-empty list")
    return data


def _build_id_prompts(
    refined_instance: dict[str, Any], use_new_subject: bool
) -> tuple[str, str, str]:
    """Return (generation_id_prompt, filename_id_prompt, subject_text); same logic as gen_benchmark."""
    refined_style = refined_instance.get("style", "")
    if isinstance(refined_style, str):
        refined_style = refined_style.strip()
    refined_subject = refined_instance.get("subject", "")
    subject_for_generation = (
        refined_instance.get("new_subject", refined_subject)
        if use_new_subject
        else refined_subject
    )
    if refined_style:
        generation_id_prompt = f"{refined_style} {subject_for_generation}"
        filename_id_prompt = f"{refined_style} {refined_subject}"
    else:
        generation_id_prompt = subject_for_generation
        filename_id_prompt = refined_subject
    subject_text = refined_subject
    return generation_id_prompt, filename_id_prompt, subject_text


def run_generation(
    yaml_text: str,
    device: str,
    model_path: str,
    precision: str,
    window_length: int,
    seed: int,
    use_new_subject: bool,
    output_root: str,
) -> tuple[Any, Any, str, str]:
    """Generate the sequence and merge; return gallery, merged PIL, note, save dir."""
    import gen_benchmark
    from merge_images import merge_pil_horizontally

    try:
        inst = _parse_instance_yaml(yaml_text)
    except Exception as e:
        return [], None, f"YAML parse error: {e}", ""

    msg, _ = _load_sd_stack(device.strip(), model_path.strip(), precision)
    pipe = _SD_STATE["pipe"]
    unet_controller = _SD_STATE["unet_controller"]

    gen_id, file_id, subject_text = _build_id_prompts(inst, use_new_subject)
    frame_prompt_list = [str(s) for s in inst["settings"]]
    wl = int(window_length)
    sd = int(seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        os.path.expanduser(output_root.strip() or "./results/gradio_ui"),
        f"run_{stamp}",
    )
    os.makedirs(save_dir, exist_ok=True)

    try:
        images, _story = gen_benchmark.main_ben(
            unet_controller,
            pipe,
            save_dir,
            gen_id,
            frame_prompt_list,
            sd,
            wl,
            original_id_prompt=file_id if file_id != gen_id else None,
        )
    except Exception as e:
        return [], None, f"{msg}\nGeneration failed: {e}", save_dir

    merged = merge_pil_horizontally(images, gen_id, frame_prompt_list)
    merged_path = os.path.join(save_dir, "merged_display.jpg")
    merged.save(merged_path, "JPEG", quality=95)

    gallery = [(img, cap) for img, cap in zip(images, frame_prompt_list)]
    note = (
        f"{msg}\nSave directory: {save_dir}\n"
        f"id_prompt: {gen_id}\nFrames: {len(images)}\nMerged image: {merged_path}"
    )
    return gallery, merged, note, save_dir


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="1Prompt1Story · Gradio") as demo:
        gr.Markdown(
            "## 1Prompt1Story UI\n"
            "1. Use OpenAI to produce one YAML entry matching the refined benchmark schema "
            "(`concept_token`, `settings`, `style`, `subject`, `new_subject`).\n"
            "2. Load the diffusion stack, then generate the frame sequence using `gen_benchmark.py` logic.\n"
            "3. Preview a horizontal strip with captions via `merge_images.merge_pil_horizontally` "
            "(same layout as `merge_images.py`)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Or use OPENAI_API_KEY in the environment",
                )
                base_url = gr.Textbox(
                    label="API Base URL (optional)",
                    placeholder="Leave empty for default; e.g. https://api.openai.com/v1",
                )
                oa_model = gr.Textbox(value="gpt-4o-mini", label="OpenAI model")
                seq_len = gr.Slider(1, 16, value=5, step=1, label="settings sequence length")
                theme = gr.Textbox(
                    label="Theme / creative hint",
                    placeholder="e.g. mechanical butterflies in a rainforest, ink-wash teapot …",
                )
                btn_prompt = gr.Button("Step 1: Generate YAML with OpenAI")
            with gr.Column(scale=2):
                yaml_box = gr.Code(
                    label="Entry YAML (editable)",
                    language="yaml",
                    lines=24,
                )
                prompt_status = gr.Textbox(label="Prompt step status", lines=2)

        gr.Markdown("### Image generation (same settings as gen_benchmark)")
        with gr.Row():
            device = gr.Textbox(value="cuda:0", label="Device")
            sd_model = gr.Textbox(
                value="stabilityai/stable-diffusion-xl-base-1.0",
                label="Model path / Hugging Face ID",
            )
            precision = gr.Dropdown(choices=["fp16", "fp32"], value="fp16", label="Precision")
        with gr.Row():
            window_length = gr.Slider(2, 20, value=10, step=1, label="window_length")
            seed = gr.Number(value=42, label="Random seed", precision=0)
            use_new_subject = gr.Checkbox(
                value=True,
                label="Use new_subject for generation (same as --use_new_subject)",
            )
            output_root = gr.Textbox(
                value="./results/gradio_ui",
                label="Output root directory",
            )
        with gr.Row():
            btn_load = gr.Button("Load diffusion model")
            load_status = gr.Textbox(label="Load status", lines=2)
            btn_gen = gr.Button(
                "Step 2: Generate frames from YAML and merge",
                variant="primary",
            )

        gen_status = gr.Textbox(label="Generation status", lines=4)
        save_dir_txt = gr.Textbox(label="Last save directory", lines=1)

        with gr.Row():
            merged_img = gr.Image(
                label="Merged strip (merge_pil_horizontally)",
                type="pil",
            )
            gallery = gr.Gallery(label="Per-frame images", columns=4, height="auto")

        btn_prompt.click(
            generate_benchmark_entry_openai,
            inputs=[api_key, base_url, oa_model, seq_len, theme],
            outputs=[yaml_box, prompt_status],
        )

        btn_load.click(
            lambda d, m, p: _load_sd_stack(d, m, p)[0],
            inputs=[device, sd_model, precision],
            outputs=[load_status],
        )

        btn_gen.click(
            run_generation,
            inputs=[
                yaml_box,
                device,
                sd_model,
                precision,
                window_length,
                seed,
                use_new_subject,
                output_root,
            ],
            outputs=[gallery, merged_img, gen_status, save_dir_txt],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
