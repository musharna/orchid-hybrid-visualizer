"""Orchid Hybrid Visualizer — public ZeroGPU Space (core generator).

Pick two Cattleya parent species (or a famous RHS cross) -> a CLIP-optimized appearance
prompt via the botanical phenotype engine -> SDXL + v2 ancestry LoRA -> image. Science:
real hybrids blend toward the parent midpoint in orchid-clip-v8 space (Stage 16/18).
"""

import os
import threading

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

import pipeline as P
from phenotype_engine import PhenotypeEngine
from stage3_tokens import hue_to_token
from crosses import CROSSES_WITH_DISPLAY

_ENGINE = PhenotypeEngine()
SPECIES = sorted(_ENGINE.db.keys())
_PRESETS = {display: ancestry for _stem, display, ancestry in CROSSES_WITH_DISPLAY}
_DEPTH = {"F1": 1, "F2": 2, "F3": 3}

_LORA_REPO = "mjarnold/orchid-ancestry-lora-v2"
_LORA_FILE = "orchid-ancestry-lora-v2.safetensors"
_TOKEN_DIR = os.path.join(os.path.dirname(__file__), "tokens")

_BUILT = None
_BUILD_LOCK = threading.Lock()


def _get_built():
    """Lazily build the SDXL pipeline (downloads the LoRA on first call). Thread-safe."""
    global _BUILT
    if _BUILT is None:
        with _BUILD_LOCK:
            if _BUILT is None:
                lora_path = hf_hub_download(_LORA_REPO, _LORA_FILE)
                _BUILT = P.build_pipeline(lora_path, _TOKEN_DIR)
    return _BUILT


def build_request(p1, pct1, p2, pct2, preset, depth):
    """Pure (no GPU): resolve UI inputs -> (ancestry, prompt, err). On error returns
    (None, None, msg)."""
    if preset and preset != "(none)":
        ancestry = dict(_PRESETS[preset])
    else:
        if not p1 or not p2:
            return None, None, "Pick two parent species (or choose a preset cross)."
        if p1 == p2:
            return None, None, "Parent 1 and Parent 2 must be different species."
        ancestry = {p1: float(pct1), p2: float(pct2)}
    err = P.validate_ancestry(ancestry)
    if err:
        return None, None, err
    prompt = _ENGINE.describe(ancestry, generation=_DEPTH.get(depth, 1))
    return ancestry, prompt, None


@spaces.GPU(duration=120)
def _run(ancestry, prompt, seed, num_images, lora_scale, hue_token, warmth):
    built = _get_built()
    built.pipe.to("cuda")
    return P.generate_images(
        built,
        ancestry,
        prompt=prompt,
        seed=int(seed),
        guidance_scale=5.0,
        num_images=int(num_images),
        steps=30,
        lora_scale=lora_scale,
        hue_token=hue_token,
        warmth=float(warmth),
    )


def generate(p1, pct1, p2, pct2, preset, depth, hue, warmth, seed, num_images):
    ancestry, prompt, err = build_request(p1, pct1, p2, pct2, preset, depth)
    if err:
        return [], err
    hue_token = hue_to_token(hue)
    lora_scale = 0.0 if hue == "red" else 0.6  # red arm is no_lora (stage3_tokens)
    imgs = _run(ancestry, prompt, seed, num_images, lora_scale, hue_token, warmth)
    return imgs, prompt


with gr.Blocks(title="Orchid Hybrid Visualizer") as demo:
    gr.Markdown(
        "# 🌸 Cattleya Orchid Hybrid Visualizer\n"
        "Predict the appearance of a hypothetical Cattleya cross. Pick two parent species "
        "or a famous registered cross. Backbone: hybrids blend toward the parent midpoint "
        "in orchid-clip-v8 space (validated, Stage 16/18)."
    )
    with gr.Row():
        with gr.Column():
            preset = gr.Dropdown(
                ["(none)"] + sorted(_PRESETS.keys()),
                value="(none)",
                label="Famous RHS cross (overrides the parents below)",
            )
            p1 = gr.Dropdown(SPECIES, label="Parent 1")
            pct1 = gr.Slider(0, 100, value=50, step=5, label="Parent 1 %")
            p2 = gr.Dropdown(SPECIES, label="Parent 2")
            pct2 = gr.Slider(0, 100, value=50, step=5, label="Parent 2 %")
            depth = gr.Radio(["F1", "F2", "F3"], value="F1", label="Generation depth")
            hue = gr.Radio(
                ["none", "yellow", "red"], value="none", label="Warm-color hint"
            )
            warmth = gr.Slider(0.0, 1.0, value=0.6, step=0.1, label="Warmth strength")
            seed = gr.Number(value=42, label="Seed", precision=0)
            num_images = gr.Slider(1, 3, value=2, step=1, label="Images")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Predicted hybrid", columns=2, height=600)
            prompt_box = gr.Textbox(label="Prompt used", interactive=False)
    btn.click(
        generate,
        [p1, pct1, p2, pct2, preset, depth, hue, warmth, seed, num_images],
        [gallery, prompt_box],
    )

if __name__ == "__main__":
    demo.launch()
