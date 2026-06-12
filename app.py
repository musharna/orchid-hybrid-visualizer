"""Orchid Hybrid Visualizer — pre-rendered gallery (free CPU Space).

Browse SDXL predictions of the 27 famous Cattleya crosses, each rendered once on a GPU with
the exact shipping pipeline (SDXL base + v2 ancestry LoRA at scale 0.6, F1 depth, seed 42 —
see render_gallery.py). Every entry shows the predicted-hybrid image, the parent species +
percentages, and the ~77-token CLIP prompt that produced it.

This is the free, no-GPU variant. The live interactive generator (custom parents, warm-color
control, multiple seeds) is app_live.py — it runs SDXL in-Space and needs a ZeroGPU Space,
which requires an HF PRO account.
"""

import json
import os

import gradio as gr

_GALLERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")


def load_manifest(gallery_dir: str = _GALLERY_DIR) -> list[dict]:
    """Parse gallery/manifest.json -> list of cross entries, each with an absolute
    image_path. Raises FileNotFoundError if the manifest is absent."""
    with open(os.path.join(gallery_dir, "manifest.json")) as f:
        data = json.load(f)
    entries = []
    for c in data["crosses"]:
        e = dict(c)
        e["image_path"] = os.path.join(gallery_dir, c["image"])
        entries.append(e)
    return entries


def gallery_items(entries: list[dict]) -> list[tuple]:
    """[(image_path, caption)] for gr.Gallery."""
    return [(e["image_path"], e["display"]) for e in entries]


def detail_md(entry: dict) -> str:
    """Markdown detail panel for one selected cross."""
    lines = [
        f"### {entry['display']}",
        f"**Parents:** {entry['parents']}",
        "",
        f"**Prompt used:** {entry['prompt']}",
    ]
    if entry.get("ref_url"):
        lines += [
            "",
            f"[Reference photo of the real registered cross]({entry['ref_url']})",
        ]
    return "\n".join(lines)


_HAS_MANIFEST = os.path.exists(os.path.join(_GALLERY_DIR, "manifest.json"))
_ENTRIES = load_manifest() if _HAS_MANIFEST else []

_INTRO = (
    "# 🌸 Cattleya Orchid Hybrid Visualizer\n\n"
    "AI-predicted appearances of **27 famous registered Cattleya crosses**. Each image is "
    "generated from the two parent species by a botanical phenotype engine (blending pigment "
    "channels and dominance rules into a CLIP-optimized prompt) feeding **SDXL + a custom "
    "orchid-ancestry LoRA**. Science backbone: real hybrids blend toward the parent midpoint "
    "in [orchid-clip-v8](https://huggingface.co/mjarnold/orchid-clip-v8) latent space "
    "(validated, Stage 16/18).\n\n"
    "**Click any image** to see its parents and the exact prompt that produced it. These are "
    "*predictions of hypothetical appearance*, not photographs.\n\n"
    "_Pre-rendered (seed 42, F1). A live interactive version — custom parent pairs, warm-color "
    "control, multiple seeds — exists as `app_live.py` (needs a ZeroGPU Space / HF PRO)._"
)


def _on_select(evt: gr.SelectData) -> str:
    if not _ENTRIES or evt.index is None or evt.index >= len(_ENTRIES):
        return ""
    return detail_md(_ENTRIES[evt.index])


with gr.Blocks(title="Orchid Hybrid Visualizer") as demo:
    gr.Markdown(_INTRO)
    if not _ENTRIES:
        gr.Markdown(
            "⚠️ Gallery assets not found (run render_gallery.py to populate gallery/)."
        )
    gallery = gr.Gallery(
        value=gallery_items(_ENTRIES),
        columns=3,
        height=720,
        label="Predicted Cattleya hybrids",
        show_label=True,
        object_fit="cover",
    )
    detail = gr.Markdown()
    gallery.select(_on_select, None, detail)

if __name__ == "__main__":
    demo.launch()
