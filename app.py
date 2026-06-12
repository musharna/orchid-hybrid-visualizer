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

import latentmap
from refs import species_name

_HERE = os.path.dirname(os.path.abspath(__file__))
_GALLERY_DIR = os.path.join(_HERE, "gallery")
_PARENTS_DIR = os.path.join(_HERE, "parents")


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
        if entry.get("ref_kind") == "search":
            label = f"🔍 Search real photos of {species_name(entry['display'])}"
        else:
            label = "Reference photo of the real registered cross"
        lines += ["", f"[{label}]({entry['ref_url']})"]
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


# --- parent species thumbnails (licensing-clean photos from the curated dataset) ---
def _load_parent_meta() -> dict:
    p = os.path.join(_PARENTS_DIR, "parents.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


_PARENT_META = _load_parent_meta()


def _epithet(species: str) -> str:
    return species.replace("_aurea", "")


def parent_items(entry: dict) -> list[tuple]:
    """[(thumb_path, 'C. <epithet>  ·  <pct>%')] for a cross's parent species."""
    items = []
    for sp, pct in entry.get("ancestry", {}).items():
        ep = _epithet(sp)
        meta = _PARENT_META.get(ep)
        if not meta:
            continue
        path = os.path.join(_PARENTS_DIR, meta["file"])
        if os.path.exists(path):
            items.append((path, f"C. {ep} · {int(pct)}%"))
    return items


def parent_credits(entry: dict) -> str:
    """Attribution lines for the parent photos shown (CC-BY compliance)."""
    lines = []
    for sp in entry.get("ancestry", {}):
        meta = _PARENT_META.get(_epithet(sp))
        if meta:
            lines.append(f"- *C. {_epithet(sp)}* — {meta['credit']}")
    if not lines:
        return ""
    return "**Parent photo credits:**\n" + "\n".join(lines)


# --- multi-seed variation strips (rendered on GPU; optional) ---
def _load_seeds() -> dict:
    p = os.path.join(_GALLERY_DIR, "seeds", "seeds_manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f).get("crosses", {})


_SEEDS = _load_seeds()


def seed_items(stem: str) -> list[str]:
    files = _SEEDS.get(stem, [])
    out = []
    for fn in files:
        path = os.path.join(_GALLERY_DIR, "seeds", fn)
        if os.path.exists(path):
            out.append(path)
    return out


# --- latent map ---
_COORDS = latentmap.load_coords()
_COORDS_BY_DISPLAY = {c["display"]: c for c in _COORDS}
_IMG_BY_DISPLAY = {e["display"]: e["image_path"] for e in _ENTRIES}
# default the latent dropdown to a cross that has the real-hybrid transgression overlay
_LATENT_CHOICES = [c["display"] for c in _COORDS]
_LATENT_DEFAULT = next(
    (c["display"] for c in _COORDS if c.get("hybrid")),
    (_LATENT_CHOICES[0] if _LATENT_CHOICES else None),
)


def on_gallery_select(evt: gr.SelectData):
    if not _ENTRIES or evt.index is None or evt.index >= len(_ENTRIES):
        return "", [], "", []
    e = _ENTRIES[evt.index]
    return detail_md(e), parent_items(e), parent_credits(e), seed_items(e["stem"])


def on_latent_select(display: str):
    entry = _COORDS_BY_DISPLAY.get(display)
    if entry is None:
        return None, "", None
    return (
        latentmap.build_figure(entry),
        latentmap.caption(entry),
        _IMG_BY_DISPLAY.get(display),
    )


_ABOUT = (
    "## How it works\n\n"
    "1. **Phenotype engine** — for a chosen cross, the two parent species' trait profiles "
    "(pigment channels: anthocyanin / carotenoid / co-pigment, plus genetic dominance rules) "
    "are blended at F1 depth into a ~77-token CLIP-optimized appearance description.\n"
    "2. **SDXL + ancestry LoRA** — that prompt drives Stable Diffusion XL with a custom Cattleya "
    "ancestry LoRA (scale 0.6) to render the predicted hybrid (seed 42).\n"
    "3. **Validation** — every blend is checked against real hybrids in "
    "[orchid-clip-v8](https://huggingface.co/mjarnold/orchid-clip-v8) latent space. Real hybrids "
    "sit near the **parent midpoint**, and the residual deviation is **transgressive** (novel "
    "beyond both parents), not a lean toward one parent — see the **Latent map** tab. This "
    "replicated under permutation tests and a DINOv2 backbone (Stage 16/18).\n\n"
    "These are *predictions of hypothetical appearance*, not photographs.\n\n"
    "## More in this orchid series\n\n"
    "- 🔬 [orchid-clip-v8](https://huggingface.co/mjarnold/orchid-clip-v8) — the vision backbone "
    "(1.14M photos / 5,124 species) behind the validation here\n"
    "- 🌱 [orchid-genus-id](https://huggingface.co/spaces/mjarnold/orchid-genus-id) — live genus "
    "identification from a photo\n"
    "- 🌸 this Space — hybrid appearance prediction\n\n"
    "Parent reference photos are licensing-clean (CC-BY / public-domain) from a curated dataset; "
    "per-photo credits appear with each cross. A live interactive generator (custom parent pairs, "
    "warm-color control, multiple seeds) exists as `app_live.py` and needs a ZeroGPU Space (HF PRO)."
)


with gr.Blocks(title="Orchid Hybrid Visualizer") as demo:
    gr.Markdown(_INTRO)
    if not _ENTRIES:
        gr.Markdown(
            "⚠️ Gallery assets not found (run render_gallery.py to populate gallery/)."
        )

    with gr.Tabs():
        with gr.Tab("🌸 Gallery"):
            gallery = gr.Gallery(
                value=gallery_items(_ENTRIES),
                columns=3,
                height=620,
                label="Predicted Cattleya hybrids — click one",
                show_label=True,
                object_fit="cover",
            )
            detail = gr.Markdown()
            with gr.Row():
                parents_gallery = gr.Gallery(
                    label="Parent species (real photos)",
                    columns=4,
                    height=200,
                    object_fit="cover",
                    show_label=True,
                )
                seeds_gallery = gr.Gallery(
                    label="Model variations (same cross, different draws)",
                    columns=4,
                    height=200,
                    object_fit="cover",
                    show_label=True,
                )
            credits = gr.Markdown()
            gallery.select(
                on_gallery_select,
                None,
                [detail, parents_gallery, credits, seeds_gallery],
            )

        with gr.Tab("🧬 Latent map"):
            gr.Markdown(
                "Where each predicted hybrid lands in **orchid-clip-v8** latent space. The two "
                "parents sit at the ends of the horizontal **chord**; the predicted F1 blend is the "
                "**midpoint** (0,0). For crosses with real examples in the dataset, the **real "
                "hybrid** (★) is plotted *perpendicular* to the chord — its off-chord offset is the "
                "**transgressive residual** (validated, Stage 16/18)."
            )
            if _LATENT_DEFAULT:
                latent_pick = gr.Dropdown(
                    choices=_LATENT_CHOICES, value=_LATENT_DEFAULT, label="Cross"
                )
                with gr.Row():
                    latent_plot = gr.Plot(
                        value=latentmap.build_figure(
                            _COORDS_BY_DISPLAY[_LATENT_DEFAULT]
                        ),
                        label="Parent chord-plane",
                    )
                    latent_img = gr.Image(
                        value=_IMG_BY_DISPLAY.get(_LATENT_DEFAULT),
                        label="Predicted hybrid",
                        height=460,
                    )
                latent_caption = gr.Markdown(
                    latentmap.caption(_COORDS_BY_DISPLAY[_LATENT_DEFAULT])
                )
                latent_pick.change(
                    on_latent_select,
                    latent_pick,
                    [latent_plot, latent_caption, latent_img],
                )
            else:
                gr.Markdown(
                    "⚠️ Latent coordinates not found (run build_latent_coords.py)."
                )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(_ABOUT)

if __name__ == "__main__":
    demo.launch()
