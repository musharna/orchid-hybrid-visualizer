---
title: Orchid Hybrid Visualizer
emoji: 🌸
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Predicted appearances of 27 famous Cattleya hybrids
---

# 🌸 Cattleya Orchid Hybrid Visualizer

AI-predicted appearances of **27 famous registered Cattleya crosses**. For each cross, a
botanical phenotype engine blends the two parent species' trait profiles (pigment channels +
genetic dominance rules) into a CLIP-optimized appearance description, which is rendered with
**SDXL + a custom Cattleya ancestry LoRA**.

These are _predictions of hypothetical appearance_, not photographs.

## Three tabs

- **🌸 Gallery** — browse the 27 predicted hybrids. Click one to see its **real parent-species
  photos** (CC BY / CC0 / public-domain, per-photo credits in [`parents/CREDITS.md`](parents/CREDITS.md); one fallback photo is CC BY-NC), the exact prompt used, a
  reference link, and a **variation strip** (4 different model draws of the same cross).
- **🧬 Latent map** — the science, made interactive. For each cross, the two parents sit at the
  ends of a horizontal _chord_ and the predicted F1 blend at the **midpoint**. For crosses with
  real examples in [orchid-clip-v8](https://huggingface.co/mjarnold/orchid-clip-v8) latent space,
  the **real hybrid** is plotted _perpendicular_ to the chord — its off-chord offset is the
  **transgressive residual** (novel beyond both parents, not a lean toward one parent). This
  replicated under permutation tests and a DINOv2 backbone (Stage 16/18).
- **ℹ️ About** — the phenotype-engine pipeline and links to the rest of the orchid series.

This is a free, pre-rendered gallery (seed 42, F1 depth). A live interactive generator — custom
parent pairs, warm-color control, multiple seeds — lives in `app_live.py`; it runs SDXL
in-Space and needs ZeroGPU hardware (HF PRO).

- **Base model:** `stabilityai/stable-diffusion-xl-base-1.0`
- **LoRA:** [`mjarnold/orchid-ancestry-lora-v2`](https://huggingface.co/mjarnold/orchid-ancestry-lora-v2)
- **Rendered by:** `render_gallery.py` / `render_seeds.py` (diffusers 0.31, the regime the LoRA was validated under)
- **Also in this series:** [orchid-clip-v8](https://huggingface.co/mjarnold/orchid-clip-v8) · [orchid-genus-id](https://huggingface.co/spaces/mjarnold/orchid-genus-id)

## License

**Code** is MIT. **Bundled assets are not** — the parent photographs, the
textual-inversion tokens, and the rendered gallery each keep their own terms,
and one parent photo is CC BY-NC (NonCommercial), so the repository as a whole
is not commercially reusable.

See [`LICENSE`](LICENSE) for the scope statement and
[`ASSETS-LICENSE.md`](ASSETS-LICENSE.md) for the full breakdown.
