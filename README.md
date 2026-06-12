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

The blend target is validated against real hybrids in
[orchid-clip-v8](https://huggingface.co/mjarnold/orchid-clip-v8) latent space: real hybrids sit
near the parent midpoint, and the residual deviation is transgressive (novel beyond both
parents), not a lean toward one parent (Stage 16/18).

**Click any image** to see its parent species and the exact prompt that produced it. These are
_predictions of hypothetical appearance_, not photographs.

This is a free, pre-rendered gallery (seed 42, F1 depth). A live interactive generator — custom
parent pairs, warm-color control, multiple seeds — lives in `app_live.py`; it runs SDXL
in-Space and needs ZeroGPU hardware (HF PRO).

- **Base model:** `stabilityai/stable-diffusion-xl-base-1.0`
- **LoRA:** [`mjarnold/orchid-ancestry-lora-v2`](https://huggingface.co/mjarnold/orchid-ancestry-lora-v2)
- **Rendered by:** `render_gallery.py` (diffusers 0.31, the version regime the LoRA was validated under)
