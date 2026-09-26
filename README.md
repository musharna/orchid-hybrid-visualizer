---
title: Orchid Hybrid Visualizer
emoji: 🌸
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
pinned: false
license: mit
short_description: Predicted appearances of 27 Cattleya hybrids
---

# Cattleya hybrid visualizer

Predicted appearances of 27 registered _Cattleya_ crosses. A rule-based phenotype engine
blends the two parent species' traits (pigment channels and dominance rules, which are
heuristics) into a short prompt, and SDXL with a _Cattleya_ ancestry LoRA renders it. Every
image is a prediction, not a photograph.

## Tabs

- **Gallery**: the 27 predicted hybrids. Click one for photos of the parent species (credits in
  [`parents/CREDITS.md`](parents/CREDITS.md); one is CC BY-NC), the prompt used, a reference
  link, and four more draws of the same cross.
- **Latent map**: each cross's parents in [orchid-clip-v8](https://huggingface.co/musharna/orchid-clip-v8)
  embedding space, at the two ends of a line, with the prediction drawn at their midpoint. For
  the 4 crosses with real hybrid photos, the real hybrid is plotted too. The predicted images
  themselves are not embedded.
- **About**: how the pipeline works.

## What the embeddings show

Across 1,002 registered orchid hybrids, a real hybrid's orchid-clip-v8 embedding sits closer to
its parents' midpoint than a shuffled null (cosine 0.910 vs 0.730), and this holds on DINOv2
(0.886 vs 0.539). Hybrids also sit off the line between their parents, and further off when the
parents look more different (Spearman ρ 0.52, n = 485, permutation p < 0.001; 0.65 on DINOv2).
This is visual similarity, not genetics, and weighting the blend by the dominance rules did not
beat the plain midpoint.

The gallery is pre-rendered (seed 42, F1) and runs on free CPU hardware. `app_live.py` is a live
version with custom parent pairs and more seeds; it needs a GPU (ZeroGPU) Space.

- **Base model:** `stabilityai/stable-diffusion-xl-base-1.0`
- **LoRA:** [`musharna/orchid-ancestry-lora-v2`](https://huggingface.co/musharna/orchid-ancestry-lora-v2)
- **Rendered by:** `render_gallery.py` / `render_seeds.py` (diffusers 0.31, the version the LoRA was tested with)
- **Also in this series:** [orchid-clip-v8](https://huggingface.co/musharna/orchid-clip-v8) · [orchid-genus-id](https://huggingface.co/spaces/musharna/orchid-genus-id)

## License

**Code** is MIT. **Bundled assets are not** — the parent photographs, the
textual-inversion tokens, and the rendered gallery each keep their own terms,
and one parent photo is CC BY-NC (NonCommercial), so the repository as a whole
is not commercially reusable.

See [`LICENSE`](LICENSE) for the scope statement and
[`ASSETS-LICENSE.md`](ASSETS-LICENSE.md) for the full breakdown.
