"""Render the 27 named Cattleya crosses once on a GPU, for the pre-rendered gallery Space.

Reuses the exact shipping pipeline (pipeline.build_pipeline + generate_images) so the
gallery shows precisely what the live ZeroGPU Space would produce: SDXL base + v2 ancestry
LoRA(0.6) + stage-3 token injection, F1 depth, seed 42, 30 steps, guidance 5.0.

Writes gallery/<stem>.jpg + gallery/manifest.json. The prompt stored in the manifest is the
one actually passed to the pipeline (composed here, then handed in), so it cannot drift.

Run on a GPU host (desktop) with the local SDXL base + v2 LoRA:

  venv-render/bin/python render_gallery.py \
    --base ~/orchid-sdxl/models/sdxl-base-1.0 \
    --lora ~/orchid-sdxl/output_v2/orchid-ancestry-lora-v2.safetensors \
    --out gallery
"""

from __future__ import annotations

import argparse
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import pipeline as P  # noqa: E402
from crosses import CROSSES  # noqa: E402


def _parents_label(ancestry: dict) -> str:
    """'dowiana 50% × warscewiczii 50%' — ordered by descending percentage."""
    parts = sorted(ancestry.items(), key=lambda kv: -float(kv[1]))
    return " × ".join(f"{sp} {int(pct)}%" for sp, pct in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="local SDXL base dir or Hub id")
    ap.add_argument("--lora", required=True, help="v2 ancestry LoRA safetensors path")
    ap.add_argument("--tokens", default=os.path.join(THIS_DIR, "tokens"))
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "gallery"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[render] building pipeline (base={args.base})...", flush=True)
    built = P.build_pipeline(
        os.path.expanduser(args.lora),
        os.path.expanduser(args.tokens),
        base_model=os.path.expanduser(args.base),
    )
    built.pipe.to("cuda")
    print("[render] pipeline on cuda; rendering 27 crosses...", flush=True)

    manifest = []
    for i, (stem, display, ancestry, ref_url) in enumerate(CROSSES, 1):
        prompt = built.engine.describe(ancestry, generation=1)  # F1
        imgs = P.generate_images(
            built,
            ancestry,
            prompt=prompt,  # stored == rendered, no drift
            seed=args.seed,
            guidance_scale=args.guidance,
            num_images=1,
            steps=args.steps,
            lora_scale=0.6,
            hue_token=None,
        )
        fname = f"{stem}.jpg"
        imgs[0].convert("RGB").save(os.path.join(args.out, fname), "JPEG", quality=92)
        # A verified-live direct photo -> "photo"; otherwise a stable Google Images search
        # link so every cross has a working reference (see refs.py for why).
        if ref_url:
            ref_kind = "photo"
        else:
            ref_url = search_url(display)
            ref_kind = "search"
        manifest.append(
            {
                "stem": stem,
                "display": display,
                "ancestry": ancestry,
                "parents": _parents_label(ancestry),
                "prompt": prompt,
                "image": fname,
                "ref_url": ref_url,
                "ref_kind": ref_kind,
            }
        )
        print(f"[render] {i:2d}/27 {display}  ->  {fname}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "steps": args.steps,
                "guidance_scale": args.guidance,
                "lora_scale": 0.6,
                "depth": "F1",
                "crosses": manifest,
            },
            f,
            indent=2,
        )
    print(f"[render] wrote {len(manifest)} images + manifest.json -> {args.out}")


if __name__ == "__main__":
    main()
