"""Render N variations per cross for the gallery's variation strip (GPU host, desktop).

Reuses the exact pipeline (build_pipeline + generate_images) as render_gallery.py, but draws
num_images variations in a single batched call per cross (different per-image noise, same
prompt/seed) to convey the model's range for each hybrid. Writes:

  <out>/<stem>_v{0..N-1}.jpg   and   <out>/seeds_manifest.json  ({stem: [files], n, seed})

Run on the desktop with the local SDXL base + v2 LoRA:

  venv-render/bin/python render_seeds.py \
    --base ~/orchid-sdxl/models/sdxl-base-1.0 \
    --lora ~/orchid-sdxl/output_v2/orchid-ancestry-lora-v2.safetensors \
    --out gallery/seeds --n 4
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="local SDXL base dir or Hub id")
    ap.add_argument("--lora", required=True, help="v2 ancestry LoRA safetensors path")
    ap.add_argument("--tokens", default=os.path.join(THIS_DIR, "tokens"))
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "gallery", "seeds"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=4, help="variations per cross")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[seeds] building pipeline (base={args.base})...", flush=True)
    built = P.build_pipeline(
        os.path.expanduser(args.lora),
        os.path.expanduser(args.tokens),
        base_model=os.path.expanduser(args.base),
    )
    built.pipe.to("cuda")
    print(f"[seeds] pipeline on cuda; {args.n} variations x 27 crosses...", flush=True)

    manifest = {}
    for i, (stem, display, ancestry, _ref) in enumerate(CROSSES, 1):
        prompt = built.engine.describe(ancestry, generation=1)  # F1, same as gallery
        imgs = P.generate_images(
            built,
            ancestry,
            prompt=prompt,
            seed=args.seed,
            guidance_scale=args.guidance,
            num_images=args.n,
            steps=args.steps,
            lora_scale=0.6,
            hue_token=None,
        )
        files = []
        for j, img in enumerate(imgs):
            fname = f"{stem}_v{j}.jpg"
            img.convert("RGB").save(os.path.join(args.out, fname), "JPEG", quality=90)
            files.append(fname)
        manifest[stem] = files
        print(f"[seeds] {i:2d}/27 {display}  ->  {len(files)} variations", flush=True)

    with open(os.path.join(args.out, "seeds_manifest.json"), "w") as f:
        json.dump({"n": args.n, "seed": args.seed, "crosses": manifest}, f, indent=2)
    print(f"[seeds] wrote {len(manifest)} crosses x {args.n} -> {args.out}")


if __name__ == "__main__":
    main()
