# Asset licensing

`LICENSE` (MIT) covers this repository's **source code only**. The bundled
media and model assets are third-party works, or derivatives of third-party
works, and keep their own terms. This file is the map.

## The short version

- **Code** (`*.py`) — MIT. Reuse freely.
- **Parent photos** (`parents/`) — 26 images, individually licensed. Attribution required for the CC BY ones.
- **`parents/rex.jpg` is CC BY-NC 4.0 — NonCommercial.** This one file means the
  repository *as a whole* cannot be used commercially. Remove it and the
  remaining photos are all commercially redistributable with attribution.
- **`tokens/*.safetensors`** — derived from SDXL base 1.0 (CreativeML Open RAIL++-M);
  its use-based restrictions carry over.
- **`gallery/*.jpg`** — generated output from that model + the ancestry LoRA.

## Parent reference photos (`parents/`)

Full per-photo attribution lives in [`parents/CREDITS.md`](parents/CREDITS.md).
License distribution across the 26 photos:

| license | count | commercial use | attribution required |
|---|---|---|---|
| CC BY 4.0 | 18 | yes | **yes** |
| Public Domain | 5 | yes | no |
| CC0 1.0 | 2 | yes | no |
| CC BY-NC 4.0 | 1 | **NO** | **yes** |

### The NonCommercial carve-out

- ***Cattleya rex*** — `parents/rex.jpg`, CC BY-NC 4.0, by sheylan (iNaturalist).

`CREDITS.md` notes this was a last-resort fallback — no clean redistributable
photo of that species was available in the dataset. It is fine for a
non-commercial research demo, which is what this repo is. It is *not* fine to
relicense under a permissive license that grants commercial rights.

## Model-derived assets

### `tokens/*.safetensors`

Textual-inversion embeddings (4 tokens each, 1500 steps, 768px) trained against
[`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),
which is licensed under the
[CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).

That license states downstream derivatives may be released under different
terms but must include **at minimum the same use-based restrictions**
(Attachment A). Those restrictions therefore apply to these embeddings.

### `gallery/*.jpg`

Pre-rendered output (seed 42, F1 depth) from SDXL base 1.0 plus
[`mjarnold/orchid-ancestry-lora-v2`](https://huggingface.co/mjarnold/orchid-ancestry-lora-v2),
produced by `render_gallery.py`. OpenRAIL++ does not assert ownership over
model outputs, but they originate from a restricted-use model.

## If you want to reuse this

- **Just the code?** MIT — take it.
- **Code + photos, non-commercially?** Fine; keep the CC BY attributions.
- **Anything commercial?** Drop `parents/rex.jpg` first, honour the CC BY
  attributions, and check the OpenRAIL++ restrictions for the model assets.
