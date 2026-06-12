"""SDXL generation pipeline for the Orchid Hybrid Visualizer Space (CPU-buildable,
dependency-injected pipe so the control flow is unit-testable without a GPU).

Ports scripts/inference_server_v2.py into importable functions. The actual GPU call
lives in generate_images(); app.py wraps it with @spaces.GPU.
"""

from __future__ import annotations

NEG_PROMPT = (
    "oversaturated, neon, painting, unrealistic colors, too many petals, "
    "distorted, watermark, text, logo, signature"
)


def compose_prompt(engine, ancestry: dict, override: str | None) -> str:
    if override:
        return override
    return engine.describe(ancestry)


def validate_ancestry(ancestry: dict) -> str | None:
    if not ancestry:
        return "Pick at least one parent species."
    if sum(float(v) for v in ancestry.values()) <= 0:
        return "Parent percentages must be greater than zero."
    return None


from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Built:
    pipe: Any
    engine: Any
    placeholders: dict
    warm_cache: dict = field(default_factory=dict)
    e_baseline: dict = field(default_factory=dict)


def set_token_strength(built: Built, bare: str, s: float) -> None:
    """Interpolate a hue token's placeholder rows to strength s (1=trained, 0=neutral)."""
    from stage13_warmcolor import interpolate_embedding

    cache = built.warm_cache.get(bare)
    if cache is None:
        return
    emb1 = built.pipe.text_encoder.get_input_embeddings().weight
    emb2 = built.pipe.text_encoder_2.get_input_embeddings().weight
    with torch.no_grad():
        for i, row in zip(cache["ids1"], cache["rows1"]):
            emb1.data[i] = interpolate_embedding(built.e_baseline["te1"], row, s).to(
                emb1.dtype
            )
        for i, row in zip(cache["ids2"], cache["rows2"]):
            emb2.data[i] = interpolate_embedding(built.e_baseline["te2"], row, s).to(
                emb2.dtype
            )


def generate_images(
    built: Built,
    ancestry: dict,
    *,
    prompt=None,
    seed=42,
    guidance_scale=5.0,
    num_images=1,
    steps=30,
    lora_scale=0.6,
    hue_token: str | None = None,
    warmth: float = 1.0,
):
    pipe, engine = built.pipe, built.engine
    prompt = compose_prompt(engine, ancestry, prompt)

    if hue_token and hue_token in built.placeholders:
        set_token_strength(built, hue_token, float(warmth))
        prompt = prompt + " " + " ".join(built.placeholders[hue_token])

    neg_extra = engine.negative_for(ancestry)
    neg = NEG_PROMPT + (f", {neg_extra}" if neg_extra else "")

    if lora_scale > 0:
        pipe.enable_lora()
        pipe.set_adapters(["v2"], adapter_weights=[lora_scale])
    else:
        pipe.disable_lora()

    device = getattr(getattr(pipe, "device", None), "type", "cpu")
    gen = torch.Generator(device if device == "cuda" else "cpu").manual_seed(int(seed))
    return pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=gen,
    ).images


def build_pipeline(
    lora_path: str,
    token_dir: str,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
) -> Built:
    """CPU build: SDXL base + v2 LoRA + hue-token injection + warm baseline cache.

    base_model defaults to the Hub id (the live Space pulls it at runtime); the desktop
    render passes a local models/sdxl-base-1.0 path to avoid a 7 GB download. Matches the
    inference server's load (fp16 variant, safetensors)."""
    from diffusers import StableDiffusionXLPipeline
    from phenotype_engine import PhenotypeEngine
    import stage3_tokens as st

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_lora_weights(lora_path, adapter_name="v2")

    orig1 = pipe.text_encoder.get_input_embeddings().weight.shape[0]
    orig2 = pipe.text_encoder_2.get_input_embeddings().weight.shape[0]
    placeholders = st.bulk_inject(pipe, [(n, s) for n, _arm, s in st.TOKENS])

    warm_cache, e_baseline = {}, {}
    with torch.no_grad():
        emb1 = pipe.text_encoder.get_input_embeddings().weight
        emb2 = pipe.text_encoder_2.get_input_embeddings().weight
        e_baseline["te1"] = emb1[:orig1].mean(0).float()
        e_baseline["te2"] = emb2[:orig2].mean(0).float()
        for bare, ph in placeholders.items():
            ids1 = [pipe.tokenizer.convert_tokens_to_ids(p) for p in ph]
            ids2 = [pipe.tokenizer_2.convert_tokens_to_ids(p) for p in ph]
            warm_cache[bare] = {
                "ids1": ids1,
                "ids2": ids2,
                "rows1": [emb1[i].clone().float() for i in ids1],
                "rows2": [emb2[i].clone().float() for i in ids2],
            }
    return Built(
        pipe=pipe,
        engine=PhenotypeEngine(),
        placeholders=placeholders,
        warm_cache=warm_cache,
        e_baseline=e_baseline,
    )
