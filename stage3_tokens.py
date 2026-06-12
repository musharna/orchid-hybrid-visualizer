"""Stage 3 trait-axis token registry and injection helpers.

Shared between the demo-grid generator (scripts/stage3_demo_grid.py) and
the inference server (scripts/inference_server_v2.py); the Gradio UI in
app.py also imports the dropdown/slider helpers.

PRUNED 2026-05-09 to the 2 tokens whose LAB-delta direction reproduced
on side-by-side audit: yellow (Δb=+4.58) and red (Δa=+6.25). The other 7
hue tokens trained the wrong direction (crimson/burgundy/pink/rose_pink
went toward -a) or null (lavender ≈ baseline). All 12 ordinal endpoints
were below the LAB noise floor (|Δab|<2.3) and perceptually invisible.
Diagnostic in memory/project_stage3_gate1.md. Restoring requires a
re-train with magnitude regularization (rows landed 3× normal SDXL L2).
"""

from __future__ import annotations

from pathlib import Path

# Bundled in the Space repo under ./tokens/<name>/<name>_K4_generic.safetensors
_TOK_ROOT = Path(__file__).resolve().parent / "tokens"
YELLOW = _TOK_ROOT
RED = _TOK_ROOT

N_TOKENS_PER_AXIS = 4

# (bare_root, arm_canonical, source_dir) — pruned to direction-validated only.
TOKENS: list[tuple[str, str, Path]] = [
    ("v3_caro_hue_yellow", "lora", YELLOW),
    ("v3_anth_hue_red", "no_lora", RED),
]

# UI surface — only the validated hues. ORDINAL_AXES kept as empty list so
# call sites importing it don't break; ordinal_to_token always returns None.
HUES = [
    "yellow",
    "red",
]
ORDINAL_AXES: list[str] = []


def safetensors_path(name: str, src: Path) -> Path:
    """Resolve safetensors path supporting both wave-2 nested and flat layouts."""
    nested = src / name / f"{name}_K4_generic.safetensors"
    flat = src / f"{name}_K4_generic.safetensors"
    return nested if nested.exists() else flat


def hue_to_token(hue: str | None) -> str | None:
    """Map a hue label (or 'none'/'') to its bare_root, or None."""
    if not hue or hue == "none":
        return None
    if hue in ("yellow", "orange"):
        return f"v3_caro_hue_{hue}"
    return f"v3_anth_hue_{hue}"


def ordinal_to_token(axis: str, direction: int) -> str | None:
    """direction in {-1, 0, +1} → bare_root, or None for 0 (no edit)."""
    if direction == 0 or axis not in ORDINAL_AXES:
        return None
    suffix = "min" if direction < 0 else "max"
    return f"v3_{axis}_{suffix}"


def bulk_inject(pipe, registry: list[tuple[str, Path]]) -> dict[str, list[str]]:
    """Inject all listed (bare_root, source_dir) tokens into the SDXL pipe.

    Adds N_TOKENS_PER_AXIS placeholders per bare_root to both tokenizers,
    resizes embedding tables once, and writes trained TE1/TE2 rows.

    Returns a dict mapping bare_root → ordered placeholder list (e.g.
    ['<v3_anth_hue_red_0>', ..., '<v3_anth_hue_red_3>']). Bare roots whose
    safetensors are missing are skipped silently in the return value (a
    SKIP line is still printed).
    """
    import torch
    from safetensors.torch import load_file

    all_placeholders: list[str] = []
    pending: dict[str, tuple[list[str], list, list]] = {}

    for bare_root, src in registry:
        sf = safetensors_path(bare_root, src)
        if not sf.exists():
            print(f"  [stage3_tokens] SKIP {bare_root}: {sf} missing", flush=True)
            continue
        payload = load_file(str(sf))
        bares = [f"{bare_root}_{i}" for i in range(N_TOKENS_PER_AXIS)]
        placeholders = [f"<{b}>" for b in bares]
        te1_rows = [payload[f"{b}.te1"] for b in bares]
        te2_rows = [payload[f"{b}.te2"] for b in bares]
        all_placeholders.extend(placeholders)
        pending[bare_root] = (placeholders, te1_rows, te2_rows)

    if not pending:
        return {}

    pipe.tokenizer.add_tokens(all_placeholders)
    pipe.tokenizer_2.add_tokens(all_placeholders)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))

    out: dict[str, list[str]] = {}
    with torch.no_grad():
        emb1 = pipe.text_encoder.get_input_embeddings().weight
        emb2 = pipe.text_encoder_2.get_input_embeddings().weight
        for bare_root, (placeholders, te1_rows, te2_rows) in pending.items():
            ids1 = [pipe.tokenizer.convert_tokens_to_ids(p) for p in placeholders]
            ids2 = [pipe.tokenizer_2.convert_tokens_to_ids(p) for p in placeholders]
            for pid, row in zip(ids1, te1_rows):
                emb1.data[pid] = row.to(emb1.device, dtype=emb1.dtype)
            for pid, row in zip(ids2, te2_rows):
                emb2.data[pid] = row.to(emb2.device, dtype=emb2.dtype)
            out[bare_root] = placeholders
    return out
