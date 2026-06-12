"""Stage 13 warm-color control — pure recipe/calibration math (CPU-only).

No GPU, no model, no I/O. Every function here is unit-tested against synthetic
fixtures. The GPU trainer (Phase 2) imports the loss/gate helpers; the inference
path imports the interpolation + calibration helpers.

Spec: specs/stage13_warmcolor_control_design.md
"""

from __future__ import annotations

import torch

JND = 2.3  # CIE just-noticeable LAB difference (Δa/Δb units)


def interpolate_embedding(
    e_baseline: torch.Tensor, e_token: torch.Tensor, s: float
) -> torch.Tensor:
    """Linear interpolation from baseline to trained token row at strength s.

    s=0 → e_baseline (no effect); s=1 → e_token (full token). The injection
    path (Phase 2) supplies e_baseline = per-encoder mean vocab embedding.
    """
    if not (0.0 <= s <= 1.0):
        raise ValueError(f"s must be in [0, 1], got {s}")
    if e_baseline.shape != e_token.shape:
        raise ValueError(
            f"shape mismatch: {tuple(e_baseline.shape)} vs {tuple(e_token.shape)}"
        )
    return e_baseline + s * (e_token - e_baseline)


def median_vocab_norm(embedding_weight: torch.Tensor) -> float:
    """Median L2 norm of the rows of an embedding table — the reg target."""
    return float(embedding_weight.norm(dim=-1).median())


def magnitude_reg_loss(rows: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Mean squared deviation of each row's L2 norm from target_norm.

    Pulls learned token rows back toward real-vocab scale (Stage-3 tokens
    overshot ~3×). Add to the TI reconstruction loss during Phase-2 training.
    """
    norms = rows.norm(dim=-1)
    return ((norms - target_norm) ** 2).mean()


def direction_validation_gate(
    delta_a: float,
    delta_b: float,
    target_axis: str,
    expected_sign: int = 1,
    jnd: float = JND,
) -> tuple[bool, str]:
    """Accept a trained token only if its LAB delta moves the right way + enough.

    target_axis in {"a", "b"}; expected_sign +1 for warm hues (red +a, yellow
    +b). Returns (accepted, human-readable reason). This is the gate that would
    have rejected the 28/30 wrong-direction Stage-3 tokens.
    """
    if target_axis not in ("a", "b"):
        raise ValueError(f"target_axis must be 'a' or 'b', got {target_axis!r}")
    d = delta_a if target_axis == "a" else delta_b
    signed = d * expected_sign
    if signed <= 0:
        return (False, f"wrong direction: signed Δ{target_axis}={signed:+.2f} <= 0")
    if signed < jnd:
        return (False, f"below JND: |Δ{target_axis}|={signed:.2f} < {jnd}")
    return (True, f"Δ{target_axis}={signed:+.2f} >= {jnd}")


def monotonicity(s_values: list[float], delta_values: list[float]) -> dict:
    """Monotonicity of the s → target-axis-delta curve.

    Caller passes deltas already signed in the target direction (positive =
    correct). Returns Spearman rho(s, delta) and whether every consecutive step
    is non-decreasing.

    scipy is imported lazily here so the GPU trainer (Phase 2), which only needs
    the torch-only helpers, never has to have scipy installed in its env.
    """
    if len(delta_values) < 2:
        return {"spearman": float("nan"), "sign_consistent": True}
    from scipy.stats import spearmanr

    rho = float(spearmanr(s_values, delta_values).statistic)
    diffs = [
        delta_values[i + 1] - delta_values[i] for i in range(len(delta_values) - 1)
    ]
    return {"spearman": rho, "sign_consistent": all(d >= 0 for d in diffs)}


def magnitude_vs_jnd(delta_at_s1: float, jnd: float = JND) -> bool:
    """True if the target-axis delta at full strength exceeds the JND floor."""
    return abs(delta_at_s1) >= jnd


import numpy as np


def fit_calibration(
    grid: list[tuple[float, float, float, float]],
    target_axis: str,
    baseline_abs: float,
) -> dict:
    """Reduce a measured (s, Δa, Δb, std) sweep to a per-hue s→Δ curve.

    grid rows are (s, mean_da, mean_db, std). baseline_abs is the absolute LAB
    target-axis value at s=0 (so inference can convert an absolute target into a
    delta). target_axis selects which column becomes the curve's delta.
    """
    if target_axis not in ("a", "b"):
        raise ValueError(f"target_axis must be 'a' or 'b', got {target_axis!r}")
    col = 1 if target_axis == "a" else 2
    return {
        "target_axis": target_axis,
        "baseline_abs": baseline_abs,
        "s": [row[0] for row in grid],
        "delta": [row[col] for row in grid],
    }


def invert_calibration(curve: dict, target_delta: float) -> float:
    """Map a desired target-axis delta back to a strength s, clamped to grid.

    Monotone piecewise-linear inverse; pairs are sorted by delta so np.interp
    receives an increasing x. Out-of-range targets clamp to the nearest s.
    """
    pairs = sorted(zip(curve["delta"], curve["s"]))
    ds = [p[0] for p in pairs]
    ss = [p[1] for p in pairs]
    if target_delta <= ds[0]:
        return float(ss[0])
    if target_delta >= ds[-1]:
        return float(ss[-1])
    return float(np.interp(target_delta, ds, ss))


def warmth_to_strength(warmth_pct: float, curve: dict) -> float:
    """Map a 0–100% manual warmth slider to a strength s via the calibration.

    Linear in target ΔE over [0, max achievable Δ] (the curve, not s, carries
    the perceptual evening), then inverted to s. 0% → s at zero warmth; 100% →
    s at the top of the calibrated range. Caller should treat 0% as "inject no
    token" (true baseline) since the neutral placeholder has a small residual Δ.
    """
    if not (0.0 <= warmth_pct <= 100.0):
        raise ValueError(f"warmth_pct must be in [0, 100], got {warmth_pct}")
    max_delta = max(curve["delta"])
    target = (warmth_pct / 100.0) * max_delta
    return invert_calibration(curve, target)
