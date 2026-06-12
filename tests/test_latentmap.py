"""Tests for the interactive latent map: pure figure/caption builders on fixtures, plus a
real-asset check that activates once latent_coords.json has been precomputed."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import latentmap

_HERO = {
    "stem": "C_Hero",
    "display": "C. Hero",
    "parents": [
        {"name": "alpha", "epithet": "alpha", "weight": 50, "xy": [-1.0, 0.0]},
        {"name": "beta", "epithet": "beta", "weight": 50, "xy": [1.0, 0.0]},
    ],
    "midpoint": [0.0, 0.0],
    "hybrid": [0.1, 1.0],
    "angle_deg": 77.0,
    "note": "real examples: 12",
}
_NOHYB = {
    "stem": "C_Plain",
    "display": "C. Plain",
    "parents": [
        {"name": "alpha", "epithet": "alpha", "weight": 50, "xy": [-1.0, 0.0]},
        {"name": "beta", "epithet": "beta", "weight": 50, "xy": [1.0, 0.0]},
    ],
    "midpoint": [0.0, 0.0],
    "hybrid": None,
    "angle_deg": None,
    "note": "no real examples",
}
_NOPLANE = {
    "stem": "C_X",
    "display": "C. X",
    "parents": [],
    "midpoint": None,
    "hybrid": None,
    "angle_deg": None,
    "note": "parent embeddings unavailable",
}


def test_load_coords_missing_returns_empty(tmp_path):
    assert latentmap.load_coords(str(tmp_path / "nope.json")) == []


def test_load_coords_parses(tmp_path):
    p = tmp_path / "latent_coords.json"
    p.write_text(json.dumps({"crosses": [_HERO, _NOHYB]}))
    entries = latentmap.load_coords(str(p))
    assert len(entries) == 2
    assert entries[0]["display"] == "C. Hero"


def test_has_plane():
    assert latentmap.has_plane(_HERO)
    assert not latentmap.has_plane(_NOPLANE)


def test_build_figure_hero_has_hybrid_and_transgression():
    fig = latentmap.build_figure(_HERO)
    names = [t.name for t in fig.data]
    assert "parent species" in names
    assert "predicted F1 blend" in names
    assert "real hybrid (avg)" in names
    # the off-chord arrow annotation mentions the angle
    texts = [a.text for a in fig.layout.annotations]
    assert any("off-chord" in (t or "") for t in texts)


def test_build_figure_no_hybrid_omits_hybrid_trace():
    fig = latentmap.build_figure(_NOHYB)
    names = [t.name for t in fig.data]
    assert "parent species" in names
    assert "real hybrid (avg)" not in names


def test_build_figure_no_plane_is_safe():
    fig = latentmap.build_figure(_NOPLANE)
    # should not raise; renders an annotation instead of traces
    assert len(fig.data) == 0


def test_caption_hero_mentions_transgressive():
    cap = latentmap.caption(_HERO)
    assert "transgressive" in cap
    assert "77" in cap


def test_caption_no_hybrid_explains_absence():
    cap = latentmap.caption(_NOHYB)
    assert "No real examples" in cap or "no real examples" in cap.lower()


def test_real_latent_coords_complete():
    """Real-asset check: once precomputed, latent_coords has 27 crosses and every one with a
    chord-plane builds a figure without error. Skips if not yet built."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "gallery", "latent_coords.json"
    )
    if not os.path.exists(path):
        import pytest

        pytest.skip("latent_coords.json not yet built")
    entries = latentmap.load_coords(path)
    assert len(entries) == 27
    n_real = 0
    for e in entries:
        fig = latentmap.build_figure(e)  # must not raise
        assert fig is not None
        if e.get("hybrid"):
            n_real += 1
    assert n_real >= 1, "expected at least one cross with real-hybrid overlay"
