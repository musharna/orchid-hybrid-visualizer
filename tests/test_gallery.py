"""Tests for the pre-rendered gallery app: pure manifest/format functions on a fixture,
plus a real-asset integrity check that activates once gallery/ has been rendered."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app


def _write_fixture(tmp_path):
    gdir = tmp_path / "gallery"
    gdir.mkdir()
    (gdir / "C_Test.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg\xff\xd9")
    manifest = {
        "seed": 42,
        "crosses": [
            {
                "stem": "C_Test",
                "display": "C. Test",
                "ancestry": {"dowiana": 50, "bicolor": 50},
                "parents": "dowiana 50% × bicolor 50%",
                "prompt": "a macro photograph of a Cattleya",
                "image": "C_Test.jpg",
                "ref_url": "http://example.com/ref.jpg",
            }
        ],
    }
    (gdir / "manifest.json").write_text(json.dumps(manifest))
    return str(gdir)


def test_load_manifest_parses_and_resolves_paths(tmp_path):
    entries = app.load_manifest(_write_fixture(tmp_path))
    assert len(entries) == 1
    assert entries[0]["display"] == "C. Test"
    assert entries[0]["image_path"].endswith("C_Test.jpg")
    assert os.path.isabs(entries[0]["image_path"])


def test_gallery_items_pairs_path_and_caption(tmp_path):
    entries = app.load_manifest(_write_fixture(tmp_path))
    items = app.gallery_items(entries)
    assert items == [(entries[0]["image_path"], "C. Test")]


def test_detail_md_includes_parents_prompt_and_ref(tmp_path):
    entries = app.load_manifest(_write_fixture(tmp_path))
    md = app.detail_md(entries[0])
    assert "C. Test" in md
    assert "dowiana 50% × bicolor 50%" in md
    assert "a macro photograph of a Cattleya" in md
    assert "http://example.com/ref.jpg" in md


def test_detail_md_omits_ref_link_when_absent(tmp_path):
    entries = app.load_manifest(_write_fixture(tmp_path))
    del entries[0]["ref_url"]
    md = app.detail_md(entries[0])
    assert "Reference photo" not in md


def test_real_gallery_assets_complete():
    """Real-execution check: once rendered, all 27 crosses are present and every manifest
    image file exists and is non-trivial. Skips if the gallery hasn't been rendered yet."""
    gdir = os.path.join(os.path.dirname(__file__), "..", "gallery")
    if not os.path.exists(os.path.join(gdir, "manifest.json")):
        import pytest

        pytest.skip("gallery not yet rendered")
    entries = app.load_manifest(gdir)
    assert len(entries) == 27
    for e in entries:
        assert os.path.exists(e["image_path"]), f"missing image: {e['image']}"
        assert os.path.getsize(e["image_path"]) > 2000, f"trivial image: {e['image']}"
        assert e["prompt"], f"empty prompt for {e['stem']}"
