"""Tests for the pre-rendered gallery app: pure manifest/format functions on a fixture,
plus a real-asset integrity check that activates once gallery/ has been rendered."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app
import refs


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
                "ref_kind": "photo",
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
    # ref_kind="photo" -> direct-photo label
    assert "Reference photo of the real registered cross" in md


def test_detail_md_search_kind_uses_search_label(tmp_path):
    entries = app.load_manifest(_write_fixture(tmp_path))
    e = entries[0]
    e["ref_kind"] = "search"
    e["ref_url"] = "https://www.google.com/search?tbm=isch&q=Cattleya+Test+orchid"
    md = app.detail_md(e)
    assert "Search real photos of Cattleya Test" in md
    assert e["ref_url"] in md
    # search links must NOT claim to be a single curated photo
    assert "Reference photo of the real registered cross" not in md


def test_detail_md_omits_ref_link_when_absent(tmp_path):
    entries = app.load_manifest(_write_fixture(tmp_path))
    del entries[0]["ref_url"]
    md = app.detail_md(entries[0])
    assert "Reference photo" not in md
    assert "Search real photos" not in md


def test_search_url_builds_google_images_query():
    url = refs.search_url("C. Hardyana")
    assert url.startswith("https://www.google.com/search?tbm=isch")
    assert "Cattleya" in url and "Hardyana" in url and "orchid" in url


def test_species_name_expands_prefix():
    assert (
        refs.species_name("C. Mem. Albert Heinecke") == "Cattleya Mem. Albert Heinecke"
    )


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
        # every cross has a usable reference link: a direct photo or a stable search link
        assert e.get("ref_url"), f"no ref_url for {e['stem']}"
        assert e.get("ref_kind") in ("photo", "search"), f"bad ref_kind for {e['stem']}"
        if e["ref_kind"] == "search":
            assert e["ref_url"].startswith("https://www.google.com/search?tbm=isch"), (
                f"search ref_url not a google-images link: {e['stem']}"
            )


def test_real_seed_variations_complete():
    """Real-execution check: once the multi-seed GPU render has run, every cross has its
    variation strip on disk. Skips if seeds haven't been rendered yet."""
    gdir = os.path.join(os.path.dirname(__file__), "..", "gallery")
    if not os.path.exists(os.path.join(gdir, "seeds", "seeds_manifest.json")):
        import pytest

        pytest.skip("seeds not yet rendered")
    entries = app.load_manifest(gdir)
    for e in entries:
        paths = app.seed_items(e["stem"])
        assert len(paths) == 4, f"{e['stem']} has {len(paths)} variations, expected 4"
        for p in paths:
            assert os.path.getsize(p) > 2000, f"trivial variation: {p}"


def test_real_parent_photos_licensing_clean():
    """Real-execution check: parent photos exist and carry redistributable licenses + credit.
    Skips if parents/ hasn't been assembled."""
    pdir = os.path.join(os.path.dirname(__file__), "..", "parents")
    meta_path = os.path.join(pdir, "parents.json")
    if not os.path.exists(meta_path):
        import pytest

        pytest.skip("parent photos not yet assembled")
    with open(meta_path) as f:
        meta = json.load(f)
    for ep, m in meta.items():
        assert os.path.exists(os.path.join(pdir, m["file"])), (
            f"missing parent photo: {ep}"
        )
        assert m.get("credit"), f"no credit line for {ep}"
        lic = m["license"].lower()
        # no all-rights-reserved / ND / unknown in the bundled set (rex may be cc-by-nc)
        assert "all-rights" not in lic and lic not in ("", "none"), (
            f"non-clean license for {ep}: {lic}"
        )
        assert "-nd" not in lic and "noderiv" not in lic, f"ND license for {ep}: {lic}"
