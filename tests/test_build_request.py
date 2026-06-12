import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app_live as app


def test_manual_requires_two_parents():
    _, _, err = app.build_request("dowiana", 50, None, 0, "(none)", "F1")
    assert err and "two parent" in err.lower()


def test_manual_rejects_same_species():
    _, _, err = app.build_request("dowiana", 50, "dowiana", 50, "(none)", "F1")
    assert err and "different" in err.lower()


def test_manual_valid_pair_ok():
    anc, prompt, err = app.build_request(
        "dowiana", 50, "warscewiczii", 50, "(none)", "F1"
    )
    assert (
        err is None
        and anc == {"dowiana": 50.0, "warscewiczii": 50.0}
        and len(prompt) > 10
    )


def test_preset_ok():
    name = sorted(app._PRESETS)[0]
    anc, prompt, err = app.build_request(None, 0, None, 0, name, "F2")
    assert err is None and anc and len(prompt) > 10
