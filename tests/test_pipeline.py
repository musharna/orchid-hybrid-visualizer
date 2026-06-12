import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safetensors.torch import load_file
import stage3_tokens as st


def test_bundled_hue_tokens_resolve_and_load():
    for name, src in [(n, s) for n, _arm, s in st.TOKENS]:
        p = st.safetensors_path(name, src)
        assert p.exists(), f"{name} token missing at {p}"
        payload = load_file(str(p))
        assert f"{name}_0.te1" in payload and f"{name}_3.te2" in payload


import pipeline as P
from phenotype_engine import PhenotypeEngine

_E = PhenotypeEngine()


def test_compose_prompt_nonempty_for_valid_cross():
    p = P.compose_prompt(_E, {"dowiana": 50, "warscewiczii": 50}, None)
    assert isinstance(p, str) and len(p) > 10


def test_compose_prompt_override_wins():
    assert P.compose_prompt(_E, {"dowiana": 100}, "custom text") == "custom text"


def test_validate_ancestry_rejects_empty():
    assert P.validate_ancestry({}) is not None


def test_validate_ancestry_rejects_all_zero():
    assert P.validate_ancestry({"dowiana": 0, "warscewiczii": 0}) is not None


def test_validate_ancestry_accepts_valid():
    assert P.validate_ancestry({"dowiana": 50, "warscewiczii": 50}) is None


import types


class _FakePipe:
    """Records the call surface generate_images touches; returns sentinel images."""

    def __init__(self):
        self.adapters = None
        self.lora_enabled = None
        self.last = {}

    def enable_lora(self):
        self.lora_enabled = True

    def disable_lora(self):
        self.lora_enabled = False

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (names, adapter_weights)

    def __call__(self, **kwargs):
        self.last = kwargs
        return types.SimpleNamespace(
            images=["IMG"] * kwargs.get("num_images_per_prompt", 1)
        )


def _built_with_fake():
    return P.Built(
        pipe=_FakePipe(), engine=_E, placeholders={}, warm_cache={}, e_baseline={}
    )


def test_generate_disables_lora_when_scale_zero():
    b = _built_with_fake()
    P.generate_images(b, {"dowiana": 50, "warscewiczii": 50}, lora_scale=0.0, seed=1)
    assert b.pipe.lora_enabled is False


def test_generate_sets_v2_adapter_weight():
    b = _built_with_fake()
    P.generate_images(b, {"dowiana": 50, "warscewiczii": 50}, lora_scale=0.6, seed=1)
    assert b.pipe.adapters == (["v2"], [0.6])


def test_generate_negative_prompt_includes_base_and_engine_suppression():
    b = _built_with_fake()
    P.generate_images(b, {"dowiana": 50, "warscewiczii": 50}, lora_scale=0.6, seed=1)
    neg = b.pipe.last["negative_prompt"]
    assert P.NEG_PROMPT.split(",")[0] in neg


def test_generate_returns_requested_count():
    b = _built_with_fake()
    out = P.generate_images(b, {"dowiana": 50}, num_images=2, seed=1)
    assert out == ["IMG", "IMG"]
