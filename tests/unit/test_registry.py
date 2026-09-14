"""E4-T1: model registry and metadata schema."""

import pytest

from horos.core import registry
from horos.errors import UnknownModelError

RFDETR_APACHE_KEYS = {"rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-large"}


def test_list_models_contains_only_apache_models():
    models = registry.list_models()
    keys = {m.key for m in models}
    assert RFDETR_APACHE_KEYS <= keys
    # §9: XL / 2XL must never be listed publicly
    assert "rfdetr-xl" not in keys
    assert "rfdetr-2xl" not in keys
    for m in models:
        assert m.weights_license == "Apache-2.0"
        assert m.code_license == "Apache-2.0"


def test_every_model_has_required_metadata():
    for m in registry.list_models():
        assert m.license  # R3: license is first-class
        assert m.license_url.startswith("https://")
        assert m.input_resolution > 0
        assert m.params_millions > 0
        assert m.latency_hint
        module, _, cls = m.entrypoint.partition(":")
        assert module.startswith("horos.backends."), m.entrypoint
        assert cls, m.entrypoint


def test_owlv2_models_registered_with_hf_ids():
    for key in ("owlv2-base", "owlv2-large"):
        info = registry.get_model_info(key)
        assert info.hf_id and info.hf_id.startswith("google/owlv2")


def test_unknown_model_raises_with_known_keys_listed():
    with pytest.raises(UnknownModelError) as exc_info:
        registry.get_model_info("yolo-v99")
    assert "rfdetr-nano" in str(exc_info.value)


def test_gated_models_resolve_but_require_acknowledgement():
    for key in ("rfdetr-xl", "rfdetr-2xl"):
        info = registry.get_model_info(key)
        assert info.weights_license == "PML-1.0"
        assert info.requires_acknowledgement


def test_apache_models_do_not_require_acknowledgement():
    for key in RFDETR_APACHE_KEYS:
        assert not registry.get_model_info(key).requires_acknowledgement


def test_list_models_filter_by_task():
    keys = {m.key for m in registry.list_models(task="detection")}
    assert RFDETR_APACHE_KEYS <= keys
    assert registry.list_models(task="instance_segmentation") == [
        m for m in registry.list_models() if m.task == "instance_segmentation"
    ]


SEG_KEYS = {
    "rfdetr-seg-nano", "rfdetr-seg-small", "rfdetr-seg-medium", "rfdetr-seg-large",
    "rfdetr-seg-xlarge", "rfdetr-seg-2xlarge",
}


def test_rfdetr_seg_models_are_listed_as_apache_instance_segmentation():
    """E4-T15: every RF-DETR-Seg size ships in the open rfdetr package under
    Apache 2.0 (unlike detection XL/2XL), so all of them are public."""
    seg = {m.key: m for m in registry.list_models(task="instance_segmentation")
           if m.family == "rfdetr"}
    assert set(seg) == SEG_KEYS
    for m in seg.values():
        assert m.trainable and m.weights_license == "Apache-2.0"
        assert m.resolution_step in (12, 24)
        assert m.input_resolution % m.resolution_step == 0
    assert seg["rfdetr-seg-nano"].input_resolution == 312
    assert seg["rfdetr-seg-2xlarge"].input_resolution == 768
    # detection listings (and the detection default) are untouched
    assert not (SEG_KEYS & {m.key for m in registry.list_models(task="detection")})
    assert registry.get_model_info("rfdetr-nano").resolution_step == 64


def test_only_sam_models_are_promptable():
    """The annotator's SAM picker lists promptable models only: RF-DETR-Seg is
    a trained segmenter, not an interactive one, and used to fail on first click."""
    from horos.core.registry import list_models

    promptable = {m.key for m in list_models() if m.promptable}
    assert promptable and all(k.startswith("sam") for k in promptable)
    assert all(not m.promptable for m in list_models() if m.family == "rfdetr")
    assert all(m.task == "instance_segmentation" for m in list_models() if m.promptable)
