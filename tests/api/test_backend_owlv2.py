"""E3-T1: OWLv2 backend. The dev/CI environment has no transformers/torch, so
these tests cover the contract that holds WITHOUT the dependency: registry
metadata, prompt validation, explicit train/export refusals, lazy loading.
Real-inference tests run only where transformers is installed."""

import importlib.util
import subprocess
import sys

import pytest
from helpers.data import make_image

from horos.backends.owlv2 import OWLv2Backend
from horos.core.registry import get_model_info
from horos.errors import BackendError

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def backend():
    return OWLv2Backend(get_model_info("owlv2-base"))


def test_registry_metadata():
    info = get_model_info("owlv2-base")
    assert info.family == "owlv2"
    assert info.weights_license == "Apache-2.0"
    assert info.hf_id == "google/owlv2-base-patch16-ensemble"
    assert not info.requires_acknowledgement


def test_construction_is_lazy():
    """Building the backend must not pull in the heavy deps (R1b).

    Checked in a clean subprocess, like the R1b invariant in
    tests/test_invariants.py: `sys.modules` is process-global, and pytest
    imports every test module before running anything, so a module-level
    `pytest.importorskip("torch")` elsewhere in the suite (test_infer_many.py,
    legitimately) puts torch there long before this test runs. An in-process
    assertion therefore passes alone and fails in the full suite while saying
    nothing about this backend.
    """
    script = """
import sys
from horos.backends.owlv2 import OWLv2Backend
from horos.core.registry import get_model_info
backend = OWLv2Backend(get_model_info("owlv2-base"))
backend.configure_prompts(["forklift"])
leaked = sorted({'torch', 'transformers'} & set(sys.modules))
assert not leaked, f"constructing the backend imported: {leaked}"
print("lazy")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "lazy" in result.stdout


def test_prompts_are_required(backend, tmp_path):
    with pytest.raises(BackendError, match="prompts"):
        backend.infer_one(make_image(tmp_path / "a.png"))


def test_empty_prompts_rejected(backend):
    with pytest.raises(BackendError, match="non-empty"):
        backend.configure_prompts(["", "  "])
    with pytest.raises(BackendError, match="non-empty"):
        backend.configure_prompts([])


def test_prompts_are_stripped(backend):
    backend.configure_prompts(["  forklift ", "person"])
    assert backend._prompts == ["forklift", "person"]


def test_train_is_refused(backend):
    from horos.backends.base import TrainSpec

    with pytest.raises(BackendError, match="not trainable"):
        next(
            backend.train(
                TrainSpec(dataset_dir=".", output_dir=".", epochs=1, batch_size=1)
            )
        )


def test_export_is_refused(backend, tmp_path):
    from horos.backends.base import ExportSpec

    with pytest.raises(BackendError, match="not supported"):
        next(
            backend.export(tmp_path, ExportSpec(format="onnx", output_dir=tmp_path))
        )


def test_missing_dependency_is_translated(backend, tmp_path, monkeypatch):
    if HAS_TRANSFORMERS:
        pytest.skip("transformers installed — the import cannot fail here")
    backend.configure_prompts(["forklift"])
    with pytest.raises(BackendError, match="owlv2"):
        backend.infer_one(make_image(tmp_path / "a.png"))


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_real_inference_smoke(backend, tmp_path):
    """Fixture image + known prompt: assert the pipeline produces a sane
    prediction shape (weights download on first run)."""
    backend.configure_prompts(["square"])
    prediction = backend.infer_one(make_image(tmp_path / "a.png", 64, 48), threshold=0.05)
    assert prediction.width == 64 and prediction.height == 48
    for inst in prediction.instances:
        x, y, w, h = inst.bbox
        assert 0 <= x <= 64 and 0 <= y <= 48 and w >= 0 and h >= 0
        assert 0.0 <= inst.score <= 1.0
        assert inst.category_id == 0
