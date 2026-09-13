"""E10-T2: the image-embedder contract and the DINOv2 backend. Everything
testable without the ML deps runs always; the real model runs where
transformers is installed (weights download on first use)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from helpers.data import make_image
from helpers.fake_backend import FakeEmbedder

from horos.backends import get_backend
from horos.backends.base import ExportSpec, ImageEmbedder, TrainSpec
from horos.backends.dinov2 import DINOv2Backend
from horos.core.registry import get_model_info, list_models
from horos.errors import BackendError

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def test_registry_lists_dinov2_small_as_an_apache_embedding_model():
    info = get_model_info("dinov2-small")
    assert info.family == "dinov2" and info.task == "embedding"
    assert info.code_license == "Apache-2.0" and info.weights_license == "Apache-2.0"
    assert info.hf_id == "facebook/dinov2-small" and not info.requires_acknowledgement
    assert not info.trainable
    assert [m.key for m in list_models(task="embedding")] == ["dinov2-small"]
    # detection listings (the training UI) never show the encoder
    assert "dinov2-small" not in {m.key for m in list_models(task="detection")}
    # §9: the non-Apache successor is not registered
    assert not any(m.key.startswith("dinov3") for m in list_models())


def test_backend_resolves_lazily_and_implements_the_contract():
    backend = get_backend("dinov2-small", device="cpu")
    assert isinstance(backend, DINOv2Backend) and isinstance(backend, ImageEmbedder)
    assert backend._model is None  # construction loads nothing (R1b)


def test_encoder_refuses_detector_and_trainer_roles(tmp_path):
    backend = DINOv2Backend(get_model_info("dinov2-small"))
    image = make_image(tmp_path / "a.png")
    with pytest.raises(BackendError, match="does not train"):
        next(backend.train(TrainSpec(dataset_dir=tmp_path, output_dir=tmp_path, epochs=1,
                                     batch_size=1)))
    with pytest.raises(BackendError, match="does not detect"):
        backend.infer_one(image)
    with pytest.raises(BackendError, match="does not detect"):
        next(backend.infer_batch([image]))
    with pytest.raises(BackendError, match="does not export"):
        next(backend.export(image, ExportSpec(format="onnx", output_dir=tmp_path)))
    assert backend.embed_batch([]) == []


def test_fake_embedder_is_deterministic_and_normalised(tmp_path):
    fake = FakeEmbedder()
    a = make_image(tmp_path / "a.png", color=(200, 10, 10))
    b = make_image(tmp_path / "b.png", color=(10, 200, 10))
    vecs = np.asarray(fake.embed_batch([a, b, a]))
    assert vecs.shape == (3, fake.embedding_dim)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], vecs[2])
    assert vecs[0] @ vecs[1] < 0.99


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_real_dinov2_embeds_images(tmp_path):
    """Runs in a subprocess: importing transformers here would pollute
    sys.modules for the lazy-loading assertions of later test modules."""
    import json
    import subprocess
    import sys

    same = make_image(tmp_path / "same.png", 96, 96, color=(30, 60, 200))
    twin = make_image(tmp_path / "twin.png", 96, 96, color=(30, 60, 200))
    other = make_image(tmp_path / "other.png", 96, 96, color=(220, 220, 40))
    script = (
        "import json, sys\n"
        "from horos.backends import get_backend\n"
        "b = get_backend('dinov2-small', device='cpu')\n"
        "vecs = b.embed_batch([sys.argv[1], sys.argv[2], sys.argv[3]])\n"
        "print(json.dumps({'dim': b.embedding_dim, 'vecs': vecs}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(same), str(twin), str(other)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    body = json.loads(proc.stdout.strip().splitlines()[-1])
    vecs = np.asarray(body["vecs"])
    assert vecs.shape == (3, body["dim"]) and body["dim"] == 384
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert vecs[0] @ vecs[1] == pytest.approx(1.0, abs=1e-4)
    assert vecs[0] @ vecs[2] < vecs[0] @ vecs[1]
