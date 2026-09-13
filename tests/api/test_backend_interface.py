"""E4-T2: the abstract backend interface contract, exercised via FakeBackend."""

from pathlib import Path

import pytest
from helpers.fake_backend import FakeBackend

from horos.backends.base import (
    ModelBackend,
    PredictionReady,
    RunCompleted,
    RunStarted,
    TrainSpec,
)
from horos.core.registry import get_model_info


@pytest.fixture
def backend():
    return FakeBackend(get_model_info("rfdetr-nano"))


def test_interface_has_three_method_groups():
    for method in ("train", "infer_one", "infer_batch", "export"):
        assert hasattr(ModelBackend, method)


def test_cannot_instantiate_abstract_backend():
    with pytest.raises(TypeError):
        ModelBackend(get_model_info("rfdetr-nano"))  # type: ignore[abstract]


def test_train_streams_events_and_terminates_with_completed(backend, tmp_path):
    spec = TrainSpec(dataset_dir=tmp_path, output_dir=tmp_path, epochs=3, batch_size=2)
    events = list(backend.train(spec))
    assert isinstance(events[0], RunStarted)
    assert isinstance(events[-1], RunCompleted)
    assert "checkpoint" in events[-1].result
    assert any(e.type == "metrics" for e in events)


def test_infer_batch_yields_prediction_per_image(backend):
    images = [Path("a.jpg"), Path("b.jpg")]
    events = list(backend.infer_batch(images))
    predictions = [e for e in events if isinstance(e, PredictionReady)]
    assert len(predictions) == len(images)
    assert predictions[0].prediction.instances[0].score == pytest.approx(0.9)
    assert isinstance(events[-1], RunCompleted)


def test_infer_many_defaults_to_one_prediction_per_image_in_order(backend, tmp_path):
    """Batched inference is optional for a backend: the base implementation
    loops over infer_one, so callers can always ask for many at once."""
    from helpers.data import make_image

    paths = [make_image(tmp_path / f"{n}.png", 32, 24) for n in range(3)]
    many = backend.infer_many(paths, threshold=0.2, masks=False)
    assert [p.image for p in many] == [str(p) for p in paths]
    assert [p.model_dump() for p in many] == [
        backend.infer_one(p, threshold=0.2).model_dump() for p in paths
    ]


def test_backend_carries_model_info(backend):
    assert backend.info.key == "rfdetr-nano"
    assert backend.info.license == "Apache-2.0"
