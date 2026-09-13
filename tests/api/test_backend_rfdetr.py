"""E4-T4: RF-DETR adapter unit tests — the pure mapping logic, no ML runtime.

The full training path is exercised end-to-end by test_train_e2e.py; these
tests pin down the spec→rfdetr translation and result conversion, which is
where a silent regression would corrupt runs without failing loudly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from horos.backends.base import TrainSpec
from horos.backends.rfdetr import (
    _MODEL_CLASSES,
    _best_checkpoint,
    _BestTracker,
    _default_weights_filename,
    _detections_to_instances,
    _repoint_checkpoint_monitor,
    _train_kwargs,
)
from horos.core.registry import list_models


def _spec(**overrides):
    defaults = dict(
        dataset_dir=Path("/data/ds"),
        output_dir=Path("/data/out"),
        epochs=5,
        batch_size=2,
    )
    defaults.update(overrides)
    return TrainSpec(**defaults)


def test_every_registered_rfdetr_model_has_a_class_mapping():
    keys = {m.key for m in list_models() if m.family == "rfdetr"}
    assert keys == set(_MODEL_CLASSES)


def test_train_kwargs_maps_spec_fields():
    kwargs = _train_kwargs(
        _spec(resolution=384, device="cpu", seed=7, resume_from=Path("/ckpt.pth"))
    )
    assert kwargs["dataset_dir"] == str(Path("/data/ds"))
    assert kwargs["output_dir"] == str(Path("/data/out"))
    assert kwargs["epochs"] == 5 and kwargs["batch_size"] == 2
    assert kwargs["resolution"] == 384 and kwargs["device"] == "cpu"
    assert kwargs["seed"] == 7 and kwargs["resume"] == str(Path("/ckpt.pth"))


def test_train_kwargs_omits_unset_optionals_and_silences_rfdetr_reporting():
    kwargs = _train_kwargs(_spec())
    for absent in ("resolution", "device", "seed", "resume"):
        assert absent not in kwargs
    # horos owns progress reporting (R4)
    assert kwargs["tensorboard"] is False
    assert kwargs["progress_bar"] is None
    assert kwargs["early_stopping"] is False


def test_train_kwargs_extra_overrides_win():
    kwargs = _train_kwargs(
        _spec(extra={"epochs": 99, "tensorboard": True, "lr": 1e-5})
    )
    assert kwargs["epochs"] == 99  # expert override beats the derived value (E5-S5)
    assert kwargs["tensorboard"] is True
    assert kwargs["lr"] == 1e-5


def test_detections_convert_to_coco_xywh():
    detections = SimpleNamespace(
        xyxy=[(10.0, 20.0, 30.0, 60.0), (0.0, 0.0, 5.0, 5.0)],
        confidence=[0.9, 0.4],
        class_id=[2, 0],
    )
    instances = _detections_to_instances(detections)
    assert instances[0].bbox == (10.0, 20.0, 20.0, 40.0)
    assert instances[0].score == 0.9 and instances[0].category_id == 2
    assert instances[1].bbox == (0.0, 0.0, 5.0, 5.0)


def test_detections_tolerate_missing_confidence_and_class():
    detections = SimpleNamespace(
        xyxy=[(1.0, 1.0, 2.0, 2.0)], confidence=None, class_id=None
    )
    (instance,) = _detections_to_instances(detections)
    assert instance.score == 1.0 and instance.category_id == 0


def test_best_checkpoint_preference_order(tmp_path):
    assert _best_checkpoint(tmp_path) is None
    (tmp_path / "last.ckpt").touch()
    assert _best_checkpoint(tmp_path).name == "last.ckpt"
    (tmp_path / "checkpoint_best_ema.pth").touch()
    assert _best_checkpoint(tmp_path).name == "checkpoint_best_ema.pth"
    (tmp_path / "checkpoint_best_total.pth").touch()
    assert _best_checkpoint(tmp_path).name == "checkpoint_best_total.pth"


def test_epoch_metrics_split_train_and_val_sides():
    from horos.backends.rfdetr import _epoch_metrics

    snapshot = {
        "val/loss": 9.2,
        "val/mAP_50": 0.31,
        "loss": 8.8,           # bare keys are validation-time aggregates
        "train/loss": 7.1,
        "train/lr": 1e-4,
    }
    val_side = _epoch_metrics(snapshot, train_side=False)
    train_side = _epoch_metrics(snapshot, train_side=True)
    assert set(val_side) == {"val/loss", "val/mAP_50", "loss"}
    assert set(train_side) == {"train/loss", "train/lr"}


def test_epoch_metrics_skip_unconvertible_values():
    from horos.backends.rfdetr import _epoch_metrics

    picked = _epoch_metrics(
        {"val/loss": "not-a-number", "val/mAP": 0.5}, train_side=False
    )
    assert picked == {"val/mAP": 0.5}


def test_detections_carry_class_names():
    """Fine-tuned rfdetr emits 0-based label indices, not dataset category
    ids — the class NAME is the portable identity and must ride along."""
    detections = SimpleNamespace(
        xyxy=[(1.0, 1.0, 2.0, 2.0), (3.0, 3.0, 4.0, 4.0)],
        confidence=[0.9, 0.8],
        class_id=[0, 7],  # 7 is out of range for a 1-class model
    )
    instances = _detections_to_instances(detections, ["balloon"])
    assert instances[0].category_name == "balloon"
    assert instances[1].category_name is None  # unknown label stays nameless
    unnamed = _detections_to_instances(detections, None)
    assert all(i.category_name is None for i in unnamed)


def test_default_weights_filename_reads_config_default():
    field = SimpleNamespace(default="rf-detr-small.pth")
    cls = SimpleNamespace(
        _model_config_class=SimpleNamespace(model_fields={"pretrain_weights": field})
    )
    assert _default_weights_filename(cls) == "rf-detr-small.pth"


def test_default_weights_filename_tolerates_missing_pieces():
    assert _default_weights_filename(SimpleNamespace()) is None  # no config class
    no_field = SimpleNamespace(_model_config_class=SimpleNamespace(model_fields={}))
    assert _default_weights_filename(no_field) is None
    none_default = SimpleNamespace(
        _model_config_class=SimpleNamespace(
            model_fields={"pretrain_weights": SimpleNamespace(default=None)}
        )
    )
    assert _default_weights_filename(none_default) is None


def test_train_kwargs_checkpoint_criterion_mapping(tmp_path):
    base = dict(dataset_dir=tmp_path, output_dir=tmp_path, epochs=1, batch_size=2)
    default = _train_kwargs(TrainSpec(**base))
    assert "smooth_alpha" not in default  # "map" keeps rfdetr's defaults

    smoothed = _train_kwargs(TrainSpec(**base, checkpoint_criterion="smoothed_map"))
    assert smoothed["smooth_alpha"] == pytest.approx(0.6)

    # an expert extra override still wins (extra is applied last)
    custom = _train_kwargs(
        TrainSpec(**base, checkpoint_criterion="smoothed_map",
                  extra={"smooth_alpha": 0.3})
    )
    assert custom["smooth_alpha"] == pytest.approx(0.3)

    # "loss" is handled by re-pointing the checkpoint callback, not a kwarg
    loss = _train_kwargs(TrainSpec(**base, checkpoint_criterion="loss"))
    assert "smooth_alpha" not in loss


def test_repoint_checkpoint_monitor_uses_the_mangled_init():
    calls = {}

    class ModelCheckpoint:  # the mangled name must match PL's class name
        def __init_monitor_mode(self, mode):
            calls["mode"] = mode

    class Sub(ModelCheckpoint):
        monitor = "val/mAP_50_95"

    callback = Sub()
    _repoint_checkpoint_monitor(callback, "val/loss", "min")
    assert callback.monitor == "val/loss"
    assert calls["mode"] == "min"


def _fake_best_cb(**overrides):
    defaults = dict(
        best_model_score=None, _best_ema=0.0, _best_raw_regular=0.0,
        _smooth_alpha=0.0, _monitor_ema=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_best_tracker_reports_regular_improvements_once():
    tracker = _BestTracker()
    tracker.callback = _fake_best_cb(best_model_score=0.10)
    assert tracker.observe(0) == {"best/epoch": 0.0, "best/is_ema": 0.0}
    assert tracker.observe(1) is None  # unchanged: nothing new to report
    tracker.callback.best_model_score = 0.30
    assert tracker.observe(2) == {"best/epoch": 2.0, "best/is_ema": 0.0}


def test_best_tracker_prefers_ema_on_strict_improvement():
    tracker = _BestTracker()
    tracker.callback = _fake_best_cb(
        best_model_score=0.20, _monitor_ema="val/ema_mAP_50_95"
    )
    assert tracker.observe(0) == {"best/epoch": 0.0, "best/is_ema": 0.0}
    tracker.callback._best_ema = 0.25  # EMA overtakes on strict >
    assert tracker.observe(3) == {"best/epoch": 3.0, "best/is_ema": 1.0}
    # a tie goes to the regular track (mirrors on_fit_end's strict >)
    tracker.callback.best_model_score = 0.25
    assert tracker.observe(4) == {"best/epoch": 4.0, "best/is_ema": 0.0}


def test_best_tracker_loss_mode_never_picks_ema():
    tracker = _BestTracker()
    tracker.callback = _fake_best_cb(best_model_score=9.5)  # _monitor_ema=None
    assert tracker.observe(1) == {"best/epoch": 1.0, "best/is_ema": 0.0}
    tracker.callback.best_model_score = 8.2
    assert tracker.observe(5) == {"best/epoch": 5.0, "best/is_ema": 0.0}


def test_best_tracker_smoothing_compares_raw_regular():
    tracker = _BestTracker()
    tracker.callback = _fake_best_cb(
        best_model_score=0.18,      # smoothed value
        _best_raw_regular=0.30,     # raw value at the smoothed-best epoch
        _smooth_alpha=0.6,
        _monitor_ema="val/ema_mAP_50_95",
        _best_ema=0.25,             # below the RAW regular: regular must win
    )
    assert tracker.observe(2) == {"best/epoch": 2.0, "best/is_ema": 0.0}


def test_segmentation_masks_become_polygons_for_final_instances_only():
    """E4-T15: RF-DETR-Seg masks are polygonised for detections at or above
    the score floor; low-confidence candidates stay boxes."""
    import numpy as np

    mask_a = np.zeros((40, 40), dtype=bool)
    mask_a[10:30, 5:25] = True  # a 20×20 block
    mask_b = np.zeros((40, 40), dtype=bool)
    mask_b[0:4, 0:4] = True
    detections = SimpleNamespace(
        xyxy=[(5.0, 10.0, 25.0, 30.0), (0.0, 0.0, 4.0, 4.0)],
        confidence=[0.9, 0.1],
        class_id=[0, 0],
        mask=np.stack([mask_a, mask_b]),
    )
    strong, weak = _detections_to_instances(detections, ["box"], masks=True, min_score=0.5)
    assert strong.segmentation and len(strong.segmentation[0]) >= 6  # ≥ 3 vertices
    xs, ys = strong.segmentation[0][0::2], strong.segmentation[0][1::2]
    assert 4 <= min(xs) <= 6 and 24 <= max(xs) <= 26
    assert 9 <= min(ys) <= 11 and 29 <= max(ys) <= 31
    assert weak.segmentation is None  # below the floor: not polygonised
    # without masks requested (or a detection-only model) nothing changes
    plain = _detections_to_instances(detections, ["box"])
    assert all(i.segmentation is None for i in plain)


def test_every_seg_size_maps_to_an_rfdetr_seg_class():
    for key, cls in _MODEL_CLASSES.items():
        if "seg" in key:
            assert cls.startswith("RFDETRSeg"), (key, cls)
    assert _MODEL_CLASSES["rfdetr-seg-nano"] == "RFDETRSegNano"
