"""E6-T1: single and batch inference bound to a trained run, plus the
evaluation event stream over the run's own dataset snapshot."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.evaluate import (
    _reset_backend_cache,
    evaluation_events,
    get_eval_report,
    infer_image,
)
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status
from horos.errors import ProjectError

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"

pycocotools = pytest.importorskip("pycocotools", reason="training stack not installed")


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    _reset_backend_cache()


@pytest.fixture
def trained(tmp_path):
    """A project with one completed fake run."""
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=1)
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    assert training_status(project, record.run_id).run.state == "completed"
    return project, record.run_id


def test_infer_image_uses_the_runs_checkpoint(trained, tmp_path):
    project, run_id = trained
    image = next((project.root / "runs" / run_id / "dataset" / "train").glob("*.png"))
    prediction = infer_image(project, run_id, image)
    assert prediction.instances and prediction.instances[0].score == 0.9


def test_infer_rejects_missing_file_and_unknown_run(trained):
    project, run_id = trained
    with pytest.raises(ProjectError, match="No such image"):
        infer_image(project, run_id, "does-not-exist.png")
    with pytest.raises(ProjectError, match="No such training run"):
        infer_image(project, "nope", "also-irrelevant.png")


def test_infer_rejects_unfinished_runs(trained):
    project, run_id = trained
    slow = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=60,
                       extra={"sleep_per_epoch": 0.5}),
    )
    try:
        with pytest.raises(ProjectError, match="no usable checkpoint"):
            infer_image(project, slow.run_id, "x.png")
    finally:
        from horos.api.train import stop_training

        stop_training(project, slow.run_id)
        deadline = time.monotonic() + 30
        while training_status(project, slow.run_id).run.state in ("pending", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.2)


def test_evaluation_streams_and_persists_a_report(trained):
    project, run_id = trained
    events = list(evaluation_events(project, run_id, split="valid"))
    assert events[0].type == "started" and events[0].total == 1  # 1 valid image
    assert events[-1].type == "completed"
    report = events[-1].result
    assert report["split"] == "valid" and report["num_images"] == 1
    # the fake predicts category 0 which is not in the gt → zero AP, but the
    # pipeline is intact and every gt class is reported
    assert report["map_50"] == 0.0
    assert {c["name"] for c in report["per_class"]} == {"forklift", "pallet"}

    persisted = get_eval_report(project, run_id, "valid")
    assert persisted.run_id == run_id and persisted.map_50 == 0.0


def test_evaluation_refuses_missing_split(trained):
    project, run_id = trained
    with pytest.raises(ProjectError, match="no 'nope' split"):
        list(evaluation_events(project, run_id, split="nope"))


def test_report_before_any_evaluation_is_a_clear_error(trained):
    project, run_id = trained
    with pytest.raises(ProjectError, match="no persisted evaluation"):
        get_eval_report(project, run_id, "test")


def test_evaluation_maps_predictions_by_class_name(trained, monkeypatch):
    """The gt uses COCO category ids; backends emit their own label indices.
    A backend that names its classes must be scored by name (the balloon-det
    bug: label 0 vs category id 1 made every real detection score zero)."""
    import json

    from horos.backends.base import ImagePrediction, PredictedInstance

    project, run_id = trained
    gt_path = project.root / "runs" / run_id / "dataset" / "valid" / "_annotations.coco.json"
    gt = json.loads(gt_path.read_text("utf-8"))
    ann_by_image = {}
    for ann in gt["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    name_by_id = {c["id"]: c["name"] for c in gt["categories"]}
    file_to_id = {i["file_name"]: i["id"] for i in gt["images"]}

    class EchoGtBackend:
        """Perfect predictions — but identified by NAME with label ids 0-based."""

        def infer_one(self, image, *, threshold=0.5):
            image_id = file_to_id[image.name]
            instances = [
                PredictedInstance(
                    bbox=tuple(a["bbox"]),
                    score=0.99,
                    category_id=0,  # model-internal label, deliberately wrong as an id
                    category_name=name_by_id[a["category_id"]],
                )
                for a in ann_by_image.get(image_id, [])
            ]
            return ImagePrediction(image=str(image), instances=instances)

    import horos.api.evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module, "_load_run_backend",
        lambda project, run_id, device=None: (EchoGtBackend(), None),
    )
    events = list(evaluate_module.evaluation_events(project, run_id, split="valid"))
    report = events[-1].result
    assert report["map_50"] == pytest.approx(1.0)  # names matched, ids ignored


def test_evaluation_persists_mask_polygons_for_the_overlay(trained, monkeypatch):
    """A segmentation run's masks reach the persisted detections and the
    error items the evaluate page's overlay is drawn from (E6-T6)."""
    import json

    from horos.api.error_analysis import image_errors
    from horos.api.evaluate import load_detections
    from horos.backends.base import ImagePrediction, PredictedInstance

    project, run_id = trained
    gt_path = project.root / "runs" / run_id / "dataset" / "valid" / "_annotations.coco.json"
    gt = json.loads(gt_path.read_text("utf-8"))
    name_by_id = {c["id"]: c["name"] for c in gt["categories"]}
    file_to_id = {i["file_name"]: i["id"] for i in gt["images"]}
    ann_by_image = {}
    for ann in gt["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def outline(bbox):
        x, y, w, h = bbox
        return [x + 1, y + 1, x + w - 1, y + 1, x + w - 1, y + h - 1, x + 1, y + h - 1]

    class SegEchoBackend:
        def infer_one(self, image, *, threshold=0.5):
            anns = ann_by_image.get(file_to_id[image.name], [])
            return ImagePrediction(image=str(image), instances=[
                PredictedInstance(bbox=tuple(a["bbox"]), score=0.95, category_id=0,
                                  category_name=name_by_id[a["category_id"]],
                                  segmentation=[outline(a["bbox"])])
                for a in anns
            ])

    import horos.api.evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module, "_load_run_backend",
        lambda project, run_id, device=None: (SegEchoBackend(), None),
    )
    list(evaluate_module.evaluation_events(project, run_id, split="valid"))
    detections = load_detections(project, run_id, "valid")
    assert detections and all(len(d["segmentation"][0]) == 8 for d in detections)
    image_id = detections[0]["image_id"]
    errors = image_errors(project, run_id, "valid", image_id, threshold=0.5, iou=0.5)
    assert errors.items and all(i.kind == "tp" and i.segmentation for i in errors.items)
