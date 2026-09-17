"""E8-T4: the model card ships the confidence to run the artifact at.

The number comes from the evaluation's F-score sweep (E6-T10), so what ships
next to the weights is what the evaluate page showed. A run with no evaluation
ships the reason instead of an invented value."""

from __future__ import annotations

import json
import time

import pytest
from helpers.runs import completed_fake_run

from horos.api.evaluate import _write_detections, evaluate_run
from horos.api.export import _suggested_threshold, start_model_export
from horos.api.jobs import job_status

pytest.importorskip("pycocotools", reason="training stack not installed")


@pytest.fixture
def run(tmp_path):
    return completed_fake_run(tmp_path, epochs=1)


def _good_detections(project, run_id, split):
    """Detections that find every ground-truth box, at spread-out scores, so
    the sweep has a real peak to report."""
    from horos.api.evaluate import eval_ground_truth

    gt, _, _ = eval_ground_truth(project, run_id, split)
    dets = [
        {"image_id": a["image_id"], "category_id": a["category_id"],
         "bbox": list(a["bbox"]), "score": 0.9 - 0.05 * n}
        for n, a in enumerate(gt["annotations"])
    ]
    dets += [  # junk well below the true boxes, so a threshold is worth picking
        {"image_id": gt["images"][0]["id"], "category_id": gt["categories"][0]["id"],
         "bbox": [0, 0, 2, 2], "score": 0.05 + 0.01 * n}
        for n in range(6)
    ]
    _write_detections(project, run_id, split, dets)


def test_a_run_with_no_evaluation_says_so_instead_of_inventing_one(run):
    project, record = run
    section = _suggested_threshold(project, record.run_id)
    assert section["confidence"] is None
    assert "run 'horos evaluate'" in section["note"]


def test_a_model_that_never_hits_reports_why(run):
    project, record = run
    # the fake backend only ever predicts an unknown class: nothing is correct
    evaluate_run(project, record.run_id, split="valid")
    section = _suggested_threshold(project, record.run_id)
    assert section["confidence"] is None
    assert "valid" in section["note"]


def test_the_threshold_and_what_it_came_from_are_recorded(run):
    project, record = run
    report = evaluate_run(project, record.run_id, split="valid")
    _good_detections(project, record.run_id, "valid")
    section = _suggested_threshold(project, record.run_id)

    assert 0.05 <= section["confidence"] <= 0.95
    assert section["metric"] == "F1"
    assert section["f_score"] > 0.9  # these detections are near perfect
    assert section["plateau"][0] <= section["confidence"] <= section["plateau"][1]
    came_from = section["derived_from"]
    assert came_from["split"] == "valid" and came_from["iou"] == 0.5
    assert came_from["labels"] == report.labels == "current"
    assert came_from["images"] == report.num_images
    assert came_from["instances"] == report.num_instances
    assert came_from["evaluated_at"] == report.created_at
    # every class the evaluation scored gets its own operating point
    assert {c["name"] for c in section["per_class"]} == {
        c.name for c in report.per_class if c.instances
    }
    for entry in section["per_class"]:
        assert 0.05 <= entry["confidence"] <= 0.95
        assert entry["enough_data"] is (entry["instances"] >= 10)


def test_the_test_split_wins_over_valid(run):
    project, record = run
    evaluate_run(project, record.run_id, split="valid")
    _good_detections(project, record.run_id, "valid")
    assert _suggested_threshold(project, record.run_id)["derived_from"]["split"] == "valid"
    # the sample project has no test split, so give the run one to prefer
    snapshot = project.root / "runs" / record.run_id / "dataset"
    (snapshot / "test").mkdir(exist_ok=True)
    for name in ("_annotations.coco.json",):
        (snapshot / "test" / name).write_text((snapshot / "valid" / name).read_text("utf-8"))
    for image in (snapshot / "valid").glob("*.png"):
        (snapshot / "test" / image.name).write_bytes(image.read_bytes())
    evaluate_run(project, record.run_id, split="test", labels="snapshot")
    _good_detections(project, record.run_id, "test")
    assert _suggested_threshold(project, record.run_id)["derived_from"]["split"] == "test"


def test_the_exported_card_carries_it(run):
    project, record = run
    evaluate_run(project, record.run_id, split="valid")
    _good_detections(project, record.run_id, "valid")
    job_id = start_model_export(project, record.run_id, format="onnx")
    deadline = time.monotonic() + 30
    while job_status(project, job_id).state in ("pending", "running"):
        assert time.monotonic() < deadline, "export did not finish"
        time.sleep(0.1)
    assert job_status(project, job_id).state == "completed"
    card = json.loads(
        (project.root / "runs" / record.run_id / "exports" / "onnx" / "model_card.json")
        .read_text("utf-8")
    )
    assert card["threshold"]["confidence"] == _suggested_threshold(
        project, record.run_id
    )["confidence"]
    assert card["threshold"]["derived_from"]["split"] == "valid"
