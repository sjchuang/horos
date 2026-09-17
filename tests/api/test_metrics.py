"""E6-T3: COCO metrics agree with the reference implementation on fixtures
with known answers."""

from __future__ import annotations

import json

import pytest

from horos.api.evaluate import _compute_metrics

pytest.importorskip("pycocotools", reason="training stack not installed")

# two images, two classes, three gt boxes — small enough to reason about
GT = {
    "images": [
        {"id": 1, "file_name": "a.png", "width": 100, "height": 100},
        {"id": 2, "file_name": "b.png", "width": 100, "height": 100},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30],
         "area": 900, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20],
         "area": 400, "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 20, 40, 40],
         "area": 1600, "iscrowd": 0},
    ],
    "categories": [
        {"id": 1, "name": "block", "supercategory": "none"},
        {"id": 2, "name": "cone", "supercategory": "none"},
    ],
}


def _det(image_id, category_id, bbox, score=0.95):
    return {"image_id": image_id, "category_id": category_id,
            "bbox": bbox, "score": score}


@pytest.fixture
def gt_path(tmp_path):
    # kept as a fixture so every test still names one; the metrics take the
    # ground truth itself now (it may be assembled from the project's current
    # labels and never written as a COCO file)
    path = tmp_path / "_annotations.coco.json"
    path.write_text(json.dumps(GT), "utf-8")
    return path


def _report(gt_path, detections):
    return _compute_metrics(None, "run-x", "test", GT, detections)


def test_perfect_predictions_score_full_marks(gt_path):
    detections = [
        _det(ann["image_id"], ann["category_id"], list(ann["bbox"]))
        for ann in GT["annotations"]
    ]
    report = _report(gt_path, detections)
    assert report.map_5095 == pytest.approx(1.0)
    assert report.map_50 == pytest.approx(1.0)
    assert report.mar_100 == pytest.approx(1.0)
    for cls in report.per_class:
        assert cls.ap == pytest.approx(1.0) and cls.ap50 == pytest.approx(1.0)
    assert {c.name for c in report.per_class} == {"block", "cone"}
    assert report.num_images == 2 and report.num_instances == 3


def test_one_class_missed_entirely_halves_map50(gt_path):
    detections = [
        _det(ann["image_id"], 1, list(ann["bbox"]))
        for ann in GT["annotations"]
        if ann["category_id"] == 1
    ]
    report = _report(gt_path, detections)
    by_name = {c.name: c for c in report.per_class}
    assert by_name["block"].ap50 == pytest.approx(1.0)
    assert by_name["cone"].ap50 == pytest.approx(0.0)
    assert report.map_50 == pytest.approx(0.5)  # mean over the two classes


def test_low_iou_boxes_pass_at_50_but_fail_at_75(gt_path):
    # shift each box by ~30% of its size: IoU lands between 0.5 and 0.75
    detections = []
    for ann in GT["annotations"]:
        x, y, w, h = ann["bbox"]
        detections.append(
            _det(ann["image_id"], ann["category_id"],
                 [x + 0.15 * w, y + 0.15 * h, w, h])
        )
    report = _report(gt_path, detections)
    assert report.map_50 == pytest.approx(1.0)
    assert report.map_75 == pytest.approx(0.0)
    assert 0.0 < report.map_5095 < 1.0


def test_instance_counts_come_from_the_ground_truth(gt_path):
    report = _report(gt_path, [_det(1, 1, [10, 10, 30, 30])])
    by_name = {c.name: c for c in report.per_class}
    assert by_name["block"].instances == 2
    assert by_name["cone"].instances == 1


def test_pr_curve_has_the_101_standard_recall_points(gt_path):
    detections = [
        _det(ann["image_id"], ann["category_id"], list(ann["bbox"]))
        for ann in GT["annotations"]
    ]
    report = _report(gt_path, detections)
    for cls in report.per_class:
        assert len(cls.pr_curve_50) == 101
        assert all(0.0 <= v <= 1.0 for v in cls.pr_curve_50)


def test_no_detections_yield_a_zero_report_not_a_crash(gt_path):
    report = _report(gt_path, [])
    assert report.map_5095 == 0.0 and report.map_50 == 0.0
    assert {c.name for c in report.per_class} == {"block", "cone"}
    assert all(c.ap == 0.0 for c in report.per_class)
