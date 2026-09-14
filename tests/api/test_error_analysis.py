"""E6-T4: confusion matrix and per-class miss / false-positive analysis.

The matching core is a pure function over COCO-style dicts, so most cases are
hand-built fixtures with an obvious answer; the last tests go through a real
project with a completed fake run and persisted detections."""

from __future__ import annotations

import json

import pytest
from helpers.runs import completed_fake_run

from horos.api.error_analysis import (
    BACKGROUND,
    analyze_detections,
    analyze_errors,
    box_iou,
    match_image,
)
from horos.api.evaluate import _write_detections, load_detections
from horos.errors import ProjectError

GT = {
    "images": [
        {"id": 1, "file_name": "a.png", "width": 100, "height": 100},
        {"id": 2, "file_name": "b.png", "width": 100, "height": 100},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20]},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 20, 40, 40]},
    ],
    "categories": [
        {"id": 1, "name": "block"},
        {"id": 2, "name": "cone"},
    ],
}
NAMES = {1: "block", 2: "cone"}


def _det(image_id, category_id, bbox, score=0.9):
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}


def _perfect():
    return [_det(a["image_id"], a["category_id"], list(a["bbox"])) for a in GT["annotations"]]


def _analysis(detections, threshold=0.5, iou=0.5):
    analysis, per_image = analyze_detections(
        GT, detections, threshold=threshold, iou=iou, run_id="r", split="test"
    )
    return analysis, per_image


# ------------------------------------------------------------------ geometry


def test_box_iou_of_identical_disjoint_and_half_overlapping_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0
    # half overlap in x: inter 50, union 150
    assert box_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


# ------------------------------------------------------------------ matching


def test_perfect_predictions_are_all_true_positives():
    analysis, per_image = _analysis(_perfect())
    assert (analysis.tp, analysis.fp, analysis.fn, analysis.confused) == (3, 0, 0, 0)
    assert analysis.classes == ["block", "cone", BACKGROUND]
    # diagonal only: block×2, cone×1
    assert analysis.matrix == [[2, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert all(img.errors == 0 for img in per_image)
    for cls in analysis.per_class:
        assert cls.recall == pytest.approx(1.0) and cls.precision == pytest.approx(1.0)


def test_wrong_class_on_a_matching_box_is_a_confusion_pair():
    detections = _perfect()
    detections[1]["category_id"] = 1  # the cone in image 1 is called a block
    analysis, _ = _analysis(detections)
    assert analysis.confused == 1 and analysis.fn == 0 and analysis.fp == 0
    assert analysis.matrix[1][0] == 1  # row cone (gt) -> column block (pred)
    pair = analysis.confused_pairs[0]
    assert (pair.gt_name, pair.pred_name, pair.count) == ("cone", "block", 1)
    by_name = {c.name: c for c in analysis.per_class}
    assert by_name["cone"].confused_as == 1 and by_name["block"].confused_from == 1
    # the confusion still counts as a cone instance and a block prediction
    assert by_name["cone"].instances == 1 and by_name["cone"].recall == 0.0
    assert by_name["block"].precision == pytest.approx(2 / 3)


def test_unmatched_ground_truth_is_a_miss_in_the_background_column():
    detections = [d for d in _perfect() if d["image_id"] == 1]  # image 2 gets nothing
    analysis, per_image = _analysis(detections)
    assert analysis.fn == 1 and analysis.matrix[0][2] == 1  # block -> background
    image_two = next(img for img in per_image if img.image_id == 2)
    assert image_two.fn == 1 and image_two.errors == 1
    assert image_two.items[0].kind == "fn" and image_two.items[0].gt_name == "block"
    assert image_two.missed_area == pytest.approx(1600 / 10000)


def test_unmatched_prediction_is_a_false_positive_in_the_background_row():
    detections = _perfect() + [_det(2, 2, [70, 70, 20, 20], 0.8)]
    analysis, per_image = _analysis(detections)
    assert analysis.fp == 1 and analysis.matrix[2][1] == 1  # background -> cone
    by_name = {c.name: c for c in analysis.per_class}
    assert by_name["cone"].fp == 1 and by_name["cone"].precision == pytest.approx(0.5)
    image_two = next(img for img in per_image if img.image_id == 2)
    fp = [i for i in image_two.items if i.kind == "fp"]
    assert len(fp) == 1 and fp[0].pred_name == "cone" and fp[0].score == 0.8


def test_low_iou_overlap_does_not_match():
    detections = [_det(1, 1, [30, 30, 30, 30])]  # IoU with [10,10,30,30] is 100/1700
    analysis, _ = _analysis(detections)
    assert analysis.fp == 1 and analysis.fn == 3 and analysis.tp == 0


def test_threshold_drops_low_confidence_predictions():
    detections = _perfect()
    detections[0]["score"] = 0.3
    high, _ = _analysis(detections, threshold=0.5)
    low, _ = _analysis(detections, threshold=0.2)
    assert high.tp == 2 and high.fn == 1
    assert low.tp == 3 and low.fn == 0


def test_greedy_matching_prefers_the_confident_prediction_and_same_class():
    # two predictions on the same gt box: the confident one takes it, the
    # other becomes a false positive
    items = match_image(
        [{"category_id": 1, "bbox": [10, 10, 30, 30]}],
        [
            {"category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.6},
            {"category_id": 1, "bbox": [11, 11, 30, 30], "score": 0.9},
        ],
        iou_threshold=0.5,
        names=NAMES,
    )
    kinds = {round(i.score, 1): i.kind for i in items}
    assert kinds == {0.9: "tp", 0.6: "fp"}

    # a prediction that overlaps two gt boxes above the threshold prefers the
    # same-class box even when the other has slightly higher IoU
    items = match_image(
        [
            {"category_id": 2, "bbox": [10, 10, 30, 30]},  # other class, IoU 1.0
            {"category_id": 1, "bbox": [12, 12, 30, 30]},  # same class, IoU < 1
        ],
        [{"category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}],
        iou_threshold=0.5,
        names=NAMES,
    )
    matched = next(i for i in items if i.kind != "fn")
    assert matched.kind == "tp" and matched.gt_bbox == (12, 12, 30, 30)


def test_matching_carries_the_polygons_of_predictions_and_missed_truth():
    """A segmentation run's detections keep their mask outline; the overlay
    draws it (E6-T6). RLE masks have no outline and are dropped."""
    diamond = [25.0, 10.0, 40.0, 25.0, 25.0, 40.0, 10.0, 25.0]
    items = match_image(
        [
            {"category_id": 1, "bbox": [10, 10, 30, 30], "segmentation": [diamond]},
            {"category_id": 2, "bbox": [50, 50, 20, 20],
             "segmentation": {"counts": [], "size": [1, 1]}},
        ],
        [{"category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9,
          "segmentation": [[11.0, 11.0, 39.0, 11.0, 39.0, 39.0, 11.0, 39.0]]}],
        iou_threshold=0.5,
        names={1: "a", 2: "b"},
    )
    by_kind = {i.kind: i for i in items}
    assert by_kind["tp"].segmentation == [[11.0, 11.0, 39.0, 11.0, 39.0, 39.0, 11.0, 39.0]]
    assert by_kind["fn"].segmentation is None  # RLE, nothing to draw
    missed = match_image(
        [{"category_id": 1, "bbox": [10, 10, 30, 30], "segmentation": [diamond]}], [],
        iou_threshold=0.5, names={1: "a"},
    )
    assert missed[0].kind == "fn" and missed[0].segmentation == [diamond]


def test_predictions_of_a_class_unknown_to_the_split_do_not_crash():
    detections = _perfect() + [_det(1, 99, [0, 0, 5, 5], 0.9)]
    analysis, _ = _analysis(detections)
    assert "unknown:99" in analysis.classes
    assert analysis.fp == 1


def test_bad_fractions_are_refused():
    with pytest.raises(ProjectError, match="threshold"):
        _analysis(_perfect(), threshold=1.5)
    with pytest.raises(ProjectError, match="iou"):
        _analysis(_perfect(), iou=0.0)


# ------------------------------------------------------- project entry point


def test_analysis_needs_persisted_detections(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    with pytest.raises(ProjectError, match="run an evaluation first"):
        analyze_errors(project, run.run_id, "valid")


def test_persisted_detections_round_trip_and_feed_the_analysis(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    gt = json.loads(
        (project.root / "runs" / run.run_id / "dataset" / "valid" / "_annotations.coco.json")
        .read_text("utf-8")
    )
    ann = gt["annotations"][0]
    detections = [
        _det(ann["image_id"], ann["category_id"], list(ann["bbox"]), 0.9),
        _det(ann["image_id"], ann["category_id"], [0, 0, 3, 3], 0.2),  # below 0.5
    ]
    path = _write_detections(project, run.run_id, "valid", detections)
    assert path.name == "valid.detections.json"
    assert load_detections(project, run.run_id, "valid") == detections

    analysis = analyze_errors(project, run.run_id, "valid", threshold=0.5)
    assert analysis.run_id == run.run_id and analysis.split == "valid"
    assert analysis.tp == 1 and analysis.fp == 0 and analysis.fn == 0
    lowered = analyze_errors(project, run.run_id, "valid", threshold=0.1)
    assert lowered.fp == 1
