"""E6-T10: the suggested operating confidence threshold.

The sweep is derived from ONE matching pass, so the first duty of this file is
to prove it agrees with `analyze_detections` — the tested matcher — at every
grid point. The rest covers the recommendation rule (peak, plateau, per class)
and the project entry point."""

from __future__ import annotations

import json

import pytest
from helpers.runs import completed_fake_run

from horos.api.error_analysis import analyze_detections
from horos.api.evaluate import _write_detections
from horos.api.threshold import (
    BASELINE,
    MIN_INSTANCES,
    suggest_threshold,
    sweep_detections,
)
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
    "categories": [{"id": 1, "name": "block"}, {"id": 2, "name": "cone"}],
}


def _det(image_id, category_id, bbox, score):
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}


def _perfect(score=0.9):
    return [
        _det(a["image_id"], a["category_id"], list(a["bbox"]), score)
        for a in GT["annotations"]
    ]


# --------------------------------------------- the one-pass sweep is exact


def test_every_grid_point_matches_a_full_analysis_pass():
    # the three真 boxes at descending confidences plus two false positives
    detections = [
        _det(1, 1, [10, 10, 30, 30], 0.91),
        _det(1, 2, [50, 50, 20, 20], 0.62),
        _det(2, 1, [20, 20, 40, 40], 0.33),
        _det(1, 1, [70, 5, 20, 20], 0.55),   # matches nothing
        _det(2, 2, [20, 20, 40, 40], 0.44),  # right box, wrong class -> confusion
    ]
    advice = sweep_detections(GT, detections, iou=0.5)
    assert len(advice.points) == 91
    for point in advice.points:
        reference, _ = analyze_detections(
            GT, detections, threshold=point.threshold, iou=0.5
        )
        assert (point.tp, point.fp, point.fn, point.confused) == (
            reference.tp, reference.fp, reference.fn, reference.confused
        ), f"threshold {point.threshold}"


def test_precision_and_recall_follow_the_counts():
    advice = sweep_detections(GT, _perfect(), iou=0.5)
    low = next(p for p in advice.points if p.threshold == pytest.approx(0.5))
    assert (low.tp, low.fp, low.fn) == (3, 0, 0)
    assert low.precision == pytest.approx(1.0)
    assert low.recall == pytest.approx(1.0)
    assert low.f_score == pytest.approx(1.0)


# ------------------------------------------------------ the recommendation


def test_a_perfect_run_recommends_the_middle_of_its_plateau():
    # every threshold up to 0.90 is perfect, so the plateau is 0.05..0.90
    advice = sweep_detections(GT, _perfect(0.9), iou=0.5)
    assert advice.confident
    assert advice.plateau == (0.05, 0.9)
    assert advice.recommended == pytest.approx(0.47)  # the plateau's middle grid point
    assert advice.best.f_score == pytest.approx(1.0)
    assert "F1 peaks at 1.000" in advice.reason
    # the 0.50 default is on that plateau, so the advice says so instead of
    # asking the user to move the slider for nothing
    assert "nothing needs changing" in advice.reason


def test_the_threshold_lands_above_a_noisy_low_confidence_band():
    # three correct boxes above 0.8, twelve false positives below 0.4
    detections = _perfect(0.85) + [
        _det(1, 1, [60 + i, 2, 10, 10], 0.2 + 0.01 * i) for i in range(12)
    ]
    advice = sweep_detections(GT, detections, iou=0.5)
    assert advice.recommended > 0.32
    assert advice.best.fp == 0 and advice.best.tp == 3
    # the 0.50 default is reported for comparison, and here it is just as good
    assert advice.baseline.threshold == pytest.approx(BASELINE)


def test_recall_weighting_lowers_the_threshold_precision_weighting_raises_it():
    # a correct box at 0.30 and a false positive at 0.60: keeping the correct
    # one costs precision, dropping it costs recall
    detections = [
        _det(1, 1, [10, 10, 30, 30], 0.9),
        _det(1, 2, [50, 50, 20, 20], 0.9),
        _det(2, 1, [20, 20, 40, 40], 0.3),
        _det(2, 2, [70, 70, 20, 20], 0.6),
    ]
    recall_first = sweep_detections(GT, detections, iou=0.5, beta=2.0)
    precision_first = sweep_detections(GT, detections, iou=0.5, beta=0.5)
    assert recall_first.recommended < precision_first.recommended
    assert recall_first.best.recall >= precision_first.best.recall
    assert precision_first.best.precision >= recall_first.best.precision
    assert "F2 peaks" in recall_first.reason


def test_per_class_thresholds_are_independent_and_flag_thin_classes():
    # block's detections are confident, cone's are not
    detections = [
        _det(1, 1, [10, 10, 30, 30], 0.95),
        _det(2, 1, [20, 20, 40, 40], 0.92),
        _det(1, 2, [50, 50, 20, 20], 0.20),
    ]
    advice = sweep_detections(GT, detections, iou=0.5)
    by_name = {c.name: c for c in advice.per_class}
    assert by_name["cone"].recommended < 0.2  # its only detection sits at 0.20
    assert by_name["block"].recommended > by_name["cone"].recommended
    # two and one ground-truth boxes: far too few to advise on
    assert not any(c.enough_data for c in advice.per_class)
    assert MIN_INSTANCES > 2
    assert "Too little ground truth" in " ".join(advice.notes)


def test_a_run_that_detects_nothing_keeps_the_default_and_says_so():
    advice = sweep_detections(GT, [], iou=0.5, split="valid")
    assert not advice.confident
    assert advice.recommended == pytest.approx(BASELINE)
    assert "detects nothing" in advice.reason


def test_an_empty_split_keeps_the_default():
    empty = {"images": [], "annotations": [], "categories": GT["categories"]}
    advice = sweep_detections(empty, [], iou=0.5, split="valid")
    assert not advice.confident and "no ground truth" in advice.reason


def test_tuning_on_the_test_split_is_called_out():
    on_test = sweep_detections(GT, _perfect(), iou=0.5, split="test")
    on_valid = sweep_detections(GT, _perfect(), iou=0.5, split="valid")
    assert any("test split" in n for n in on_test.notes)
    assert not any("test split" in n for n in on_valid.notes)


def test_bad_inputs_are_refused():
    with pytest.raises(ProjectError, match="beta"):
        sweep_detections(GT, _perfect(), beta=0.0)
    with pytest.raises(ProjectError, match="iou"):
        sweep_detections(GT, _perfect(), iou=0.0)


# ------------------------------------------------------- project entry point


def test_advice_needs_a_prior_evaluation(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    with pytest.raises(ProjectError, match="run an evaluation first"):
        suggest_threshold(project, run.run_id, "valid")


def test_advice_reads_the_runs_persisted_detections(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    gt = json.loads(
        (project.root / "runs" / run.run_id / "dataset" / "valid" / "_annotations.coco.json")
        .read_text("utf-8")
    )
    ann = gt["annotations"][0]
    _write_detections(project, run.run_id, "valid", [
        _det(ann["image_id"], ann["category_id"], list(ann["bbox"]), 0.88),
        _det(ann["image_id"], ann["category_id"], [0, 0, 3, 3], 0.12),
    ])
    advice = suggest_threshold(project, run.run_id, "valid")
    assert advice.run_id == run.run_id and advice.split == "valid"
    assert advice.confident
    # the 0.12 false positive is excluded by any sensible threshold
    assert advice.recommended > 0.12
    assert advice.best.fp == 0
