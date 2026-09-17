"""E6-T13: an evaluation scores the project's labels as they are now.

Correcting a wrong box in a held-out set is the point of error analysis, and
the correction has to count on the next evaluation — relabeling is not a
reason to retrain. `labels="snapshot"` keeps the frozen export for
reproducing an older number. See horos/api/evaluate.py's module docstring."""

from __future__ import annotations

import json

import pytest
from helpers.runs import completed_fake_run

from horos.api.error_analysis import analyze_errors
from horos.api.evaluate import (
    DEFAULT_LABELS,
    _write_detections,
    eval_ground_truth,
    evaluate_run,
    get_eval_report,
)
from horos.core.dataset import Annotation
from horos.errors import ProjectError

pytest.importorskip("pycocotools", reason="training stack not installed")


def _boxes(project, image_id, count, category_id=2):
    """Replace one photo's labels with `count` non-overlapping boxes."""
    stored = project.load_annotations(image_id)
    project.save_annotations(
        image_id,
        [
            Annotation(id=n + 1, image_id=image_id, category_id=category_id,
                       bbox=(2 + 9 * n, 2, 6, 6))
            for n in range(count)
        ],
        expected_version=stored.version,
    )


def test_current_labels_are_the_default(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    assert DEFAULT_LABELS == "current"
    # c.png is the valid split and carried one box at training time
    _boxes(project, 3, 3)
    report = evaluate_run(project, run.run_id, split="valid")
    assert report.labels == "current"
    assert report.num_images == 1 and report.num_instances == 3
    assert "as they are now" in report.notes[0]


def test_the_snapshot_is_still_available_unchanged(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    _boxes(project, 3, 3)
    report = evaluate_run(project, run.run_id, split="valid", labels="snapshot")
    assert report.labels == "snapshot"
    assert report.num_instances == 1  # the one box the run trained beside
    assert "trained with" in report.notes[0]


def test_a_photo_labeled_into_the_set_since_the_run_joins_it(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    # a photo that was in no set at training time, labeled and placed now
    record = project.add_image(
        project.image_path(project.get_image(3)), width=32, height=32
    )
    _boxes(project, record.id, 2)
    project.update_image_splits({record.id: "valid"})
    report = evaluate_run(project, run.run_id, split="valid")
    assert report.num_images == 2
    assert any("labeled into this set since" in n for n in report.notes)


def test_a_photo_the_run_trained_on_is_held_back(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    # a reshuffle moved a training photo into valid: scoring on it would be
    # marking the model's own homework
    project.update_image_splits({1: "valid"})
    report = evaluate_run(project, run.run_id, split="valid")
    assert report.num_images == 1  # c.png only, not a.png
    assert any("the run trained on them" in n for n in report.notes)


def test_skipped_photos_and_unlabeled_set_members_are_left_out(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    record = project.add_image(
        project.image_path(project.get_image(3)), width=32, height=32
    )
    _boxes(project, record.id, 1)
    project.update_image_splits({record.id: "valid"})
    project.set_excluded([record.id], True)  # unfit for training (E10-T16)
    other = project.add_image(
        project.image_path(project.get_image(3)), width=32, height=32
    )
    project.update_image_splits({other.id: "valid"})  # a member with no labels
    report = evaluate_run(project, run.run_id, split="valid")
    assert report.num_images == 1


def test_a_set_with_nothing_left_to_score_is_refused_before_inference(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    project.set_excluded([3], True)
    with pytest.raises(ProjectError, match="nothing to score"):
        evaluate_run(project, run.run_id, split="valid")


# ------------------------------------------- the analyses see the same boxes


def test_error_analysis_re_matches_the_boxes_the_metrics_used(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    _boxes(project, 3, 4)
    report = evaluate_run(project, run.run_id, split="valid")
    gt, labels, resolve = eval_ground_truth(project, run.run_id, "valid")
    assert labels == "current"
    assert len(gt["annotations"]) == report.num_instances == 4
    # the photo comes from the project now, not from the run's frozen copy
    assert resolve("c.png") == project.image_path(project.get_image(3))
    analysis = analyze_errors(project, run.run_id, "valid", threshold=0.5)
    assert analysis.tp + analysis.fn + analysis.confused == report.num_instances

    # labels edited AFTER the evaluation must not move the analysis: those
    # detections were matched against the boxes above
    _boxes(project, 3, 9)
    again = analyze_errors(project, run.run_id, "valid", threshold=0.5)
    assert again.tp + again.fn + again.confused == 4
    assert get_eval_report(project, run.run_id, "valid").num_instances == 4


def test_an_evaluation_from_before_this_existed_reads_its_snapshot(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    # an old run has detections but no persisted ground truth
    snapshot = json.loads(
        (project.root / "runs" / run.run_id / "dataset" / "valid"
         / "_annotations.coco.json").read_text("utf-8")
    )
    _write_detections(project, run.run_id, "valid", [])
    _boxes(project, 3, 5)  # the project has moved on since
    gt, labels, resolve = eval_ground_truth(project, run.run_id, "valid")
    assert labels == "snapshot"
    assert len(gt["annotations"]) == len(snapshot["annotations"]) == 1
    assert resolve("c.png").parent.name == "valid"
