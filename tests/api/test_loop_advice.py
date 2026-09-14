"""E10-S5: after each round the loop says whether another one is worth it —
a rule-based verdict with its reason, driven by the round history."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS

from horos.api.loop import loop_advice, update_loop_settings
from horos.api.project import create_project
from horos.core.dataset import Annotation, Category
from horos.core.rounds import LoopRound, PickedImage, SelectionRecord, save_round


def _project(tmp_path, total=40, labeled=10):
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box")])
    colours = list(COLOURS.values())
    for n in range(1, total + 1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64, 48, colours[n % len(colours)])
        project.add_image(path, width=64, height=48)
    for image_id in range(1, labeled + 1):
        project.save_annotations(
            image_id, [Annotation(id=1, image_id=image_id, category_id=1, bbox=(1, 1, 5, 5))],
            expected_version=0,
        )
    return project


def _closed_round(project, number, *, metric, labeled_before, labeled_after, key="val/mAP_50"):
    """A closed round with a recorded metric — what training and review leave behind."""
    picks = [PickedImage(image_id=i, score=0.5, reason="t") for i in range(1, 3)]
    record = LoopRound(
        number=number, labeled_before=labeled_before, labeled_after=labeled_after,
        selection=SelectionRecord(strategy="diversity", requested=2, pool_size=30, picks=picks),
        metrics={key: metric},
    )
    record = record.advance("labeling").advance("training").advance("reviewing").advance("closed")
    save_round(project, record)


def test_no_round_yet_asks_for_the_first_one(tmp_path):
    a = loop_advice(_project(tmp_path))
    assert a.verdict == "first_round" and a.pool_size == 30 and a.metric is None


def test_one_round_is_not_a_trend(tmp_path):
    project = _project(tmp_path)
    _closed_round(project, 1, metric=0.40, labeled_before=10, labeled_after=30)
    a = loop_advice(project)
    assert a.verdict == "continue" and "not a trend" in a.title
    assert a.metric == 0.40 and a.labels_spent == 20 and a.gain_per_100 is None


def test_clear_gains_say_keep_going_with_gain_per_100(tmp_path):
    project = _project(tmp_path)
    _closed_round(project, 1, metric=0.40, labeled_before=10, labeled_after=30)
    _closed_round(project, 2, metric=0.50, labeled_before=30, labeled_after=50)
    a = loop_advice(project)
    assert a.verdict == "continue" and a.delta == pytest.approx(0.10)
    assert a.gain_per_100 == pytest.approx(0.5)  # +0.10 for 20 labels
    assert "still pay off" in a.title and "per 100 labels" in a.reason


def test_rounds_headlined_by_different_metrics_are_not_compared(tmp_path):
    """demo_project: round 5 was headlined by eval/valid, round 6 by the new
    eval/test — no delta exists, and the advice must say so, not crash."""
    project = _project(tmp_path)
    _closed_round(project, 1, metric=0.70, labeled_before=10, labeled_after=30,
                  key="eval/valid/map_5095")
    _closed_round(project, 2, metric=0.68, labeled_before=30, labeled_after=50,
                  key="eval/test/map_5095")
    a = loop_advice(project)
    assert a.verdict == "continue" and a.delta is None and a.gain_per_100 is None
    assert "not comparable" in a.reason and "eval/valid/map_5095" in a.reason


def test_two_flat_rounds_say_flattening(tmp_path):
    project = _project(tmp_path)
    _closed_round(project, 1, metric=0.60, labeled_before=10, labeled_after=30)
    _closed_round(project, 2, metric=0.605, labeled_before=30, labeled_after=50)
    _closed_round(project, 3, metric=0.608, labeled_before=50, labeled_after=70)
    a = loop_advice(project)
    assert a.verdict == "flattening" and a.rounds_with_metric == 3
    assert "bigger round" in a.reason or "larger model" in a.reason


def test_a_drop_says_check_the_labels(tmp_path):
    project = _project(tmp_path)
    _closed_round(project, 1, metric=0.60, labeled_before=10, labeled_after=30)
    _closed_round(project, 2, metric=0.52, labeled_before=30, labeled_after=50)
    a = loop_advice(project)
    assert a.verdict == "check_labels" and "-0.080" in a.reason


def test_goal_reached_beats_everything(tmp_path):
    project = _project(tmp_path)
    update_loop_settings(project, target=0.55)
    _closed_round(project, 1, metric=0.60, labeled_before=10, labeled_after=30)
    a = loop_advice(project)
    assert a.verdict == "target_reached" and a.target == 0.55
    # loss-type metrics compare the other way round
    project2 = _project(tmp_path / "b")
    update_loop_settings(project2, target=1.0)
    _closed_round(project2, 1, metric=0.8, labeled_before=10, labeled_after=30, key="loss")
    assert loop_advice(project2).verdict == "target_reached"


def test_empty_pool_says_nothing_left(tmp_path):
    project = _project(tmp_path, total=12, labeled=12)
    _closed_round(project, 1, metric=0.6, labeled_before=0, labeled_after=12)
    assert loop_advice(project).verdict == "nothing_left"


def test_target_setting_validates(tmp_path):
    project = _project(tmp_path)
    import json

    from horos.api.loop import get_loop_settings

    # a goal of 0 (or below) would be met by any model: it is "no goal", not an error
    assert update_loop_settings(project, target=-1).target is None
    assert update_loop_settings(project, target=0).target is None
    assert update_loop_settings(project, target=None).target is None
    # files written by earlier versions may still carry 0
    path = project.root / "loop.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "target": 0.0}))
    assert get_loop_settings(project).target is None
    _closed_round(project, 1, metric=0.6, labeled_before=10, labeled_after=30)
    assert loop_advice(project).verdict != "target_reached"


def _evaluated_round(project, number, *, train, test, labeled_before, labeled_after, metric):
    _closed_round(project, number, metric=metric, labeled_before=labeled_before,
                  labeled_after=labeled_after, key="eval/test/map_5095")
    from horos.core.rounds import load_round, save_round

    record = load_round(project, number)
    metrics = {**record.metrics, "eval/train/map_50": train, "eval/test/map_50": test}
    save_round(project, record.model_copy(update={"metrics": metrics}))


def test_a_wide_train_test_gap_keeps_the_loop_going_even_past_the_goal(tmp_path):
    """The user saw 'Goal reached — export' while train sat 0.15 above test:
    a model that still memorises its photos is not done, goal or not."""
    project = _project(tmp_path)
    update_loop_settings(project, target=0.6)
    _evaluated_round(project, 1, train=0.95, test=0.79, labeled_before=10, labeled_after=30,
                     metric=0.68)
    a = loop_advice(project)
    assert a.verdict == "continue" and a.gap == pytest.approx(0.16)
    assert "train is still well above test" in a.title and "goal of 0.6 is met" in a.reason
    # once the curves meet, the goal counts
    _evaluated_round(project, 2, train=0.83, test=0.80, labeled_before=30, labeled_after=50,
                     metric=0.70)
    a = loop_advice(project)
    assert a.verdict == "target_reached" and a.gap == pytest.approx(0.03)
