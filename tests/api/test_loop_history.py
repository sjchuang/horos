"""E10-T9: per-round history — labels spent, the round's validation metric
and the change against the previous round, so "is another round worth it"
has an answer (E10-S5)."""

from __future__ import annotations

import time

from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeDetector, FakeEmbedder
from helpers.runs import FAKE, ensure_worker_can_import_helpers

from horos.api.loop import close_round, loop_history, loop_status, select_round, train_round
from horos.api.project import create_project
from horos.api.train import training_status
from horos.core.dataset import Annotation, Category


def _project(tmp_path, total=32, labeled=24):
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box"), Category(id=2, name="pallet")])
    colours = list(COLOURS.values())
    for n in range(1, total + 1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, colours[n % len(colours)])
        project.add_image(path, width=64 + n, height=48)
    for image_id in range(1, labeled + 1):
        cat = 2 if image_id % 4 == 0 else 1
        project.save_annotations(
            image_id, [Annotation(id=1, image_id=image_id, category_id=cat, bbox=(10, 10, 30, 20))],
            expected_version=0,
        )
    return project


def _label(project, image_ids):
    for image_id in image_ids:
        stored = project.load_annotations(image_id)
        project.save_annotations(
            image_id, [Annotation(id=1, image_id=image_id, category_id=1, bbox=(5, 5, 20, 20))],
            expected_version=stored.version,
        )


def _cycle(project, *, count, epochs):
    """select → label every pick → train → wait → close; returns the round."""
    record = select_round(project, count=count, embedder=FakeEmbedder(),
                          embedding_model="fake-embedder", detector=FakeDetector(),
                          preannotate=False)
    _label(project, record.image_ids)
    record = train_round(project, record.number, entrypoint_override=FAKE, epochs=epochs)
    deadline = time.monotonic() + 90
    while training_status(project, record.train_run_id).run.state in ("queued", "pending",
                                                                       "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    return close_round(project, record.number)


def test_history_tracks_labels_spent_metrics_and_deltas(tmp_path):
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    assert loop_history(project) == []

    # the fake backend's final loss is 1/epochs: round 2 trains longer → improves
    first = _cycle(project, count=2, epochs=1)
    time.sleep(1.05)  # run ids carry a per-second timestamp prefix
    second = _cycle(project, count=3, epochs=2)

    history = loop_history(project)
    assert [h.number for h in history] == [1, 2]
    one, two = history
    assert one.labels_spent == 2 and two.labels_spent == 3
    assert one.labeled_before == 24 and two.labeled_before == 26
    assert one.metric_key == two.metric_key and one.metric_key is not None
    assert one.metric is not None and two.metric is not None
    assert one.delta is None and one.improved is None  # nothing to compare with
    assert two.delta is not None and two.improved is True  # lower loss than round 1
    assert one.train_run_id == first.train_run_id and two.train_run_id == second.train_run_id
    assert loop_status(project).rounds == history


def test_open_round_reports_labels_so_far(tmp_path):
    project = _project(tmp_path)
    record = select_round(project, count=3, embedder=FakeEmbedder(),
                          embedding_model="fake-embedder", detector=FakeDetector(),
                          preannotate=False)
    _label(project, record.image_ids[:1])
    entry = loop_history(project)[0]
    assert entry.state == "labeling" and entry.labels_spent == 1 and entry.labeled == 1
    assert entry.metric is None and entry.delta is None


def test_headline_prefers_validation_map_over_loss():
    from horos.api.loop import _headline

    rfdetr_like = {"loss": 6.8, "val/loss": 5.9, "val/mAP_50": 0.57, "val/ema_mAP_50_95": 0.59,
                   "val/mAP_50_95": 0.54}
    assert _headline(rfdetr_like) == ("val/ema_mAP_50_95", 0.59)
    # the loop's own split evaluation wins once it exists — it is computed the
    # same way every round, unlike the trainer's EMA numbers
    assert _headline({**rfdetr_like, "eval/valid/map_5095": 0.71})[0] == "eval/valid/map_5095"
    assert _headline({"loss": 1.0}) == ("loss", 1.0)
    assert _headline({}) == (None, None)


def test_completed_round_evaluates_every_split_for_the_learning_curve(tmp_path, monkeypatch):
    """E10-S5: after training, the round's model is evaluated on train,
    valid and (when present) test; the summary exposes them as `curve`."""
    from types import SimpleNamespace

    from horos.api import loop as loop_mod
    from horos.errors import ProjectError

    calls = []

    def fake_evaluate_run(project, run_id, *, split="test", labels="current", device=None):
        calls.append((split, labels))
        if split == "test":
            raise ProjectError(f"Run {run_id} has no 'test' split in its dataset snapshot.")
        return SimpleNamespace(map_50={"train": 0.9, "valid": 0.6}[split], map_5095=0.4)

    import horos.api.evaluate as eval_mod

    monkeypatch.setattr(eval_mod, "evaluate_run", fake_evaluate_run)
    # the background thread must not race the assertions: run it inline
    monkeypatch.setattr(loop_mod, "_evaluate_in_background",
                        lambda project, number: loop_mod.evaluate_round_splits(project, number))
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    record = _cycle(project, count=2, epochs=1)
    # the training line is scored on the run's own snapshot — the set the model
    # actually saw; the held-out lines keep the current-labels default (E6-T13)
    assert sorted(calls) == [("test", "current"), ("train", "snapshot"), ("valid", "current")]
    row = loop_history(project)[0]
    assert row.curve == {"train": 0.9, "valid": 0.6}
    assert row.evaluating is False and row.labeled_total == 26
    # the curve's x axis is what the model saw, not every label
    assert row.train_images == record.training["holdout"]["train_images"] < row.labeled_total
    assert record.metrics["eval/train/map_5095"] == 0.4
    assert "evaluation_notes" not in record.training  # a missing test split is not a failure


def test_training_score_survives_a_train_set_the_run_saw_entirely(tmp_path, monkeypatch):
    """Regression: scoring the train split against the project's current labels
    holds back every photo the run trained on (E6-T13's reshuffle guard), which
    empties the set by construction and left the learning curve with no training
    line. The train split is scored on the run's snapshot instead."""
    from types import SimpleNamespace

    from horos.api import loop as loop_mod
    from horos.errors import ProjectError

    def fake_evaluate_run(project, run_id, *, split="test", labels="current", device=None):
        if split == "train" and labels == "current":
            raise ProjectError(
                "The project's 'train' set has no labeled photo this run did not "
                "train on, so there is nothing to score."
            )
        return SimpleNamespace(map_50={"train": 0.9, "valid": 0.6, "test": 0.5}[split],
                               map_5095=0.4)

    import horos.api.evaluate as eval_mod

    monkeypatch.setattr(eval_mod, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(loop_mod, "_evaluate_in_background",
                        lambda project, number: loop_mod.evaluate_round_splits(project, number))
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    record = _cycle(project, count=2, epochs=1)
    assert "evaluation_notes" not in record.training
    assert loop_history(project)[0].curve == {"train": 0.9, "valid": 0.6, "test": 0.5}
