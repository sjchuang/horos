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
    assert _headline({"loss": 1.0}) == ("loss", 1.0)
    assert _headline({}) == (None, None)
