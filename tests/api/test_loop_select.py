"""E10-T6: round selection — the loop's core acceptance gate. With fake
backends: zero labels → a diverse batch; labels + a scorer → a PAL batch;
no embedding model → random, and the round says so. Every pick has a reason."""

from __future__ import annotations

import time

import pytest
from helpers.data import make_image
from helpers.fake_backend import (
    COLOURS,
    FakeDetector,
    FakeEmbedder,
    dominant_colour,
    fake_get_backend,
)

from horos.api.jobs import job_status
from horos.api.loop import (
    close_round,
    get_round,
    loop_status,
    select_round,
    select_round_events,
    start_round_job,
)
from horos.api.project import create_project
from horos.core.dataset import Annotation, Category
from horos.core.rounds import load_round
from horos.errors import BackendError, ProjectError


def _project(tmp_path, layout: list[tuple[str, str]], *, labeled: dict[int, str] | None = None):
    """`layout` = [(colour, split), ...] in image-id order (ids start at 1);
    `labeled` = {image_id: class_name} confirmed boxes to add."""
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box"), Category(id=2, name="pallet")])
    for n, (colour, split) in enumerate(layout, start=1):
        # size varies a little so the fake embedder separates images within a colour
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, COLOURS[colour])
        project.add_image(path, width=64 + n, height=48, split=split)
    for image_id, name in (labeled or {}).items():
        cat = next(c for c in project.categories if c.name == name)
        project.save_annotations(
            image_id,
            [Annotation(id=1, image_id=image_id, category_id=cat.id, bbox=(10, 10, 30, 20))],
            expected_version=0,
        )
    return project


def _select(project, **kw):
    kw.setdefault("embedder", FakeEmbedder())
    kw.setdefault("embedding_model", "fake-embedder")
    kw.setdefault("detector", FakeDetector())  # never load a real scorer in tests
    return select_round(project, **kw)


# ------------------------------------------------------------- cold start


def test_cold_start_uses_diversity_and_spreads_over_colours(tmp_path):
    layout = [("red", "train")] * 4 + [("green", "train")] * 4 + [("blue", "train")] * 4
    project = _project(tmp_path, layout)
    assert loop_status(project).next_strategy == "diversity"

    record = _select(project, count=3)
    assert record.number == 1 and record.state == "labeling"
    sel = record.selection
    assert sel.strategy == "diversity" and sel.requested == 3 and sel.pool_size == 12
    assert sel.embedding_model == "fake-embedder"
    assert sel.scorer == "fake-detector"  # only used to pre-annotate the picks (E10-T7)
    assert len(set(sel.image_ids)) == 3
    colours = {dominant_colour(project.image_path(project.get_image(i))) for i in sel.image_ids}
    assert colours == {"red", "green", "blue"}
    assert all(p.reason for p in sel.picks)
    assert any("no labeled images yet" in n for n in sel.notes)

    status = loop_status(project)
    assert status.current.number == 1
    assert status.pool_size == 9  # the open round's picks are reserved
    assert status.rounds[0].picked == 3 and status.rounds[0].labeled == 0


def test_pool_excludes_labeled_validation_and_test_images(tmp_path):
    layout = [("red", "train"), ("green", "valid"), ("blue", "test"), ("grey", "train"),
              ("red", "train")]
    project = _project(tmp_path, layout, labeled={1: "box"})
    record = _select(project, count=10, strategy="diversity")
    assert sorted(record.selection.image_ids) == [4, 5]  # never 1 (labeled), 2, 3
    assert record.selection.requested == 2  # clamped to the pool


def test_percent_is_resolved_and_recorded(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 10)
    record = _select(project, percent=30)
    assert record.selection.requested == 3 and record.selection.requested_percent == 30


# ---------------------------------------------------------------- PAL


def test_labels_and_a_scorer_give_a_pal_round(tmp_path):
    layout = (
        [("red", "train")] * 4 + [("green", "train")] * 4 + [("blue", "train")] * 3
        + [("grey", "train")] * 3
    )
    labeled = {1: "box", 2: "box", 9: "pallet"}
    project = _project(tmp_path, layout, labeled=labeled)
    assert loop_status(project).next_strategy == "pal"

    detector = FakeDetector()
    record = _select(project, count=4, detector=detector, detector_label="fake-run")
    sel = record.selection
    assert sel.strategy == "pal" and sel.scorer == "fake-run"
    assert len(set(sel.image_ids)) == 4
    # labeled images were scored to fit the classifiers, unlabeled ones to rank
    assert len(detector.seen) == len(labeled) + (len(layout) - len(labeled))
    pal_picks = [p for p in sel.picks if p.reason.startswith("PAL for class")]
    assert pal_picks, sel.picks
    # the fence-sitting green images outrank the confident red ones
    kinds = [dominant_colour(project.image_path(project.get_image(p.image_id))) for p in pal_picks]
    assert "green" in kinds and "red" not in kinds[: len(pal_picks) // 2 + 1]
    assert any("PAL class budgets" in n for n in sel.notes)
    assert all(i not in sel.image_ids for i in labeled)


def test_pal_top_up_comes_from_diversity_when_the_scorer_sees_nothing(tmp_path):
    # only one unlabeled image has a detection; the rest are grey
    layout = [("red", "train"), ("green", "train")] + [("grey", "train")] * 6
    project = _project(tmp_path, layout, labeled={1: "box"})
    record = _select(project, count=4, detector=FakeDetector())
    sel = record.selection
    assert sel.strategy == "pal"
    assert len(set(sel.image_ids)) == 4
    assert sum(p.reason.startswith("PAL") for p in sel.picks) == 1
    assert sum("farthest" in p.reason or "most typical" in p.reason for p in sel.picks) == 3
    assert any("added by diversity" in n for n in sel.notes)


def test_pal_without_labels_is_refused_explicitly(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 3)
    with pytest.raises(ProjectError, match="PAL needs labeled images"):
        _select(project, strategy="pal", detector=FakeDetector())
    assert loop_status(project).current is None  # nothing half-opened


# ------------------------------------------------------------ fallbacks


class _NoEmbedder(FakeEmbedder):
    def embed_batch(self, images):
        raise BackendError("DINOv2 weights not available", backend="dinov2")


def test_no_embedding_model_falls_back_to_random_and_says_so(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 6)
    record = _select(project, count=2, embedder=_NoEmbedder())
    sel = record.selection
    assert sel.strategy == "random" and sel.embedding_model is None
    assert len(set(sel.image_ids)) == 2
    assert all("random pick" in p.reason for p in sel.picks)
    assert any("unavailable" in n for n in sel.notes)
    assert any("fell back to random" in n for n in sel.notes)


def test_explicit_random_strategy(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 6)
    record = _select(project, count=2, strategy="random", seed=1)
    assert record.selection.strategy == "random"


# ------------------------------------------------------------- lifecycle


def test_only_one_open_round_and_close_reopens_the_pool(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 6)
    first = _select(project, count=2)
    with pytest.raises(ProjectError, match="still 'labeling'"):
        _select(project, count=2)
    closed = close_round(project, first.number)
    assert closed.state == "closed" and closed.closed_at
    assert close_round(project, first.number).state == "closed"  # idempotent
    second = _select(project, count=2)
    assert second.number == 2
    assert get_round(project, 1).state == "closed"
    with pytest.raises(ProjectError, match="No round 9"):
        get_round(project, 9)


def test_embedding_crash_becomes_a_random_round(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 4)

    class Boom(FakeEmbedder):
        def embed_batch(self, images):
            raise RuntimeError("disk on fire")

    events = list(select_round_events(project, count=2, embedder=Boom(),
                                      embedding_model="boom", detector=FakeDetector()))
    # the embedding stream ends with `failed`; selection treats that as "no
    # embeddings" and falls back rather than losing the round
    assert events[-1].type == "completed"
    record = load_round(project, 1)
    assert record.selection.strategy == "random"
    assert any("unavailable" in n for n in record.selection.notes)


def test_scorer_failure_closes_the_round_and_ends_with_failed(tmp_path):
    project = _project(tmp_path, [("red", "train")] * 4, labeled={1: "box"})

    class Broken(FakeDetector):
        def infer_one(self, image, *, threshold=0.5):
            raise BackendError("CUDA fell over", backend="fake-detector")

    events = list(select_round_events(project, count=2, embedder=FakeEmbedder(),
                                      embedding_model="fake-embedder", detector=Broken()))
    assert events[-1].type == "failed" and "CUDA fell over" in events[-1].message
    assert load_round(project, 1).state == "closed"  # the loop is not stuck
    assert loop_status(project).current is None
    with pytest.raises(ProjectError, match="Round selection failed"):
        _select(project, count=2, detector=Broken())


def test_cancel_closes_the_round(tmp_path):
    from threading import Event

    project = _project(tmp_path, [("red", "train")] * 4)
    cancel = Event()
    cancel.set()
    events = list(select_round_events(project, count=2, embedder=FakeEmbedder(),
                                      embedding_model="fake-embedder", cancel=cancel))
    assert events[-1].type == "completed" and events[-1].result["cancelled"] is True
    assert load_round(project, 1).state == "closed"
    assert loop_status(project).current is None


def test_background_job_selects_a_round(tmp_path, monkeypatch):
    project = _project(tmp_path, [("red", "train")] * 5)
    monkeypatch.setattr("horos.backends.get_backend", fake_get_backend)
    job_id = start_round_job(project, count=2, embedding_model="fake-embedder")
    deadline = time.monotonic() + 10
    while job_status(project, job_id).state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert job_status(project, job_id).state == "completed"
    assert load_round(project, 1).state == "labeling"
    with pytest.raises(ProjectError, match="still 'labeling'"):
        start_round_job(project, count=1)
