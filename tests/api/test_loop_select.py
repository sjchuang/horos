"""E10-T6: round selection — the loop's core acceptance gate. With fake
backends: zero labels → a diverse batch; labels + a scorer → a PAL batch;
no embedding model → random, and the round says so. Every pick has a reason."""

from __future__ import annotations

import time
from pathlib import Path

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
    `labeled` = {image_id: class_name} confirmed boxes to add. Only labeled
    photos belong to a set, so the layout's split is applied to those alone
    (as an importer keeps a source's split for the photos it labeled)."""
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box"), Category(id=2, name="pallet")])
    for n, (colour, split) in enumerate(layout, start=1):
        # size varies a little so the fake embedder separates images within a colour
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, COLOURS[colour])
        project.add_image(path, width=64 + n, height=48,
                          split=split if n in (labeled or {}) else None)
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


def test_pool_excludes_labeled_images_whatever_their_set(tmp_path):
    """Only labeled photos belong to a set, so the pool is simply every
    unlabeled, unskipped photo — including ones a source once filed under
    valid/ or test/ (they are in no set until labeled)."""
    layout = [("red", "train"), ("green", "valid"), ("blue", "test"), ("grey", "valid"),
              ("red", "test")]
    project = _project(tmp_path, layout, labeled={1: "box", 2: "box", 3: "pallet"})
    by_id = {r.id: r for r in project.list_images()}
    assert {by_id[i].split for i in (1, 2, 3)} == {"train", "valid", "test"}  # labeled: in sets
    assert by_id[4].split is None and by_id[5].split is None  # unlabeled: in none
    record = _select(project, count=10, strategy="diversity")
    assert sorted(record.selection.image_ids) == [4, 5]  # never a labeled photo
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


def test_big_pools_are_scored_on_a_seeded_sample(tmp_path):
    """Scoring 10k photos one by one took 14 minutes; the scorer looks at a
    seeded random sample of the pool (the Scan setting), says so, and the
    round still fills — diversity tops up from the whole pool."""
    layout = [("red", "train")] * 30
    project = _project(tmp_path, layout, labeled={1: "box", 2: "box", 3: "box"})
    detector = FakeDetector()
    # the cap is a multiple of the round size: 4 photos × 2 = 8 scored
    record = _select(project, count=4, detector=detector, scan_factor=2)
    labeled_files = {"1.png", "2.png", "3.png"}  # compare by name: Windows paths
    scored_unlabeled = [p for p in detector.seen if Path(p).name not in labeled_files]
    assert len(scored_unlabeled) == 8
    assert any("scored 8 of 27 unlabeled photos (2× the round" in n for n in record.selection.notes)
    assert len(record.image_ids) == 4
    # 0 means no cap: every unlabeled photo is scored
    from horos.api.loop import close_round

    close_round(project, record.number)
    detector.seen.clear()
    record = _select(project, count=2, detector=detector, scan_factor=0)
    assert len(detector.seen) == 3 + 27  # labeled + the whole pool (round 1 stayed unlabeled)
    assert not any("scored" in n and "of" in n for n in record.selection.notes)


def test_rare_class_gets_lookalikes_of_its_few_photos(tmp_path):
    """Class balance beyond the scorer: 'pallet' has one label on a grey
    photo and the fake detector never proposes it on grey photos, so PAL
    alone would never pick a grey photo. Look-alikes of the pallet photo
    (grey pool photos, by embedding) take a share of the round, with a
    reason that says so. Balance off = the paper's picks only."""
    layout = [("red", "train")] * 20 + [("grey", "train")] * 6
    labeled = {i: "box" for i in range(1, 11)}
    labeled[23] = "pallet"  # the one grey photo with a label
    project = _project(tmp_path, layout, labeled=labeled)
    record = _select(project, count=8)
    grey_pool = {21, 22, 24, 25, 26}
    lookalikes = [p for p in record.selection.picks if "rare class 'pallet'" in p.reason]
    assert lookalikes and all(p.image_id in grey_pool for p in lookalikes)
    assert len(lookalikes) <= 2  # at most a quarter of the round
    assert any(n.startswith("class balance:") for n in record.selection.notes)
    assert len(record.image_ids) == 8
    from horos.api.loop import close_round

    close_round(project, record.number)
    plain = _select(project, count=8, balance=False)
    assert not any("rare class" in p.reason for p in plain.selection.picks)


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


def test_status_reports_the_running_selection_so_a_reload_can_reattach(tmp_path, monkeypatch):
    """The page keeps the job id only in memory; after a reload it asks the
    loop status, which names the job in flight — and nothing once it is done."""
    import threading

    from horos.api import loop as loop_api

    project = _project(tmp_path, [("red", "train")] * 5)
    monkeypatch.setattr("horos.backends.get_backend", fake_get_backend)
    gate = threading.Event()
    real = loop_api.embedding_events

    def slow(*args, **kwargs):
        assert gate.wait(10), "test gate never opened"
        yield from real(*args, **kwargs)

    monkeypatch.setattr(loop_api, "embedding_events", slow)
    job_id = start_round_job(project, count=2, embedding_model="fake-embedder")
    try:
        deadline = time.monotonic() + 10  # the job thread opens the round, then waits at the gate
        while loop_status(project).current is None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        status = loop_status(project)
        assert status.current.state == "selecting"
        assert status.job is not None
        assert (status.job.job_id, status.job.kind) == (job_id, "loop-select")
        assert status.rounds and status.rounds[0].state == "selecting"
    finally:
        gate.set()  # never leave the job registry holding a stuck "running" job
    deadline = time.monotonic() + 10
    while job_status(project, job_id).state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    status = loop_status(project)
    assert status.job is None and status.current.state == "labeling"


def test_interrupted_selection_is_closed_by_hand_and_leaves_no_history(tmp_path):
    """A server restart mid-pick strands a round in 'selecting' with no job;
    the page offers Start over, which closes it, and an empty round is not a
    row in the history."""
    from horos.core.rounds import create_round

    project = _project(tmp_path, [("red", "train")] * 5)
    create_round(project, labeled_before=0)
    status = loop_status(project)
    assert status.current.state == "selecting" and status.job is None
    with pytest.raises(ProjectError, match="still 'selecting'"):
        start_round_job(project, count=1)
    close_round(project, 1)
    status = loop_status(project)
    assert status.current is None and status.rounds == []
    record = select_round(project, count=2, embedder=FakeEmbedder(), detector=FakeDetector(),
                          embedding_model="fake-embedder", preannotate=False)
    assert record.number == 2 and [r.number for r in loop_status(project).rounds] == [2]
