"""E10-T3: the per-project embedding store — incremental, invalidated by
file changes, atomic, and reported through R4 events."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from helpers.data import make_image, write_sample_coco_dir
from helpers.fake_backend import FakeEmbedder

from horos.api.dataset import import_dataset
from horos.api.embeddings import (
    embedding_events,
    embedding_status,
    load_embeddings,
    start_embedding_job,
)
from horos.api.jobs import job_status
from horos.api.project import create_project
from horos.backends.base import ImageEmbedder
from horos.errors import BackendError, ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _run(project, backend, **kw):
    events = list(embedding_events(project, "fake-embedder", backend=backend, **kw))
    assert events[0].type == "started"
    assert events[-1].type in ("completed", "failed"), events[-1]
    return events


def test_fresh_project_reports_everything_missing(project):
    status = embedding_status(project, "fake-embedder")
    assert (status.total_images, status.embedded, status.stale, status.missing) == (3, 0, 0, 3)
    assert not status.complete
    assert load_embeddings(project, [1, 2], "fake-embedder") is None


def test_first_run_embeds_all_and_second_run_embeds_nothing(project):
    fake = FakeEmbedder()
    events = _run(project, fake, batch=2)
    assert events[0].total == 3
    progress = [e for e in events if e.type == "progress"]
    assert [p.current for p in progress] == [2, 3]
    assert events[-1].result["embedded"] == 3 and events[-1].result["cancelled"] is False
    assert sum(len(c) for c in fake.calls) == 3

    status = embedding_status(project, "fake-embedder")
    assert status.complete and status.embedded == 3 and status.dim == fake.embedding_dim
    again = _run(project, fake)
    assert again[-1].result["embedded"] == 0
    assert sum(len(c) for c in fake.calls) == 3  # the encoder was not called again


def test_changed_image_is_re_embedded_alone(project):
    fake = FakeEmbedder()
    _run(project, fake)
    before = load_embeddings(project, [1, 2, 3], "fake-embedder")
    record = project.get_image(2)
    path = project.image_path(record)
    make_image(path, record.width, record.height, color=(1, 250, 1))
    future = time.time() + 5
    os.utime(path, (future, future))

    assert embedding_status(project, "fake-embedder").stale == 1
    assert load_embeddings(project, [2], "fake-embedder") is None  # stale = not usable
    events = _run(project, fake)
    assert events[-1].result["embedded"] == 1
    assert fake.calls[-1] == [str(path)]
    after = load_embeddings(project, [1, 2, 3], "fake-embedder")
    assert np.allclose(before[0], after[0]) and np.allclose(before[2], after[2])
    assert not np.allclose(before[1], after[1])


def test_deleted_image_rows_are_dropped_on_the_next_write(project):
    fake = FakeEmbedder()
    _run(project, fake)
    project.remove_images([3])
    events = _run(project, fake)
    assert events[-1].result["images_with_embeddings"] == 2
    assert embedding_status(project, "fake-embedder").total_images == 2
    assert load_embeddings(project, [1, 2], "fake-embedder").shape == (2, fake.embedding_dim)
    assert load_embeddings(project, [3], "fake-embedder") is None


def test_load_returns_rows_in_the_requested_order(project):
    fake = FakeEmbedder()
    _run(project, fake)
    forward = load_embeddings(project, [1, 3], "fake-embedder")
    backward = load_embeddings(project, [3, 1], "fake-embedder")
    assert np.allclose(forward[0], backward[1]) and np.allclose(forward[1], backward[0])
    assert load_embeddings(project, [], "fake-embedder").shape == (0, fake.embedding_dim)


def test_cancel_keeps_the_batches_already_embedded(project):
    from threading import Event

    fake = FakeEmbedder()
    cancel = Event()
    stream = embedding_events(project, "fake-embedder", backend=fake, cancel=cancel, batch=1)
    assert next(stream).type == "started"
    assert next(stream).type == "progress"  # first image done
    cancel.set()
    rest = list(stream)
    assert rest[-1].type == "completed" and rest[-1].result["cancelled"] is True
    assert rest[-1].result["embedded"] == 1
    status = embedding_status(project, "fake-embedder")
    assert (status.embedded, status.missing) == (1, 2)


class _Broken(ImageEmbedder):
    family = "broken"

    def __init__(self):
        super().__init__(None)

    @property
    def embedding_dim(self):
        return 3

    def embed_batch(self, images):
        raise BackendError("encoder exploded", backend="broken")


class _WrongCount(_Broken):
    def embed_batch(self, images):
        return [[1.0, 0.0, 0.0]] * (len(images) + 1)


def test_backend_failures_end_the_stream_with_failed(project):
    events = list(embedding_events(project, "broken", backend=_Broken()))
    assert events[-1].type == "failed" and "exploded" in events[-1].message
    assert embedding_status(project, "broken").embedded == 0  # nothing half-written
    events = list(embedding_events(project, "wrong", backend=_WrongCount()))
    assert events[-1].type == "failed" and "vectors for" in events[-1].message


def test_corrupt_store_is_explicit(project):
    fake = FakeEmbedder()
    _run(project, fake)
    (project.root / "embeddings" / "fake-embedder" / "index.json").write_text("{x", "utf-8")
    with pytest.raises(ProjectError, match="Corrupt embedding store"):
        embedding_status(project, "fake-embedder")


def test_background_job_runs_the_stream(project, monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr("horos.backends.get_backend", lambda key, **kw: fake)
    job_id = start_embedding_job(project, "fake-embedder")
    deadline = time.monotonic() + 10
    while job_status(project, job_id).state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    status = job_status(project, job_id)
    assert status.state == "completed"
    assert embedding_status(project, "fake-embedder").complete
