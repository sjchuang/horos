"""E10-T16: an annotator skips a photo as unfit for training — and similar
photos with it. Skipped images stay in the project and the annotator but
leave the selection pool, the dataset snapshot, statistics and training."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeDetector, FakeEmbedder

from horos.api.dataset import dataset_stats
from horos.api.embeddings import embedding_events
from horos.api.loop import (
    loop_history,
    loop_status,
    restore_images,
    round_queue,
    select_round,
    similar_images,
    skip_images,
)
from horos.api.project import create_project
from horos.core.dataset import Annotation, Category
from horos.errors import ProjectError


def _project(tmp_path):
    """12 images: ids 1-4 red, 5-8 green, 9-12 blue (the fake embedder
    separates colours, so 'similar' = same colour)."""
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box")])
    for n, colour in enumerate(["red"] * 4 + ["green"] * 4 + ["blue"] * 4, start=1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64, 48, COLOURS[colour])
        project.add_image(path, width=64, height=48)
    list(embedding_events(project, "fake-embedder", backend=FakeEmbedder()))
    return project


def test_skip_removes_images_from_pool_snapshot_and_stats(tmp_path):
    project = _project(tmp_path)
    project.save_annotations(
        1, [Annotation(id=1, image_id=1, category_id=1, bbox=(1, 1, 5, 5))], expected_version=0
    )
    assert loop_status(project).pool_size == 11

    result = skip_images(project, [1, 2], note="blurry")
    assert result.skipped == [1, 2] and result.skipped_images == 2
    status = loop_status(project)
    assert status.skipped_images == 2 and status.pool_size == 10
    assert status.labeled_images == 0  # a skipped image's labels do not train
    records = {r.id: r for r in project.list_images()}
    assert records[1].excluded and records[1].exclude_note == "blurry"
    assert not records[3].excluded
    snapshot = project.to_dataset()
    assert {i.id for i in snapshot.images} == set(range(3, 13))
    assert snapshot.annotations == []
    assert dataset_stats(project).num_images == 10
    assert len(project.to_dataset(include_excluded=True).images) == 12

    again = skip_images(project, [2, 3])
    assert again.skipped == [3] and again.unchanged == 1


def test_restore_brings_images_back(tmp_path):
    project = _project(tmp_path)
    skip_images(project, [5, 6])
    result = restore_images(project, [5, 6, 7])
    assert result.skipped_images == 0 and result.unchanged == 1
    assert not any(r.excluded for r in project.list_images())
    assert loop_status(project).pool_size == 12
    with pytest.raises(ProjectError, match="at least one"):
        skip_images(project, [])
    with pytest.raises(ProjectError, match="No image"):
        skip_images(project, [99])


def test_similar_images_come_from_the_same_colour(tmp_path):
    project = _project(tmp_path)
    similar = similar_images(project, 1, threshold=0.99, model="fake-embedder")
    assert [s.image.id for s in similar] == [2, 3, 4]
    assert all(s.similarity >= 0.99 for s in similar)
    # a loose threshold reaches the other colours too, most similar first
    loose = similar_images(project, 1, threshold=0.0, model="fake-embedder", limit=5)
    assert len(loose) == 5 and loose[0].image.id in (2, 3, 4)
    assert loose == sorted(loose, key=lambda s: -s.similarity)
    # skipped and labeled images are not offered again
    skip_images(project, [2])
    project.save_annotations(
        3, [Annotation(id=1, image_id=3, category_id=1, bbox=(1, 1, 5, 5))], expected_version=0
    )
    left = similar_images(project, 1, threshold=0.99, model="fake-embedder")
    assert [s.image.id for s in left] == [4]
    with_labeled = similar_images(project, 1, threshold=0.99, model="fake-embedder",
                                  include_labeled=True)
    assert [s.image.id for s in with_labeled] == [3, 4]
    assert next(s for s in with_labeled if s.image.id == 3).annotated


def test_similar_needs_an_embedding_and_validates_threshold(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(ProjectError, match="no current dinov2-small embedding"):
        similar_images(project, 1)  # the default model was never run here
    with pytest.raises(ProjectError, match="threshold"):
        similar_images(project, 1, threshold=1.5, model="fake-embedder")
    with pytest.raises(ProjectError, match="No image with id 99"):
        similar_images(project, 99, model="fake-embedder")


def test_round_progress_counts_skipped_picks(tmp_path):
    project = _project(tmp_path)
    record = select_round(project, count=4, embedder=FakeEmbedder(),
                          embedding_model="fake-embedder", detector=FakeDetector(),
                          preannotate=False)
    first, second = record.image_ids[:2]
    skip_images(project, [first])
    project.save_annotations(
        second, [Annotation(id=1, image_id=second, category_id=1, bbox=(1, 1, 5, 5))],
        expected_version=0,
    )
    row = loop_history(project)[0]
    assert (row.picked, row.labeled, row.skipped) == (4, 1, 1)
    queue = round_queue(project, record.number)
    assert queue[-1].image.id == first and queue[-1].excluded  # skipped sinks to the end
    assert queue[-2].image.id == second and queue[-2].annotated
    # the skipped image cannot be picked again by a later round
    assert loop_status(project).pool_size == 12 - 4  # picks reserved; skipped not in pool
