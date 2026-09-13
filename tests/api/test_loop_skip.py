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
    assign_round,
    loop_history,
    loop_status,
    refill_round,
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
    skip_images(project, [first], embedder=FakeEmbedder(), detector=FakeDetector())
    project.save_annotations(
        second, [Annotation(id=1, image_id=second, category_id=1, bbox=(1, 1, 5, 5))],
        expected_version=0,
    )
    row = loop_history(project)[0]
    assert (row.picked, row.labeled, row.skipped) == (5, 1, 1)  # one replacement added
    queue = round_queue(project, record.number)
    assert queue[-1].image.id == first and queue[-1].excluded  # skipped sinks to the end
    assert queue[0].image.id == second and queue[0].annotated  # labeled keeps its place
    replacement = [p.image_id for p in loop_status(project).current.selection.picks][-1]
    assert queue[-2].image.id == replacement  # the replacement is appended, not inserted
    # the skipped image cannot be picked again by a later round
    assert loop_status(project).pool_size == 12 - 5  # picks reserved; skipped not in pool


def _fakes():
    return dict(embedder=FakeEmbedder(), embedding_model="fake-embedder", detector=FakeDetector())


def test_skipping_round_picks_refills_the_round_to_its_size(tmp_path):
    project = _project(tmp_path)
    record = select_round(project, count=4, preannotate=False, **_fakes())
    assign_round(project, record.number, ["ann", "bob"])
    victims = record.image_ids[:2]  # owned by ann and bob
    result = skip_images(project, victims, embedder=FakeEmbedder(), detector=FakeDetector())
    assert result.skipped == victims and len(result.replacements) == 2
    after = loop_history(project)[0]
    assert (after.picked, after.skipped) == (6, 2)
    round_now = loop_status(project).current
    active = [p for p in round_now.selection.picks if p.image_id in result.replacements]
    assert all(p.reason.startswith("replacement for a skipped photo") for p in active)
    assert sorted(p.assigned_to for p in active) == ["ann", "bob"]  # inherited
    assert not set(result.replacements) & set(victims)
    assert any("added to replace" in n for n in round_now.selection.notes)
    # replacements got the round's pre-labels (the fake detector sees red/green/blue)
    assert round_now.preannotation["images"] == 2
    # the queue offers the replacements and sinks the skipped ones
    queue = round_queue(project, record.number)
    assert [i.image.id for i in queue if i.excluded] == victims
    assert loop_status(project).pool_size == 12 - 6


def test_skipping_outside_the_round_or_with_refill_off_adds_nothing(tmp_path):
    project = _project(tmp_path)
    record = select_round(project, count=3, preannotate=False, **_fakes())
    outsider = next(i for i in range(1, 13) if i not in record.image_ids)
    assert skip_images(project, [outsider]).replacements == []
    assert len(loop_status(project).current.image_ids) == 3
    result = skip_images(project, record.image_ids[:1], refill=False)
    assert result.replacements == []
    assert len(loop_status(project).current.image_ids) == 3
    # the manual retry tops it up
    refilled = refill_round(project, record.number, embedder=FakeEmbedder(),
                            detector=FakeDetector())
    assert len(refilled.image_ids) == 4


def test_refill_with_an_empty_pool_leaves_a_note(tmp_path):
    project = _project(tmp_path)
    record = select_round(project, count=12, preannotate=False, **_fakes())  # the whole pool
    result = skip_images(project, record.image_ids[:1], embedder=FakeEmbedder(),
                         detector=FakeDetector())
    assert result.replacements == []
    notes = loop_status(project).current.selection.notes
    assert any("could not be replaced" in n for n in notes)
    from horos.api.loop import close_round

    close_round(project, record.number)
    with pytest.raises(ProjectError, match="only a round being labeled"):
        refill_round(project, record.number)
