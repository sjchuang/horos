"""E10-T20: photo groups for the Dataset page — k-means over the embedding
store, with the counts a user needs before skipping a whole group."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeEmbedder

from horos.api.clusters import cluster_images
from horos.api.embeddings import embedding_events
from horos.api.loop import skip_images
from horos.api.project import create_project
from horos.core.dataset import Annotation, Category
from horos.errors import ProjectError


def _project(tmp_path, *, embed=True):
    """12 images: ids 1-4 red, 5-8 green, 9-12 blue."""
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box")])
    for n, colour in enumerate(["red"] * 4 + ["green"] * 4 + ["blue"] * 4, start=1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64, 48, COLOURS[colour])
        project.add_image(path, width=64, height=48)
    if embed:
        list(embedding_events(project, "fake-embedder", backend=FakeEmbedder()))
    return project


def test_groups_follow_the_embedding_and_report_labels_and_skips(tmp_path):
    project = _project(tmp_path)
    project.save_annotations(
        1, [Annotation(id=1, image_id=1, category_id=1, bbox=(1, 1, 5, 5))], expected_version=0
    )
    skip_images(project, [9, 10], note="blank")

    result = cluster_images(project, k=3, model="fake-embedder", samples=2)
    assert result.k == 3 and result.requested_k == 3
    assert result.total_images == 12 and result.embedded == 12
    groups = {frozenset(c.image_ids) for c in result.clusters}
    assert groups == {frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8}), frozenset({9, 10, 11, 12})}
    by_first = {min(c.image_ids): c for c in result.clusters}
    assert by_first[1].labeled == 1 and by_first[1].labeled_ids == [1]
    assert by_first[1].skipped == 0 and by_first[5].labeled_ids == []
    assert by_first[9].labeled == 0 and by_first[9].skipped == 2
    assert all(len(c.samples) == 2 for c in result.clusters)
    assert all(s.id in c.image_ids for c in result.clusters for s in c.samples)
    assert all(c.cohesion > 0.99 for c in result.clusters)


def test_automatic_k_and_stable_seed(tmp_path):
    project = _project(tmp_path)
    auto = cluster_images(project, model="fake-embedder")
    assert auto.requested_k is None and auto.k == 2  # sqrt(12 / 2) rounds to 2
    again = cluster_images(project, model="fake-embedder")
    assert [c.image_ids for c in auto.clusters] == [c.image_ids for c in again.clusters]


def test_photos_without_an_embedding_are_left_out_and_counted(tmp_path):
    project = _project(tmp_path)
    extra = make_image(tmp_path / "src" / "late.png", 64, 48, COLOURS["grey"])
    late = project.add_image(extra, width=64, height=48)
    result = cluster_images(project, k=3, model="fake-embedder")
    assert result.total_images == 13 and result.embedded == 12
    assert all(late.id not in c.image_ids for c in result.clusters)


def test_without_any_embedding_the_error_says_what_to_run(tmp_path):
    project = _project(tmp_path, embed=False)
    with pytest.raises(ProjectError, match="run the embedding job"):
        cluster_images(project, model="fake-embedder")


def test_parameters_are_validated(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(ProjectError, match="k must be within"):
        cluster_images(project, k=0, model="fake-embedder")
    with pytest.raises(ProjectError, match="samples must be"):
        cluster_images(project, samples=0, model="fake-embedder")
