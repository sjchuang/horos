"""E10-T20: spherical k-means over embeddings — deterministic, cosine, largest group first."""

import numpy as np
import pytest

from horos.core.clustering import auto_k, kmeans
from horos.errors import ProjectError


def _three_colours(rng):
    reds = rng.normal([1, 0, 0], 0.05, size=(6, 3))
    greens = rng.normal([0, 1, 0], 0.05, size=(4, 3))
    blues = rng.normal([0, 0, 1], 0.05, size=(2, 3))
    return np.vstack([reds, greens, blues])


def test_separates_three_tight_colour_groups_largest_first():
    x = _three_colours(np.random.default_rng(1))
    clusters = kmeans(x, 3, seed=0)
    assert [len(c.members) for c in clusters] == [6, 4, 2]
    assert sorted(clusters[0].members) == list(range(0, 6))
    assert sorted(clusters[1].members) == list(range(6, 10))
    assert sorted(clusters[2].members) == list(range(10, 12))
    assert all(c.cohesion > 0.99 for c in clusters)
    # members are ordered closest-to-centre first; the medoid is the first
    assert clusters[0].medoid == clusters[0].members[0]


def test_same_seed_same_groups():
    x = _three_colours(np.random.default_rng(2))
    assert kmeans(x, 3, seed=7) == kmeans(x, 3, seed=7)


def test_scale_does_not_matter_only_direction():
    x = _three_colours(np.random.default_rng(3))
    scaled = x * np.linspace(1, 50, len(x))[:, None]
    assert [sorted(c.members) for c in kmeans(scaled, 3)] == [
        sorted(c.members) for c in kmeans(x, 3)
    ]


def test_k_is_clipped_to_the_row_count_and_empty_input_is_empty():
    x = np.eye(3)
    assert len(kmeans(x, 10)) == 3
    assert kmeans(np.zeros((0, 3)), 2) == []
    with pytest.raises(ProjectError, match="k must be at least 1"):
        kmeans(x, 0)


def test_auto_k_is_sqrt_n_over_two_within_bounds():
    assert auto_k(1) == 1
    assert auto_k(5) == 2
    assert auto_k(50) == 5
    assert auto_k(200) == 10
    assert auto_k(100_000) == 24
