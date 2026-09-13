"""E10-T4: k-center greedy diversity selection and round sizing."""

import numpy as np
import pytest

from horos.core.selection import (
    kcenter_greedy,
    normalize,
    random_picks,
    resolve_count,
)
from horos.errors import ProjectError


def _clusters(rng, centres, per_cluster=10, noise=0.05):
    """Points around a few unit-vector centres; returns (features, cluster_id)."""
    feats, ids = [], []
    for cid, centre in enumerate(centres):
        for _ in range(per_cluster):
            feats.append(np.asarray(centre) + rng.normal(0, noise, len(centre)))
            ids.append(cid)
    return np.asarray(feats, dtype=np.float32), np.asarray(ids)


CENTRES = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0)]


def test_cold_start_covers_every_cluster_before_repeating_one():
    rng = np.random.default_rng(0)
    feats, ids = _clusters(rng, CENTRES)
    picks = kcenter_greedy(feats, 4)
    assert len(picks) == 4
    assert sorted(ids[[p.index for p in picks]]) == [0, 1, 2, 3]
    assert "most typical" in picks[0].reason
    assert all("farthest" in p.reason for p in picks[1:])
    assert picks[1].score >= picks[2].score >= picks[3].score  # farthest-first


def test_labeled_images_steer_picks_away_from_their_cluster():
    rng = np.random.default_rng(1)
    feats, ids = _clusters(rng, CENTRES)
    covered = feats[ids == 0]  # cluster 0 is already labeled
    picks = kcenter_greedy(feats, 3, covered=covered)
    picked_clusters = sorted(ids[[p.index for p in picks]])
    assert picked_clusters == [1, 2, 3]
    assert all("labeled images" in p.reason for p in picks)
    assert all(p.score > 0.5 for p in picks)  # far from anything covered


def test_picks_are_distinct_and_capped_by_the_pool():
    feats = np.eye(5, dtype=np.float32)
    picks = kcenter_greedy(feats, 50)
    assert len(picks) == 5
    assert len({p.index for p in picks}) == 5
    assert kcenter_greedy(feats, 0) == []
    assert kcenter_greedy(np.zeros((0, 5), dtype=np.float32), 3) == []


def test_selection_is_deterministic():
    rng = np.random.default_rng(2)
    feats, _ = _clusters(rng, CENTRES)
    a = [p.index for p in kcenter_greedy(feats, 6)]
    b = [p.index for p in kcenter_greedy(feats, 6)]
    assert a == b


def test_normalize_handles_zero_rows_and_rejects_bad_shapes():
    arr = normalize(np.asarray([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(arr[0], [0.6, 0.8]) and np.allclose(arr[1], [0.0, 0.0])
    with pytest.raises(ProjectError, match="2-D"):
        normalize(np.zeros(3))


def test_random_fallback_says_so():
    picks = random_picks(10, 4, seed=3)
    assert len({p.index for p in picks}) == 4
    assert all("random pick" in p.reason for p in picks)
    assert random_picks(0, 4) == []


def test_resolve_count_fixed_percent_and_limits():
    assert resolve_count(100) == 20  # the default round size
    assert resolve_count(100, count=15) == 15
    assert resolve_count(100, percent=10) == 10
    assert resolve_count(7, percent=10) == 1  # ceil, never zero
    assert resolve_count(5, count=50) == 5  # capped by the pool
    with pytest.raises(ProjectError, match="not both"):
        resolve_count(10, count=1, percent=1)
    with pytest.raises(ProjectError, match="No unlabeled"):
        resolve_count(0)
    with pytest.raises(ProjectError, match="percent"):
        resolve_count(10, percent=150)
    with pytest.raises(ProjectError, match="positive"):
        resolve_count(10, count=0)
