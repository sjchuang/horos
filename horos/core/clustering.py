"""Grouping photos by embedding (E10-T20).

Spherical k-means over L2-normalised embedding vectors: cosine geometry, the
same space the loop's diversity and skip-similar features use. Pure numpy,
deterministic under a seed (k-means++ seeding, Lloyd iterations until the
assignment settles), so the same photos give the same groups every time the
Dataset page asks.

The number of groups is the user's choice; when they do not choose,
`auto_k` takes the common sqrt(n / 2) rule of thumb, clamped to a range a
person can actually scan on one page.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from horos.core.selection import normalize
from horos.errors import ProjectError

#: bounds for the automatic group count
AUTO_K_MIN = 2
AUTO_K_MAX = 24
#: Lloyd iterations before we stop even if a few points still flip
MAX_ITER = 50


def auto_k(n: int) -> int:
    """A group count a person can scan: sqrt(n / 2), clamped to [2, 24]."""
    if n <= 1:
        return 1
    return int(min(AUTO_K_MAX, max(AUTO_K_MIN, round((n / 2) ** 0.5))))


@dataclass(frozen=True)
class Cluster:
    #: row indices (into the feature matrix) of the members, closest to the
    #: centre first
    members: list[int]
    #: mean cosine similarity of the members to the centre; 1.0 = identical
    cohesion: float

    @property
    def medoid(self) -> int:
        return self.members[0]


def kmeans(features: np.ndarray, k: int, *, seed: int = 0) -> list[Cluster]:
    """Group the rows of `features` into `k` clusters, largest first.

    Rows are L2-normalised so the distance is cosine. `k` is clipped to the
    number of rows; empty clusters that k-means occasionally leaves behind
    are dropped, so fewer than `k` groups may come back."""
    x = normalize(features)
    n = x.shape[0]
    if n == 0:
        return []
    if k < 1:
        raise ProjectError(f"k must be at least 1, got {k}")
    k = min(k, n)
    rng = np.random.default_rng(seed)

    # k-means++ seeding: each next centre is drawn with probability
    # proportional to its squared distance from the nearest centre so far
    centres = np.empty((k, x.shape[1]), dtype=np.float32)
    centres[0] = x[rng.integers(n)]
    dist = np.clip(1.0 - x @ centres[0], 0.0, 2.0)
    for i in range(1, k):
        weights = dist**2
        total = float(weights.sum())
        if total <= 0:  # every remaining point sits on a centre already
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=weights / total))
        centres[i] = x[idx]
        dist = np.minimum(dist, np.clip(1.0 - x @ centres[i], 0.0, 2.0))

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(MAX_ITER):
        sims = x @ centres.T  # (n, k) cosine similarities
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for i in range(k):
            members = x[labels == i]
            if len(members):
                centres[i] = normalize(members.mean(axis=0, keepdims=True))[0]

    sims = x @ centres.T
    clusters: list[Cluster] = []
    for i in range(k):
        rows = np.flatnonzero(labels == i)
        if not len(rows):
            continue
        order = rows[np.argsort(-sims[rows, i], kind="stable")]
        clusters.append(
            Cluster(
                members=[int(r) for r in order],
                cohesion=round(float(sims[rows, i].mean()), 4),
            )
        )
    clusters.sort(key=lambda c: (-len(c.members), c.members[0]))
    return clusters
