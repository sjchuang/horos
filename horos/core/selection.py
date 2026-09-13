"""Batch selection for the active-learning loop — cold-start diversity and
batch sizing (E10-T4). The model-based acquisition (LIUS + GUIDE from
"Portable Active Learning for Object Detection", CVPR 2026) lives in
horos/core/pal.py (E10-T5).

Pure numpy: no backend, no project I/O. The API layer (horos/api/loop.py)
brings embeddings here and stores what comes back. Every function returns
`Pick`s whose `reason` explains the score in words — the loop must be able
to answer "why this image" without re-running anything (E10-S8).

Conventions
-----------
- Features are L2-normalised row vectors; distance is cosine distance
  (1 − cosine similarity), so 0 = identical, 1 = orthogonal, 2 = opposite.
- Scores are "higher = more worth labeling". Diversity scores are distances
  to the nearest already-covered image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from horos.errors import ProjectError

DEFAULT_ROUND_SIZE = 20


@dataclass(frozen=True)
class Pick:
    index: int  # row into the caller's feature matrix / id list
    score: float
    reason: str


# ------------------------------------------------------------- batch size


def resolve_count(
    pool_size: int,
    *,
    count: int | None = None,
    percent: float | None = None,
    default: int = DEFAULT_ROUND_SIZE,
) -> int:
    """How many images the round gets. A fixed `count` is the default form;
    `percent` of the unlabeled pool is the alternative (confirmed decision).
    Never more than the pool holds, never fewer than one when the pool is
    non-empty."""
    if count is not None and percent is not None:
        raise ProjectError("Give either a count or a percent for the round, not both")
    if pool_size <= 0:
        raise ProjectError("No unlabeled images left to select from")
    if percent is not None:
        if not 0 < percent <= 100:
            raise ProjectError(f"percent must be in (0, 100], got {percent}")
        wanted = math.ceil(pool_size * percent / 100)
    else:
        wanted = default if count is None else count
        if wanted <= 0:
            raise ProjectError(f"count must be positive, got {wanted}")
    return max(1, min(int(wanted), pool_size))


# --------------------------------------------------------------- features


def normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalise rows; a zero vector stays zero instead of becoming NaN."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ProjectError(f"features must be a 2-D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) cosine distances between normalised rows."""
    return np.clip(1.0 - a @ b.T, 0.0, 2.0)


# ------------------------------------------------------------ diversity (E10-T4)


def kcenter_greedy(
    features: np.ndarray,
    k: int,
    *,
    covered: np.ndarray | None = None,
) -> list[Pick]:
    """Farthest-first traversal: each pick is the pool image farthest from
    everything already covered (the labeled set plus earlier picks), so a
    batch spreads over the feature space instead of piling onto one cluster.

    `features` are the unlabeled pool, `covered` the already-labeled images
    (may be None or empty for a cold start). The score is the pick's cosine
    distance to its nearest covered image at the time it was chosen — the
    first cold-start pick has nothing to measure against and is the image
    closest to the pool's centre, scored as the pool's mean spread.
    Deterministic: the same inputs always give the same batch.
    """
    pool = normalize(features)
    n = len(pool)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    have_cover = covered is not None and len(covered) > 0
    if have_cover:
        cover = normalize(covered)
        nearest = cosine_distance(pool, cover).min(axis=1)
    else:
        nearest = np.full(n, np.inf, dtype=np.float32)

    picks: list[Pick] = []
    chosen = np.zeros(n, dtype=bool)
    for _ in range(k):
        if not np.isfinite(nearest).any():
            # cold start: anchor on the most central image (a representative
            # sample), then let farthest-first take over
            centre = pool.mean(axis=0, keepdims=True)
            to_centre = cosine_distance(pool, centre)[:, 0]
            idx = int(np.argmin(to_centre))
            spread = float(to_centre.mean())
            reason = (
                f"first pick with no labeled images to compare against: the most "
                f"typical image of the pool (mean spread {spread:.2f})"
            )
            score = spread
        else:
            masked = np.where(chosen, -np.inf, nearest)
            idx = int(np.argmax(masked))
            score = float(nearest[idx])
            what = "labeled images and earlier picks" if have_cover else "earlier picks"
            reason = f"farthest from all {what} (nearest distance {score:.2f})"
        chosen[idx] = True
        picks.append(Pick(index=idx, score=score, reason=reason))
        # every remaining image is now at most this far from a covered one
        dist_to_new = cosine_distance(pool, pool[idx : idx + 1])[:, 0]
        nearest = np.minimum(nearest, dist_to_new)
    return picks


def random_picks(pool_size: int, k: int, *, seed: int | None = None) -> list[Pick]:
    """The fallback when no embedding model is available: uniform without
    replacement, and the reason says so — the loop never hides a downgrade."""
    if pool_size <= 0 or k <= 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool_size, size=min(k, pool_size), replace=False)
    return [
        Pick(index=int(i), score=0.0, reason="random pick (no embedding model available)")
        for i in idx
    ]
