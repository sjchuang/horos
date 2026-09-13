"""E10-T5 speed: the RF-DETR backend scores many photos per forward pass.
Batched predictions must match the one-at-a-time ones — same candidates at
the same floor — or the active-learning ranking would depend on batch size.
Real model; skipped where torch is not installed (CI is torch-free)."""

from __future__ import annotations

import pytest
from helpers.data import make_image

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def backend():
    from horos.backends import get_backend

    return get_backend("rfdetr-nano")


def test_infer_many_matches_infer_one_and_batches(backend, tmp_path):
    colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (120, 120, 120), (250, 250, 30)]
    paths = [make_image(tmp_path / f"{n}.png", 96 + 8 * n, 72, c) for n, c in enumerate(colours)]
    backend.INFER_BATCH = 2  # three forward passes for five photos
    many = backend.infer_many(paths, threshold=0.05, masks=False)
    assert [p.image for p in many] == [str(p) for p in paths]
    for path, pred in zip(paths, many, strict=True):
        single = backend.infer_one(path, threshold=0.05)
        assert (pred.width, pred.height) == (single.width, single.height)
        assert len(pred.candidates) == len(single.candidates)
        for a, b in zip(pred.candidates, single.candidates, strict=True):
            assert a.category_id == b.category_id
            assert abs(a.score - b.score) < 0.02
            assert all(abs(x - y) <= 2.0 for x, y in zip(a.bbox, b.bbox, strict=True))
        assert all(c.segmentation is None for c in pred.candidates)
