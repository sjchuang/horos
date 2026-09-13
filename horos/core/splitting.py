"""Which set a labeled photo belongs to (E1-T8, revised with E10).

Only labeled photos are members of train / valid / test. A photo enters the
project with no split; the first time it carries a confirmed annotation it is
assigned by a stable hash of its id, so:

- the sets grow with the labels in the project's ratios (default 70/10/20),
- a photo never changes set, and the test set is never trained on,
- every machine and every run of the code agrees on the assignment (R7).

Unlabeled photos have `split=None`: they are the pool the active-learning loop
picks from and they never reach a training snapshot or an export split.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator

from horos.core.dataset import Split

DEFAULT_SPLIT_SEED = 42


class SplitRatios(BaseModel):
    """Shares of the labeled photos per set; must sum to 1."""

    train: float = Field(default=0.7, ge=0.0, le=1.0)
    valid: float = Field(default=0.1, ge=0.0, le=1.0)
    test: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitRatios:
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total:g}")
        if self.train <= 0:
            raise ValueError("the train share must be above 0 — a model needs photos to learn from")
        return self


def bucket(seed: int, image_id: int) -> float:
    """A stable point in [0, 1) for a photo: sha256 of seed and id, so the
    assignment is identical on every platform and never depends on the order
    photos were labeled in."""
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_for(image_id: int, ratios: SplitRatios, seed: int = DEFAULT_SPLIT_SEED) -> Split:
    """The set a photo falls into: the lowest buckets are test, then valid,
    the rest train — the same cut the loop used before the split moved here,
    so existing projects keep their assignments."""
    b = bucket(seed, image_id)
    if b < ratios.test:
        return "test"
    if b < ratios.test + ratios.valid:
        return "valid"
    return "train"
