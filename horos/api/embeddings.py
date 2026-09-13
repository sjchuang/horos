"""Project embedding store (E10-T3): one vector per image, per embedding
model, kept on disk and updated incrementally.

Layout (R7: pathlib only):

    <root>/embeddings/<model_key>/vectors.npy   float32 matrix, one row per image
    <root>/embeddings/<model_key>/index.json    image_id → row, file stamp

Design:

- Incremental: only images with no row, or whose file changed (mtime/size),
  are run through the encoder. Rows of deleted images are dropped on the
  next write. Embedding 10 000 images once is minutes; re-embedding them
  every round would make the loop unusable.
- Atomic: the matrix is written to a temp file and replaced, the index via
  fsutil.atomic_write_text — a crash mid-write leaves the previous state.
- R4: `embedding_events` is an event stream (started → progress per batch →
  completed/failed); `start_embedding_job` runs it in the background with
  polling, like autolabel.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from threading import Event as CancelEvent
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from horos.api.jobs import start_job
from horos.api.manifest import capability
from horos.core.fsutil import atomic_write_text, replace_with_retry
from horos.core.project import Project
from horos.errors import ProjectError

if TYPE_CHECKING:
    from horos.backends.base import Event, ImageEmbedder

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingStatus",
    "embedding_status",
    "embedding_events",
    "start_embedding_job",
    "load_embeddings",
]

DEFAULT_EMBEDDING_MODEL = "dinov2-small"
EMBEDDINGS_DIR = "embeddings"
_VECTORS = "vectors.npy"
_INDEX = "index.json"
#: images handed to the encoder per progress event
BATCH = 32


class EmbeddingStatus(BaseModel):
    model: str
    total_images: int
    #: images with a current vector
    embedded: int
    #: images whose file changed since they were embedded
    stale: int
    #: images with no vector yet
    missing: int
    dim: int | None = None

    @property
    def complete(self) -> bool:
        return self.stale == 0 and self.missing == 0


class _Row(BaseModel):
    row: int
    mtime_ns: int
    size: int


class _Index(BaseModel):
    model: str
    dim: int | None = None
    rows: dict[int, _Row] = Field(default_factory=dict)


# ------------------------------------------------------------------- disk


def _dir(project: Project, model: str) -> Path:
    return project.root / EMBEDDINGS_DIR / model


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _load(project: Project, model: str) -> tuple[_Index, np.ndarray | None]:
    base = _dir(project, model)
    index_path = base / _INDEX
    if not index_path.is_file():
        return _Index(model=model), None
    try:
        index = _Index.model_validate_json(index_path.read_text(encoding="utf-8"))
        matrix = np.load(base / _VECTORS) if (base / _VECTORS).is_file() else None
    except (ValueError, OSError) as exc:
        raise ProjectError(f"Corrupt embedding store at {base}: {exc}") from exc
    highest = max((r.row for r in index.rows.values()), default=-1)
    if matrix is not None and matrix.shape[0] <= highest:
        raise ProjectError(
            f"Corrupt embedding store at {base}: index refers to rows the matrix lacks"
        )
    return index, matrix


def _save(project: Project, index: _Index, matrix: np.ndarray) -> None:
    base = _dir(project, model=index.model)
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f".{_VECTORS}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with tmp.open("wb") as fh:
            np.save(fh, matrix.astype(np.float32, copy=False))
        replace_with_retry(tmp, base / _VECTORS)
    finally:
        tmp.unlink(missing_ok=True)
    atomic_write_text(base / _INDEX, json.dumps(index.model_dump(mode="json"), indent=1))


def _classify(project: Project, index: _Index):
    """(current, stale, missing) image records against the stored index."""
    current, stale, missing = [], [], []
    for record in project.list_images():
        path = project.image_path(record)
        row = index.rows.get(record.id)
        if row is None:
            missing.append(record)
            continue
        try:
            mtime, size = _stamp(path)
        except OSError:
            missing.append(record)  # file gone; the validator reports that separately
            continue
        (current if (mtime, size) == (row.mtime_ns, row.size) else stale).append(record)
    return current, stale, missing


# -------------------------------------------------------------------- API


@capability(
    "loop.embeddings.status",
    summary="How many project images have a current embedding for a model",
    web_route="/api/v1/loop/embeddings",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' reports embedding coverage as part of its status.",
)
def embedding_status(project: Project, model: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingStatus:
    index, _ = _load(project, model)
    current, stale, missing = _classify(project, index)
    return EmbeddingStatus(
        model=model,
        total_images=len(current) + len(stale) + len(missing),
        embedded=len(current),
        stale=len(stale),
        missing=len(missing),
        dim=index.dim,
    )


def embedding_events(
    project: Project,
    model: str = DEFAULT_EMBEDDING_MODEL,
    *,
    device: str | None = None,
    backend: ImageEmbedder | None = None,
    cancel: CancelEvent | None = None,
    batch: int = BATCH,
) -> Iterator[Event]:
    """Bring the store up to date as an R4 event stream: embed every missing
    or stale image, drop rows of deleted images, write once at the end (and
    once on cancel, so partial work is kept). `backend` is injectable for
    tests; `cancel` is honoured between batches."""
    from horos.backends.base import ProgressUpdated, RunCompleted, RunFailed, RunStarted

    index, matrix = _load(project, model)
    current, stale, missing = _classify(project, index)
    todo = stale + missing
    yield RunStarted(
        total=len(todo),
        config={"model": model, "embedded": len(current), "stale": len(stale),
                "missing": len(missing)},
    )
    try:
        # compact: keep only rows of images that still exist and are current
        keep_ids = [r.id for r in current]
        rows: dict[int, np.ndarray] = {}
        if matrix is not None:
            for image_id in keep_ids:
                rows[image_id] = matrix[index.rows[image_id].row]
        stamps: dict[int, tuple[int, int]] = {
            i: (index.rows[i].mtime_ns, index.rows[i].size) for i in keep_ids
        }
        dim = index.dim

        def flush(cancelled: bool, done: int) -> Iterator[Event]:
            nonlocal dim
            ordered = sorted(rows)
            new_index = _Index(model=model, dim=dim, rows={})
            if ordered:
                mat = np.stack([rows[i] for i in ordered]).astype(np.float32)
                for n, image_id in enumerate(ordered):
                    mtime, size = stamps[image_id]
                    new_index.rows[image_id] = _Row(row=n, mtime_ns=mtime, size=size)
            else:
                mat = np.zeros((0, dim or 0), dtype=np.float32)
            _save(project, new_index, mat)
            yield RunCompleted(
                result={"cancelled": cancelled, "embedded": done, "total": len(todo),
                        "images_with_embeddings": len(ordered), "model": model}
            )

        if not todo:
            yield from flush(False, 0)
            return
        if backend is None:
            from horos.backends import get_backend

            backend = get_backend(model, device=device)  # type: ignore[assignment]
        done = 0
        for start in range(0, len(todo), batch):
            if cancel is not None and cancel.is_set():
                yield from flush(True, done)
                return
            chunk = todo[start : start + batch]
            paths = [project.image_path(r) for r in chunk]
            pre = [_stamp(p) for p in paths]  # stamp BEFORE embedding: a write during
            vectors = backend.embed_batch(paths)  # the encode shows up as stale next time
            if len(vectors) != len(chunk):
                raise ProjectError(
                    f"{model} returned {len(vectors)} vectors for {len(chunk)} images"
                )
            for record, vec, stamp in zip(chunk, vectors, pre, strict=True):
                arr = np.asarray(vec, dtype=np.float32)
                if dim is None:
                    dim = int(arr.shape[0])
                elif arr.shape[0] != dim:
                    raise ProjectError(
                        f"{model} returned a {arr.shape[0]}-d vector; the store holds {dim}-d"
                    )
                rows[record.id] = arr
                stamps[record.id] = stamp
            done += len(chunk)
            yield ProgressUpdated(
                current=done, total=len(todo), phase="embedding",
                message=f"{done} / {len(todo)} images embedded",
            )
        yield from flush(False, done)
    except Exception as exc:  # noqa: BLE001 — the stream must end with an event (R4)
        logger.exception("embedding run failed")
        yield RunFailed(error_code=getattr(exc, "code", "backend_error"), message=str(exc))


@capability(
    "loop.embeddings.start",
    summary="Start a background job embedding every image that lacks a current vector",
    web_route="/api/v1/loop/embeddings",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos loop select' embeds what it needs in the foreground.",
)
def start_embedding_job(
    project: Project, model: str = DEFAULT_EMBEDDING_MODEL, *, device: str | None = None
) -> str:
    return start_job(
        project,
        "embeddings",
        lambda cancel: embedding_events(project, model, device=device, cancel=cancel),
    )


def load_embeddings(
    project: Project, image_ids: list[int], model: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray | None:
    """Vectors for `image_ids` in that order, or None when any is missing or
    stale — callers then run `embedding_events` first, or fall back."""
    index, matrix = _load(project, model)
    if matrix is None:
        return None if image_ids else np.zeros((0, index.dim or 0), dtype=np.float32)
    records = {r.id: r for r in project.list_images()}
    picked = []
    for image_id in image_ids:
        row = index.rows.get(image_id)
        record = records.get(image_id)
        if row is None or record is None:
            return None
        try:
            if _stamp(project.image_path(record)) != (row.mtime_ns, row.size):
                return None
        except OSError:
            return None
        picked.append(matrix[row.row])
    if not picked:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32)
    return np.stack(picked)
