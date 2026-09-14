"""Media inference for the evaluate gallery (E6-T2 + E6-S1/S2).

A "media item" is one uploaded photo, GIF, or video run through a trained
run's model. Frames are extracted server-side (imageio + imageio-ffmpeg,
confirmed E6 decision), inference runs per frame at a low threshold, and
everything is persisted under `runs/<run_id>/eval/media/<media_id>/` so the
gallery survives reloads and the confidence slider filters instantly without
re-running the model.

Frame decoding is capped: long videos are stride-sampled down to at most
`MAX_FRAMES` frames — the gallery is for eyeballing predictions, not for
exhaustive per-frame analytics.
"""

from __future__ import annotations

import logging
import math
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from horos.api import jobs
from horos.api.evaluate import _load_run_backend
from horos.api.manifest import capability
from horos.api.train import _run_dir
from horos.backends.base import (
    ProgressUpdated,
    RunCompleted,
    RunFailed,
    RunStarted,
    WarningRaised,
)
from horos.core.fsutil import atomic_write_text, rmtree_retry
from horos.core.project import Project
from horos.errors import ProjectError

logger = logging.getLogger(__name__)

__all__ = [
    "MediaFrame",
    "MediaItem",
    "media_inference_events",
    "start_media_inference",
    "list_media",
    "get_media",
    "delete_media",
]

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

#: hard cap on extracted frames per media item; longer inputs are
#: stride-sampled down to this
MAX_FRAMES = 120
#: inference floor — the UI's confidence slider filters above this instantly
THRESHOLD_FLOOR = 0.05

_MEDIA_JSON = "media.json"


class MediaFrame(BaseModel):
    index: int
    file_name: str  # relative to the media directory (e.g. "frames/00003.jpg")
    width: int
    height: int
    #: PredictedInstance dumps: bbox (COCO xywh), score, category_id/-name
    instances: list[dict[str, Any]] = Field(default_factory=list)


class MediaItem(BaseModel):
    media_id: str
    run_id: str
    source_name: str
    kind: Literal["image", "video"]
    state: Literal["running", "completed", "failed"] = "running"
    created_at: str = ""
    threshold_floor: float = THRESHOLD_FLOOR
    num_frames: int = 0
    error: str | None = None
    frames: list[MediaFrame] = Field(default_factory=list)


def _media_root(project: Project, run_id: str) -> Path:
    return _run_dir(project, run_id) / "eval" / "media"


def media_dir(project: Project, run_id: str, media_id: str) -> Path:
    path = _media_root(project, run_id) / media_id
    if not (path / _MEDIA_JSON).is_file():
        raise ProjectError(f"Run {run_id} has no media item '{media_id}'.")
    return path


def _write_item(item_dir: Path, item: MediaItem) -> None:
    atomic_write_text(item_dir / _MEDIA_JSON, item.model_dump_json(indent=2))


def _iter_frames(source: Path):
    """(index, ndarray) frames of an image, GIF, or video, stride-sampled to
    at most MAX_FRAMES. Returns (iterator, expected_count_or_None)."""
    import imageio.v3 as iio  # lazy: only media inference needs it

    if source.suffix.lower() in IMAGE_SUFFIXES:
        return iter([(0, iio.imread(source))]), 1

    total = None
    try:
        n_images = iio.improps(source).n_images
        if n_images and math.isfinite(n_images):
            total = int(n_images)
    except Exception:  # noqa: BLE001 — metadata is best effort per container
        total = None
    stride = max(1, math.ceil(total / MAX_FRAMES)) if total else 1

    def frames():
        kept = 0
        for index, frame in enumerate(iio.imiter(source)):
            if index % stride:
                continue
            yield index, frame
            kept += 1
            if kept >= MAX_FRAMES:
                return

    expected = min(math.ceil(total / stride), MAX_FRAMES) if total else None
    return frames(), expected


def media_inference_events(
    project: Project,
    run_id: str,
    source: Path,
    *,
    media_id: str,
    source_name: str | None = None,
    device: str | None = None,
    cancel: threading.Event | None = None,
) -> Any:
    """R4 stream: extract frames, infer each one, persist the media item."""
    backend, _ = _load_run_backend(project, run_id, device)
    source = Path(source)
    if not source.is_file():
        raise ProjectError(f"No such media file: {source}")
    suffix = source.suffix.lower()
    if suffix not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
        raise ProjectError(
            f"Unsupported media type '{suffix}'. Supported: images "
            f"({', '.join(sorted(IMAGE_SUFFIXES))}) and videos/GIFs "
            f"({', '.join(sorted(VIDEO_SUFFIXES))})."
        )
    kind = "video" if suffix in VIDEO_SUFFIXES else "image"
    name = source_name or source.name

    item_dir = _media_root(project, run_id) / media_id
    frames_dir = item_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # keep our own copy BEFORE returning the stream: the caller (a web route
    # holding a temp upload) may delete its file as soon as this returns
    stored = item_dir / f"source{suffix}"
    shutil.copyfile(source, stored)
    source = stored
    item = MediaItem(
        media_id=media_id,
        run_id=run_id,
        source_name=name,
        kind=kind,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_item(item_dir, item)

    def stream():
        from PIL import Image

        yield RunStarted(run_id=run_id, config={"media_id": media_id, "source": name})
        try:
            frames, expected = _iter_frames(source)
            for saved, (index, array) in enumerate(frames):
                if cancel is not None and cancel.is_set():
                    break
                image = Image.fromarray(array).convert("RGB")
                frame_name = f"frames/{saved:05d}.jpg"
                image.save(item_dir / frame_name, quality=88)
                from horos.api.labels import resolve_prediction_names

                prediction = resolve_prediction_names(project, backend.infer_one(
                    item_dir / frame_name, threshold=THRESHOLD_FLOOR
                ))
                item.frames.append(
                    MediaFrame(
                        index=index,
                        file_name=frame_name,
                        width=image.width,
                        height=image.height,
                        instances=[i.model_dump() for i in prediction.instances],
                    )
                )
                yield ProgressUpdated(
                    run_id=run_id,
                    current=saved + 1,
                    total=expected,
                    phase=f"inference {name}",
                )
            if kind == "video" and len(item.frames) == MAX_FRAMES:
                yield WarningRaised(
                    message=f"{name}: sampled down to {MAX_FRAMES} frames "
                    f"(the gallery caps extraction; use the Python API for "
                    f"exhaustive per-frame inference)."
                )
            if not item.frames:
                raise ProjectError(f"Could not decode any frame from {name}.")
        except Exception as exc:  # noqa: BLE001 — R4: the stream reports itself
            logger.exception("media inference %s failed", media_id)
            item.state = "failed"
            item.error = str(exc)
            item.num_frames = len(item.frames)
            _write_item(item_dir, item)
            yield RunFailed(
                run_id=run_id,
                error_code=getattr(exc, "code", "backend_error"),
                message=str(exc),
            )
            return
        item.state = "completed"
        item.num_frames = len(item.frames)
        _write_item(item_dir, item)
        yield RunCompleted(run_id=run_id, result={"media_id": media_id,
                                                  "frames": item.num_frames})

    return stream()


@capability(
    "infer.media",
    summary="Run a trained run's model over a photo, GIF, or video (as a job)",
    web_route="/api/v1/train/runs/<run_id>/media",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos infer' covers scripted use; frame galleries are a UI concern.",
)
def start_media_inference(
    project: Project,
    run_id: str,
    source: Path | str,
    *,
    source_name: str | None = None,
    device: str | None = None,
) -> tuple[str, str]:
    """Background media inference; returns (media_id, job_id)."""
    source = Path(source)
    # validate up front so the caller gets errors synchronously, not in the job
    _load_run_backend(project, run_id, device)
    if not source.is_file():
        raise ProjectError(f"No such media file: {source}")
    media_id = uuid.uuid4().hex[:12]
    job_id = jobs.start_job(
        project,
        "media-inference",
        lambda cancel: media_inference_events(
            project,
            run_id,
            source,
            media_id=media_id,
            source_name=source_name,
            device=device,
            cancel=cancel,
        ),
    )
    return media_id, job_id


@capability(
    "infer.media_list",
    summary="List the media items predicted for a run (the evaluate gallery)",
    web_route="/api/v1/train/runs/<run_id>/media",
    web_methods=("GET",),
    cli=None,
    not_cli_because="The gallery is a UI concern.",
)
def list_media(project: Project, run_id: str) -> list[MediaItem]:
    """Newest first; frame predictions elided (fetch one item for those)."""
    root = _media_root(project, run_id)
    if not root.is_dir():
        return []
    items = []
    for item_dir in root.iterdir():
        path = item_dir / _MEDIA_JSON
        if not path.is_file():
            continue
        try:
            item = MediaItem.model_validate_json(path.read_text("utf-8"))
        except ValueError:
            continue
        item.frames = item.frames[:1]  # thumbnail only — keep the listing light
        items.append(item)
    return sorted(items, key=lambda m: m.created_at, reverse=True)


@capability(
    "infer.media_get",
    summary="Full predictions of one media item, frame by frame",
    web_route="/api/v1/train/runs/<run_id>/media/<media_id>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="The gallery is a UI concern.",
)
def get_media(project: Project, run_id: str, media_id: str) -> MediaItem:
    path = media_dir(project, run_id, media_id) / _MEDIA_JSON
    return MediaItem.model_validate_json(path.read_text("utf-8"))


@capability(
    "infer.media_delete",
    summary="Delete one media item (frames and predictions)",
    web_route="/api/v1/train/runs/<run_id>/media/<media_id>",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="The gallery is a UI concern.",
)
def delete_media(project: Project, run_id: str, media_id: str) -> bool:
    # a browser may still be fetching a frame: on Windows that handle blocks
    # the delete for an instant (R7; see horos.core.fsutil)
    rmtree_retry(media_dir(project, run_id, media_id))
    return True
