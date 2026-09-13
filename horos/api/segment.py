"""Interactive segmentation (SAM-T2): click / box → candidate shape.

Design decisions (confirmed 2026-09-12):

- The encoder output of an image is cached (LRU, `EMBEDDING_CACHE_SIZE`
  entries) keyed by project, image, model and device, and invalidated when
  the image file changes. The first click on an image pays the encoder;
  every later click costs only the prompt decoder. `prefetch_embedding`
  lets the UI warm the cache when the tool is picked or the image opens.
- `segment_image` returns a CANDIDATE — polygon, mask box, score — and
  writes nothing. Accepting it goes through the ordinary annotation save
  (annotate.save_annotations), so versions, optimistic locking and
  multi-annotator conflicts have exactly one implementation.
- Default model: SAM 2.1 hiera-tiny (Apache 2.0). SAM v1 is selectable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.dataset import Annotation, clamp_to_image
from horos.core.project import Project
from horos.errors import ProjectError

if TYPE_CHECKING:
    from horos.backends.base import ImageEmbedding, PromptableSegmenter

logger = logging.getLogger(__name__)

__all__ = [
    "SegmentRequest",
    "SegmentCandidate",
    "PrefetchResult",
    "DEFAULT_SEGMENTER",
    "EMBEDDING_CACHE_SIZE",
    "segment_image",
    "ConvertResult",
    "boxes_to_polygons",
    "boxes_to_polygons_events",
    "start_boxes_to_polygons",
    "prefetch_embedding",
]

DEFAULT_SEGMENTER = "sam2.1-tiny"
EMBEDDING_CACHE_SIZE = 8
OUTPUT_MODES = ("polygon", "bbox")


class SegmentRequest(BaseModel):
    """One prompt from the annotator, in image pixels."""

    points: list[tuple[float, float]] = Field(default_factory=list)
    #: 1 = positive (this is the object), 0 = negative; one per point
    labels: list[int] = Field(default_factory=list)
    #: rough COCO-xywh box drawn by the user
    box: tuple[float, float, float, float] | None = None
    output: Literal["polygon", "bbox"] = "polygon"
    model: str = DEFAULT_SEGMENTER
    #: cap on the polygon's control points (None = the mask's full outline);
    #: SAM outlines can be far more detailed than a label needs
    max_points: int | None = Field(default=None, ge=3)


class SegmentCandidate(BaseModel):
    """What the user sees before accepting: never written on its own."""

    image_id: int
    model: str
    shape_type: Literal["polygon", "rectangle"]
    #: flat [x1, y1, ...] for polygons; the mask's box corners for rectangles
    points: list[list[float]] = Field(default_factory=list)
    polygon: list[float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    score: float = 0.0
    area: int = 0
    #: True when the image embedding came from the cache (no encoder run)
    embedding_cached: bool = False
    elapsed_ms: float = 0.0


class PrefetchResult(BaseModel):
    image_id: int
    model: str
    embedding_cached: bool  # already there before this call
    elapsed_ms: float = 0.0


# ------------------------------------------------------------------ cache


class EmbeddingCache:
    """LRU of image embeddings; a changed image file (mtime/size) misses."""

    def __init__(self, capacity: int = EMBEDDING_CACHE_SIZE):
        self.capacity = capacity
        self._items: OrderedDict[tuple, tuple[tuple[int, int], ImageEmbedding]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _stamp(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def peek(self, key: tuple, path: Path) -> bool:
        with self._lock:
            item = self._items.get(key)
        return item is not None and item[0] == self._stamp(path)

    def get_or_compute(self, key: tuple, path: Path, compute) -> tuple[ImageEmbedding, bool]:
        """(embedding, was_cached). `compute()` runs the encoder outside the
        lock so a slow embed never blocks lookups for other images."""
        stamp = self._stamp(path)
        with self._lock:
            item = self._items.get(key)
            if item is not None and item[0] == stamp:
                self._items.move_to_end(key)
                self.hits += 1
                return item[1], True
            self.misses += 1
        embedding = compute()
        with self._lock:
            self._items[key] = (stamp, embedding)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return embedding, False

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


_CACHE = EmbeddingCache()
_SEGMENTERS: dict[tuple[str, str | None], Any] = {}
_SEGMENTERS_LOCK = threading.Lock()


def _segmenter(model: str, device: str | None) -> PromptableSegmenter:
    key = (model, device)
    with _SEGMENTERS_LOCK:
        backend = _SEGMENTERS.get(key)
    if backend is None:
        from horos.backends import get_backend
        from horos.backends.base import PromptableSegmenter

        backend = get_backend(model, device=device)
        if not isinstance(backend, PromptableSegmenter) and not (
            hasattr(backend, "embed") and hasattr(backend, "segment")
        ):
            raise ProjectError(
                f"Model '{model}' is not an interactive segmenter — pick a SAM model "
                f"(e.g. {DEFAULT_SEGMENTER})."
            )
        with _SEGMENTERS_LOCK:
            _SEGMENTERS[key] = backend
    return backend


def _reset_segmenters() -> None:  # tests only
    with _SEGMENTERS_LOCK:
        _SEGMENTERS.clear()
    _CACHE.clear()


def _cache_key(project: Project, image_id: int, model: str, device: str | None) -> tuple:
    return (str(project.root), image_id, model, device or "auto")


def _embedding_for(
    project: Project, image_id: int, model: str, device: str | None, backend
) -> tuple[ImageEmbedding, bool, Path]:
    record = project.get_image(image_id)
    path = project.image_path(record)
    if not path.is_file():
        raise ProjectError(f"Image file for id {image_id} is missing: {path}")
    embedding, cached = _CACHE.get_or_compute(
        _cache_key(project, image_id, model, device), path, lambda: backend.embed(path)
    )
    return embedding, cached, path


# ------------------------------------------------------------------ API


@capability(
    "segment.prefetch",
    summary="Warm the image-embedding cache so the first click answers instantly",
    web_route="/api/v1/images/<int:image_id>/segment/prefetch",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Interactive annotation aid; the CLI has no canvas to click on.",
)
def prefetch_embedding(
    project: Project,
    image_id: int,
    *,
    model: str = DEFAULT_SEGMENTER,
    device: str | None = None,
    backend: PromptableSegmenter | None = None,
) -> PrefetchResult:
    backend = backend or _segmenter(model, device)
    started = time.perf_counter()
    _, cached, _ = _embedding_for(project, image_id, model, device, backend)
    return PrefetchResult(
        image_id=image_id, model=model, embedding_cached=cached,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@capability(
    "segment.interactive",
    summary="Turn clicks / a rough box on an image into a candidate polygon or box",
    web_route="/api/v1/images/<int:image_id>/segment",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Interactive annotation aid; the CLI has no canvas to click on.",
)
def segment_image(
    project: Project,
    image_id: int,
    request: SegmentRequest,
    *,
    device: str | None = None,
    backend: PromptableSegmenter | None = None,
) -> SegmentCandidate:
    """Decode one prompt against the (cached) image embedding and return the
    candidate shape. Nothing is written: accept it through save_annotations."""
    from horos.backends.base import SegmentPrompt

    try:
        prompt = SegmentPrompt(
            points=request.points, labels=request.labels, box=request.box
        ).validated()
    except ValueError as exc:
        raise ProjectError(f"Invalid segment prompt: {exc}") from exc
    if request.output not in OUTPUT_MODES:
        raise ProjectError(f"output must be one of {OUTPUT_MODES}")
    record = project.get_image(image_id)
    for x, y in prompt.points:
        if not (0 <= x <= record.width and 0 <= y <= record.height):
            raise ProjectError(
                f"Point ({x:.0f}, {y:.0f}) lies outside the {record.width}×{record.height} image"
            )
    backend = backend or _segmenter(request.model, device)
    started = time.perf_counter()
    embedding, cached, _ = _embedding_for(project, image_id, request.model, device, backend)
    result = backend.segment(embedding, prompt)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    if result.bbox is None or (request.output == "polygon" and not result.polygon):
        return SegmentCandidate(
            image_id=image_id, model=request.model, shape_type="polygon", points=[],
            polygon=None, bbox=None, score=result.score, area=0,
            embedding_cached=cached, elapsed_ms=elapsed,
        )
    x, y, w, h = result.bbox
    if request.output == "bbox":
        points = [[x, y], [x + w, y + h]]
        shape_type: Literal["polygon", "rectangle"] = "rectangle"
    else:
        flat = result.polygon or []
        if request.max_points is not None:
            from horos.backends.sam.polygonize import simplify_polygon

            flat = simplify_polygon(flat, request.max_points)
        points = [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
        shape_type = "polygon"
    return SegmentCandidate(
        image_id=image_id, model=request.model, shape_type=shape_type, points=points,
        polygon=flat if shape_type == "polygon" else result.polygon, bbox=result.bbox,
        score=result.score, area=result.area, embedding_cached=cached, elapsed_ms=elapsed,
    )


# ------------------------------------------------ boxes as prompts (SAM-T6)
#
# A box annotation is already a good SAM prompt: the user (or OWLv2) said
# "the object is in here". Rewriting boxes as polygons therefore needs no
# clicks — one embedding per image, one decoder pass per box. The annotation
# keeps its id and class; the geometry changes and, because a machine drew
# it, the annotation becomes a pending auto pre-label scored with SAM's
# predicted IoU (E10-T11). A box SAM cannot segment stays a box, unchanged
# (counted as skipped, never dropped).


class ConvertResult(BaseModel):
    image_id: int
    version: int
    converted: int
    skipped: int
    annotations: list[Annotation] = Field(default_factory=list)


def _resolve_categories(project: Project, categories) -> set[int] | None:
    """Category ids from a mixed list of ids and names; None means all."""
    if categories is None:
        return None
    by_name = {c.name: c.id for c in project.categories}
    known = {c.id for c in project.categories}
    ids: set[int] = set()
    for item in categories:
        if isinstance(item, bool):
            raise ProjectError(f"Invalid category {item!r}")
        if isinstance(item, int):
            if item not in known:
                raise ProjectError(f"Unknown category id {item} — available: {sorted(known)}")
            ids.add(item)
        elif isinstance(item, str) and item in by_name:
            ids.add(by_name[item])
        else:
            raise ProjectError(
                f"Unknown category {item!r} — available: {sorted(by_name)}"
            )
    return ids


def _select_boxes(
    annotations: list[Annotation],
    *,
    annotation_ids: set[int] | None,
    category_ids: set[int] | None,
    include_pending: bool,
) -> list[Annotation]:
    picked = []
    for a in annotations:
        if a.segmentation:
            continue  # already a polygon
        if annotation_ids is not None and a.id not in annotation_ids:
            continue
        if category_ids is not None and a.category_id not in category_ids:
            continue
        if not include_pending and a.status == "pending":
            continue
        picked.append(a)
    return picked


@capability(
    "segment.boxes_to_polygons",
    summary="Rewrite one image's box annotations as SAM polygons (each box is the prompt)",
    web_route="/api/v1/images/<int:image_id>/segment/boxes",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos boxes-to-polygons' runs the project-wide batch; one image is an "
                    "editor action.",
)
def boxes_to_polygons(
    project: Project,
    image_id: int,
    *,
    annotation_ids: list[int] | None = None,
    categories: list[int | str] | None = None,
    include_pending: bool = True,
    model: str = DEFAULT_SEGMENTER,
    max_points: int | None = None,
    device: str | None = None,
    backend: PromptableSegmenter | None = None,
    expected_version: int | None = None,
) -> ConvertResult:
    """Turn the image's box-only annotations (optionally just `annotation_ids`
    or `categories`, ids or names) into polygons: each box prompts the
    segmenter against the image's cached embedding. Converted annotations
    come back as pending auto pre-labels with SAM's predicted IoU as score —
    machine geometry is never stored as confirmed human work (E10-T11).
    Written through the ordinary versioned save; `expected_version` guards
    against a concurrent editor (E2-T8)."""
    from horos.backends.base import SegmentPrompt

    record = project.get_image(image_id)
    stored = project.load_annotations(image_id)
    category_ids = _resolve_categories(project, categories)
    wanted = set(annotation_ids) if annotation_ids is not None else None
    targets = {
        a.id for a in _select_boxes(
            stored.annotations, annotation_ids=wanted, category_ids=category_ids,
            include_pending=include_pending,
        )
    }
    if not targets:
        return ConvertResult(image_id=image_id, version=stored.version, converted=0,
                             skipped=0, annotations=list(stored.annotations))
    backend = backend or _segmenter(model, device)
    embedding, _, _ = _embedding_for(project, image_id, model, device, backend)
    converted = skipped = 0
    out: list[Annotation] = []
    for a in stored.annotations:
        if a.id not in targets:
            out.append(a)
            continue
        result = backend.segment(embedding, SegmentPrompt(box=a.bbox).validated())
        if not result.polygon or result.bbox is None:
            skipped += 1
            out.append(a)
            continue
        polygon = [float(v) for v in result.polygon]
        if max_points is not None:
            from horos.backends.sam.polygonize import simplify_polygon

            polygon = simplify_polygon(polygon, max_points)
        # The polygon is machine-made, so the annotation stops counting as
        # human work: it becomes ("auto", "pending") carrying the segmenter's
        # predicted IoU as its score, and a person accepts it in review
        # (E10-T11). The 2026-09-13 QC of demo_project found ~14 % of such
        # polygons wrong while they were stored as confirmed manual labels.
        new = clamp_to_image(
            a.model_copy(update={
                "segmentation": [polygon], "bbox": tuple(result.bbox),
                "source": "auto", "status": "pending",
                "score": round(float(result.score), 4),
            }),
            record.width, record.height,
        )
        out.append(new)
        converted += 1
    version = stored.version
    if converted:
        saved = project.save_annotations(
            image_id, out,
            expected_version=stored.version if expected_version is None else expected_version,
        )
        version = saved.version
    return ConvertResult(image_id=image_id, version=version, converted=converted,
                         skipped=skipped, annotations=out)


def boxes_to_polygons_events(
    project: Project,
    *,
    categories: list[int | str] | None = None,
    split: str | None = None,
    include_pending: bool = True,
    model: str = DEFAULT_SEGMENTER,
    max_points: int | None = None,
    device: str | None = None,
    backend: PromptableSegmenter | None = None,
    cancel: threading.Event | None = None,
):
    """R4 stream over every image with matching boxes: started → progress per
    image → completed(result={images, converted, skipped})."""
    from horos.backends.base import (
        ProgressUpdated,
        RunCompleted,
        RunFailed,
        RunStarted,
        WarningRaised,
    )

    try:
        category_ids = _resolve_categories(project, categories)
        targets = []
        for record in project.list_images():
            if split and record.split != split:
                continue
            stored = project.load_annotations(record.id)
            boxes = _select_boxes(
                stored.annotations, annotation_ids=None, category_ids=category_ids,
                include_pending=include_pending,
            )
            if boxes:
                targets.append((record, len(boxes)))
        yield RunStarted(
            total=len(targets),
            config={"model": model, "categories": sorted(category_ids) if category_ids else None,
                    "split": split, "include_pending": include_pending,
                    "boxes": sum(n for _, n in targets)},
        )
        if not targets:
            yield WarningRaised(message="No box annotations match — nothing to convert.")
            yield RunCompleted(result={"images": 0, "converted": 0, "skipped": 0})
            return
        backend = backend or _segmenter(model, device)
        converted = skipped = 0
        for index, (record, _) in enumerate(targets):
            if cancel is not None and cancel.is_set():
                yield RunCompleted(result={"cancelled": True, "images": index,
                                           "converted": converted, "skipped": skipped})
                return
            result = boxes_to_polygons(
                project, record.id,
                categories=sorted(category_ids) if category_ids is not None else None,
                include_pending=include_pending, model=model, max_points=max_points,
                device=device, backend=backend,
            )
            converted += result.converted
            skipped += result.skipped
            yield ProgressUpdated(
                current=index + 1, total=len(targets), phase="boxes-to-polygons",
                message=f"{record.file_name}: {result.converted} converted"
                        + (f", {result.skipped} kept as box" if result.skipped else ""),
            )
        yield RunCompleted(result={"images": len(targets), "converted": converted,
                                   "skipped": skipped})
    except Exception as exc:  # noqa: BLE001 — the stream must terminate with an event (R4)
        logger.exception("boxes-to-polygons run failed")
        yield RunFailed(error_code=getattr(exc, "code", "backend_error"), message=str(exc))


@capability(
    "segment.boxes_to_polygons_batch",
    summary="Background job: rewrite the project's box annotations (a class, a split) as "
            "SAM polygons",
    web_route="/api/v1/segment/boxes",
    web_methods=("POST",),
    cli="boxes-to-polygons",
)
def start_boxes_to_polygons(
    project: Project,
    *,
    categories: list[int | str] | None = None,
    split: str | None = None,
    include_pending: bool = True,
    model: str = DEFAULT_SEGMENTER,
    max_points: int | None = None,
    device: str | None = None,
    backend: PromptableSegmenter | None = None,
) -> str:
    """Kick off the batch as a job (poll via jobs.status); returns the job id."""
    from horos.api import jobs

    _resolve_categories(project, categories)  # unknown names fail synchronously
    if split is not None and split not in ("train", "valid", "test"):
        raise ProjectError(f"split must be train, valid or test, got {split!r}")
    return jobs.start_job(
        project,
        "boxes-to-polygons",
        lambda cancel: boxes_to_polygons_events(
            project, categories=categories, split=split, include_pending=include_pending,
            model=model, max_points=max_points, device=device, backend=backend, cancel=cancel,
        ),
    )
