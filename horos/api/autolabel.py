"""Zero-shot autolabeling (E3) — business logic lives here (R2).

Design decisions (confirmed): batch runs as a background job with polling
(api/jobs.py); the editor also gets synchronous single-image assist; pre-labels
are written as `source="auto", status="pending"` annotations reviewed inside
the annotator.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from threading import Event as CancelEvent
from typing import TYPE_CHECKING

from pydantic import BaseModel

from horos.api.manifest import capability
from horos.core.dataset import Annotation, Category, clamp_to_image, default_color
from horos.core.project import Project
from horos.errors import DatasetValidationError, ProjectError

if TYPE_CHECKING:
    from horos.backends.base import (
        BoxToMaskBackend,
        Event,
        ImagePrediction,
        OpenVocabularyBackend,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "PromptSpec",
    "AssistResult",
    "build_prompt_index",
    "postprocess",
    "autolabel_events",
    "assist_image",
    "accept_pending",
    "reject_pending",
    "pending_summary",
]

DEFAULT_MODEL = "owlv2-base"
DEFAULT_REFINER = "sam-base"
DEFAULT_THRESHOLD = 0.1
DEFAULT_NMS_IOU = 0.5
OUTPUT_MODES = ("bbox", "polygon")


def _check_output(output: str) -> None:
    if output not in OUTPUT_MODES:
        raise ProjectError(f"output must be one of {OUTPUT_MODES}")


class PromptSpec(BaseModel):
    """class name -> one or more text prompts (E3-T2)."""

    prompts: dict[str, list[str]]

    def flat(self) -> tuple[list[str], list[str]]:
        """Returns (prompt_list, class_name_per_prompt_index)."""
        texts: list[str] = []
        classes: list[str] = []
        for cls, plist in self.prompts.items():
            cleaned = [p.strip() for p in plist if p.strip()]
            if not cls.strip() or not cleaned:
                raise DatasetValidationError(
                    f"Prompt spec entries need a class name and at least one "
                    f"prompt (offending entry: {cls!r}: {plist!r})"
                )
            for p in cleaned:
                texts.append(p)
                classes.append(cls.strip())
        if not texts:
            raise DatasetValidationError("Prompt spec is empty")
        return texts, classes


class AssistResult(BaseModel):
    image_id: int
    annotations: list[Annotation]
    version: int


class PendingImageSummary(BaseModel):
    image_id: int
    file_name: str
    num_pending: int
    mean_score: float
    min_score: float


def build_prompt_index(spec: PromptSpec) -> tuple[list[str], list[str]]:
    return spec.flat()


# ------------------------------------------------------------- postprocess


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def postprocess(
    prediction: ImagePrediction,
    class_by_prompt: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """Confidence filter + per-class NMS (E3-T4).

    Returns (class_name, bbox, score) tuples, highest score first. Prompts
    mapping to the same class are merged before NMS, so two prompts firing on
    the same object keep only the better box.
    """
    kept: list[tuple[str, tuple[float, float, float, float], float]] = []
    candidates = sorted(
        (i for i in prediction.instances if i.score >= threshold),
        key=lambda i: -i.score,
    )
    for inst in candidates:
        if not 0 <= inst.category_id < len(class_by_prompt):
            raise DatasetValidationError(
                f"Backend returned prompt index {inst.category_id} outside the "
                f"prompt list (0..{len(class_by_prompt) - 1})"
            )
        cls = class_by_prompt[inst.category_id]
        if any(
            c == cls and _iou(b, inst.bbox) >= nms_iou for c, b, _ in kept
        ):
            continue
        kept.append((cls, inst.bbox, inst.score))
    return kept


# ------------------------------------------------------------ category sync


def _ensure_categories(project: Project, class_names: set[str]) -> dict[str, int]:
    """{name: category id} for every requested class name, creating classes
    that do not exist. A name that is an alias of an existing class (renamed
    or merged since the model learned it) maps to that class instead of
    becoming a new one; the mapping is keyed by the name as requested."""
    categories = list(project.categories)
    by_name = {c.name: c.id for c in categories}
    changed = False
    for name in sorted(class_names):
        current = project.resolve_category_name(name)
        if current in by_name:
            by_name[name] = by_name[current]
            continue
        new_id = max((c.id for c in categories), default=0) + 1
        categories.append(
            Category(id=new_id, name=name, color=default_color(len(categories)))
        )
        by_name[name] = new_id
        changed = True
    if changed:
        project.set_categories(categories)
    return by_name


def _write_pending(
    project: Project,
    image_id: int,
    detections: list[tuple[str, tuple[float, float, float, float], float]],
    cat_ids: dict[str, int],
    polygons: list[list[float] | None] | None = None,
) -> int:
    """Replace the image's previous auto-pending annotations with this run's;
    confirmed (human) annotations are never touched. `polygons` (parallel to
    detections) attaches a segmentation where the refiner produced one — a
    detection whose mask failed keeps its box, never gets dropped silently."""
    stored = project.load_annotations(image_id)
    record = project.get_image(image_id)
    keep = [a for a in stored.annotations if a.status != "pending"]
    next_id = max((a.id for a in keep), default=0) + 1
    for n, (cls, bbox, score) in enumerate(detections):
        polygon = polygons[n] if polygons else None
        # model outputs routinely overshoot the frame by a few pixels —
        # clip into bounds; a detection entirely outside the image is noise
        annotation = clamp_to_image(
            Annotation(
                id=next_id,
                image_id=image_id,
                category_id=cat_ids[cls],
                bbox=bbox,
                segmentation=[polygon] if polygon else [],
                source="auto",
                status="pending",
                score=round(score, 4),
            ),
            record.width,
            record.height,
        )
        if annotation.bbox[2] <= 0 or annotation.bbox[3] <= 0:
            continue
        keep.append(annotation)
        next_id += 1
    project.save_annotations(image_id, keep, expected_version=stored.version)
    return sum(1 for a in keep if a.status == "pending")


# ------------------------------------------------------------- batch (E3-T3)


def autolabel_events(
    project: Project,
    spec: PromptSpec,
    *,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    split: str | None = None,
    only_unannotated: bool = True,
    output: str = "bbox",
    refiner_model: str = DEFAULT_REFINER,
    device: str | None = None,
    backend: OpenVocabularyBackend | None = None,
    refiner: BoxToMaskBackend | None = None,
    cancel: CancelEvent | None = None,
) -> Iterator[Event]:
    """Autolabel the project's images as an R4 event stream.

    `output="polygon"` runs each kept box through the SAM refiner and writes
    the mask outline as the annotation's segmentation (the box is kept when a
    mask fails — never dropped silently). `backend`/`refiner` are injectable
    for tests. `cancel` is checked between images — a cancelled run ends with
    RunCompleted carrying result["cancelled"]=True (work done so far is kept;
    pre-labels are pending anyway).
    """
    _check_output(output)
    from horos.backends.base import (
        PredictionReady,
        ProgressUpdated,
        RunCompleted,
        RunFailed,
        RunStarted,
        WarningRaised,
    )

    texts, class_by_prompt = spec.flat()
    targets = []
    for record in project.list_images():
        if split and record.split != split:
            continue
        stored = project.load_annotations(record.id)
        has_confirmed = any(a.status == "confirmed" for a in stored.annotations)
        if only_unannotated and has_confirmed:
            continue
        targets.append(record)

    yield RunStarted(
        total=len(targets),
        config={
            "model": model, "threshold": threshold, "nms_iou": nms_iou,
            "prompts": spec.prompts, "only_unannotated": only_unannotated,
            "output": output,
            **({"split": split} if split else {}),
        },
    )
    if not targets:
        # finishing silently with 0 pre-labels reads like a failure — say why
        reasons = []
        if only_unannotated:
            reasons.append(
                "every image already has confirmed annotations "
                "(uncheck 'only unannotated' to pre-label them anyway)"
            )
        if split:
            reasons.append(f"the split filter '{split}' matched nothing")
        yield WarningRaised(
            message="No target images: " + (" and ".join(reasons) or "the project has no images")
        )
        yield RunCompleted(result={"images": 0, "annotations": 0})
        return
    try:
        if backend is None:
            from horos.backends import get_backend

            backend = get_backend(model, device=device)  # type: ignore[assignment]
        if output == "polygon" and refiner is None:
            from horos.backends import get_backend

            refiner = get_backend(refiner_model, device=device)  # type: ignore[assignment]
        backend.configure_prompts(texts)
        cat_ids = _ensure_categories(project, set(class_by_prompt))

        written = 0
        for index, record in enumerate(targets):
            if cancel is not None and cancel.is_set():
                yield RunCompleted(
                    result={"cancelled": True, "images": index, "annotations": written}
                )
                return
            path = project.image_path(record)
            prediction = backend.infer_one(path, threshold=threshold)
            detections = postprocess(
                prediction, class_by_prompt, threshold=threshold, nms_iou=nms_iou
            )
            polygons = (
                refiner.polygons_for_boxes(path, [b for _, b, _ in detections])
                if refiner is not None and detections
                else None
            )
            written += _write_pending(project, record.id, detections, cat_ids, polygons)
            yield PredictionReady(index=index, prediction=prediction)
            yield ProgressUpdated(
                current=index + 1, total=len(targets), phase="autolabel",
                message=f"{record.file_name}: {len(detections)} pre-label(s)",
            )
        yield RunCompleted(result={"images": len(targets), "annotations": written})
    except Exception as exc:  # noqa: BLE001 — stream must terminate with an event (R4)
        logger.exception("autolabel run failed")
        code = getattr(exc, "code", "backend_error")
        yield RunFailed(error_code=code, message=str(exc))


@capability(
    "autolabel.start",
    summary="Start a background autolabel job over the project (poll via jobs.status)",
    web_route="/api/v1/autolabel",
    web_methods=("POST",),
    cli="autolabel",
)
def start_autolabel(
    project: Project,
    spec: PromptSpec,
    *,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    split: str | None = None,
    only_unannotated: bool = True,
    output: str = "bbox",
    device: str | None = None,
    backend: OpenVocabularyBackend | None = None,
    refiner: BoxToMaskBackend | None = None,
) -> str:
    """Kick off the batch as a background job (design decision); returns the
    job id for polling. The CLI instead consumes autolabel_events directly."""
    from horos.api import jobs

    spec.flat()  # validate before the thread starts, so errors are synchronous
    _check_output(output)
    return jobs.start_job(
        project,
        "autolabel",
        lambda cancel: autolabel_events(
            project,
            spec,
            model=model,
            threshold=threshold,
            nms_iou=nms_iou,
            split=split,
            only_unannotated=only_unannotated,
            output=output,
            device=device,
            backend=backend,
            refiner=refiner,
            cancel=cancel,
        ),
    )


# ---------------------------------------------------------- assist (single)

_ASSIST_BACKENDS: dict = {}


def _cached_backend(model: str, device: str | None):
    key = (model, device)
    if key not in _ASSIST_BACKENDS:
        from horos.backends import get_backend

        _ASSIST_BACKENDS[key] = get_backend(model, device=device)
    return _ASSIST_BACKENDS[key]


@capability(
    "autolabel.assist",
    summary="Zero-shot pre-labels for ONE image from text prompts (editor assist)",
    web_route="/api/v1/images/<int:image_id>/assist",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Single-image assist is an interactive editor feature.",
)
def assist_image(
    project: Project,
    image_id: int,
    spec: PromptSpec,
    *,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    output: str = "bbox",
    refiner_model: str = DEFAULT_REFINER,
    device: str | None = None,
    backend: OpenVocabularyBackend | None = None,
    refiner: BoxToMaskBackend | None = None,
) -> AssistResult:
    """Run the open-vocabulary model on one image and write the results as
    pending annotations (replacing previous pendings on that image)."""
    _check_output(output)
    record = next((r for r in project.list_images() if r.id == image_id), None)
    if record is None:
        raise ProjectError(f"No image with id {image_id}")
    texts, class_by_prompt = spec.flat()
    if backend is None:
        backend = _cached_backend(model, device)
    backend.configure_prompts(texts)
    path = project.image_path(record)
    prediction = backend.infer_one(path, threshold=threshold)
    detections = postprocess(
        prediction, class_by_prompt, threshold=threshold, nms_iou=nms_iou
    )
    polygons = None
    if output == "polygon" and detections:
        if refiner is None:
            refiner = _cached_backend(refiner_model, device)
        polygons = refiner.polygons_for_boxes(path, [b for _, b, _ in detections])
    cat_ids = _ensure_categories(project, set(class_by_prompt))
    _write_pending(project, image_id, detections, cat_ids, polygons)
    stored = project.load_annotations(image_id)
    return AssistResult(
        image_id=image_id, version=stored.version, annotations=stored.annotations
    )


# ------------------------------------------------------------ review (E3-T5)


@capability(
    "autolabel.review",
    summary="Accept or reject pending pre-labels on an image",
    web_route="/api/v1/images/<int:image_id>/review",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Review is interactive; scripts edit via the Python API.",
)
def review_pending(
    project: Project,
    image_id: int,
    action: str,
    *,
    ann_ids: list[int] | None = None,
    min_score: float | None = None,
) -> int:
    """One entry point for the review actions (E3-S5): action is "accept"
    (optionally at a score threshold — below it pendings are dropped) or
    "reject" (pendings removed)."""
    if action == "accept":
        return accept_pending(project, image_id, ann_ids=ann_ids, min_score=min_score)
    if action == "reject":
        return reject_pending(project, image_id, ann_ids=ann_ids)
    raise ProjectError("review action must be 'accept' or 'reject'")


def accept_pending(
    project: Project,
    image_id: int,
    *,
    ann_ids: list[int] | None = None,
    min_score: float | None = None,
) -> int:
    """Flip pending annotations to confirmed. With `min_score`, only pendings
    at or above it are accepted AND the ones below are dropped — that is the
    'accept at this threshold' review action (E3-S2/S5)."""
    stored = project.load_annotations(image_id)
    kept: list[Annotation] = []
    accepted = 0
    for ann in stored.annotations:
        if ann.status != "pending" or (ann_ids is not None and ann.id not in ann_ids):
            kept.append(ann)
            continue
        if min_score is not None and (ann.score or 0.0) < min_score:
            continue  # below the accept threshold: rejected
        kept.append(ann.model_copy(update={"status": "confirmed"}))
        accepted += 1
    project.save_annotations(image_id, kept, expected_version=stored.version)
    return accepted


def reject_pending(
    project: Project, image_id: int, *, ann_ids: list[int] | None = None
) -> int:
    """Remove pending annotations (all of them, or the given ids)."""
    stored = project.load_annotations(image_id)
    kept = [
        a
        for a in stored.annotations
        if a.status != "pending" or (ann_ids is not None and a.id not in ann_ids)
    ]
    removed = len(stored.annotations) - len(kept)
    project.save_annotations(image_id, kept, expected_version=stored.version)
    return removed


# ----------------------------------------------------------- ranking (E3-T6)


@capability(
    "autolabel.pending",
    summary="Images with pending pre-labels, least-confident first (review queue)",
    web_route="/api/v1/autolabel/pending",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Review is interactive; scripts edit via the Python API.",
)
def pending_summary(project: Project) -> list[PendingImageSummary]:
    """Uncertainty ranking: ascending mean pending score, so the annotator
    reviews the model's least confident images first (E3-S3)."""
    out: list[PendingImageSummary] = []
    for record in project.list_images():
        scores = [
            a.score or 0.0
            for a in project.load_annotations(record.id).annotations
            if a.status == "pending"
        ]
        if not scores:
            continue
        out.append(
            PendingImageSummary(
                image_id=record.id,
                file_name=record.file_name,
                num_pending=len(scores),
                mean_score=sum(scores) / len(scores),
                min_score=min(scores),
            )
        )
    out.sort(key=lambda s: s.mean_score)
    return out
