"""Annotation editing API — E2. Business logic lives here (R2).

Design decisions (confirmed):
- every action saves immediately, guarded by the per-image optimistic lock
  (version + atomic rename, E2-T8)
- multi-user coordination is optimistic locking plus advisory soft-claims:
  a session claims an image while annotating it, the queue steers other
  sessions away, and an expired claim simply stops steering — correctness
  always comes from the version check, never from the claim
- the queue is server-side so the WebUI and Python API resume identically
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.dataset import Annotation, ImageRecord, clamp_to_image
from horos.core.project import Project, _write_json_atomic
from horos.errors import DatasetValidationError, ProjectError

logger = logging.getLogger(__name__)

__all__ = [
    "AnnotationSetView",
    "QueueItem",
    "AnnotationProgress",
    "ClaimResult",
    "get_annotations",
    "save_annotations",
    "image_queue",
    "annotation_progress",
    "claim_image",
    "release_claim",
    "image_file_path",
]

QUEUE_MODES = ("unannotated_first", "file_name", "annotated", "unannotated", "pending")
CLAIM_TTL_SECONDS = 300.0
_CLAIMS_FILE = "claims.json"


class AnnotationSetView(BaseModel):
    image_id: int
    version: int
    annotations: list[Annotation]
    #: the image's record. The editor can therefore open one photo — name,
    #: set, pixels — from this single response, without waiting for the whole
    #: queue, which is what a link straight to one photo needs.
    image: ImageRecord | None = None


class QueueItem(BaseModel):
    image: ImageRecord
    num_annotations: int
    annotated: bool
    #: session id of an unexpired claim held by someone else; None otherwise
    claimed_by: str | None = None
    #: auto pre-labels awaiting review (E3-T5); mean score drives the
    #: uncertainty-first review ordering (E3-T6)
    num_pending: int = 0
    mean_pending_score: float | None = None
    #: skipped as unfit for training (E10-T16); shown, never selected or trained on
    excluded: bool = False


class AnnotationProgress(BaseModel):
    total_images: int
    annotated_images: int
    unannotated_images: int
    total_annotations: int


class ClaimResult(BaseModel):
    granted: bool
    image_id: int
    #: who holds it when not granted
    held_by: str | None = None
    expires_at: float | None = None


class _Claims(BaseModel):
    claims: dict[int, dict] = Field(default_factory=dict)


def _record(project: Project, image_id: int) -> ImageRecord:
    # Project.get_image indexes the (cached) image index by id; scanning the
    # list here cost a pass over every record on each annotation save
    try:
        return project.get_image(image_id)
    except ProjectError:
        raise ProjectError(f"No image with id {image_id}") from None


# ------------------------------------------------------------------ editing


def _normalize(project: Project, raw: list[Annotation | dict], image_id: int) -> list[Annotation]:
    """Validate incoming annotations; derive a bbox from the polygon when the
    client sends segmentation only. Explicit errors — with one deliberate
    normalization: coordinates are clipped into the image bounds (the
    out-of-frame part of a box describes nothing photographable; a box left
    entirely outside still errors as degenerate)."""
    record = project.get_image(image_id)
    category_ids = {c.id for c in project.categories}
    out: list[Annotation] = []
    for n, item in enumerate(raw, start=1):
        data = item.model_dump() if isinstance(item, Annotation) else dict(item)
        data.setdefault("id", n)
        data["image_id"] = image_id
        segmentation = data.get("segmentation") or []
        for poly in segmentation:
            if len(poly) < 6 or len(poly) % 2:
                raise DatasetValidationError(
                    f"annotation #{n}: polygon needs >=3 (x, y) points, "
                    f"got {len(poly)} values"
                )
        if data.get("bbox") is None and segmentation:
            # an object in several pieces has several rings: the box spans them all
            from horos.core.validate import polygons_extent

            data["bbox"] = polygons_extent(segmentation)
        ann = clamp_to_image(
            Annotation.model_validate(data), record.width, record.height
        )
        if ann.category_id not in category_ids:
            raise DatasetValidationError(
                f"annotation #{n}: unknown category id {ann.category_id}"
            )
        if ann.bbox[2] <= 0 or ann.bbox[3] <= 0:
            raise DatasetValidationError(
                f"annotation #{n}: degenerate box (w={ann.bbox[2]}, h={ann.bbox[3]})"
            )
        out.append(ann)
    return out


@capability(
    "annotate.get",
    summary="Read one image's annotations with their lock version",
    web_route="/api/v1/images/<int:image_id>/annotations",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Annotating is interactive; scripts use the Python API.",
)
def get_annotations(project: Project, image_id: int) -> AnnotationSetView:
    record = _record(project, image_id)
    stored = project.load_annotations(image_id)
    return AnnotationSetView(
        image_id=image_id, version=stored.version, annotations=stored.annotations,
        image=record,
    )


@capability(
    "annotate.save",
    summary="Replace one image's annotations (optimistic-locked by version)",
    web_route="/api/v1/images/<int:image_id>/annotations",
    web_methods=("PUT",),
    cli=None,
    not_cli_because="Annotating is interactive; scripts use the Python API.",
)
def save_annotations(
    project: Project,
    image_id: int,
    annotations: list[Annotation | dict],
    *,
    expected_version: int,
) -> AnnotationSetView:
    """Every UI action lands here (design: save per action). Raises
    AnnotationConflictError when someone else wrote first (E2-T8)."""
    _record(project, image_id)
    normalized = _normalize(project, annotations, image_id)
    updated = project.save_annotations(
        image_id, normalized, expected_version=expected_version
    )
    # re-read: the first confirmed annotation puts the photo in a set
    return AnnotationSetView(
        image_id=image_id, version=updated.version, annotations=updated.annotations,
        image=_record(project, image_id),
    )


# ------------------------------------------------------------------- queue


@capability(
    "annotate.queue",
    summary="The annotation image queue (unannotated-first by default)",
    web_route="/api/v1/queue",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Annotating is interactive; scripts use the Python API.",
)
def image_queue(
    project: Project,
    *,
    mode: str = "unannotated_first",
    split: str | None = None,
    category_id: int | None = None,
    session_id: str | None = None,
) -> list[QueueItem]:
    """Server-side queue so WebUI and Python API resume at the same place
    (E2-S2). `category_id` keeps only the photos that carry that class
    (confirmed or pending). `session_id` hides that session's own claims
    from claimed_by."""
    if mode not in QUEUE_MODES:
        raise ProjectError(f"queue mode must be one of {QUEUE_MODES}")
    if category_id is not None and all(c.id != category_id for c in project.categories):
        raise ProjectError(f"Unknown category id {category_id}")
    claims = _load_claims(project)
    items: list[QueueItem] = []
    for record in project.list_images():
        # "unlabeled" selects the photos in no set (only labeled photos have one)
        if split and (record.split or "unlabeled") != split:
            continue
        stored = project.load_annotations(record.id)
        if category_id is not None and all(
            a.category_id != category_id for a in stored.annotations
        ):
            continue
        holder = claims.get(record.id)
        pending_scores = [
            a.score or 0.0 for a in stored.annotations if a.status == "pending"
        ]
        confirmed = [a for a in stored.annotations if a.status == "confirmed"]
        items.append(
            QueueItem(
                image=record,
                num_annotations=len(confirmed),
                annotated=bool(confirmed),
                excluded=record.excluded,
                claimed_by=(
                    holder["session"]
                    if holder and holder["session"] != session_id
                    else None
                ),
                num_pending=len(pending_scores),
                mean_pending_score=(
                    sum(pending_scores) / len(pending_scores)
                    if pending_scores
                    else None
                ),
            )
        )
    if mode == "annotated":
        items = [i for i in items if i.annotated]
    elif mode == "unannotated":
        items = [i for i in items if not i.annotated]
    elif mode == "pending":
        # review queue: least-confident pre-labels first (E3-S3)
        items = [i for i in items if i.num_pending]
        items.sort(key=lambda i: (i.mean_pending_score or 0.0, i.image.file_name))
        return items
    if mode == "unannotated_first":
        items.sort(key=lambda i: (i.annotated, i.image.file_name))
    else:
        items.sort(key=lambda i: i.image.file_name)
    return items


@capability(
    "annotate.progress",
    summary="Annotation progress counters for the project",
    web_route="/api/v1/progress",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Annotating is interactive; scripts use the Python API.",
)
def annotation_progress(project: Project) -> AnnotationProgress:
    total = annotated = annotations = 0
    for record in project.list_images():
        total += 1
        stored = project.load_annotations(record.id)
        confirmed = [a for a in stored.annotations if a.status == "confirmed"]
        if confirmed:
            annotated += 1
            annotations += len(confirmed)
    return AnnotationProgress(
        total_images=total,
        annotated_images=annotated,
        unannotated_images=total - annotated,
        total_annotations=annotations,
    )


# ------------------------------------------------------------------- claims


def _claims_path(project: Project) -> Path:
    return project.root / _CLAIMS_FILE


def _load_claims(project: Project) -> dict[int, dict]:
    path = _claims_path(project)
    if not path.exists():
        return {}
    try:
        claims = _Claims.model_validate_json(path.read_text(encoding="utf-8")).claims
    except ValueError:
        logger.warning("claims file unreadable — starting fresh")
        return {}
    now = time.time()
    return {k: v for k, v in claims.items() if v.get("expires_at", 0) > now}


def _store_claims(project: Project, claims: dict[int, dict]) -> None:
    _write_json_atomic(
        _claims_path(project), _Claims(claims=claims).model_dump_json(indent=2)
    )


def claims_held_by_others(
    project: Project, session_id: str | None = None
) -> dict[int, str]:
    """image_id -> holding session, for unexpired claims not held by
    `session_id`. Used by destructive bulk operations (image deletion) to
    steer around images someone is actively annotating."""
    return {
        image_id: holder.get("session", "")
        for image_id, holder in _load_claims(project).items()
        if holder.get("session") != session_id
    }


@capability(
    "annotate.claim",
    summary="Soft-claim an image for a session (advisory; queue steers others away)",
    web_route="/api/v1/images/<int:image_id>/claim",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Claims coordinate interactive sessions only.",
)
def claim_image(
    project: Project,
    image_id: int,
    session_id: str | None = None,
    *,
    ttl_seconds: float = CLAIM_TTL_SECONDS,
) -> ClaimResult:
    """Claiming again from the same session renews the TTL. A claim held by
    another session is reported, not overridden — but it is advisory only:
    the optimistic lock still decides every write."""
    _record(project, image_id)
    session_id = session_id or uuid.uuid4().hex
    claims = _load_claims(project)
    holder = claims.get(image_id)
    if holder and holder["session"] != session_id:
        return ClaimResult(
            granted=False,
            image_id=image_id,
            held_by=holder["session"],
            expires_at=holder["expires_at"],
        )
    expires = time.time() + ttl_seconds
    claims[image_id] = {"session": session_id, "expires_at": expires}
    _store_claims(project, claims)
    return ClaimResult(granted=True, image_id=image_id, expires_at=expires)


@capability(
    "annotate.release",
    summary="Release a session's soft-claim on an image",
    web_route="/api/v1/images/<int:image_id>/claim",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="Claims coordinate interactive sessions only.",
)
def release_claim(project: Project, image_id: int, session_id: str) -> bool:
    """Only the holding session can release; anyone else's release is a no-op
    (returns False) — never yank a claim out from under an active annotator."""
    claims = _load_claims(project)
    holder = claims.get(image_id)
    if holder is None or holder["session"] != session_id:
        return False
    del claims[image_id]
    _store_claims(project, claims)
    return True


# --------------------------------------------------------------- image file


@capability(
    "annotate.image_file",
    summary="Absolute path of one image's file (served as bytes over the Web API)",
    web_route="/api/v1/images/<int:image_id>/file",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Scripts read the path via the Python API directly.",
)
def image_file_path(project: Project, image_id: int) -> Path:
    record = _record(project, image_id)
    # absolute: Flask's send_file resolves relative paths against the app
    # package dir, not the CWD — a project opened via a relative path 500s
    path = project.image_path(record).resolve()
    if not path.exists():
        raise ProjectError(f"Image file missing on disk for image {image_id}: {path}")
    return path
