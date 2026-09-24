"""Staged dataset uploads: the WebUI upload path with progress (E1-T10, R4).

A dataset zip — or a handful of loose photos (E1-T11) — arrives over HTTP, is
kept under <project>/uploads/<upload_id>/, and is imported by a background job
whose R4 events the page polls via /jobs/<id> (the same machinery autolabel
and evaluation use). Keeping the upload server-side means the conflict /
class-name / label confirmation dialogs retry with a policy instead of
re-uploading hundreds of megabytes.

Lifecycle of a staged upload:
  * import completed            → deleted
  * failed with a retryable code (import_conflict, label_conflict,
    class_names_required)       → kept for the retry
  * failed for any other reason → deleted
  * discarded by the user       → deleted
  * older than STALE_UPLOAD_SECONDS at the next staging → deleted
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

from pydantic import BaseModel

from horos.api import jobs
from horos.api.dataset import (
    ANNOTATION_POLICIES,
    CONFLICT_POLICIES,
    import_dataset,
    import_zip,
)
from horos.api.manifest import capability
from horos.backends.base import RunCompleted, RunFailed, RunStarted
from horos.core.formats import IMAGE_SUFFIXES
from horos.core.project import Project
from horos.errors import DatasetFormatError, HorosError, ProjectError

if TYPE_CHECKING:
    from horos.backends.base import Event

logger = logging.getLogger(__name__)

__all__ = [
    "StagedUpload",
    "stage_upload",
    "stage_photos",
    "start_upload_import",
    "discard_upload",
]

UPLOADS_DIR = "uploads"
#: loose photos are staged under this directory inside the upload's folder
PHOTOS_DIR = "photos"
STALE_UPLOAD_SECONDS = 60 * 60
#: failures the user can resolve from a dialog and retry without re-uploading
RETRYABLE_CODES = frozenset({"import_conflict", "label_conflict", "class_names_required"})
JOB_KIND = "import"


class StagedUpload(BaseModel):
    upload_id: str
    file_name: str
    size_bytes: int
    created_at: float


def _uploads_root(project: Project) -> Path:
    return project.root / UPLOADS_DIR


def _upload_dir(project: Project, upload_id: str) -> Path:
    if not upload_id or any(ch not in "0123456789abcdef" for ch in upload_id):
        raise ProjectError(f"Invalid upload id: {upload_id!r}")
    return _uploads_root(project) / upload_id


def _staged_source(directory: Path) -> Path:
    """What was staged: the photos directory of a loose-photo upload, else
    the zip. Raises when the upload is gone."""
    photos = directory / PHOTOS_DIR
    if photos.is_dir():
        return photos
    zips = sorted(p for p in directory.iterdir() if p.is_file()) if directory.is_dir() else []
    if not zips:
        raise ProjectError(
            f"No staged upload {directory.name} — it was imported, discarded, or expired"
        )
    return zips[0]


def _is_photo_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def _purge_stale(project: Project, *, now: float | None = None) -> int:
    root = _uploads_root(project)
    if not root.is_dir():
        return 0
    now = time.time() if now is None else now
    purged = 0
    for directory in root.iterdir():
        try:
            age = now - directory.stat().st_mtime
        except OSError:
            continue
        if age > STALE_UPLOAD_SECONDS:
            shutil.rmtree(directory, ignore_errors=True)
            purged += 1
    return purged


@capability(
    "dataset.stage_upload",
    summary="Store an uploaded dataset zip server-side for a progress-reporting import",
    web_route="/api/v1/dataset/upload",
    web_methods=("POST",),
    cli=None,
    not_cli_because="The CLI imports a local directory or zip path directly with 'import'.",
)
def stage_upload(
    project: Project, source: Path | str | IO[bytes], *, file_name: str | None = None
) -> StagedUpload:
    """Copy a zip (path or open binary stream) under <project>/uploads/<id>/.

    Rejects non-zip content immediately, so the client learns about a bad
    file synchronously instead of from a failed job. Stale staged uploads
    are purged on every call."""
    _purge_stale(project)
    upload_id = uuid.uuid4().hex[:12]
    directory = _upload_dir(project, upload_id)
    directory.mkdir(parents=True, exist_ok=False)
    if isinstance(source, str | Path):
        source = Path(source)
        name = file_name or source.name
        target = directory / (Path(name).name or "upload.zip")
        shutil.copyfile(source, target)
    else:
        name = file_name or "upload.zip"
        target = directory / (Path(name).name or "upload.zip")
        with target.open("wb") as out:
            shutil.copyfileobj(source, out)
    import zipfile

    if not zipfile.is_zipfile(target):
        shutil.rmtree(directory, ignore_errors=True)
        raise DatasetFormatError(f"'{name}' is not a valid zip archive")
    return StagedUpload(
        upload_id=upload_id,
        file_name=target.name,
        size_bytes=target.stat().st_size,
        created_at=time.time(),
    )


@capability(
    "dataset.stage_photos",
    summary="Store uploaded loose photos server-side for a progress-reporting import",
    web_route="/api/v1/dataset/upload",
    web_methods=("POST",),
    cli=None,
    not_cli_because="The CLI imports a local photo directory directly with 'import'.",
)
def stage_photos(
    project: Project, photos: Sequence[tuple[str, Path | str | IO[bytes]]]
) -> StagedUpload:
    """Stage photos dropped without any annotation file (E1-T11): each
    (name, path-or-stream) lands under <project>/uploads/<id>/photos/, and the
    import job reads that directory as the "images" format, so the photos join
    the project unlabeled. Names that are not photos, repeated names and an
    empty list are refused synchronously."""
    if not photos:
        raise DatasetFormatError("No photos to upload")
    names = [Path(name).name for name, _ in photos]
    rejected = [name for name in names if not name or not _is_photo_name(name)]
    if rejected:
        raise DatasetFormatError(
            f"Not photos: {', '.join(rejected[:5])}{' …' if len(rejected) > 5 else ''} — "
            f"drop photos ({', '.join(IMAGE_SUFFIXES)}) or one dataset zip"
        )
    if len(set(names)) != len(names):
        raise DatasetFormatError("The same photo name appears twice in this upload")
    _purge_stale(project)
    upload_id = uuid.uuid4().hex[:12]
    directory = _upload_dir(project, upload_id) / PHOTOS_DIR
    directory.mkdir(parents=True, exist_ok=False)
    total = 0
    for name, (_, source) in zip(names, photos, strict=True):
        target = directory / name
        if isinstance(source, str | Path):
            shutil.copyfile(Path(source), target)
        else:
            with target.open("wb") as out:
                shutil.copyfileobj(source, out)
        total += target.stat().st_size
    return StagedUpload(
        upload_id=upload_id,
        file_name=names[0] if len(names) == 1 else f"{len(names)} photos",
        size_bytes=total,
        created_at=time.time(),
    )


def upload_import_events(
    project: Project,
    upload_id: str,
    *,
    on_conflict: str = "ask",
    on_annotations: str = "ask",
    class_names: list[str] | None = None,
    require_class_names: bool = True,
) -> Iterator[Event]:
    """R4 stream for importing a staged upload: started → extracting / reading /
    checking / copying / saving progress → completed(result=ImportSummary) or
    failed(error_code, details). Not cancellable: an import is a single
    transaction whose partial state would be worse than a finished one."""
    directory = _upload_dir(project, upload_id)
    source = _staged_source(directory)
    is_photos = source.is_dir()
    yield RunStarted(
        config={
            "upload_id": upload_id,
            "file_name": f"{len(list(source.iterdir()))} photos" if is_photos else source.name,
            "size_bytes": (
                sum(p.stat().st_size for p in source.iterdir()) if is_photos
                else source.stat().st_size
            ),
            "on_conflict": on_conflict,
            "on_annotations": on_annotations,
            "class_names": class_names,
        }
    )

    # import_zip reports through a callback; bridge it into this generator
    # with a queue fed from a worker thread
    inbox: queue.Queue = queue.Queue()
    _DONE, _ERROR = object(), object()

    def work() -> None:
        try:
            if is_photos:
                summary = import_dataset(
                    project,
                    source,
                    copy_images=True,
                    boundary=source,
                    on_conflict=on_conflict,
                    on_annotations=on_annotations,
                    progress=inbox.put,
                )
            else:
                summary = import_zip(
                    project,
                    source,
                    on_conflict=on_conflict,
                    on_annotations=on_annotations,
                    class_names=class_names,
                    require_class_names=require_class_names,
                    progress=inbox.put,
                )
            inbox.put((_DONE, summary))
        except BaseException as exc:  # noqa: BLE001 — every outcome must reach the stream
            inbox.put((_ERROR, exc))

    threading.Thread(target=work, name=f"horos-import-{upload_id}", daemon=True).start()
    while True:
        item = inbox.get()
        if isinstance(item, tuple) and len(item) == 2 and item[0] in (_DONE, _ERROR):
            marker, payload = item
            break
        yield item

    if marker is _DONE:
        shutil.rmtree(directory, ignore_errors=True)
        yield RunCompleted(result=payload.model_dump())
        return
    exc = payload
    if isinstance(exc, HorosError):
        retryable = exc.code in RETRYABLE_CODES
        if not retryable:
            shutil.rmtree(directory, ignore_errors=True)
        yield RunFailed(
            error_code=exc.code,
            message=str(exc),
            details={**(exc.details or {}), "upload_id": upload_id, "retryable": retryable},
        )
        return
    logger.exception("import of staged upload %s crashed", upload_id, exc_info=exc)
    shutil.rmtree(directory, ignore_errors=True)
    yield RunFailed(
        error_code="import_error",
        message=f"{type(exc).__name__}: {exc}",
        details={"upload_id": upload_id, "retryable": False},
    )


@capability(
    "dataset.import_upload",
    summary="Import a staged upload (zip or photos) as a background job "
            "(poll /jobs/<id> for progress)",
    web_route="/api/v1/dataset/upload/<upload_id>/import",
    web_methods=("POST",),
    cli=None,
    not_cli_because="The CLI imports in the foreground and prints progress directly.",
)
def start_upload_import(
    project: Project,
    upload_id: str,
    *,
    on_conflict: str = "ask",
    on_annotations: str = "ask",
    class_names: list[str] | None = None,
    require_class_names: bool = True,
) -> str:
    """Start the import job for a staged upload; returns the job id. Parameter
    errors and a missing upload raise synchronously."""
    if on_conflict not in CONFLICT_POLICIES:
        raise ProjectError(f"on_conflict must be one of {CONFLICT_POLICIES}")
    if on_annotations not in ANNOTATION_POLICIES:
        raise ProjectError(f"on_annotations must be one of {ANNOTATION_POLICIES}")
    _staged_source(_upload_dir(project, upload_id))  # fail now, not inside the job
    return jobs.start_job(
        project,
        JOB_KIND,
        lambda cancel: upload_import_events(
            project,
            upload_id,
            on_conflict=on_conflict,
            on_annotations=on_annotations,
            class_names=class_names,
            require_class_names=require_class_names,
        ),
    )


@capability(
    "dataset.discard_upload",
    summary="Delete a staged upload the user decided not to import",
    web_route="/api/v1/dataset/upload/<upload_id>",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="The CLI never stages uploads.",
)
def discard_upload(project: Project, upload_id: str) -> bool:
    """True if a staged zip was removed, False if there was nothing to remove."""
    directory = _upload_dir(project, upload_id)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return True
