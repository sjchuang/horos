"""Category (label) management — E2-T4. Business logic lives here (R2)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.backends.base import ImagePrediction
from horos.core.dataset import Category, default_color
from horos.core.project import Project
from horos.errors import CategoryInUseError, ProjectError

if TYPE_CHECKING:
    from horos.backends.base import Event

logger = logging.getLogger(__name__)

__all__ = [
    "MergeResult", "add_category", "update_category", "delete_category",
    "delete_category_events", "start_delete_category_job", "merge_categories",
    "resolve_prediction_names",
]


class MergeResult(BaseModel):
    target: Category
    #: annotations relabelled from a source class to the target
    merged_annotations: int = 0
    images_touched: int = 0
    removed_ids: list[int] = Field(default_factory=list)


def _find(project: Project, category_id: int) -> Category:
    cat = next((c for c in project.categories if c.id == category_id), None)
    if cat is None:
        raise ProjectError(f"No category with id {category_id}")
    return cat


def _ensure_name_free(project: Project, name: str, *, ignore_id: int | None = None) -> None:
    clash = next(
        (c for c in project.categories if c.name == name and c.id != ignore_id), None
    )
    if clash is not None:
        raise ProjectError(f"A category named '{name}' already exists (id {clash.id})")


@capability(
    "labels.add",
    summary="Add a category to the project",
    web_route="/api/v1/categories",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Label management is interactive; scripts edit via the Python API.",
)
def add_category(project: Project, name: str, *, color: str | None = None) -> Category:
    name = name.strip()
    if not name:
        raise ProjectError("Category name must not be empty")
    _ensure_name_free(project, name)
    categories = list(project.categories)
    new_id = max((c.id for c in categories), default=0) + 1
    category = Category(
        id=new_id, name=name, color=color or default_color(len(categories))
    )
    project.set_categories(categories + [category])
    return category


@capability(
    "labels.update",
    summary="Rename a category or change its color",
    web_route="/api/v1/categories/<int:category_id>",
    web_methods=("PATCH",),
    cli=None,
    not_cli_because="Label management is interactive; scripts edit via the Python API.",
)
def update_category(
    project: Project,
    category_id: int,
    *,
    name: str | None = None,
    color: str | None = None,
) -> Category:
    cat = _find(project, category_id)
    if name is not None:
        name = name.strip()
        if not name:
            raise ProjectError("Category name must not be empty")
        _ensure_name_free(project, name, ignore_id=category_id)
    aliases = list(cat.aliases)
    if name is not None and name != cat.name:
        # the old name stays reachable: a model trained before the rename
        # answers with it, and that output must land on this class
        aliases = [a for a in aliases if a != name] + [cat.name]
    updated = cat.model_copy(
        update={
            **({"name": name} if name is not None else {}),
            **({"color": color} if color is not None else {}),
            "aliases": aliases,
        }
    )
    others = [
        c.model_copy(update={"aliases": [a for a in c.aliases if a != name]})
        if name is not None and name in c.aliases else c
        for c in project.categories if c.id != category_id
    ]
    project.set_categories(
        [updated if c.id == category_id else next(o for o in others if o.id == c.id)
         for c in project.categories]
    )
    return updated


def resolve_prediction_names(project: Project, prediction: ImagePrediction) -> ImagePrediction:
    """The prediction with every instance's class name mapped through the
    project's current names (renames and merges recorded as aliases), so a
    model that learned "box" shows and labels "Box" after the rename."""
    def fix(inst):
        if inst.category_name is None:
            return inst
        current = project.resolve_category_name(inst.category_name)
        return inst if current == inst.category_name else inst.model_copy(
            update={"category_name": current})

    return prediction.model_copy(update={
        "instances": [fix(i) for i in prediction.instances],
        "candidates": [fix(i) for i in prediction.candidates],
    })


def delete_category_events(
    project: Project,
    category_id: int,
    *,
    force: bool = False,
    cancel: threading.Event | None = None,
) -> Iterator[Event]:
    """R4 stream of a class deletion: started → progress while every photo's
    labels are scanned ("scanning") and while the referencing photos are
    rewritten ("deleting") → completed. Without force, a class still in use
    ends the stream with failed(category_in_use) carrying the counts, so a
    UI can ask "delete them too?" and start again with force. On a project
    of ten thousand photos the scan alone takes seconds — hence a job with
    a bar, not a request that seems to hang."""
    from horos.backends.base import ProgressUpdated, RunCompleted, RunFailed, RunStarted

    try:
        category = _find(project, category_id)
    except ProjectError as exc:
        yield RunFailed(error_code=exc.code, message=str(exc))
        return
    images = project.list_images()
    yield RunStarted(total=len(images), config={"category": category.name, "force": force})
    referencing: dict[int, int] = {}  # image_id -> count
    for n, record in enumerate(images, start=1):
        if cancel is not None and cancel.is_set():
            yield RunCompleted(result={"cancelled": True, "deleted_annotations": 0})
            return
        stored = project.load_annotations(record.id)
        hits = sum(1 for a in stored.annotations if a.category_id == category_id)
        if hits:
            referencing[record.id] = hits
        if n % 50 == 0 or n == len(images):
            yield ProgressUpdated(
                current=n, total=len(images), phase="scanning",
                message=f"{len(referencing)} photo(s) use '{category.name}'",
            )
    total = sum(referencing.values())
    if total and not force:
        exc = CategoryInUseError(
            f"'{category.name}' is used by {total} annotation(s) on {len(referencing)} "
            f"photo(s). Pass force=True to delete them too.",
            annotations=total, images=len(referencing),
        )
        yield RunFailed(error_code=exc.code, message=str(exc), details=exc.details)
        return
    for k, image_id in enumerate(referencing, start=1):
        stored = project.load_annotations(image_id)
        project.save_annotations(
            image_id,
            [a for a in stored.annotations if a.category_id != category_id],
            expected_version=stored.version,
        )
        yield ProgressUpdated(
            current=k, total=len(referencing), phase="deleting",
            message=f"{k} of {len(referencing)} photo(s) rewritten",
        )
    project.set_categories([c for c in project.categories if c.id != category_id])
    logger.info("deleted category %d and %d annotation(s)", category_id, total)
    yield RunCompleted(result={
        "cancelled": False, "deleted_annotations": total, "images": len(referencing),
        "category": category.name,
    })


@capability(
    "labels.delete",
    summary="Delete a class; force=True deletes its annotations too",
    web_route="/api/v1/categories/<int:category_id>",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="Label management is interactive; scripts edit via the Python API.",
)
def delete_category(project: Project, category_id: int, *, force: bool = False) -> int:
    """Remove a category. Returns how many annotations were deleted with it.

    Without force, deletion is refused while any annotation references the
    category — never silently orphan or reassign labels. With force=True the
    referencing annotations are deleted too (each image's version bumps, so
    concurrent annotator sessions see a conflict instead of stale state).
    Synchronous form of `delete_category_events`."""
    last = None
    for event in delete_category_events(project, category_id, force=force):
        last = event
    if last is None or last.type == "failed":
        if last is not None and last.error_code == "category_in_use":
            raise CategoryInUseError(
                last.message, annotations=last.details.get("annotations", 0),
                images=last.details.get("images", 0),
            )
        raise ProjectError(getattr(last, "message", "deletion produced no result"))
    return int(last.result.get("deleted_annotations", 0))


@capability(
    "labels.delete_job",
    summary="Delete a class as a background job with progress (poll via jobs.status)",
    web_route="/api/v1/categories/<int:category_id>/delete",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Label management is interactive; scripts call delete_category.",
)
def start_delete_category_job(project: Project, category_id: int, *, force: bool = False) -> str:
    """The deletion as a job: the page shows a bar and blocks itself while
    the scan and the rewrites run. Returns the job id."""
    from horos.api.jobs import start_job

    _find(project, category_id)  # an unknown id fails synchronously
    return start_job(
        project, "labels-delete",
        lambda cancel: delete_category_events(project, category_id, force=force, cancel=cancel),
    )


@capability(
    "labels.merge",
    summary="Merge classes: relabel their annotations to a target class and remove them",
    web_route="/api/v1/categories/merge",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Label management is interactive; scripts edit via the Python API.",
)
def merge_categories(
    project: Project, source_ids: list[int], target_id: int
) -> MergeResult:
    """Fold one or more source classes into `target_id`.

    Every annotation of a source class is relabelled to the target (boxes and
    polygons are kept as they are — nothing is deleted or de-duplicated), the
    source classes are removed, and each touched image's version bumps so a
    concurrent annotator sees a conflict instead of stale labels. The target
    keeps its id, name and color, so runs and exports referring to it stay
    valid.
    """
    target = _find(project, target_id)
    sources: list[int] = []
    for raw in source_ids:
        cid = int(raw)
        if cid == target_id:
            raise ProjectError(
                f"Cannot merge category {cid} into itself — pick a different target"
            )
        _find(project, cid)
        if cid not in sources:
            sources.append(cid)
    if not sources:
        raise ProjectError("merge_categories needs at least one source category")

    merged = touched = 0
    for record in project.list_images():
        stored = project.load_annotations(record.id)
        if not any(a.category_id in sources for a in stored.annotations):
            continue
        relabelled = [
            a.model_copy(update={"category_id": target_id}) if a.category_id in sources else a
            for a in stored.annotations
        ]
        merged += sum(1 for a in stored.annotations if a.category_id in sources)
        touched += 1
        project.save_annotations(record.id, relabelled, expected_version=stored.version)
    # the merged classes' names (and their own aliases) keep pointing at the
    # target, so an older model's output for them lands on the target
    absorbed = [n for c in project.categories if c.id in sources for n in (c.name, *c.aliases)]
    aliases = list(target.aliases) + [
        n for n in absorbed if n not in target.aliases and n != target.name
    ]
    target = target.model_copy(update={"aliases": aliases})
    project.set_categories([
        target if c.id == target_id else c for c in project.categories if c.id not in sources
    ])
    logger.info(
        "merged categories %s into %d (%d annotation(s) across %d image(s))",
        sources, target_id, merged, touched,
    )
    return MergeResult(
        target=target, merged_annotations=merged, images_touched=touched, removed_ids=sources
    )
