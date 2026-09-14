"""Category (label) management — E2-T4. Business logic lives here (R2)."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.dataset import Category, default_color
from horos.core.project import Project
from horos.errors import CategoryInUseError, ProjectError

logger = logging.getLogger(__name__)

__all__ = ["MergeResult", "add_category", "update_category", "delete_category", "merge_categories"]


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
    updated = cat.model_copy(
        update={
            **({"name": name} if name is not None else {}),
            **({"color": color} if color is not None else {}),
        }
    )
    project.set_categories(
        [updated if c.id == category_id else c for c in project.categories]
    )
    return updated


@capability(
    "labels.delete",
    summary="Delete a category (refused while annotations reference it, unless forced)",
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
    """
    category = _find(project, category_id)
    referencing: dict[int, int] = {}  # image_id -> count
    for record in project.list_images():
        stored = project.load_annotations(record.id)
        hits = sum(1 for a in stored.annotations if a.category_id == category_id)
        if hits:
            referencing[record.id] = hits
    total = sum(referencing.values())
    if total and not force:
        raise CategoryInUseError(
            f"'{category.name}' is used by {total} annotation(s) on {len(referencing)} "
            f"photo(s). Pass force=True to delete them too.",
            annotations=total, images=len(referencing),
        )
    for image_id in referencing:
        stored = project.load_annotations(image_id)
        project.save_annotations(
            image_id,
            [a for a in stored.annotations if a.category_id != category_id],
            expected_version=stored.version,
        )
    project.set_categories([c for c in project.categories if c.id != category_id])
    logger.info("deleted category %d and %d annotation(s)", category_id, total)
    return total


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
    project.set_categories([c for c in project.categories if c.id not in sources])
    logger.info(
        "merged categories %s into %d (%d annotation(s) across %d image(s))",
        sources, target_id, merged, touched,
    )
    return MergeResult(
        target=target, merged_annotations=merged, images_touched=touched, removed_ids=sources
    )
