"""Dataset operations — the public Python API (R2: business logic lives here)."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core import formats
from horos.core.dataset import Category, Dataset, default_color
from horos.core.formats import coco as coco_format
from horos.core.formats import darknet as darknet_format
from horos.core.formats import images as images_format
from horos.core.formats import labelme as labelme_format
from horos.core.formats import via as via_format
from horos.core.formats import voc as voc_format
from horos.core.formats import yolo as yolo_format
from horos.core.project import Project
from horos.core.stats import DatasetStats, compute_stats
from horos.core.validate import (
    ValidationReport,
    clamp_fix,
    polygon_bbox_fix,
    validate_dataset,
)
from horos.errors import (
    ClassNamesRequiredError,
    DatasetFormatError,
    ImportConflictError,
    LabelConflictError,
    ProjectError,
)

if TYPE_CHECKING:
    from horos.backends.base import Event

CONFLICT_POLICIES = ("ask", "overwrite", "skip", "rename")
#: what to do when the import brings labels for a photo that already has some
ANNOTATION_POLICIES = ("ask", "replace", "merge", "skip")

#: R4 progress sink for import: receives ProgressUpdated (and WarningRaised)
#: events; import_dataset itself emits no started/completed — the caller that
#: owns the run (a job, the CLI) frames the stream.
ProgressCallback = Callable[["Event"], None]

logger = logging.getLogger(__name__)


class _Reporter:
    """Throttled phase/tick emitter so a 5k-image import does not produce 5k
    events per phase: at most ~50 ticks per phase plus the final one."""

    def __init__(self, progress: ProgressCallback | None):
        self._progress = progress
        self._phase = ""
        self._total: int | None = None
        self._step = 1

    def phase(self, name: str, total: int | None = None, message: str = "") -> None:
        self._phase, self._total = name, total
        self._step = max(1, (total or 0) // 50)
        self._emit(0, message)

    def tick(self, current: int, message: str = "") -> None:
        if current == self._total or current % self._step == 0:
            self._emit(current, message)

    def warn(self, message: str) -> None:
        if self._progress is not None:
            from horos.backends.base import WarningRaised

            self._progress(WarningRaised(message=message))

    def _emit(self, current: int, message: str) -> None:
        if self._progress is None:
            return
        from horos.backends.base import ProgressUpdated

        self._progress(
            ProgressUpdated(
                current=current, total=self._total, phase=self._phase, message=message
            )
        )

__all__ = [
    "ImportSummary",
    "import_dataset",
    "import_zip",
    "export_dataset",
    "convert_dataset",
    "validate_project",
    "fix_validation_issues",
    "delete_images",
    "dataset_stats",
    "resplit",
    "list_images",
]


class ImportSummary(BaseModel):
    format: str
    num_images: int
    num_annotations: int
    num_categories: int
    instances_per_category: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    #: same name AND same content as an existing image — skipped automatically
    duplicates_skipped: int = 0
    #: file names that already existed with different content
    conflict_files: list[str] = Field(default_factory=list)
    #: how the conflicts were resolved (per the on_conflict policy)
    overwritten: int = 0
    conflicts_skipped: int = 0
    renamed: int = 0
    #: photos already in the project whose labels came from this import — the
    #: file was not re-imported (same name and content, or not in the source
    #: at all), only its annotations
    images_matched: int = 0
    #: names of matched photos that already carried labels (the on_annotations
    #: decision applies to exactly these)
    annotation_conflict_files: list[str] = Field(default_factory=list)
    annotations_replaced: int = 0
    annotations_merged: int = 0
    #: matched photos whose existing labels were left alone (on_annotations="skip")
    annotations_kept: int = 0


def _read_any(
    source: Path, format: str | None, *, class_names: list[str] | None = None
) -> tuple[str, Dataset, dict[int, Path]]:
    detected = format or formats.detect_format(source)
    if detected is None:
        raise DatasetFormatError(
            f"Could not detect a supported dataset format under {source}. Expected "
            f"a COCO '_annotations.coco.json', a YOLO 'data.yaml', Pascal VOC "
            f"<annotation> XML files, Darknet label .txt files next to images, "
            f"a VIA 'via_region_data.json', or LabelMe per-image JSON files."
        )
    if detected == "coco":
        dataset, image_paths = coco_format.read_coco(source)
    elif detected == "yolo":
        dataset, image_paths = yolo_format.read_yolo(source)
    elif detected == "voc":
        dataset, image_paths = voc_format.read_voc(source)
    elif detected == "darknet":
        dataset, image_paths = darknet_format.read_darknet(source, class_names=class_names)
    elif detected == "via":
        dataset, image_paths = via_format.read_via(source, class_names=class_names)
    elif detected == "labelme":
        dataset, image_paths = labelme_format.read_labelme(source)
    elif detected == "images":
        dataset, image_paths = images_format.read_images(source)
    else:
        raise DatasetFormatError(f"Unsupported dataset format '{detected}'")
    return detected, dataset, image_paths


def _assign_splits(
    ids: list[int], *, train: float, valid: float, test: float, seed: int
) -> dict[int, str]:
    """Shuffle-and-cut split assignment shared by resplit and import."""
    import random

    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * train)
    n_valid = round(n * valid)
    assignment: dict[int, str] = {}
    for pos, image_id in enumerate(ids):
        if pos < n_train:
            assignment[image_id] = "train"
        elif pos < n_train + n_valid:
            assignment[image_id] = "valid"
        else:
            assignment[image_id] = "test"
    return assignment


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


@capability(
    "dataset.import",
    summary="Import a COCO / YOLO / VOC / Darknet / VIA / LabelMe dataset, or plain photos "
            "(format auto-detected)",
    web_route="/api/v1/dataset/import",
    web_methods=("POST",),
    cli="import",
)
def import_dataset(
    project: Project,
    source: Path | str,
    *,
    format: str | None = None,
    copy_images: bool = True,
    boundary: Path | None = None,
    on_conflict: str = "ask",
    on_annotations: str = "ask",
    class_names: list[str] | None = None,
    require_class_names: bool = False,
    progress: ProgressCallback | None = None,
) -> ImportSummary:
    """Import a dataset directory (or annotation file) into the project.

    A directory of photos with no annotation file is a legitimate import
    (format "images"): the photos join the project unlabeled and in no set,
    ready for the annotator or the loop's pool.

    Categories are merged by name with any the project already has. Images are
    copied into the project by default; copy_images=False stores absolute-path
    references instead. When `boundary` is given, any annotation-referenced
    file resolving outside that directory is refused — import_zip passes the
    extraction dir so an uploaded archive cannot pull in server-side files.

    File-name conflicts: an incoming image whose name already exists in the
    project with identical content is skipped automatically; with different
    content it is resolved per `on_conflict` — "ask" (default) raises
    ImportConflictError listing the names without touching the project,
    "overwrite" replaces the image and its annotations, "skip" keeps the
    existing one, "rename" imports under an auto-suffixed name.

    Labels for a photo the project already has are applied to that photo
    instead of being dropped with the duplicate — what uploading a label file
    for photos uploaded earlier looks like. A photo matches when the source
    ships the same file (same name and content), or names it without shipping
    it (a bare annotation file) and the recorded size agrees. When the matched
    photo already carries labels, `on_annotations` decides: "ask" (default)
    raises LabelConflictError listing the names and writes nothing, "replace"
    swaps the labels, "merge" keeps both sets, "skip" leaves the photo alone.

    `class_names` supplies Darknet class names when no _darknet.labels exists
    (placeholder index names plus a warning otherwise); `require_class_names`
    makes that case raise ClassNamesRequiredError instead — the WebUI upload
    path uses it to show an editable name list.

    `progress` receives R4 ProgressUpdated events (phases: reading annotations,
    checking for duplicates, copying images, saving annotations, applying
    default split) and WarningRaised for reader warnings, throttled to ~50
    ticks per phase.
    """
    if on_conflict not in CONFLICT_POLICIES:
        raise ProjectError(f"on_conflict must be one of {CONFLICT_POLICIES}")
    if on_annotations not in ANNOTATION_POLICIES:
        raise ProjectError(f"on_annotations must be one of {ANNOTATION_POLICIES}")
    source = Path(source)
    if not source.exists():
        raise DatasetFormatError(f"Dataset source does not exist: {source}")
    report = _Reporter(progress)
    report.phase("reading annotations", message=format or "detecting format")
    detected, dataset, image_paths = _read_any(source, format, class_names=class_names)
    report.phase(
        "reading annotations",
        message=f"{detected}: {len(dataset.images)} images, "
        f"{len(dataset.annotations)} annotations",
    )
    pre_warnings: list[str] = list(dataset.reader_warnings)
    for message in pre_warnings:
        report.warn(message)
    if (
        detected == "darknet"
        and class_names is None
        and darknet_format.find_labels_file(source) is None
    ):
        if require_class_names:
            # raised before the empty-category drop below: the name list is
            # positional (index -> name), so it must cover every index
            raise ClassNamesRequiredError(
                "This Darknet dataset has no _darknet.labels file — provide "
                "class names to import it.",
                default_names=[c.name for c in dataset.categories],
            )
        pre_warnings.append(
            f"No _darknet.labels found — class indices 0..{len(dataset.categories) - 1} "
            f"used as class names; rename them later or re-import with class_names"
        )
    if (
        detected == "via"
        and class_names is None
        and require_class_names
        and not via_format.has_class_attributes(source)
    ):
        # same prompt flow as Darknet: the WebUI shows an editable name list
        raise ClassNamesRequiredError(
            "This VIA dataset has no class attribute on its regions — name "
            "the class to import it.",
            default_names=[c.name for c in dataset.categories],
        )

    # drop categories nothing references (e.g. Roboflow's auto-added empty
    # supercategory) — they would pollute the class list and stats forever
    used_category_ids = {ann.category_id for ann in dataset.annotations}
    empty_categories = [c for c in dataset.categories if c.id not in used_category_ids]
    if empty_categories:
        dataset.categories = [
            c for c in dataset.categories if c.id in used_category_ids
        ]
        kept_names = {c.name for c in dataset.categories}
        gone = sorted({c.name for c in empty_categories} - kept_names)
        label = f"categor{'y' if len(empty_categories) == 1 else 'ies'}"
        pre_warnings.append(
            f"Dropped {len(empty_categories)} empty {label} with no annotations"
            + (f": {', '.join(gone)}" if gone else " (same-named supercategory)")
        )
    if boundary is not None:
        bound = boundary.resolve()
        for path in image_paths.values():
            if not path.is_relative_to(bound):
                raise DatasetFormatError(
                    f"Dataset references a file outside the archive: {path}"
                )

    # merge categories by name
    categories = list(project.categories)
    id_by_name = {c.name: c.id for c in categories}
    category_map: dict[int, int] = {}
    for cat in dataset.categories:
        if cat.name not in id_by_name:
            new_id = max((c.id for c in categories), default=0) + 1
            categories.append(
                Category(id=new_id, name=cat.name, color=default_color(len(categories)))
            )
            id_by_name[cat.name] = new_id
        category_map[cat.id] = id_by_name[cat.name]
    project.set_categories(categories)

    # conflict prescan — nothing is written until every decision is known
    existing_by_name = {r.file_name: r for r in project.list_images()}
    actions: dict[int, str] = {}  # image.id -> duplicate | overwrite | skip | rename
    conflict_files: list[str] = []
    #: incoming image id -> existing record id, for photos the project already
    #: has that this import brings labels for (the photo itself is not copied)
    matched: dict[int, int] = {}
    size_mismatch: list[str] = []
    labels_by_image: dict[int, list] = {}
    for ann in dataset.annotations:
        labels_by_image.setdefault(ann.image_id, []).append(ann)
    report.phase("checking for duplicates", total=len(dataset.images))
    for position, image in enumerate(dataset.images, start=1):
        report.tick(position)
        src = image_paths.get(image.id)
        existing = existing_by_name.get(image.file_name)
        if existing is None:
            continue
        brings_labels = bool(labels_by_image.get(image.id))
        if src is None or not src.exists():
            # a bare label file naming a photo uploaded earlier: no bytes to
            # compare, so the recorded size is the guard against a namesake
            if not brings_labels:
                continue
            if (image.width, image.height) != (existing.width, existing.height):
                size_mismatch.append(
                    f"'{image.file_name}' ({image.width}x{image.height})"
                    f" vs the project's photo ({existing.width}x{existing.height})"
                )
                continue
            matched[image.id] = existing.id
        elif _sha256(src) == _sha256(project.image_path(existing)):
            if brings_labels:
                matched[image.id] = existing.id
            else:
                actions[image.id] = "duplicate"
        else:
            conflict_files.append(image.file_name)
            actions[image.id] = on_conflict
    if conflict_files and on_conflict == "ask":
        raise ImportConflictError(
            f"{len(conflict_files)} image(s) already exist with different content: "
            f"{', '.join(conflict_files[:10])}"
            f"{' …' if len(conflict_files) > 10 else ''}. Nothing was imported — "
            f"retry with on_conflict='overwrite', 'skip', or 'rename'.",
            conflicts=conflict_files,
        )
    # the same labels arriving again are a duplicate, not a conflict: the
    # identical zip uploaded twice must never prompt
    label_conflicts: list[str] = []
    for old_id, existing_id in list(matched.items()):
        current = project.load_annotations(existing_id).annotations
        if not current:
            continue
        if _same_labels(current, labels_by_image[old_id], category_map):
            del matched[old_id]
            actions[old_id] = "duplicate"
            continue
        label_conflicts.append(dataset.image_by_id(old_id).file_name)
    if label_conflicts and on_annotations == "ask":
        raise LabelConflictError(
            f"{len(label_conflicts)} photo(s) already have labels: "
            f"{', '.join(label_conflicts[:10])}"
            f"{' …' if len(label_conflicts) > 10 else ''}. Nothing was imported — "
            f"retry with on_annotations='replace', 'merge', or 'skip'.",
            conflicts=label_conflicts,
        )

    warnings = pre_warnings
    for detail in size_mismatch:
        warnings.append(
            f"Labels skipped for {detail}: the sizes disagree, so this is a "
            f"different photo with the same name"
        )
    # this index is edited image by image and saved at the end, so it must be
    # a private copy, never the project's shared cache (see _load_image_index)
    index = project._load_image_index(fresh=True)
    image_map: dict[int, int] = {}
    overwritten_ids: set[int] = set()
    duplicates_skipped = conflicts_skipped = renamed = 0
    report.phase("copying images" if copy_images else "registering images",
                 total=len(dataset.images))
    for position, image in enumerate(dataset.images, start=1):
        report.tick(position)
        if image.id in matched:
            continue  # the project has this photo; only its labels arrive
        src = image_paths.get(image.id)
        if src is None or not src.exists():
            warnings.append(
                f"Image file missing for '{image.file_name}' — record skipped"
            )
            continue
        action = actions.get(image.id)
        if action == "duplicate":
            duplicates_skipped += 1
            continue
        if action == "skip":
            conflicts_skipped += 1
            continue
        if action == "overwrite":
            record = project.replace_image(
                existing_by_name[image.file_name].id,
                src,
                width=image.width,
                height=image.height,
                split=image.split,
                copy=copy_images,
                _index=index,
            )
            overwritten_ids.add(record.id)
        else:  # new image, or conflict resolved by rename (add_image auto-suffixes)
            record = project.add_image(
                src,
                width=image.width,
                height=image.height,
                split=image.split,
                copy=copy_images,
                _index=index,
            )
            if action == "rename":
                renamed += 1
        image_map[image.id] = record.id
    project._save_image_index(index)

    imported_annotations = 0
    instances: dict[str, int] = {}
    replaced = merged = kept = 0
    #: every photo this import writes labels to: the new ones and the matched
    targets = {**image_map, **matched}
    report.phase("saving annotations", total=len(targets))
    for position, (old_image_id, new_image_id) in enumerate(targets.items(), start=1):
        report.tick(position)
        current = project.load_annotations(new_image_id)
        is_match = old_image_id in matched
        if is_match and current.annotations:
            if on_annotations == "skip":
                kept += 1
                continue
            if on_annotations == "replace":
                replaced += 1
            else:
                merged += 1
        # merged labels continue the existing numbering; everything else starts at 1
        base = list(current.annotations) if is_match and on_annotations == "merge" else []
        annotations = list(base)
        next_id = max((a.id for a in base), default=0) + 1
        for ann in labels_by_image.get(old_image_id, []):
            new_cat = category_map[ann.category_id]
            annotations.append(
                ann.model_copy(
                    update={
                        "id": next_id,
                        "image_id": new_image_id,
                        "category_id": new_cat,
                    }
                )
            )
            next_id += 1
            name = next(c.name for c in categories if c.id == new_cat)
            instances[name] = instances.get(name, 0) + 1
            imported_annotations += 1
        if annotations or new_image_id in overwritten_ids:
            # an overwritten image must not keep its old annotations, so an
            # empty incoming set still saves
            project.save_annotations(
                new_image_id, annotations, expected_version=current.version,
                assign_split=False,  # one assignment pass below, not one per photo
            )

    # photos whose source named no split join one now if they are labeled —
    # by the project's stable hash and ratios; unlabeled ones stay in no set
    # until they are labeled (core/splitting.py)
    report.phase("assigning splits", total=len(targets))
    assigned = project.assign_splits(sorted(targets.values()))
    back = {new: old for old, new in targets.items()}
    for new_id, split in assigned.items():
        dataset.image_by_id(back[new_id]).split = split  # type: ignore[assignment]
    # a source directory may have named a split for photos it never labeled;
    # only labeled photos are set members, so those leave it
    by_id = {r.id: r for r in project.list_images()}
    cleared = {
        new_id: None for new_id in image_map.values()
        if by_id[new_id].split is not None and not project._has_confirmed(new_id)
    }
    if cleared:
        project.update_image_splits(cleared)
        for new_id in cleared:
            dataset.image_by_id(back[new_id]).split = None
    unassigned = sum(1 for img in dataset.images if img.id in image_map and img.split is None)
    if assigned:
        r = project.split_ratios
        warnings.append(
            f"{len(assigned)} labeled photo(s) had no split in the source — assigned "
            f"train/valid/test {r.train:g}/{r.valid:g}/{r.test:g} by stable hash"
        )
    if unassigned:
        warnings.append(
            f"{unassigned} photo(s) have no labels and are in no split yet; each joins "
            f"one when it is first labeled"
        )

    split_counts: dict[str, int] = {}
    for image in dataset.images:
        if image.id in image_map:
            key = image.split or "unassigned"
            split_counts[key] = split_counts.get(key, 0) + 1

    logger.info(
        "imported %s dataset from %s: %d images, %d annotations, %d photos matched",
        detected, source, len(image_map), imported_annotations, len(matched),
    )
    return ImportSummary(
        format=detected,
        num_images=len(image_map),
        num_annotations=imported_annotations,
        num_categories=len(categories),
        instances_per_category=instances,
        split_counts=split_counts,
        warnings=warnings,
        duplicates_skipped=duplicates_skipped,
        conflict_files=conflict_files,
        overwritten=len(overwritten_ids),
        conflicts_skipped=conflicts_skipped,
        renamed=renamed,
        images_matched=len(matched),
        annotation_conflict_files=label_conflicts,
        annotations_replaced=replaced,
        annotations_merged=merged,
        annotations_kept=kept,
    )


def _label_key(ann, category_id: int) -> tuple:
    return (
        category_id,
        tuple(round(v, 3) for v in ann.bbox),
        tuple(tuple(round(v, 3) for v in poly) for poly in ann.segmentation),
        ann.iscrowd,
        ann.source,
        ann.status,
    )


def _same_labels(existing, incoming, category_map: dict[int, int]) -> bool:
    """Do the incoming labels (in source category ids) equal the stored ones?
    Ids are ignored — they are renumbered on every import."""
    if len(existing) != len(incoming):
        return False
    have = sorted(_label_key(a, a.category_id) for a in existing)
    want = sorted(_label_key(a, category_map[a.category_id]) for a in incoming)
    return have == want


def _safe_extract(
    zip_path: Path, target: Path, progress: ProgressCallback | None = None
) -> None:
    report = _Reporter(progress)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for member in members:
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise DatasetFormatError(
                    f"Zip contains an unsafe path: {member.filename!r}"
                )
        report.phase("extracting", total=len(members), message=zip_path.name)
        for position, member in enumerate(members, start=1):
            zf.extract(member, target)
            report.tick(position)


@capability(
    "dataset.import_zip",
    summary="Import a zipped dataset of any supported format (extract, then import)",
    not_web_because=(
        "The Web path stages the zip (dataset.stage_upload) and imports it as a "
        "polled job (dataset.import_upload) so the UI can show progress."
    ),
    cli=None,
    not_cli_because="The CLI imports directories directly via 'import'.",
)
def import_zip(
    project: Project,
    zip_path: Path | str,
    *,
    on_conflict: str = "ask",
    on_annotations: str = "ask",
    class_names: list[str] | None = None,
    require_class_names: bool = False,
    progress: ProgressCallback | None = None,
) -> ImportSummary:
    """Extract a dataset zip to a temp dir and import it (always copies).
    `progress` gets an "extracting" phase first, then import_dataset's phases."""
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise DatasetFormatError(f"{zip_path} is not a valid zip archive")
    with tempfile.TemporaryDirectory(prefix="horos_upload_") as tmp:
        _safe_extract(zip_path, Path(tmp), progress)
        return import_dataset(
            project,
            Path(tmp),
            copy_images=True,
            boundary=Path(tmp),
            on_conflict=on_conflict,
            on_annotations=on_annotations,
            class_names=class_names,
            require_class_names=require_class_names,
            progress=progress,
        )


def filter_dataset_categories(
    dataset: Dataset, names: list[str], *, include_background: bool = False
) -> Dataset:
    """Keep only the named categories and their annotations.

    Images with no annotation of a selected class are dropped by default, so
    "3 of 4 classes" really means a smaller dataset (and the image-count
    rules — epochs, augmentation, patience — see that). With
    `include_background=True` every image stays and those pictures become
    negatives: their other-class objects are background as far as this
    dataset is concerned."""
    known = {c.name for c in dataset.categories}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ProjectError(
            f"Unknown categor{'y' if len(unknown) == 1 else 'ies'} "
            f"{unknown} — available: {sorted(known)}"
        )
    if not names:
        raise ProjectError("Select at least one category")
    keep_ids = {c.id for c in dataset.categories if c.name in names}
    annotations = [a for a in dataset.annotations if a.category_id in keep_ids]
    images = dataset.images
    if not include_background:
        with_objects = {a.image_id for a in annotations}
        images = [i for i in dataset.images if i.id in with_objects]
    return dataset.model_copy(
        update={
            "categories": [c for c in dataset.categories if c.id in keep_ids],
            "images": images,
            "annotations": annotations,
        }
    )


def subset_dataset(
    dataset: Dataset,
    *,
    image_ids: list[int] | None = None,
    confirmed_only: bool = False,
) -> Dataset:
    """Restrict a snapshot to `image_ids` (None = all) and, with
    `confirmed_only`, drop pending pre-labels — a training run must never
    learn from geometry nobody reviewed (E10-T8/E10-T11). Categories are
    kept intact so class ids stay stable across rounds."""
    keep = None if image_ids is None else set(image_ids)
    images = [i for i in dataset.images if keep is None or i.id in keep]
    present = {i.id for i in images}
    annotations = [
        a for a in dataset.annotations
        if a.image_id in present and (not confirmed_only or a.status == "confirmed")
    ]
    return Dataset(
        categories=list(dataset.categories), images=images, annotations=annotations,
        reader_warnings=list(dataset.reader_warnings),
    )


@capability(
    "dataset.export",
    summary="Export the project's dataset as COCO, YOLO or LabelMe",
    web_route="/api/v1/dataset/export",
    web_methods=("POST",),
    cli="export",
)
def export_dataset(
    project: Project,
    out_dir: Path | str,
    *,
    format: str = "coco",
    categories: list[str] | None = None,
    include_background: bool = False,
    image_ids: list[int] | None = None,
    confirmed_only: bool = False,
) -> Path:
    """Write the project dataset to `out_dir` in the requested format,
    optionally restricted to the named categories (see
    filter_dataset_categories for `include_background`), to `image_ids`,
    and to confirmed annotations only (see subset_dataset)."""
    dataset = subset_dataset(
        project.to_dataset(), image_ids=image_ids, confirmed_only=confirmed_only
    )
    if categories is not None:
        dataset = filter_dataset_categories(
            dataset, categories, include_background=include_background
        )
    image_paths = {i.id: project.image_path(i) for i in dataset.images}
    out_dir = Path(out_dir)
    if format == "coco":
        written = coco_format.write_coco(
            dataset, out_dir, image_paths=image_paths, copy_images=True
        )
        return written[0].parent.parent if len(written) > 1 else written[0]
    if format == "yolo":
        return yolo_format.write_yolo(dataset, out_dir, image_paths=image_paths)
    if format == "labelme":
        return labelme_format.write_labelme(dataset, out_dir, image_paths=image_paths)
    raise DatasetFormatError(
        f"Unsupported export format '{format}' ({'|'.join(formats.WRITABLE_FORMATS)})"
    )


@capability(
    "dataset.convert",
    summary="Convert a dataset to COCO, YOLO or LabelMe without creating a project",
    not_web_because="Server-side path-to-path conversion is a CLI/scripting concern.",
    cli="convert",
)
def convert_dataset(
    source: Path | str,
    out_dir: Path | str,
    *,
    to_format: str,
    from_format: str | None = None,
) -> Path:
    """One-shot format conversion (E1-S2): read `source`, write to `out_dir`."""
    detected, dataset, image_paths = _read_any(Path(source), from_format)
    if to_format == detected:
        raise DatasetFormatError(f"Source is already in format '{to_format}'")
    if to_format == "coco":
        written = coco_format.write_coco(
            dataset, Path(out_dir), image_paths=image_paths, copy_images=True
        )
        return written[0].parent.parent if len(written) > 1 else written[0]
    if to_format == "yolo":
        return yolo_format.write_yolo(dataset, Path(out_dir), image_paths=image_paths)
    if to_format == "labelme":
        return labelme_format.write_labelme(dataset, Path(out_dir), image_paths=image_paths)
    raise DatasetFormatError(
        f"Unsupported target format '{to_format}' ({'|'.join(formats.WRITABLE_FORMATS)})"
    )


@capability(
    "dataset.validate",
    summary="Validate the project dataset and report structured issues",
    web_route="/api/v1/dataset/validation",
    web_methods=("GET",),
    cli="validate",
)
def validate_project(project: Project) -> ValidationReport:
    """Run the dataset validator over the project's current data (E1-T6)."""
    dataset = project.to_dataset()
    image_paths = {i.id: project.image_path(i) for i in dataset.images}
    return validate_dataset(dataset, image_paths=image_paths)


class DeleteImagesSummary(BaseModel):
    deleted: list[int]
    #: images skipped because another session holds an active annotation claim
    skipped_claimed: list[int]


@capability(
    "dataset.delete_images",
    summary="Delete images (and their annotations) from the project",
    web_route="/api/v1/images/delete",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Bulk deletion is interactive (grid selection); scripts use the Python API.",
)
def delete_images(
    project: Project, image_ids: list[int], *, session_id: str | None = None
) -> DeleteImagesSummary:
    """Remove images with their annotation files (project-owned image files are
    deleted; externally referenced ones only drop the reference). Images
    another session is actively annotating (an unexpired claim not held by
    `session_id`) are skipped and reported, never yanked out from under the
    annotator. Unknown ids fail the whole call before anything is deleted."""
    from horos.api.annotate import claims_held_by_others

    unique_ids = list(dict.fromkeys(image_ids))
    claimed = claims_held_by_others(project, session_id)
    doomed = [i for i in unique_ids if i not in claimed]
    skipped = [i for i in unique_ids if i in claimed]
    removed = project.remove_images(doomed) if doomed else []
    logger.info(
        "deleted %d image(s); skipped %d claimed by other sessions",
        len(removed), len(skipped),
    )
    return DeleteImagesSummary(
        deleted=[record.id for record in removed], skipped_claimed=skipped
    )


class ClearDatasetSummary(BaseModel):
    deleted_images: int
    deleted_annotations: int
    deleted_categories: int
    #: images left in place because another session is annotating them
    skipped_claimed: list[int] = Field(default_factory=list)


@capability(
    "dataset.clear",
    summary="Delete every image and annotation of the project (classes optional)",
    web_route="/api/v1/dataset",
    web_methods=("DELETE",),
    cli="clear",
)
def clear_dataset(
    project: Project,
    *,
    confirm: str,
    keep_categories: bool = True,
    session_id: str | None = None,
) -> ClearDatasetSummary:
    """Empty the dataset: every image (project-owned files deleted, external
    references dropped) with its annotations, and the class list unless
    `keep_categories`. Training runs keep their own snapshots and are not
    touched. `confirm` must equal the project name — the guard every caller
    (UI, CLI, scripts) has to pass for a call this destructive. Images
    another session is annotating right now are skipped, like delete_images."""
    if confirm != project.manifest.name:
        raise ProjectError(
            f"Refusing to clear the dataset: confirm must equal the project name "
            f"({project.manifest.name!r})."
        )
    records = project.list_images()
    annotations = sum(len(project.load_annotations(r.id).annotations) for r in records)
    summary = delete_images(project, [r.id for r in records], session_id=session_id)
    deleted_annotations = annotations
    if summary.skipped_claimed:
        kept = sum(
            len(project.load_annotations(i).annotations) for i in summary.skipped_claimed
        )
        deleted_annotations -= kept
    deleted_categories = 0
    if not keep_categories:
        if summary.skipped_claimed:
            logger.info("class list kept: %d image(s) still annotated by others",
                        len(summary.skipped_claimed))
        else:
            deleted_categories = len(project.categories)
            project.set_categories([])
    logger.info(
        "cleared dataset: %d image(s), %d annotation(s), %d class(es)",
        len(summary.deleted), deleted_annotations, deleted_categories,
    )
    return ClearDatasetSummary(
        deleted_images=len(summary.deleted),
        deleted_annotations=deleted_annotations,
        deleted_categories=deleted_categories,
        skipped_claimed=summary.skipped_claimed,
    )


class FixedBox(BaseModel):
    image_id: int
    annotation_id: int
    before: tuple[float, float, float, float]
    after: tuple[float, float, float, float]


class ValidationFixResult(BaseModel):
    num_fixed: int
    fixed: list[FixedBox]
    #: the validation state after fixing — what remains needs a human
    report: ValidationReport


@capability(
    "dataset.validate_fix",
    summary="Repair auto-fixable boxes: clamp edge overshoots, refit boxes to their polygons",
    web_route="/api/v1/dataset/validation/fix",
    web_methods=("POST",),
    cli="validate",  # exposed as `horos validate --fix`
)
def fix_validation_issues(project: Project) -> ValidationFixResult:
    """Repair every issue the validator marked `fixable`: boxes past the image
    edge by at most FIXABLE_OVERSHOOT pixels are clamped back in (polygons
    included), and a bbox that drifted from the polygons it belongs to is
    recomputed from them. Uses exactly the validator's `clamp_fix` /
    `polygon_bbox_fix` decisions, so the set of repairs equals the set of
    `fixable` issues in the report — larger overshoots and fragment polygons
    are left alone for a human (E1-S4: never a silent pass).

    Each change is reported with the before/after box; writes go through the
    project's optimistic-locked per-image annotation files.
    """
    fixed: list[FixedBox] = []
    for record in project.list_images():
        stored = project.load_annotations(record.id)
        updated: list = []
        changed = False
        for ann in stored.annotations:
            before = ann.bbox
            refit = polygon_bbox_fix(ann)
            if refit is not None:
                ann = refit
            clamped = clamp_fix(ann, record.width, record.height)
            if clamped is not None:
                ann = clamped
            if ann.bbox != before:
                fixed.append(
                    FixedBox(
                        image_id=record.id, annotation_id=ann.id, before=before, after=ann.bbox
                    )
                )
                changed = True
            updated.append(ann)
        if changed:
            project.save_annotations(
                record.id, updated, expected_version=stored.version
            )
    return ValidationFixResult(
        num_fixed=len(fixed), fixed=fixed, report=validate_project(project)
    )


@capability(
    "dataset.stats",
    summary="Compute dataset statistics (feeds hyperparameter derivation)",
    web_route="/api/v1/dataset/stats",
    web_methods=("GET",),
    cli="stats",
)
def dataset_stats(
    project: Project,
    *,
    categories: list[str] | None = None,
    include_background: bool = False,
    image_ids: list[int] | None = None,
    confirmed_only: bool = False,
) -> DatasetStats:
    """Class distribution, relative object area, image sizes, splits (E1-T7).

    With `categories`, the statistics describe the data a run on those classes
    would train on (the Train page shows them live as classes are toggled);
    `image_ids` / `confirmed_only` narrow it the way a loop round's training
    snapshot is narrowed (E10-T8)."""
    dataset = subset_dataset(
        project.to_dataset(), image_ids=image_ids, confirmed_only=confirmed_only
    )
    if categories is not None:
        dataset = filter_dataset_categories(
            dataset, categories, include_background=include_background
        )
    return compute_stats(dataset)


@capability(
    "dataset.resplit",
    summary="Set the train/valid/test ratios and give labeled photos without a split "
            "their set; reshuffle=True re-draws every labeled photo",
    web_route="/api/v1/dataset/split",
    web_methods=("POST",),
    cli="split",
)
def resplit(
    project: Project,
    *,
    train: float | None = None,
    valid: float | None = None,
    test: float | None = None,
    seed: int | None = None,
    reshuffle: bool = False,
) -> dict[str, int]:
    """Only labeled photos belong to a set. The ratios (and hash seed) given
    here become the project's policy; then every labeled photo that has no
    split yet gets one by the stable hash. With `reshuffle`, every labeled
    photo is re-drawn at random under `seed` instead — that moves photos
    past models trained on into test, so callers must warn before using it.
    Unlabeled photos are never assigned. No symlinks: the split is an
    attribute on the image record (R7)."""
    from horos.core.splitting import SplitRatios

    current = project.split_ratios
    try:
        ratios = SplitRatios(
            train=current.train if train is None else train,
            valid=current.valid if valid is None else valid,
            test=current.test if test is None else test,
        )
    except ValueError as exc:
        raise ProjectError(f"Invalid split ratios: {exc}") from exc
    project.set_split_policy(ratios, seed)
    images = project.list_images()
    if not images:
        raise ProjectError("Project has no images to split")
    if reshuffle:
        labeled = [i.id for i in images if project._has_confirmed(i.id)]
        assignment: dict[int, str | None] = dict(_assign_splits(
            labeled, train=ratios.train, valid=ratios.valid, test=ratios.test,
            seed=project.manifest.split_seed,
        ))
        for image in images:
            if image.id not in assignment:
                assignment[image.id] = None  # unlabeled: in no set
        project.update_image_splits(assignment)
    else:
        project.assign_splits()
    counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0, "unassigned": 0}
    for image in project.list_images():
        counts[image.split or "unassigned"] += 1
    return counts


@capability(
    "dataset.images",
    summary="List the project's images with size and split",
    web_route="/api/v1/images",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Covered by 'stats'; a raw image list is a UI need.",
)
def list_images(project: Project):
    """All image records in the project."""
    return project.list_images()
