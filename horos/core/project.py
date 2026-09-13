"""Project: the single authority over on-disk structure (E1-T1).

Layout (all paths via pathlib, R7):

    <root>/
      horos.json          project manifest: name, version, categories
      images.json         image index: records + next id
      images/             copied image files (project owns its data by default)
      annotations/        one JSON per image: {"image_id", "version", "annotations"}
      runs/               training runs (E5/E7)

Per-image annotation files are the concurrency unit (E2-T8): writes are
optimistic-locked on a version counter and applied with an atomic replace —
no fcntl, works on Windows (R7).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from pydantic import BaseModel, Field

from horos.core.dataset import Annotation, Category, ImageRecord, default_color
from horos.core.fsutil import atomic_write_text
from horos.errors import AnnotationConflictError, ProjectError

MANIFEST_NAME = "horos.json"


def occupied_by(root: Path | str) -> list[str]:
    """Names that make `root` unusable as a new project directory.

    Dotfiles do not count: `git init` (or an editor's dot-directory) must not
    force the project into a pointless nested subdirectory. The guard exists to
    protect a directory of real files from gaining images/, annotations/ and
    runs/, and dot-entries are not that."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
IMAGE_INDEX_NAME = "images.json"
STRUCTURE_VERSION = 1


class ProjectManifest(BaseModel):
    name: str
    structure_version: int = STRUCTURE_VERSION
    created_at: float = Field(default_factory=time.time)
    categories: list[Category] = Field(default_factory=list)


class ImageIndex(BaseModel):
    next_image_id: int = 1
    images: list[ImageRecord] = Field(default_factory=list)


class AnnotationFile(BaseModel):
    image_id: int
    version: int = 0
    annotations: list[Annotation] = Field(default_factory=list)


def _write_json_atomic(path: Path, text: str) -> None:
    """Atomic write: per-writer tmp file + os.replace, retried through a
    Windows sharing violation (R7; see horos.core.fsutil)."""
    atomic_write_text(path, text)


class Project:
    def __init__(self, root: Path, manifest: ProjectManifest):
        self.root = Path(root)
        self.manifest = manifest

    # ------------------------------------------------------------------ paths
    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def image_path(self, record: ImageRecord) -> Path:
        if record.external_path:
            return Path(record.external_path)
        return self.images_dir / Path(record.file_name)

    def _annotation_path(self, image_id: int) -> Path:
        return self.annotations_dir / f"{image_id}.json"

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, root: Path | str, name: str | None = None) -> Project:
        root = Path(root)
        if (root / MANIFEST_NAME).exists():
            raise ProjectError(f"A horos project already exists at {root}")
        occupants = occupied_by(root)
        if occupants:
            listed = ", ".join(occupants[:5]) + (" …" if len(occupants) > 5 else "")
            raise ProjectError(
                f"Refusing to create a project in non-empty directory {root} "
                f"(contains {listed})"
            )
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "annotations", "runs"):
            (root / sub).mkdir()
        manifest = ProjectManifest(name=name or root.name)
        project = cls(root, manifest)
        project.save_manifest()
        project._save_image_index(ImageIndex())
        return project

    @classmethod
    def open(cls, root: Path | str) -> Project:
        root = Path(root)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.exists():
            raise ProjectError(f"No horos project at {root} (missing {MANIFEST_NAME})")
        try:
            manifest = ProjectManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise ProjectError(f"Corrupt project manifest at {manifest_path}: {exc}") from exc
        project = cls(root, manifest)
        project.validate_structure()
        return project

    def validate_structure(self) -> None:
        missing = [
            d
            for d in (self.images_dir, self.annotations_dir)
            if not d.is_dir()
        ]
        if missing:
            raise ProjectError(
                f"Project at {self.root} is missing directories: "
                + ", ".join(str(m) for m in missing)
            )
        if not (self.root / IMAGE_INDEX_NAME).exists():
            raise ProjectError(f"Project at {self.root} is missing {IMAGE_INDEX_NAME}")

    def save_manifest(self) -> None:
        _write_json_atomic(
            self.root / MANIFEST_NAME, self.manifest.model_dump_json(indent=2)
        )

    # ------------------------------------------------------------- categories
    @property
    def categories(self) -> list[Category]:
        return list(self.manifest.categories)

    def set_categories(self, categories: list[Category]) -> None:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        colored: list[Category] = []
        for i, cat in enumerate(categories):
            if cat.id in seen_ids:
                raise ProjectError(f"Duplicate category id {cat.id}")
            if cat.name in seen_names:
                raise ProjectError(f"Duplicate category name '{cat.name}'")
            seen_ids.add(cat.id)
            seen_names.add(cat.name)
            colored.append(
                cat if cat.color else cat.model_copy(update={"color": default_color(i)})
            )
        self.manifest.categories = colored
        self.save_manifest()

    # ------------------------------------------------------------------ images
    def _load_image_index(self) -> ImageIndex:
        path = self.root / IMAGE_INDEX_NAME
        return ImageIndex.model_validate_json(path.read_text(encoding="utf-8"))

    def _save_image_index(self, index: ImageIndex) -> None:
        _write_json_atomic(self.root / IMAGE_INDEX_NAME, index.model_dump_json(indent=2))

    def list_images(self) -> list[ImageRecord]:
        return self._load_image_index().images

    def get_image(self, image_id: int) -> ImageRecord:
        record = next((i for i in self.list_images() if i.id == image_id), None)
        if record is None:
            raise ProjectError(f"No image with id {image_id} in project {self.root}")
        return record

    def add_image(
        self,
        source: Path,
        *,
        width: int,
        height: int,
        split: str = "train",
        copy: bool = True,
        _index: ImageIndex | None = None,
    ) -> ImageRecord:
        """Register one image, copying the file into the project by default.

        With copy=False the project stores an absolute-path reference instead
        (fast, but the project breaks if the source moves). `_index` lets bulk
        importers batch the index write.
        """
        index = _index if _index is not None else self._load_image_index()
        file_name = self._free_file_name(source.name, index)
        record = ImageRecord(
            id=index.next_image_id,
            file_name=file_name,
            width=width,
            height=height,
            split=split,  # type: ignore[arg-type]
            external_path=None if copy else str(Path(source).resolve()),
        )
        if copy:
            shutil.copy2(source, self.images_dir / file_name)
        index.images.append(record)
        index.next_image_id += 1
        if _index is None:
            self._save_image_index(index)
        return record

    def replace_image(
        self,
        image_id: int,
        source: Path,
        *,
        width: int,
        height: int,
        split: str = "train",
        copy: bool = True,
        _index: ImageIndex | None = None,
    ) -> ImageRecord:
        """Overwrite an existing record's file and metadata, keeping its id and
        file_name (the import 'overwrite' conflict policy). The caller is
        responsible for replacing the image's annotations."""
        index = _index if _index is not None else self._load_image_index()
        record = next((r for r in index.images if r.id == image_id), None)
        if record is None:
            raise ProjectError(f"No image with id {image_id} to replace")
        record.width = width
        record.height = height
        record.split = split  # type: ignore[assignment]
        target = self.images_dir / record.file_name
        if copy:
            shutil.copy2(source, target)
            record.external_path = None
        else:
            record.external_path = str(Path(source).resolve())
            # a stale copied file would shadow the new reference
            target.unlink(missing_ok=True)
        if _index is None:
            self._save_image_index(index)
        return record

    def remove_images(self, image_ids: list[int]) -> list[ImageRecord]:
        """Delete images from the project: index entry, annotation file, and —
        for project-owned images — the copied file under images/. Externally
        referenced images (copy=False imports) keep their source file; only
        the reference is dropped. Unknown ids fail before anything is touched.
        """
        index = self._load_image_index()
        by_id = {record.id: record for record in index.images}
        missing = [i for i in image_ids if i not in by_id]
        if missing:
            raise ProjectError(
                f"No image(s) with id(s) {sorted(missing)} in project {self.root}"
            )
        removed: list[ImageRecord] = []
        doomed = set(image_ids)
        for image_id in image_ids:
            record = by_id[image_id]
            self._annotation_path(image_id).unlink(missing_ok=True)
            if not record.external_path:
                (self.images_dir / record.file_name).unlink(missing_ok=True)
            removed.append(record)
        index.images = [r for r in index.images if r.id not in doomed]
        self._save_image_index(index)
        return removed

    def set_excluded(self, image_ids: list[int], excluded: bool, *, note: str = "") -> int:
        """Mark images as skipped (unfit for training) or bring them back.
        Unknown ids fail before anything is written. Returns how many records
        changed state."""
        index = self._load_image_index()
        by_id = {record.id: record for record in index.images}
        missing = [i for i in image_ids if i not in by_id]
        if missing:
            raise ProjectError(
                f"No image(s) with id(s) {sorted(missing)} in project {self.root}"
            )
        changed = 0
        for image_id in image_ids:
            record = by_id[image_id]
            if record.excluded != excluded:
                changed += 1
            record.excluded = excluded
            record.exclude_note = note if excluded else ""
        self._save_image_index(index)
        return changed

    def update_image_splits(self, split_by_id: dict[int, str]) -> None:
        index = self._load_image_index()
        for record in index.images:
            if record.id in split_by_id:
                record.split = split_by_id[record.id]  # type: ignore[assignment]
        self._save_image_index(index)

    def _free_file_name(self, name: str, index: ImageIndex) -> str:
        taken = {i.file_name for i in index.images}
        candidate = name
        stem, dot, suffix = name.rpartition(".")
        n = 1
        while candidate in taken or (self.images_dir / candidate).exists():
            candidate = f"{stem}_{n}.{suffix}" if dot else f"{name}_{n}"
            n += 1
        return candidate

    # -------------------------------------------------------------- assembly
    def to_dataset(self, *, include_excluded: bool = False):
        """Assemble the in-memory Dataset snapshot (for stats/validation/
        export/training). Images an annotator skipped as unfit for training
        are left out unless `include_excluded` is set (E10-T16)."""
        from horos.core.dataset import Dataset

        index = self._load_image_index()
        images = [r for r in index.images if include_excluded or not r.excluded]
        annotations: list[Annotation] = []
        for record in images:
            annotations.extend(self.load_annotations(record.id).annotations)
        return Dataset(
            categories=self.manifest.categories,
            images=images,
            annotations=annotations,
        )

    # ------------------------------------------------------------- annotations
    def load_annotations(self, image_id: int) -> AnnotationFile:
        path = self._annotation_path(image_id)
        if not path.exists():
            return AnnotationFile(image_id=image_id)
        return AnnotationFile.model_validate_json(path.read_text(encoding="utf-8"))

    def save_annotations(
        self,
        image_id: int,
        annotations: list[Annotation],
        *,
        expected_version: int,
    ) -> AnnotationFile:
        """Optimistic-locked write (E2-T8 foundation).

        `expected_version` must equal the stored version; otherwise someone
        else wrote first and the caller gets AnnotationConflictError with the
        current state to re-base on.
        """
        current = self.load_annotations(image_id)
        if current.version != expected_version:
            raise AnnotationConflictError(
                f"Annotations for image {image_id} were modified by another session "
                f"(stored version {current.version}, expected {expected_version}). "
                f"Reload and reapply your changes."
            )
        updated = AnnotationFile(
            image_id=image_id,
            version=current.version + 1,
            annotations=annotations,
        )
        _write_json_atomic(
            self._annotation_path(image_id), updated.model_dump_json(indent=2)
        )
        return updated
