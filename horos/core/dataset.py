"""Dataset / Split / Annotation data models (E1-T2).

Pure pydantic — no I/O here. Formats (COCO/YOLO) serialize these; the Project
persists them. bbox convention throughout horos is COCO absolute-pixel xywh.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Split = Literal["train", "valid", "test"]
SPLITS: tuple[Split, ...] = ("train", "valid", "test")

#: How an annotation came to exist. Autolabel output starts as ("auto", "pending")
#: and must stay distinguishable from human work in the data model (E3-T5).
AnnotationSource = Literal["manual", "auto"]
AnnotationStatus = Literal["confirmed", "pending"]

_DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def default_color(index: int) -> str:
    return _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)]


class Category(BaseModel):
    id: int
    name: str
    color: str = ""
    #: former names of this class (renames, merged classes): a model trained
    #: before the rename still answers with them, and its output must land
    #: on this class rather than create a new one
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("category name must not be empty")
        return v


class ImageRecord(BaseModel):
    id: int
    file_name: str  # POSIX-style path relative to the project's images/ dir (R7)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    #: the set a labeled photo belongs to; None until it carries a confirmed
    #: annotation (see core/splitting.py) — unlabeled photos are in no set
    split: Split | None = None
    #: Absolute path for images referenced in place (import with copy=False).
    #: None for images owned by the project (the default).
    external_path: str | None = None
    #: skipped by an annotator as unfit for training (E10-T16): stays in the
    #: project and the annotator, but leaves the selection pool, the dataset
    #: snapshot, the statistics and every training run
    excluded: bool = False
    exclude_note: str = ""


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    #: COCO xywh, absolute pixels.
    bbox: tuple[float, float, float, float]
    #: List of polygons, each a flat [x1, y1, x2, y2, ...] in absolute pixels.
    segmentation: list[list[float]] = Field(default_factory=list)
    iscrowd: int = 0
    source: AnnotationSource = "manual"
    status: AnnotationStatus = "confirmed"
    #: Autolabel confidence when source == "auto"; None for manual annotations.
    score: float | None = None

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


class Dataset(BaseModel):
    """An in-memory dataset snapshot: the unit formats read and write."""

    categories: list[Category] = Field(default_factory=list)
    images: list[ImageRecord] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    #: Non-fatal notes a format reader wants surfaced to the user (e.g. data a
    #: format carries that horos does not import). Import copies these into
    #: ImportSummary.warnings — nothing is dropped silently.
    reader_warnings: list[str] = Field(default_factory=list)

    # -- lookups ---------------------------------------------------------------
    def category_by_id(self, category_id: int) -> Category | None:
        return next((c for c in self.categories if c.id == category_id), None)

    def category_by_name(self, name: str) -> Category | None:
        return next((c for c in self.categories if c.name == name), None)

    def image_by_id(self, image_id: int) -> ImageRecord | None:
        return next((i for i in self.images if i.id == image_id), None)

    def annotations_for(self, image_id: int) -> list[Annotation]:
        return [a for a in self.annotations if a.image_id == image_id]

    def images_in_split(self, split: Split) -> list[ImageRecord]:
        return [i for i in self.images if i.split == split]

    # -- id helpers ------------------------------------------------------------
    def next_image_id(self) -> int:
        return max((i.id for i in self.images), default=0) + 1

    def next_annotation_id(self) -> int:
        return max((a.id for a in self.annotations), default=0) + 1

    def next_category_id(self) -> int:
        return max((c.id for c in self.categories), default=0) + 1


def clamp_to_image(annotation: Annotation, width: int, height: int) -> Annotation:
    """Clip an annotation's bbox and polygons into the image bounds.

    Detector and segmenter outputs (OWLv2 boxes in particular) routinely
    extend a few pixels past the frame; the out-of-frame part describes
    nothing photographable, so clipping loses no information. Returns a new
    Annotation; a box entirely outside the image comes back with zero width
    or height — callers decide whether that is a drop or an error.
    """
    x, y, w, h = annotation.bbox
    x0 = min(max(x, 0.0), float(width))
    y0 = min(max(y, 0.0), float(height))
    x1 = min(max(x + w, 0.0), float(width))
    y1 = min(max(y + h, 0.0), float(height))
    segmentation = [
        [
            min(max(value, 0.0), float(width if i % 2 == 0 else height))
            for i, value in enumerate(polygon)
        ]
        for polygon in annotation.segmentation
    ]
    return annotation.model_copy(
        update={
            "bbox": (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0)),
            "segmentation": segmentation,
        }
    )
