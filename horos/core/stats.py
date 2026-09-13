"""Dataset statistics (E1-T7).

These numbers feed the rule-based hyperparameter derivation (E5-T1): image
count, per-class instance counts, object relative-area distribution, image
size distribution, and class imbalance.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel

from horos.core.dataset import Dataset

_AREA_BINS = 10


class ClassStats(BaseModel):
    category_id: int
    name: str
    instances: int
    images: int  # images containing at least one instance


class SizeCount(BaseModel):
    width: int
    height: int
    count: int


class RelativeAreaStats(BaseModel):
    """Object bbox area / image area, over all annotations."""

    minimum: float
    maximum: float
    mean: float
    median: float
    #: 10 equal-width bins over [0, 1]: histogram[0] counts areas < 0.1, etc.
    histogram: list[int]


class DatasetStats(BaseModel):
    num_images: int
    num_annotations: int
    #: categories with at least one annotation — empty classes are not listed
    num_categories: int
    per_class: list[ClassStats]
    split_counts: dict[str, int]
    image_sizes: list[SizeCount]
    relative_area: RelativeAreaStats | None = None  # None when no annotations
    #: most populous class count / least populous (1.0 = perfectly balanced)
    imbalance_ratio: float | None = None
    annotated_images: int = 0
    unannotated_images: int = 0


def compute_stats(dataset: Dataset) -> DatasetStats:
    images_by_id = {i.id: i for i in dataset.images}

    instances: dict[int, int] = {c.id: 0 for c in dataset.categories}
    image_sets: dict[int, set[int]] = {c.id: set() for c in dataset.categories}
    relative_areas: list[float] = []
    annotated: set[int] = set()

    for ann in dataset.annotations:
        if ann.category_id in instances:
            instances[ann.category_id] += 1
            image_sets[ann.category_id].add(ann.image_id)
        annotated.add(ann.image_id)
        image = images_by_id.get(ann.image_id)
        if image is not None:
            image_area = image.width * image.height
            if image_area > 0:
                relative_areas.append(min(max(ann.area / image_area, 0.0), 1.0))

    # classes with no annotations (and hence no images) stay off the list —
    # they are noise in the summary and in hyperparameter derivation (E5)
    per_class = [
        ClassStats(
            category_id=c.id,
            name=c.name,
            instances=instances[c.id],
            images=len(image_sets[c.id]),
        )
        for c in dataset.categories
        if instances[c.id] > 0
    ]

    split_counts: dict[str, int] = {}
    size_counts: dict[tuple[int, int], int] = {}
    for image in dataset.images:
        if image.split is not None:  # unlabeled photos are in no set
            split_counts[image.split] = split_counts.get(image.split, 0) + 1
        key = (image.width, image.height)
        size_counts[key] = size_counts.get(key, 0) + 1

    relative_area = None
    if relative_areas:
        histogram = [0] * _AREA_BINS
        for area in relative_areas:
            histogram[min(int(area * _AREA_BINS), _AREA_BINS - 1)] += 1
        relative_area = RelativeAreaStats(
            minimum=min(relative_areas),
            maximum=max(relative_areas),
            mean=statistics.fmean(relative_areas),
            median=statistics.median(relative_areas),
            histogram=histogram,
        )

    positive_counts = [c.instances for c in per_class if c.instances > 0]
    imbalance = (
        max(positive_counts) / min(positive_counts) if positive_counts else None
    )

    return DatasetStats(
        num_images=len(dataset.images),
        num_annotations=len(dataset.annotations),
        num_categories=len(per_class),
        per_class=per_class,
        split_counts=split_counts,
        image_sizes=[
            SizeCount(width=w, height=h, count=n)
            for (w, h), n in sorted(size_counts.items(), key=lambda kv: -kv[1])
        ],
        relative_area=relative_area,
        imbalance_ratio=imbalance,
        annotated_images=len(annotated),
        unannotated_images=len(dataset.images) - len(annotated),
    )
