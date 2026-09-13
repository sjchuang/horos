"""Dataset content fingerprints (E7-T2).

Design decision (confirmed): the fingerprint is computed from the canonical
*content* of a Dataset — per split, every image's file name and size with its
annotations (class NAME, rounded bbox, rounded polygon, iscrowd) — not from the
bytes of any particular on-disk format. Two consequences the rest of E7 relies
on:

- the same data hashes identically whether it is read from a run's COCO
  snapshot or assembled live from the project (image and annotation ids are
  renumbered on export, so they are deliberately left out);
- the digest is reported per split, so a comparability warning can say "the
  train split changed but valid/test are identical" instead of a bare
  "different dataset".

Mosaic composites (E5, `mosaic_*.jpg` synthesized into the train snapshot) are
excluded: they are derived from the split's own images and a mosaic count
change is a hyperparameter, not a dataset change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from horos.core.dataset import Dataset

METHOD = (
    "sha256 over canonical per-split (file name, size, class name, bbox, polygon) "
    "records; ids ignored, mosaic composites excluded"
)
_MOSAIC_PREFIX = "mosaic_"
_DECIMALS = 3


class DatasetFingerprint(BaseModel):
    #: combined digest: the split digests plus the class list
    digest: str
    #: one digest per non-empty split
    splits: dict[str, str] = Field(default_factory=dict)
    classes: list[str] = Field(default_factory=list)
    num_images: int = 0
    num_annotations: int = 0
    method: str = METHOD


class FingerprintDiff(BaseModel):
    identical: bool
    #: splits whose digest differs, or that exist on one side only
    changed_splits: list[str] = Field(default_factory=list)
    classes_changed: bool = False

    def describe(self) -> str:
        if self.identical:
            return "identical dataset"
        parts = []
        if self.classes_changed:
            parts.append("class set differs")
        if self.changed_splits:
            parts.append(f"{', '.join(self.changed_splits)} split(s) differ")
        return "; ".join(parts) or "dataset differs"


def _round(values) -> list[float]:
    return [round(float(v), _DECIMALS) for v in values]


def fingerprint_dataset(dataset: Dataset) -> DatasetFingerprint:
    names = {c.id: c.name for c in dataset.categories}
    by_image: dict[int, list] = {}
    for ann in dataset.annotations:
        by_image.setdefault(ann.image_id, []).append(
            [
                names.get(ann.category_id, f"#{ann.category_id}"),
                _round(ann.bbox),
                [_round(poly) for poly in ann.segmentation],
                int(ann.iscrowd),
            ]
        )
    split_records: dict[str, list] = {}
    num_images = num_annotations = 0
    for image in dataset.images:
        if Path(image.file_name).name.startswith(_MOSAIC_PREFIX):
            continue
        anns = sorted(by_image.get(image.id, []), key=json.dumps)
        num_images += 1
        num_annotations += len(anns)
        split_records.setdefault(image.split or "unassigned", []).append(
            [image.file_name, image.width, image.height, anns]
        )
    splits: dict[str, str] = {}
    for split, records in sorted(split_records.items()):
        records.sort(key=lambda r: r[0])
        payload = json.dumps(records, separators=(",", ":"), sort_keys=True)
        splits[split] = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    classes = sorted(names.values())
    combined = hashlib.sha256()
    combined.update(json.dumps(classes).encode("utf-8"))
    for split, digest in sorted(splits.items()):
        combined.update(f"{split}={digest}".encode())
    return DatasetFingerprint(
        digest="sha256:" + combined.hexdigest(),
        splits=splits,
        classes=classes,
        num_images=num_images,
        num_annotations=num_annotations,
    )


def fingerprint_snapshot(snapshot_dir: Path) -> DatasetFingerprint | None:
    """Fingerprint a run's exported COCO snapshot (`<run>/dataset`), for runs
    recorded before fingerprints were persisted. None when there is none."""
    from horos.core.formats.coco import read_coco

    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir() or not any(snapshot_dir.rglob("_annotations.coco.json")):
        return None
    dataset, _ = read_coco(snapshot_dir)
    return fingerprint_dataset(dataset)


def compare_fingerprints(
    left: DatasetFingerprint, right: DatasetFingerprint
) -> FingerprintDiff:
    if left.digest == right.digest:
        return FingerprintDiff(identical=True)
    changed = sorted(
        split
        for split in set(left.splits) | set(right.splits)
        if left.splits.get(split) != right.splits.get(split)
    )
    return FingerprintDiff(
        identical=False,
        changed_splits=changed,
        classes_changed=left.classes != right.classes,
    )
