"""COCO JSON reader/writer (E1-T3).

Supports both a single annotation JSON and the Roboflow-style split layout
(train/valid/test subdirectories, each with `_annotations.coco.json`).
"""

from __future__ import annotations

import json
from pathlib import Path

from horos.core.dataset import (
    SPLITS,
    Annotation,
    Category,
    Dataset,
    ImageRecord,
    Split,
    default_color,
)
from horos.errors import DatasetFormatError

COCO_CONVENTION_NAME = "_annotations.coco.json"

# Aliases seen in the wild for the validation split directory.
_SPLIT_DIR_ALIASES: dict[str, Split] = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}


def find_annotation_files(root: Path) -> list[tuple[Split | None, Path]]:
    """Locate COCO annotation JSONs under `root`.

    Returns (split, json_path) pairs; split is None for a single flat file.
    """
    root = Path(root)
    if root.is_file():
        return [(None, root)]
    split_files: list[tuple[Split | None, Path]] = []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and child.name.lower() in _SPLIT_DIR_ALIASES:
            candidate = child / COCO_CONVENTION_NAME
            if candidate.exists():
                split_files.append((_SPLIT_DIR_ALIASES[child.name.lower()], candidate))
    if split_files:
        return split_files
    flat = root / COCO_CONVENTION_NAME
    if flat.exists():
        return [(None, flat)]
    json_files = sorted(root.glob("*.json"))
    if len(json_files) == 1:
        return [(None, json_files[0])]
    if not json_files:
        raise DatasetFormatError(
            f"No COCO annotation file found under {root} "
            f"(looked for {COCO_CONVENTION_NAME}, split subdirectories, or a single *.json)"
        )
    raise DatasetFormatError(
        f"Multiple candidate annotation JSONs under {root}: "
        + ", ".join(p.name for p in json_files)
        + ". Point directly at the right one."
    )


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"Cannot parse {path} as COCO JSON: {exc}") from exc
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise DatasetFormatError(f"{path} is missing the COCO '{key}' list")
    return data


def read_coco(source: Path | str) -> tuple[Dataset, dict[int, Path]]:
    """Read a COCO dataset from a JSON file or a dataset directory.

    Returns the Dataset plus {image_id: absolute_image_path} for every image
    (paths resolved relative to each annotation file's directory).
    """
    files = find_annotation_files(Path(source))
    dataset = Dataset()
    image_paths: dict[int, Path] = {}
    cat_id_by_name: dict[str, int] = {}
    with_keypoints = 0

    for split, json_path in files:
        data = _load_json(json_path)
        # categories: merge across splits by name, preserving first-seen ids
        local_to_global: dict[int, int] = {}
        for cat in data["categories"]:
            name = str(cat["name"])
            if name not in cat_id_by_name:
                new_id = dataset.next_category_id()
                cat_id_by_name[name] = new_id
                dataset.categories.append(
                    Category(
                        id=new_id,
                        name=name,
                        color=default_color(len(dataset.categories)),
                    )
                )
            local_to_global[int(cat["id"])] = cat_id_by_name[name]

        image_id_map: dict[int, int] = {}
        for img in data["images"]:
            new_id = dataset.next_image_id()
            image_id_map[int(img["id"])] = new_id
            try:
                record = ImageRecord(
                    id=new_id,
                    file_name=str(img["file_name"]),
                    width=int(img["width"]),
                    height=int(img["height"]),
                    split=split,  # None for a flat file: assigned when labeled
                )
            except (KeyError, ValueError) as exc:
                raise DatasetFormatError(
                    f"{json_path}: invalid image entry {img!r}: {exc}"
                ) from exc
            dataset.images.append(record)
            image_paths[new_id] = (json_path.parent / str(img["file_name"])).resolve()

        for ann in data["annotations"]:
            if ann.get("keypoints"):
                with_keypoints += 1
            try:
                bbox = tuple(float(v) for v in ann["bbox"])
                if len(bbox) != 4:
                    raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
                segmentation = ann.get("segmentation") or []
                if not isinstance(segmentation, list) or (
                    segmentation and not isinstance(segmentation[0], list)
                ):
                    # RLE crowd masks are not supported; keep the box, drop the mask.
                    segmentation = []
                dataset.annotations.append(
                    Annotation(
                        id=dataset.next_annotation_id(),
                        image_id=image_id_map[int(ann["image_id"])],
                        category_id=local_to_global[int(ann["category_id"])],
                        bbox=bbox,  # type: ignore[arg-type]
                        segmentation=[[float(v) for v in poly] for poly in segmentation],
                        iscrowd=int(ann.get("iscrowd", 0)),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise DatasetFormatError(
                    f"{json_path}: invalid annotation entry (id={ann.get('id')}): {exc}"
                ) from exc

    if with_keypoints:
        dataset.reader_warnings.append(
            f"{with_keypoints} annotation(s) carry COCO keypoints — keypoint tasks "
            f"are not supported yet, so the keypoints were NOT imported (boxes and "
            f"polygons were)"
        )

    return dataset, image_paths


def write_coco(
    dataset: Dataset,
    out_dir: Path,
    *,
    image_paths: dict[int, Path] | None = None,
    split_layout: bool = True,
    copy_images: bool = False,
) -> list[Path]:
    """Write COCO JSON(s). With split_layout, produces the Roboflow-style
    train/valid/test directories, each with `_annotations.coco.json`."""
    import shutil

    out_dir = Path(out_dir)
    written: list[Path] = []
    split_groups: list[tuple[Split | None, list]] = (
        [(s, dataset.images_in_split(s)) for s in SPLITS]
        if split_layout
        else [(None, list(dataset.images))]
    )
    for split, images in split_groups:
        if split_layout and not images:
            continue
        target_dir = out_dir / split if split else out_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        image_ids = {i.id for i in images}
        payload = {
            "info": {"description": "exported by horos"},
            "licenses": [],
            "categories": [
                {"id": c.id, "name": c.name, "supercategory": "none"}
                for c in dataset.categories
            ],
            "images": [
                {
                    "id": i.id,
                    "file_name": i.file_name,
                    "width": i.width,
                    "height": i.height,
                }
                for i in images
            ],
            # Annotation ids are renumbered from 1 per written file. horos
            # stores them per image (each image's first box is id 1), but a
            # COCO file needs ids unique across the WHOLE file: pycocotools
            # indexes annotations by id in a dict, so duplicates silently
            # overwrite each other — every consumer (metrics, and torchvision
            # -style CocoDetection loaders, which fetch targets via
            # loadAnns(getAnnIds(...))) then reads other images' boxes as
            # this image's ground truth. That corruption is invisible in the
            # file and only shows up as inexplicably poor training.
            "annotations": [
                {
                    "id": new_id,
                    "image_id": a.image_id,
                    "category_id": a.category_id,
                    "bbox": list(a.bbox),
                    "area": a.area,
                    "segmentation": a.segmentation,
                    "iscrowd": a.iscrowd,
                }
                for new_id, a in enumerate(
                    (a for a in dataset.annotations if a.image_id in image_ids),
                    start=1,
                )
            ],
        }
        json_path = target_dir / COCO_CONVENTION_NAME
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(json_path)
        if copy_images and image_paths:
            for record in images:
                src = image_paths.get(record.id)
                if src and src.exists():
                    shutil.copy2(src, target_dir / record.file_name)
    return written
