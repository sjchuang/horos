"""YOLO format reader/writer (E1-T4), including `data.yaml`.

Conventions handled:
- data.yaml with `train`/`val`/`test` image-dir keys and `names` (list or dict)
- labels alongside images: <split>/images/x.jpg ↔ <split>/labels/x.txt
- box lines `cls cx cy w h` (normalized) and segmentation lines
  `cls x1 y1 x2 y2 ...` (normalized polygon, ≥3 points)

Known format limitation (inherent to YOLO, documented for E1-T5): a segmented
object carries no independent bbox — on read, the bbox is derived from the
polygon's extent. Only the first polygon of a multi-polygon annotation is
written.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

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

_SPLIT_KEYS: dict[str, Split] = {"train": "train", "val": "valid", "test": "test"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml(root: Path) -> Path:
    root = Path(root)
    if root.is_file():
        return root
    for name in ("data.yaml", "data.yml"):
        direct = root / name
        if direct.exists():
            return direct
    matches = sorted(root.rglob("data.yaml")) + sorted(root.rglob("data.yml"))
    if not matches:
        raise DatasetFormatError(f"No data.yaml found under {root}")
    return matches[0]


def _parse_names(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(n) for n in raw]
    if isinstance(raw, dict):
        try:
            return [str(raw[k]) for k in sorted(raw, key=int)]
        except (ValueError, KeyError) as exc:
            raise DatasetFormatError(f"data.yaml names dict is not 0..n keyed: {raw}") from exc
    raise DatasetFormatError(f"data.yaml 'names' must be a list or dict, got {type(raw)}")


def _strip_parent_prefix(value: str) -> str:
    while value.startswith(("../", "..\\")):
        value = value[3:]
    return value


def _resolve_dir(yaml_path: Path, base: str | None, value: str) -> Path:
    root = yaml_path.parent / base if base else yaml_path.parent
    candidates = [root / value]
    # tolerate paths written relative to the yaml itself
    candidates.append(yaml_path.parent / value)
    # Roboflow exports put data.yaml at the dataset root but write paths like
    # `../train/images` (relative to an assumed enclosing datasets dir) —
    # retry with the leading ../ segments stripped, anchored at the yaml.
    stripped = _strip_parent_prefix(value)
    if stripped != value:
        candidates.append(yaml_path.parent / stripped)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    raise DatasetFormatError(f"data.yaml points at missing directory: {value}")


def _labels_dir_for(images_dir: Path) -> Path:
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / (images_dir.name.replace("images", "labels") or "labels")


def _parse_label_line(line: str, path: Path, lineno: int) -> tuple[int, list[float]]:
    tokens = line.split()
    try:
        cls = int(tokens[0])
        values = [float(t) for t in tokens[1:]]
    except (ValueError, IndexError) as exc:
        raise DatasetFormatError(f"{path}:{lineno}: unparsable label line: {line!r}") from exc
    if len(values) == 4:
        return cls, values
    if len(values) >= 6 and len(values) % 2 == 0:
        return cls, values
    raise DatasetFormatError(
        f"{path}:{lineno}: expected 4 box values or an even number (>=6) of "
        f"polygon coordinates, got {len(values)}"
    )


def read_yolo(source: Path | str) -> tuple[Dataset, dict[int, Path]]:
    """Read a YOLO dataset via its data.yaml.

    Returns the Dataset plus {image_id: absolute_image_path}. Image dimensions
    are read from the image files themselves (YOLO stores none).
    """
    yaml_path = find_data_yaml(Path(source))
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DatasetFormatError(f"Cannot parse {yaml_path}: {exc}") from exc
    if "names" not in config:
        raise DatasetFormatError(f"{yaml_path} has no 'names' entry")

    names = _parse_names(config["names"])
    dataset = Dataset(
        categories=[
            Category(id=i, name=name, color=default_color(i))
            for i, name in enumerate(names)
        ]
    )
    image_paths: dict[int, Path] = {}
    base = config.get("path")

    for key, split in _SPLIT_KEYS.items():
        if key not in config or not config[key]:
            continue
        images_dir = _resolve_dir(yaml_path, base, str(config[key]))
        labels_dir = _labels_dir_for(images_dir)
        for image_file in sorted(images_dir.iterdir()):
            if image_file.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(image_file) as im:
                    width, height = im.size
            except OSError as exc:
                raise DatasetFormatError(f"Cannot read image {image_file}: {exc}") from exc
            record = ImageRecord(
                id=dataset.next_image_id(),
                file_name=image_file.name,
                width=width,
                height=height,
                split=split,
            )
            dataset.images.append(record)
            image_paths[record.id] = image_file.resolve()

            label_file = labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                continue
            for lineno, line in enumerate(
                label_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                cls, values = _parse_label_line(line.strip(), label_file, lineno)
                if not 0 <= cls < len(names):
                    raise DatasetFormatError(
                        f"{label_file}:{lineno}: class index {cls} outside names list "
                        f"(0..{len(names) - 1})"
                    )
                if len(values) == 4:
                    cx, cy, w, h = values
                    bbox = (
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        w * width,
                        h * height,
                    )
                    segmentation: list[list[float]] = []
                else:
                    xs = [v * width for v in values[0::2]]
                    ys = [v * height for v in values[1::2]]
                    flat: list[float] = []
                    for x, y in zip(xs, ys, strict=True):
                        flat.extend((x, y))
                    bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                    segmentation = [flat]
                dataset.annotations.append(
                    Annotation(
                        id=dataset.next_annotation_id(),
                        image_id=record.id,
                        category_id=cls,
                        bbox=bbox,
                        segmentation=segmentation,
                    )
                )
    return dataset, image_paths


def write_yolo(
    dataset: Dataset,
    out_dir: Path,
    *,
    image_paths: dict[int, Path] | None = None,
    copy_images: bool = True,
) -> Path:
    """Write the Roboflow/Ultralytics-style layout:

        out/train/images, out/train/labels, ... , out/data.yaml

    Class indices are contiguous 0-based in category-id order; the mapping is
    recorded in data.yaml's names list. Floats are written at full repr
    precision so a COCO→YOLO→COCO round trip stays lossless (E1-T5).
    """
    import shutil

    out_dir = Path(out_dir)
    ordered = sorted(dataset.categories, key=lambda c: c.id)
    class_index = {c.id: i for i, c in enumerate(ordered)}

    yaml_splits: dict[str, str] = {}
    for split in SPLITS:
        images = dataset.images_in_split(split)
        if not images:
            continue
        key = {"train": "train", "valid": "val", "test": "test"}[split]
        images_dir = out_dir / split / "images"
        labels_dir = out_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        yaml_splits[key] = f"{split}/images"
        for record in images:
            lines: list[str] = []
            for ann in dataset.annotations_for(record.id):
                cls = class_index[ann.category_id]
                if ann.segmentation:
                    # YOLO-seg takes one polygon per object: for a multi-part
                    # object the largest ring stands in (the box is unaffected)
                    from horos.core.validate import polygon_area

                    poly = max(ann.segmentation, key=polygon_area)
                    coords: list[float] = []
                    for x, y in zip(poly[0::2], poly[1::2], strict=True):
                        coords.extend((x / record.width, y / record.height))
                    lines.append(" ".join([str(cls), *(repr(v) for v in coords)]))
                else:
                    x, y, w, h = ann.bbox
                    cx = (x + w / 2) / record.width
                    cy = (y + h / 2) / record.height
                    lines.append(
                        " ".join(
                            [
                                str(cls),
                                repr(cx),
                                repr(cy),
                                repr(w / record.width),
                                repr(h / record.height),
                            ]
                        )
                    )
            label_path = labels_dir / f"{Path(record.file_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")
            if copy_images and image_paths:
                src = image_paths.get(record.id)
                if src and src.exists():
                    shutil.copy2(src, images_dir / record.file_name)

    yaml_path = out_dir / "data.yaml"
    payload = {
        **yaml_splits,
        "nc": len(ordered),
        "names": [c.name for c in ordered],
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path
