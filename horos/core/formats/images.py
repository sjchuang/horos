"""Loose photos with no annotation file (import-only).

The upload path accepts photos on their own, not only a labeled dataset: they
join the project unlabeled, which is exactly what the active-learning loop's
pool is made of (E10) and what a photo needs before anyone can annotate it.

Detection keeps this last of all formats, so a real annotation layout is never
mistaken for a pile of photos; the reader still says in a warning that it
found no labels, because "no annotations" must never look like a silent loss.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from horos.core.dataset import Dataset, ImageRecord
from horos.errors import DatasetFormatError

from . import IMAGE_SUFFIXES, split_from_dir_name


def find_images(root: Path | str) -> list[Path]:
    """Every photo under root (or root itself when it is one), sorted."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def looks_like_images(root: Path | str) -> bool:
    return bool(find_images(root))


def read_images(root: Path | str) -> tuple[Dataset, dict[int, Path]]:
    """Photos with no annotations. Sizes are probed from the files."""
    files = find_images(root)
    if not files:
        raise DatasetFormatError(f"No photos found under {root}")
    dataset = Dataset()
    image_paths: dict[int, Path] = {}
    for image_file in files:
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
            split=split_from_dir_name(image_file.parent.name),
        )
        dataset.images.append(record)
        image_paths[record.id] = image_file.resolve()
    dataset.reader_warnings.append(
        f"No annotation file found — imported {len(files)} photo(s) with no labels"
    )
    return dataset, image_paths
