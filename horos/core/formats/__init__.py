"""Dataset format codecs. Each format reads to / writes from `horos.core.dataset.Dataset`.

COCO, YOLO and LabelMe support read and write; Pascal VOC, Darknet, and VIA
(VGG Image Annotator) are import-only (design decision: users bring legacy
data in, horos exports COCO/YOLO, plus LabelMe so a dataset can go back to the
labelme tool for edits).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

#: "images" is the no-annotation fallback: a pile of photos is a legitimate
#: import (they join the project unlabeled), but never a guess over a real format
DatasetFormat = Literal["coco", "yolo", "voc", "darknet", "via", "labelme", "images"]

#: formats horos can write (export / convert targets)
WRITABLE_FORMATS: tuple[str, ...] = ("coco", "yolo", "labelme")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_SPLIT_BY_DIR = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def split_from_dir_name(name: str) -> str | None:
    """Map a containing-directory name to a split; None when the directory is
    not one of train / valid / val / test — the source said nothing, so the
    photo joins a set only once it is labeled (core/splitting.py)."""
    return _SPLIT_BY_DIR.get(name.lower())


def _looks_like_voc(root: Path) -> bool:
    for xml_path in root.rglob("*.xml"):
        try:
            head = xml_path.read_text(encoding="utf-8", errors="ignore")[:2048]
        except OSError:
            continue
        if "<annotation" in head:
            return True
    return False


def _looks_like_darknet(root: Path) -> bool:
    if any(root.rglob("_darknet.labels")):
        return True
    # bare YOLO-style label lines next to same-stem images, without a data.yaml
    for txt in root.rglob("*.txt"):
        if not any(txt.with_suffix(ext).exists() for ext in IMAGE_SUFFIXES):
            continue
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            try:
                int(tokens[0])
                [float(t) for t in tokens[1:]]
            except ValueError:
                break
            if len(tokens) >= 5:
                return True
            break
    return False


def detect_format(root: Path) -> DatasetFormat | None:
    """Best-effort format detection for a dataset directory (used by zip import)."""
    root = Path(root)
    if any(root.rglob("_annotations.coco.json")) or any(root.glob("*.coco.json")):
        return "coco"
    if any(root.rglob("data.yaml")) or any(root.rglob("data.yml")):
        return "yolo"
    from . import via as via_format

    if via_format.find_via_files(root):
        return "via"
    if _looks_like_voc(root):
        return "voc"
    if _looks_like_darknet(root):
        return "darknet"
    from . import labelme as labelme_format

    # one JSON per image carrying "shapes" + "imagePath"; must precede the
    # single-JSON COCO fallback below (a one-image LabelMe dir has one JSON too)
    if labelme_format.looks_like_labelme(root):
        return "labelme"
    # A bare COCO export: a single .json next to an images dir
    json_files = [p for p in root.glob("*.json")]
    if len(json_files) == 1:
        return "coco"
    from . import images as images_format

    # last resort only: photos with nothing describing them
    if images_format.looks_like_images(root):
        return "images"
    return None
