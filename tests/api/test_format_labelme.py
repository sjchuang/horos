"""LabelMe import/export — one `<stem>.json` per image (E1 formats)."""

from __future__ import annotations

import base64
import io
import json

import pytest
from helpers.data import make_image, write_sample_coco_dir
from PIL import Image

from horos.api.dataset import export_dataset, import_dataset
from horos.api.project import create_project
from horos.core.formats import detect_format
from horos.core.formats.coco import read_coco
from horos.core.formats.labelme import read_labelme, write_labelme
from horos.errors import DatasetFormatError


def _shape(label, points, shape_type, group_id=None):
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
    }


def _write(directory, stem, shapes, *, width=64, height=48, image=True, **extra):
    """One LabelMe JSON (+ its image unless image=False)."""
    directory.mkdir(parents=True, exist_ok=True)
    if image:
        make_image(directory / f"{stem}.jpg", width, height)
    payload = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{stem}.jpg",
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    payload.update(extra)
    (directory / f"{stem}.json").write_text(json.dumps(payload), "utf-8")
    return directory / f"{stem}.json"


@pytest.fixture
def flat_dataset(tmp_path):
    """The hospital_beds layout: everything in one directory, rectangles only."""
    root = tmp_path / "beds"
    _write(root, "img1", [
        _shape("bed", [[10, 10], [30, 25]], "rectangle", group_id="0"),
        _shape("person", [[40, 5], [20, 30]], "rectangle", group_id=""),  # reversed corners
    ])
    _write(root, "img2", [_shape("person", [[1, 1], [11, 21]], "rectangle")])
    return root


def test_detection_recognizes_labelme(flat_dataset):
    assert detect_format(flat_dataset) == "labelme"


def test_one_image_directory_is_labelme_not_a_bare_coco_json(tmp_path):
    root = tmp_path / "one"
    _write(root, "only", [_shape("a", [[0, 0], [5, 5]], "rectangle")])
    assert detect_format(root) == "labelme"


def test_read_rectangles(flat_dataset):
    dataset, image_paths = read_labelme(flat_dataset)
    assert [c.name for c in dataset.categories] == ["bed", "person"]
    assert len(dataset.images) == 2 and len(dataset.annotations) == 3
    assert all(i.split is None for i in dataset.images)  # flat dir → no set until labeled
    assert all(i.width == 64 and i.height == 48 for i in dataset.images)
    boxes = {(a.image_id, a.category_id): a.bbox for a in dataset.annotations}
    img1 = next(i.id for i in dataset.images if i.file_name == "img1.jpg")
    assert boxes[(img1, 0)] == (10.0, 10.0, 20.0, 15.0)
    assert boxes[(img1, 1)] == (20.0, 5.0, 20.0, 25.0)  # corners normalised
    assert all(a.segmentation == [] for a in dataset.annotations)
    assert set(image_paths) == {i.id for i in dataset.images}
    assert dataset.reader_warnings == []


def test_split_from_directory_name_and_polygons(tmp_path):
    root = tmp_path / "ds"
    _write(root / "train", "a", [
        _shape("cat", [[10, 10], [30, 10], [20, 30]], "polygon"),
    ])
    _write(root / "val", "b", [_shape("cat", [[1, 1], [9, 1], [5, 9]], "polygon")])
    _write(root / "test", "c", [])  # annotated as empty: still an image
    dataset, _ = read_labelme(root)
    assert {i.file_name: i.split for i in dataset.images} == {
        "a.jpg": "train", "b.jpg": "valid", "c.jpg": "test"
    }
    a_id = next(i.id for i in dataset.images if i.file_name == "a.jpg")
    polygon = next(a for a in dataset.annotations if a.image_id == a_id)
    assert polygon.segmentation == [[10.0, 10.0, 30.0, 10.0, 20.0, 30.0]]
    assert polygon.bbox == (10.0, 10.0, 20.0, 20.0)


def test_windows_image_path_and_stem_fallback(tmp_path):
    root = tmp_path / "ds"
    # imagePath written by LabelMe on Windows, pointing at a sibling folder
    make_image(root / "images" / "w.jpg", 32, 16)
    _write(root / "ann", "w", [_shape("x", [[0, 0], [4, 4]], "rectangle")],
           image=False, imagePath="..\\images\\w.jpg", imageWidth=32, imageHeight=16)
    # imagePath stale, but a same-stem image sits next to the JSON
    _write(root / "ann", "s", [_shape("x", [[0, 0], [4, 4]], "rectangle")],
           imagePath="gone/elsewhere.jpg")
    dataset, paths = read_labelme(root)
    names = {i.file_name for i in dataset.images}
    assert names == {"w.jpg", "s.jpg"}
    assert all(p.is_file() for p in paths.values())


def test_embedded_image_data_restores_a_missing_image(tmp_path):
    root = tmp_path / "ds"
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), (1, 2, 3)).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    _write(root, "e", [_shape("x", [[0, 0], [4, 4]], "rectangle")],
           image=False, imageData=encoded, imageWidth=20, imageHeight=10)
    dataset, paths = read_labelme(root)
    assert (root / "e.jpg").is_file()
    assert dataset.images[0].width == 20 and dataset.images[0].height == 10
    assert any("restored e.jpg from the embedded imageData" in w for w in dataset.reader_warnings)
    assert paths[dataset.images[0].id] == (root / "e.jpg").resolve()


def test_missing_image_without_image_data_is_an_error(tmp_path):
    root = tmp_path / "ds"
    _write(root, "m", [_shape("x", [[0, 0], [4, 4]], "rectangle")], image=False)
    with pytest.raises(DatasetFormatError, match="m.json: image file 'm.jpg' not found"):
        read_labelme(root)


def test_images_without_json_become_unannotated_images(flat_dataset):
    make_image(flat_dataset / "unlabeled.jpg", 64, 48)
    dataset, paths = read_labelme(flat_dataset)
    assert len(dataset.images) == 3
    extra = next(i for i in dataset.images if i.file_name == "unlabeled.jpg")
    assert dataset.annotations_for(extra.id) == []
    assert any("1 image(s) have no LabelMe JSON" in w for w in dataset.reader_warnings)


def test_circle_becomes_enclosing_box_and_other_shapes_are_skipped(tmp_path):
    root = tmp_path / "ds"
    _write(root, "s", [
        _shape("disc", [[10, 10], [13, 14]], "circle"),  # r = 5
        _shape("path", [[0, 0], [5, 5]], "line"),
        _shape("path", [[0, 0], [5, 5], [9, 9]], "linestrip"),
        _shape("dot", [[3, 3]], "point"),
        _shape("keep", [[0, 0], [2, 2]], "rectangle"),
    ])
    dataset, _ = read_labelme(root)
    assert [c.name for c in dataset.categories] == ["disc", "dot", "keep", "path"]
    assert len(dataset.annotations) == 2  # circle + rectangle
    circle = next(a for a in dataset.annotations if a.category_id == 0)
    assert circle.bbox == (5.0, 5.0, 10.0, 10.0)
    joined = "\n".join(dataset.reader_warnings)
    assert "1 circle shape(s) imported as their enclosing bounding box" in joined
    assert "1 'line' shape(s) skipped" in joined
    assert "1 'linestrip' shape(s) skipped" in joined
    assert "1 'point' shape(s) skipped" in joined


def test_group_id_merges_polygons_into_one_instance(tmp_path):
    root = tmp_path / "ds"
    _write(root, "g", [
        _shape("cat", [[0, 0], [4, 0], [4, 4]], "polygon", group_id=1),
        _shape("cat", [[10, 10], [14, 10], [14, 14]], "polygon", group_id="1"),  # same group
        _shape("dog", [[20, 20], [24, 20], [24, 24]], "polygon", group_id=1),  # other label
        _shape("cat", [[30, 30], [34, 30], [34, 34]], "polygon", group_id=2),
        _shape("cat", [[40, 40], [44, 44]], "rectangle", group_id=1),  # boxes never merge
        _shape("cat", [[50, 50], [54, 50], [54, 54]], "polygon"),  # ungrouped
    ])
    dataset, _ = read_labelme(root)
    assert len(dataset.annotations) == 5
    merged = next(a for a in dataset.annotations if len(a.segmentation) == 2)
    assert merged.category_id == 0  # cat
    assert merged.bbox == (0.0, 0.0, 14.0, 14.0)  # union of both parts
    assert sum(1 for a in dataset.annotations if not a.segmentation) == 1


def test_bad_geometry_is_reported_with_context(tmp_path):
    root = tmp_path / "ds"
    _write(root, "bad", [_shape("x", [[0, 0], [5, 5]], "polygon")])
    with pytest.raises(DatasetFormatError, match=r"bad.json: shape #0: polygon needs ≥3"):
        read_labelme(root)


def test_unlabeled_shape_is_an_error(tmp_path):
    root = tmp_path / "ds"
    _write(root, "nolabel", [_shape("", [[0, 0], [5, 5]], "rectangle")])
    with pytest.raises(DatasetFormatError, match="nolabel.json: shape #0: shape has no label"):
        read_labelme(root)


def test_non_labelme_json_is_ignored_with_a_warning(flat_dataset):
    (flat_dataset / "notes.json").write_text(json.dumps({"hello": "world"}), "utf-8")
    dataset, _ = read_labelme(flat_dataset)
    assert len(dataset.images) == 2
    assert any("1 .json file(s) under the source are not LabelMe" in w
               for w in dataset.reader_warnings)


def test_macosx_zip_junk_is_ignored(flat_dataset):
    junk = flat_dataset / "__MACOSX"
    junk.mkdir()
    (junk / "._img1.json").write_text("garbage", "utf-8")
    dataset, _ = read_labelme(flat_dataset)
    assert len(dataset.images) == 2


def _canonical(dataset):
    cats = {c.id: c.name for c in dataset.categories}
    images = {i.id: i for i in dataset.images}
    rows = []
    for a in dataset.annotations:
        image = images[a.image_id]
        rows.append((
            image.file_name,
            image.split,
            cats[a.category_id],
            tuple(round(v, 6) for v in a.bbox),
            tuple(sorted(tuple(round(v, 6) for v in p) for p in a.segmentation)),
        ))
    return sorted(rows)


def test_write_then_read_is_lossless(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    original, paths = read_coco(coco_dir)
    # a two-part instance must survive as one annotation via group_id
    original.annotations[3].segmentation.append([20.0, 20.0, 28.0, 20.0, 28.0, 28.0])
    original.annotations[3].bbox = (2.0, 2.0, 26.0, 26.0)

    out = write_labelme(original, tmp_path / "labelme", image_paths=paths)
    assert out == tmp_path / "labelme"
    assert (out / "train" / "a.json").is_file() and (out / "train" / "a.png").is_file()
    payload = json.loads((out / "valid" / "c.json").read_text("utf-8"))
    assert payload["version"] == "5.3.1" and payload["imagePath"] == "c.png"
    assert [s["shape_type"] for s in payload["shapes"]] == ["polygon", "polygon"]
    assert payload["shapes"][0]["group_id"] == payload["shapes"][1]["group_id"] is not None

    back, _ = read_labelme(out)
    assert _canonical(back) == _canonical(original)
    assert {c.name for c in back.categories} == {c.name for c in original.categories}
    assert {(i.file_name, i.split) for i in back.images} == {
        (i.file_name, i.split) for i in original.images
    }


def test_import_and_export_through_the_api(flat_dataset, tmp_path):
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, flat_dataset)
    assert summary.format == "labelme"
    assert summary.num_images == 2 and summary.num_annotations == 3
    assert summary.instances_per_category == {"bed": 1, "person": 2}

    out = export_dataset(project, tmp_path / "out", format="labelme")
    written = sorted(p.name for p in out.rglob("*.json"))
    assert written == ["img1.json", "img2.json"]
