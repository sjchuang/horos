"""E1-S1/S2/S3 API paths: import (dir/zip, copy/reference), convert."""

import zipfile

import pytest
from helpers.data import write_sample_coco_dir, write_sample_yolo_dir

from horos.api.dataset import convert_dataset, import_dataset, import_zip
from horos.api.project import create_project, open_project
from horos.errors import DatasetFormatError


def test_three_line_workflow(tmp_path):
    # E1-S1: existing COCO dir -> horos project in three lines
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, coco_dir)
    assert summary.format == "coco"
    assert summary.num_images == 3
    assert summary.num_annotations == 4
    assert summary.instances_per_category == {"forklift": 2, "pallet": 2}
    assert summary.split_counts == {"train": 2, "valid": 1}


def test_import_copies_images_by_default(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    for record in project.list_images():
        assert (project.images_dir / record.file_name).exists()
        assert record.external_path is None


def test_import_reference_mode(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir, copy_images=False)
    for record in project.list_images():
        assert not (project.images_dir / record.file_name).exists()
        assert record.external_path
        assert project.image_path(record).exists()


def test_import_persists_annotations_per_image(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    reopened = open_project(project.root)
    ds = reopened.to_dataset()
    assert len(ds.annotations) == 4
    assert len([a for a in ds.annotations if a.segmentation]) == 1


def test_import_yolo_autodetected(tmp_path):
    yolo_dir = write_sample_yolo_dir(tmp_path / "yolo")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, yolo_dir)
    assert summary.format == "yolo"
    assert summary.num_images == 3


def test_import_without_split_info_assigns_labeled_photos_by_hash(tmp_path):
    # a flat export (no split dirs) must not land every image in "train":
    # labeled photos join train/valid/test by the project's stable hash and
    # ratios (70/10/20), and the import says so in a warning
    from helpers.data import make_image

    from horos.core.dataset import Annotation, Category, Dataset, ImageRecord
    from horos.core.formats.coco import write_coco

    dataset = Dataset(
        categories=[Category(id=1, name="forklift", color="#e6194b")],
        images=[
            ImageRecord(id=i, file_name=f"img{i}.png", width=64, height=48)
            for i in range(1, 11)
        ],
        annotations=[
            Annotation(id=i, image_id=i, category_id=1, bbox=(4.0, 4.0, 16.0, 12.0))
            for i in range(1, 11)
        ],
    )
    staging = tmp_path / "staging"
    image_paths = {
        r.id: make_image(staging / r.file_name, r.width, r.height)
        for r in dataset.images
    }
    src = tmp_path / "coco"
    write_coco(
        dataset, src, image_paths=image_paths, split_layout=False, copy_images=True
    )

    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, src)
    assert sum(summary.split_counts.values()) == 10 and "unassigned" not in summary.split_counts
    assert set(summary.split_counts) == {"train", "valid", "test"}
    assert any("stable hash" in w for w in summary.warnings)
    counts: dict[str, int] = {}
    for record in project.list_images():
        counts[record.split] = counts.get(record.split, 0) + 1
    assert counts == summary.split_counts


def test_import_leaves_unlabeled_photos_in_no_set(tmp_path):
    # even when the source directory is called "train": only labeled photos
    # are set members
    from helpers.data import make_image

    from horos.core.dataset import Annotation, Category, Dataset, ImageRecord
    from horos.core.formats.coco import write_coco

    dataset = Dataset(
        categories=[Category(id=1, name="forklift", color="#e6194b")],
        images=[ImageRecord(id=i, file_name=f"img{i}.png", width=64, height=48, split="train")
                for i in range(1, 5)],
        annotations=[Annotation(id=1, image_id=1, category_id=1, bbox=(4.0, 4.0, 16.0, 12.0))],
    )
    staging = tmp_path / "staging"
    image_paths = {r.id: make_image(staging / r.file_name, r.width, r.height)
                   for r in dataset.images}
    write_coco(dataset, tmp_path / "coco", image_paths=image_paths, split_layout=True,
               copy_images=True)
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, tmp_path / "coco")
    assert summary.split_counts == {"train": 1, "unassigned": 3}
    assert any("no labels" in w for w in summary.warnings)
    assert [i.split for i in project.list_images()] == ["train", None, None, None]


def test_import_explicit_splits_are_preserved(tmp_path):
    # sources that ship their own split layout keep it untouched
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, coco_dir)
    assert summary.split_counts == {"train": 2, "valid": 1}
    assert not any("stable hash" in w for w in summary.warnings)


def test_import_zip(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    zip_path = tmp_path / "upload.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in coco_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(coco_dir))
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, zip_path)
    assert summary.num_images == 3


def test_import_zip_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", "boom")
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="unsafe path"):
        import_zip(project, zip_path)


def test_import_undetectable_format_is_explicit(tmp_path):
    (tmp_path / "mystery").mkdir()
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="Could not detect"):
        import_dataset(project, tmp_path / "mystery")


def test_convert_coco_to_yolo_and_back(tmp_path):
    # E1-S2: CVAT-style conversion without a project
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    yolo_yaml = convert_dataset(coco_dir, tmp_path / "yolo", to_format="yolo")
    assert yolo_yaml.name == "data.yaml"
    convert_dataset(tmp_path / "yolo", tmp_path / "coco2", to_format="coco")
    from horos.core.formats.coco import read_coco

    dataset, _ = read_coco(tmp_path / "coco2")
    assert len(dataset.annotations) == 4


def test_convert_to_same_format_is_rejected(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    with pytest.raises(DatasetFormatError, match="already in format"):
        convert_dataset(coco_dir, tmp_path / "out", to_format="coco")


def test_import_zip_roboflow_layout(tmp_path):
    # Regression: Roboflow YOLO zips (data.yaml with ../<split>/images paths)
    from test_format_yolo import _rewrite_yaml_roboflow_style

    yolo_dir = write_sample_yolo_dir(tmp_path / "yolo")
    _rewrite_yaml_roboflow_style(yolo_dir)
    zip_path = tmp_path / "upload.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in yolo_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(yolo_dir))
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, zip_path)
    assert summary.format == "yolo"
    assert summary.num_images == 3


def test_import_zip_refuses_files_outside_archive(tmp_path):
    # A crafted data.yaml must not be able to pull server-side files into the project
    from helpers.data import make_image

    outside = tmp_path / "outside" / "images"
    make_image(outside / "secret.png")
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.yaml", f"train: {outside}\nnames: ['x']\n")
    project = create_project(tmp_path / "proj")
    with pytest.raises(DatasetFormatError, match="outside the archive"):
        import_zip(project, zip_path)


def _zip_dir(src, zip_path):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src))
    return zip_path


def test_import_zip_voc(tmp_path):
    from helpers.data import make_image

    src = tmp_path / "voc"
    make_image(src / "train" / "img1.png", 64, 48)
    (src / "train" / "img1.xml").write_text(
        "<annotation><filename>img1.png</filename>"
        "<size><width>64</width><height>48</height></size>"
        "<object><name>helmet</name>"
        "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>25</ymax></bndbox>"
        "</object></annotation>",
        encoding="utf-8",
    )
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, _zip_dir(src, tmp_path / "voc.zip"))
    assert summary.format == "voc"
    assert summary.num_images == 1
    assert summary.instances_per_category == {"helmet": 1}
    assert summary.split_counts == {"train": 1}


def test_import_zip_darknet(tmp_path):
    from helpers.data import make_image

    src = tmp_path / "darknet"
    make_image(src / "test" / "img1.png")
    (src / "test" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (src / "test" / "_darknet.labels").write_text("helmet\n", encoding="utf-8")
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, _zip_dir(src, tmp_path / "darknet.zip"))
    assert summary.format == "darknet"
    assert summary.num_images == 1
    assert summary.instances_per_category == {"helmet": 1}
    assert summary.split_counts == {"test": 1}


def test_import_zip_darknet_without_labels_requires_names(tmp_path):
    # the WebUI path (require_class_names=True) gets a structured 422-able error
    from helpers.data import make_image

    from horos.errors import ClassNamesRequiredError

    src = tmp_path / "darknet"
    make_image(src / "img1.png")
    (src / "img1.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    zip_path = _zip_dir(src, tmp_path / "darknet.zip")
    project = create_project(tmp_path / "proj")
    with pytest.raises(ClassNamesRequiredError) as exc:
        import_zip(project, zip_path, require_class_names=True)
    assert exc.value.default_names == ["0", "1"]
    # supplying names resolves it; the unused index-0 class is dropped as empty
    summary = import_zip(project, zip_path, class_names=["helmet", "vest"])
    assert summary.instances_per_category == {"vest": 1}
    assert summary.num_categories == 1
    assert any("empty categor" in w and "helmet" in w for w in summary.warnings)


def test_import_drops_empty_categories_with_warning(tmp_path):
    import json

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    for ann_file in coco_dir.rglob("_annotations.coco.json"):
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        data["categories"].append(
            {"id": 99, "name": "roboflow-superclass", "supercategory": "none"}
        )
        ann_file.write_text(json.dumps(data), encoding="utf-8")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, coco_dir)
    assert summary.num_categories == 2
    assert {c.name for c in project.categories} == {"forklift", "pallet"}
    assert any(
        "empty categor" in w and "roboflow-superclass" in w for w in summary.warnings
    )


def test_import_coco_keypoints_warns_not_silent(tmp_path):
    import json

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    for ann_file in coco_dir.rglob("_annotations.coco.json"):
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        for ann in data["annotations"]:
            ann["keypoints"] = [1.0, 2.0, 2, 3.0, 4.0, 1]
        ann_file.write_text(json.dumps(data), encoding="utf-8")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, coco_dir)
    # boxes still imported, keypoints explicitly reported as not imported
    assert summary.num_annotations == 4
    assert any("keypoint" in w and "NOT imported" in w for w in summary.warnings)
