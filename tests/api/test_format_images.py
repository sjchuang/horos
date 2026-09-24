"""E1-T11: photos with no annotation file are the "images" format — detected
last of all formats, read with sizes probed from the files, and imported as
unlabeled photos in no set."""

import zipfile

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.api.dataset import import_dataset, import_zip
from horos.api.project import create_project
from horos.core.formats import detect_format
from horos.core.formats.images import find_images, read_images
from horos.errors import DatasetFormatError


@pytest.fixture
def photos_dir(tmp_path):
    root = tmp_path / "photos"
    make_image(root / "one.jpg", 64, 48)
    make_image(root / "nested" / "two.png", 32, 32)
    make_image(root / "nested" / "three.webp", 40, 20)
    (root / "notes.txt").write_text("not a photo", encoding="utf-8")
    return root


def test_detection_falls_back_to_images_only_when_nothing_describes_them(photos_dir, tmp_path):
    assert detect_format(photos_dir) == "images"
    # a real layout always wins over the fallback
    assert detect_format(write_sample_coco_dir(tmp_path / "coco")) == "coco"
    # nothing at all is still undetectable, never "images"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert detect_format(empty) is None
    (empty / "readme.txt").write_text("hi", encoding="utf-8")
    assert detect_format(empty) is None


def test_read_images_probes_sizes_and_warns_about_the_missing_labels(photos_dir):
    dataset, paths = read_images(photos_dir)
    # deterministic: sorted by path, so nested/ comes before the top-level photo
    assert [i.file_name for i in dataset.images] == ["three.webp", "two.png", "one.jpg"]
    assert {(i.width, i.height) for i in dataset.images} == {(64, 48), (32, 32), (40, 20)}
    assert dataset.annotations == [] and dataset.categories == []
    assert all(paths[i.id].is_file() for i in dataset.images)
    assert any("no labels" in w for w in dataset.reader_warnings)


def test_read_images_refuses_an_unreadable_photo(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "broken.jpg").write_bytes(b"not really a jpeg")
    with pytest.raises(DatasetFormatError, match="Cannot read image"):
        read_images(root)
    with pytest.raises(DatasetFormatError, match="No photos"):
        read_images(tmp_path / "missing")


def test_a_single_photo_file_is_an_import_too(tmp_path):
    photo = make_image(tmp_path / "solo.png", 20, 10)
    assert find_images(photo) == [photo]
    assert detect_format(photo) == "images"
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, photo)
    assert summary.format == "images" and summary.num_images == 1
    assert project.list_images()[0].split is None


def test_import_photos_leaves_them_unlabeled_and_in_no_set(photos_dir, tmp_path):
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, photos_dir)
    assert summary.format == "images"
    assert summary.num_images == 3 and summary.num_annotations == 0
    assert summary.num_categories == 0 and summary.instances_per_category == {}
    assert summary.split_counts == {"unassigned": 3}
    assert any("No annotation file found" in w for w in summary.warnings)
    records = project.list_images()
    assert [r.split for r in records] == [None, None, None]
    assert all((project.images_dir / r.file_name).is_file() for r in records)
    assert all(project.load_annotations(r.id).annotations == [] for r in records)


def test_import_photos_under_a_split_directory_still_join_no_set(tmp_path):
    # the directory name says "train", but only labeled photos are set members
    root = tmp_path / "src"
    make_image(root / "train" / "a.png")
    make_image(root / "valid" / "b.png")
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, root)
    assert summary.split_counts == {"unassigned": 2}
    assert [r.split for r in project.list_images()] == [None, None]


def test_explicit_format_images_ignores_a_stray_label_file(tmp_path):
    root = tmp_path / "src"
    make_image(root / "a.png")
    (root / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # looks like Darknet
    assert detect_format(root) == "darknet"
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, root, format="images")
    assert summary.format == "images" and summary.num_annotations == 0


def test_import_zip_of_photos(photos_dir, tmp_path):
    zip_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in photos_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(photos_dir))
    project = create_project(tmp_path / "proj")
    summary = import_zip(project, zip_path)
    assert summary.format == "images" and summary.num_images == 3
    # the same photos again are duplicates, silently
    again = import_zip(project, zip_path)
    assert again.num_images == 0 and again.duplicates_skipped == 3
    assert again.images_matched == 0 and again.annotation_conflict_files == []
