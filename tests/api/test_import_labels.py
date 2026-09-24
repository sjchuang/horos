"""E1-T11: labels arriving for photos the project already has attach to them
instead of vanishing with the duplicate, and existing labels are never
replaced without a decision (on_annotations: ask / replace / merge / skip)."""

import json

import pytest
from helpers.data import make_image, sample_dataset, write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.core.dataset import Annotation
from horos.core.formats.coco import write_coco
from horos.errors import LabelConflictError, ProjectError


@pytest.fixture
def photos(tmp_path):
    """Three photos with the sample dataset's names, imported with no labels."""
    root = tmp_path / "photos"
    for record in sample_dataset().images:
        make_image(root / record.file_name, record.width, record.height)
    project = create_project(tmp_path / "proj")
    summary = import_dataset(project, root)
    assert summary.format == "images" and summary.num_images == 3
    return project


def _by_name(project):
    return {r.file_name: r for r in project.list_images()}


def _label(project, name, *, category_id=1, bbox=(1.0, 1.0, 5.0, 5.0)):
    """A hand-drawn box on one photo, as the annotator would save it."""
    record = _by_name(project)[name]
    current = project.load_annotations(record.id)
    project.save_annotations(
        record.id,
        [Annotation(id=1, image_id=record.id, category_id=category_id, bbox=bbox)],
        expected_version=current.version,
    )
    return record


def test_labels_attach_to_unlabeled_photos_and_they_join_a_set(photos, tmp_path):
    summary = import_dataset(photos, write_sample_coco_dir(tmp_path / "coco"))
    assert summary.num_images == 0  # no photo was copied
    assert summary.images_matched == 3
    assert summary.num_annotations == 4
    assert summary.annotation_conflict_files == []
    assert summary.duplicates_skipped == 0
    assert summary.instances_per_category == {"forklift": 2, "pallet": 2}
    by_name = _by_name(photos)
    assert len(photos.load_annotations(by_name["a.png"].id).annotations) == 2
    assert all(r.split is not None for r in by_name.values())
    assert summary.split_counts == {}  # counts describe copied photos only
    assert any("assigned" in w for w in summary.warnings)


def test_same_labels_again_are_a_duplicate_not_a_conflict(photos, tmp_path):
    coco = write_sample_coco_dir(tmp_path / "coco")
    import_dataset(photos, coco)
    again = import_dataset(photos, coco)  # default on_annotations="ask" must not raise
    assert again.images_matched == 0
    assert again.duplicates_skipped == 3
    assert again.num_annotations == 0
    assert len(photos.load_annotations(_by_name(photos)["a.png"].id).annotations) == 2


def test_ask_names_only_the_photos_with_labels_and_writes_nothing(photos, tmp_path):
    photos.set_categories(sample_dataset().categories)
    _label(photos, "a.png")
    before = {r.id: photos.load_annotations(r.id).version for r in photos.list_images()}
    with pytest.raises(LabelConflictError) as exc:
        import_dataset(photos, write_sample_coco_dir(tmp_path / "coco"))
    assert exc.value.conflicts == ["a.png"]
    assert exc.value.details == {"conflicts": ["a.png"]}
    assert exc.value.code == "label_conflict"
    # nothing was written — not even for the photos that had no labels
    assert {r.id: photos.load_annotations(r.id).version for r in photos.list_images()} == before
    assert len(photos.load_annotations(_by_name(photos)["b.png"].id).annotations) == 0


def test_replace_swaps_the_labels_of_the_conflicting_photo_only(photos, tmp_path):
    photos.set_categories(sample_dataset().categories)
    _label(photos, "a.png", bbox=(9.0, 9.0, 3.0, 3.0))
    summary = import_dataset(
        photos, write_sample_coco_dir(tmp_path / "coco"), on_annotations="replace"
    )
    assert summary.images_matched == 3
    assert summary.annotation_conflict_files == ["a.png"]
    assert summary.annotations_replaced == 1
    assert summary.annotations_merged == summary.annotations_kept == 0
    assert summary.num_annotations == 4
    labels = photos.load_annotations(_by_name(photos)["a.png"].id).annotations
    assert [a.bbox for a in labels] == [(4.0, 4.0, 16.0, 12.0), (20.0, 8.0, 8.0, 8.0)]
    assert [a.id for a in labels] == [1, 2]


def test_merge_keeps_both_sets_with_continued_ids(photos, tmp_path):
    photos.set_categories(sample_dataset().categories)
    _label(photos, "a.png", bbox=(9.0, 9.0, 3.0, 3.0))
    summary = import_dataset(
        photos, write_sample_coco_dir(tmp_path / "coco"), on_annotations="merge"
    )
    assert summary.annotations_merged == 1 and summary.annotations_replaced == 0
    labels = photos.load_annotations(_by_name(photos)["a.png"].id).annotations
    assert [a.bbox for a in labels] == [
        (9.0, 9.0, 3.0, 3.0), (4.0, 4.0, 16.0, 12.0), (20.0, 8.0, 8.0, 8.0),
    ]
    assert [a.id for a in labels] == [1, 2, 3]


def test_skip_leaves_the_labeled_photo_alone_and_labels_the_rest(photos, tmp_path):
    photos.set_categories(sample_dataset().categories)
    _label(photos, "a.png", bbox=(9.0, 9.0, 3.0, 3.0))
    summary = import_dataset(
        photos, write_sample_coco_dir(tmp_path / "coco"), on_annotations="skip"
    )
    assert summary.annotations_kept == 1
    assert summary.images_matched == 3
    assert summary.num_annotations == 2  # b.png and c.png received theirs
    by_name = _by_name(photos)
    assert [a.bbox for a in photos.load_annotations(by_name["a.png"].id).annotations] == [
        (9.0, 9.0, 3.0, 3.0)
    ]
    assert len(photos.load_annotations(by_name["b.png"].id).annotations) == 1


def test_bad_policy_is_explicit(photos, tmp_path):
    with pytest.raises(ProjectError, match="on_annotations must be one of"):
        import_dataset(photos, write_sample_coco_dir(tmp_path / "coco"), on_annotations="yes")


def _bare_label_file(tmp_path, *, resize=None):
    """A COCO json alone — no photos next to it — naming the sample photos."""
    dataset = sample_dataset()
    if resize:
        record = dataset.image_by_id(1)
        record.width, record.height = resize
    root = tmp_path / "labels_only"
    root.mkdir()
    write_coco(dataset, root, image_paths={}, split_layout=False, copy_images=False)
    return root


def test_a_bare_label_file_matches_photos_by_name(photos, tmp_path):
    summary = import_dataset(photos, _bare_label_file(tmp_path))
    assert summary.num_images == 0 and summary.images_matched == 3
    assert summary.num_annotations == 4
    assert not any("missing" in w for w in summary.warnings)
    assert all(r.split is not None for r in photos.list_images())


def test_a_bare_label_file_with_another_size_is_a_different_photo(photos, tmp_path):
    summary = import_dataset(photos, _bare_label_file(tmp_path, resize=(640, 480)))
    assert summary.images_matched == 2  # a.png (640x480 in the file) is not ours
    assert summary.num_annotations == 2
    assert any("a.png" in w and "640x480" in w and "64x48" in w for w in summary.warnings)
    assert photos.load_annotations(_by_name(photos)["a.png"].id).annotations == []


def test_labels_for_a_conflicting_file_follow_the_file_decision(photos, tmp_path):
    # different bytes under the same name: that is on_conflict's call, and a
    # skipped file must not have its labels applied to the photo it is not
    coco = write_sample_coco_dir(tmp_path / "coco")
    make_image(coco / "train" / "a.png", 64, 48, color=(1, 2, 3))
    summary = import_dataset(photos, coco, on_conflict="skip")
    assert summary.conflicts_skipped == 1
    assert summary.images_matched == 2
    assert photos.load_annotations(_by_name(photos)["a.png"].id).annotations == []


def test_coco_written_without_photos_is_readable(tmp_path):
    # guards the helper above: write_coco with no image paths still emits the json
    root = _bare_label_file(tmp_path)
    data = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
    assert len(data["images"]) == 3 and len(data["annotations"]) == 4
