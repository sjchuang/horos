"""E1-T1: Project object and on-disk structure — create, load, validate."""

import pytest
from helpers.data import make_image

from horos.core.dataset import Annotation, Category
from horos.core.project import Project
from horos.errors import AnnotationConflictError, ProjectError


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / "proj", name="demo")


def test_create_builds_expected_structure(project):
    assert (project.root / "horos.json").exists()
    assert (project.root / "images.json").exists()
    assert project.images_dir.is_dir()
    assert project.annotations_dir.is_dir()
    assert project.runs_dir.is_dir()
    assert project.manifest.name == "demo"


def test_create_refuses_existing_project(project):
    with pytest.raises(ProjectError, match="already exists"):
        Project.create(project.root)


def test_create_refuses_nonempty_directory(tmp_path):
    (tmp_path / "junk.txt").write_text("x")
    with pytest.raises(ProjectError, match="non-empty"):
        Project.create(tmp_path)


def test_open_roundtrip(project):
    reopened = Project.open(project.root)
    assert reopened.manifest.name == "demo"


def test_open_missing_project_is_explicit(tmp_path):
    with pytest.raises(ProjectError, match="No horos project"):
        Project.open(tmp_path / "nope")


def test_open_corrupt_manifest_is_explicit(project):
    (project.root / "horos.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError, match="Corrupt project manifest"):
        Project.open(project.root)


def test_open_detects_missing_directories(project):
    import shutil

    shutil.rmtree(project.annotations_dir)
    with pytest.raises(ProjectError, match="missing directories"):
        Project.open(project.root)


def test_categories_roundtrip_and_get_default_colors(project):
    project.set_categories([Category(id=1, name="forklift")])
    reopened = Project.open(project.root)
    assert reopened.categories[0].color  # default color assigned
    with pytest.raises(ProjectError, match="Duplicate category name"):
        project.set_categories(
            [Category(id=1, name="a"), Category(id=2, name="a")]
        )


def test_add_image_copies_by_default(project, tmp_path):
    src = make_image(tmp_path / "src" / "cat.png", 64, 48)
    record = project.add_image(src, width=64, height=48)
    assert (project.images_dir / record.file_name).exists()
    assert record.external_path is None
    assert project.image_path(record).parent == project.images_dir


def test_add_image_reference_mode_keeps_source_in_place(project, tmp_path):
    src = make_image(tmp_path / "src" / "cat.png", 64, 48)
    record = project.add_image(src, width=64, height=48, copy=False)
    assert not (project.images_dir / record.file_name).exists()
    assert project.image_path(record) == src.resolve()


def test_add_image_resolves_name_collisions(project, tmp_path):
    a = make_image(tmp_path / "one" / "cat.png")
    b = make_image(tmp_path / "two" / "cat.png")
    r1 = project.add_image(a, width=64, height=48)
    r2 = project.add_image(b, width=64, height=48)
    assert r1.file_name != r2.file_name
    assert r1.id != r2.id


def test_annotation_save_load_roundtrip(project, tmp_path):
    src = make_image(tmp_path / "cat.png")
    record = project.add_image(src, width=64, height=48)
    ann = Annotation(id=1, image_id=record.id, category_id=1, bbox=(1, 2, 3, 4))
    saved = project.save_annotations(record.id, [ann], expected_version=0)
    assert saved.version == 1
    loaded = project.load_annotations(record.id)
    assert loaded.version == 1
    assert loaded.annotations == [ann]


def test_annotation_optimistic_lock_conflict(project, tmp_path):
    # E2-T8 foundation: stale writers must get a conflict, not clobber.
    src = make_image(tmp_path / "cat.png")
    record = project.add_image(src, width=64, height=48)
    ann = Annotation(id=1, image_id=record.id, category_id=1, bbox=(1, 2, 3, 4))
    project.save_annotations(record.id, [ann], expected_version=0)
    with pytest.raises(AnnotationConflictError, match="another session"):
        project.save_annotations(record.id, [ann], expected_version=0)


def test_to_dataset_assembles_everything(project, tmp_path):
    project.set_categories([Category(id=1, name="forklift")])
    src = make_image(tmp_path / "cat.png")
    record = project.add_image(src, width=64, height=48)
    ann = Annotation(id=1, image_id=record.id, category_id=1, bbox=(1, 2, 3, 4))
    project.save_annotations(record.id, [ann], expected_version=0)
    ds = project.to_dataset()
    assert len(ds.images) == 1 and len(ds.annotations) == 1
    assert ds.categories[0].name == "forklift"


# ------------------------------------------- image index cache (performance)
#
# list_images() re-parsed images.json on every call, and one web request makes
# dozens of them (~25 ms each on a 20 000-photo project). The parse is cached
# per Project instance and keyed on the file's (mtime_ns, size). These tests
# pin the three properties that make that safe.


def test_repeated_reads_reuse_one_parse_until_the_file_changes(project, tmp_path):
    project.add_image(make_image(tmp_path / "a.png"), width=8, height=8)
    first = project._load_image_index()
    assert project._load_image_index() is first  # cache hit: same object
    project.add_image(make_image(tmp_path / "b.png"), width=8, height=8)
    assert project._load_image_index() is not first  # the file moved on
    assert len(project.list_images()) == 2


def test_a_write_by_another_instance_is_picked_up(project, tmp_path):
    record = project.add_image(make_image(tmp_path / "a.png"), width=8, height=8)
    assert project.list_images()[0].excluded is False  # fills the cache
    other = Project.open(project.root)  # stands in for a second process
    other.set_excluded([record.id], True)
    assert project.list_images()[0].excluded is True


def test_mutators_never_edit_the_cached_records(project, tmp_path):
    record = project.add_image(make_image(tmp_path / "a.png"), width=8, height=8)
    cached = project._load_image_index()
    project.set_excluded([record.id], True)
    # the mutator worked on its own copy, so the object that was cached when it
    # ran is untouched — a half-finished edit can never leak into a reader
    assert cached.images[0].excluded is False
    assert project.list_images()[0].excluded is True


def test_list_images_hands_out_a_list_the_caller_may_keep(project, tmp_path):
    project.add_image(make_image(tmp_path / "a.png"), width=8, height=8)
    images = project.list_images()
    images.clear()  # a caller's own list, not the project's
    assert len(project.list_images()) == 1


def test_get_image_finds_records_and_reports_unknown_ids(project, tmp_path):
    record = project.add_image(make_image(tmp_path / "a.png"), width=8, height=8)
    assert project.get_image(record.id).file_name == record.file_name
    assert project.get_image(record.id).file_name == record.file_name  # cached path
    with pytest.raises(ProjectError, match="No image with id"):
        project.get_image(4242)
