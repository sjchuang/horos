"""E2-T4: category management (add, rename, recolor, delete)."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.labels import (
    add_category,
    delete_category,
    merge_categories,
    update_category,
)
from horos.api.project import create_project, open_project
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_add_category(project):
    cat = add_category(project, "person", color="#ff0000")
    assert cat.name == "person" and cat.color == "#ff0000"
    reopened = open_project(project.root)
    assert "person" in {c.name for c in reopened.categories}


def test_add_assigns_fresh_id_and_default_color(project):
    existing_ids = {c.id for c in project.categories}
    cat = add_category(project, "person")
    assert cat.id not in existing_ids
    assert cat.color.startswith("#")


def test_duplicate_name_is_rejected(project):
    with pytest.raises(ProjectError, match="already exists"):
        add_category(project, "forklift")


def test_empty_name_is_rejected(project):
    with pytest.raises(ProjectError, match="not be empty"):
        add_category(project, "   ")


def test_rename(project):
    target = next(c for c in project.categories if c.name == "forklift")
    updated = update_category(project, target.id, name="lift_truck")
    assert updated.name == "lift_truck"
    assert {c.name for c in open_project(project.root).categories} == {
        "lift_truck",
        "pallet",
    }


def test_rename_to_taken_name_is_rejected(project):
    target = next(c for c in project.categories if c.name == "forklift")
    with pytest.raises(ProjectError, match="already exists"):
        update_category(project, target.id, name="pallet")


def test_recolor_keeps_name(project):
    target = project.categories[0]
    updated = update_category(project, target.id, color="#123456")
    assert updated.color == "#123456" and updated.name == target.name


def test_delete_unreferenced(project):
    cat = add_category(project, "person")
    assert delete_category(project, cat.id) == 0
    assert cat.id not in {c.id for c in open_project(project.root).categories}


def test_delete_referenced_is_refused(project):
    target = next(c for c in project.categories if c.name == "forklift")
    with pytest.raises(ProjectError, match="force=True") as info:
        delete_category(project, target.id)
    # a distinct code with the counts: the UI asks "delete them too?" only for this
    from horos.errors import CategoryInUseError

    assert isinstance(info.value, CategoryInUseError) and info.value.code == "category_in_use"
    assert info.value.details["annotations"] >= 1 and info.value.details["images"] >= 1
    assert f"'{target.name}' is used by" in str(info.value)


def test_forced_delete_cascades_and_bumps_versions(project):
    target = next(c for c in project.categories if c.name == "forklift")
    affected = [
        r.id
        for r in project.list_images()
        if any(
            a.category_id == target.id
            for a in project.load_annotations(r.id).annotations
        )
    ]
    versions_before = {i: project.load_annotations(i).version for i in affected}
    deleted = delete_category(project, target.id, force=True)
    assert deleted == 2
    for image_id in affected:
        stored = project.load_annotations(image_id)
        assert all(a.category_id != target.id for a in stored.annotations)
        assert stored.version == versions_before[image_id] + 1


def test_unknown_id_is_explicit(project):
    with pytest.raises(ProjectError, match="No category"):
        update_category(project, 999, name="x")


def _count(project, category_id):
    return sum(
        1
        for r in project.list_images()
        for a in project.load_annotations(r.id).annotations
        if a.category_id == category_id
    )


def test_merge_relabels_annotations_and_removes_sources(project):
    forklift = next(c for c in project.categories if c.name == "forklift")
    pallet = next(c for c in project.categories if c.name == "pallet")
    before = _count(project, forklift.id) + _count(project, pallet.id)
    touched = [
        r.id for r in project.list_images()
        if any(a.category_id == pallet.id for a in project.load_annotations(r.id).annotations)
    ]
    versions = {i: project.load_annotations(i).version for i in touched}

    result = merge_categories(project, [pallet.id], forklift.id)
    assert result.target.id == forklift.id and result.target.name == "forklift"
    assert result.merged_annotations == 2 and result.images_touched == len(touched) == 2
    assert result.removed_ids == [pallet.id]
    # nothing lost: every former pallet box is now a forklift box
    assert _count(project, forklift.id) == before and _count(project, pallet.id) == 0
    assert [c.name for c in open_project(project.root).categories] == ["forklift"]
    for image_id in touched:
        assert project.load_annotations(image_id).version == versions[image_id] + 1


def test_merge_several_sources_at_once_and_keep_target_color(project):
    forklift = next(c for c in project.categories if c.name == "forklift")
    pallet = next(c for c in project.categories if c.name == "pallet")
    crate = add_category(project, "crate")  # unreferenced source: still removed
    result = merge_categories(project, [pallet.id, crate.id, pallet.id], forklift.id)
    assert result.removed_ids == [pallet.id, crate.id]
    assert result.target.color == forklift.color
    assert {c.id for c in project.categories} == {forklift.id}


def test_merge_input_validation(project):
    forklift = next(c for c in project.categories if c.name == "forklift")
    with pytest.raises(ProjectError, match="into itself"):
        merge_categories(project, [forklift.id], forklift.id)
    with pytest.raises(ProjectError, match="at least one source"):
        merge_categories(project, [], forklift.id)
    with pytest.raises(ProjectError, match="No category"):
        merge_categories(project, [999], forklift.id)
    with pytest.raises(ProjectError, match="No category"):
        merge_categories(project, [forklift.id], 999)
    # nothing changed after the refusals
    assert len(project.categories) == 2
