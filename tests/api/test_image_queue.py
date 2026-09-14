"""E2-T7: server-side image queue — ordering, filters, claim visibility."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.annotate import claim_image, image_queue
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_unannotated_first_default(project):
    queue = image_queue(project)
    assert len(queue) == 3
    flags = [item.annotated for item in queue]
    assert flags == sorted(flags)  # unannotated (False) sorts first


def test_file_name_mode(project):
    queue = image_queue(project, mode="file_name")
    names = [i.image.file_name for i in queue]
    assert names == sorted(names)


def test_filter_modes(project):
    assert all(i.annotated for i in image_queue(project, mode="annotated"))
    assert all(not i.annotated for i in image_queue(project, mode="unannotated"))


def test_split_filter(project):
    queue = image_queue(project, split="valid")
    assert queue and all(i.image.split == "valid" for i in queue)


def test_class_filter_keeps_photos_carrying_that_class(project):
    forklift, pallet = (c.id for c in project.categories)
    assert [i.image.file_name for i in image_queue(project, category_id=forklift)] == [
        "a.png", "b.png",
    ]
    assert [i.image.file_name for i in image_queue(project, category_id=pallet)] == [
        "a.png", "c.png",
    ]
    # filters combine: pallet photos in the valid set
    assert [
        i.image.file_name for i in image_queue(project, category_id=pallet, split="valid")
    ] == ["c.png"]


def test_class_filter_rejects_unknown_class(project):
    with pytest.raises(ProjectError, match="Unknown category id 99"):
        image_queue(project, category_id=99)


def test_bad_mode_is_explicit(project):
    with pytest.raises(ProjectError, match="queue mode"):
        image_queue(project, mode="random")


def test_claims_visible_to_other_sessions_only(project):
    target = image_queue(project)[0].image.id
    claim_image(project, target, "session-a")
    seen_by_b = next(
        i for i in image_queue(project, session_id="session-b") if i.image.id == target
    )
    assert seen_by_b.claimed_by == "session-a"
    seen_by_a = next(
        i for i in image_queue(project, session_id="session-a") if i.image.id == target
    )
    assert seen_by_a.claimed_by is None  # your own claim never steers you away
