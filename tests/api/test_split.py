"""E1-T8: split management. Only labeled photos belong to train / valid /
test: a photo joins a set the first time it carries a confirmed annotation,
by a stable hash and the project's ratios; unlabeled photos are in no set.
"Re-split" assigns the unassigned; reshuffle re-draws every labeled photo."""

import json

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.api.dataset import import_dataset, resplit
from horos.api.project import create_project, open_project
from horos.core.dataset import Annotation, Category
from horos.core.splitting import SplitRatios, bucket, split_for
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project


def test_import_preserves_existing_splits(project):
    splits = {i.file_name: i.split for i in project.list_images()}
    assert sorted(splits.values()) == ["train", "train", "valid"]


def test_resplit_is_deterministic_under_seed(project):
    counts1 = resplit(project, train=0.4, valid=0.3, test=0.3, seed=7)
    first = {i.id: i.split for i in project.list_images()}
    counts2 = resplit(project, train=0.4, valid=0.3, test=0.3, seed=7)
    second = {i.id: i.split for i in project.list_images()}
    assert first == second
    assert counts1 == counts2


def test_resplit_ratio_shapes_assignment(project):
    # without reshuffle the photos already in a set stay put
    kept = {i.id: i.split for i in project.list_images()}
    assert resplit(project, train=1.0, valid=0.0, test=0.0, seed=1)["unassigned"] == 0
    assert {i.id: i.split for i in project.list_images()} == kept
    assert project.split_ratios == SplitRatios(train=1.0, valid=0.0, test=0.0)
    counts = resplit(project, reshuffle=True)
    assert counts == {"train": 3, "valid": 0, "test": 0, "unassigned": 0}
    assert all(i.split == "train" for i in project.list_images())


def _bare_project(tmp_path, n=6):
    project = create_project(tmp_path / "bare")
    project.set_categories([Category(id=1, name="box")])
    for k in range(1, n + 1):
        project.add_image(make_image(tmp_path / "src" / f"{k}.png", 32, 32), width=32, height=32)
    return project


def _label(project, image_id, status="confirmed"):
    current = project.load_annotations(image_id)
    project.save_annotations(
        image_id,
        [Annotation(id=1, image_id=image_id, category_id=1, bbox=(1, 1, 5, 5), status=status,
                    source="manual" if status == "confirmed" else "auto")],
        expected_version=current.version,
    )


def test_unlabeled_photos_are_in_no_set_and_join_one_when_first_labeled(tmp_path):
    project = _bare_project(tmp_path)
    assert all(i.split is None for i in project.list_images())
    # a pending pseudo-label is not a label
    _label(project, 1, status="pending")
    assert project.get_image(1).split is None
    _label(project, 1)
    # the only labeled photo must be trainable, whatever its bucket says
    assert project.get_image(1).split == "train"
    # from then on the stable hash decides, whatever the order of labeling
    for image_id in (4, 2, 3):
        _label(project, image_id)
    for image_id in (2, 3, 4):
        assert project.get_image(image_id).split is not None
    ratios, seed = project.split_ratios, project.manifest.split_seed
    hashed = {i: split_for(i, ratios, seed) for i in (2, 3, 4)}
    forced = [i for i in (2, 3, 4) if project.get_image(i).split != hashed[i]]
    assert len(forced) <= 2  # at most the "each set has a member" guard moved one per set
    assert project.get_image(5).split is None and project.get_image(6).split is None
    # once three or more photos are labeled, valid and test each have a member
    splits = {i.split for i in project.list_images() if i.split}
    assert splits == {"train", "valid", "test"}


def test_assign_only_touches_unassigned_and_reshuffle_only_labeled(tmp_path):
    project = _bare_project(tmp_path, n=8)
    for image_id in range(1, 6):
        _label(project, image_id)
    before = {i.id: i.split for i in project.list_images()}
    counts = resplit(project, train=0.5, valid=0.25, test=0.25)
    assert counts["unassigned"] == 3 and sum(counts.values()) == 8
    assert {i.id: i.split for i in project.list_images()} == before  # nothing moved
    shuffled = resplit(project, reshuffle=True, seed=3)
    assert shuffled["unassigned"] == 3  # the unlabeled photos stay out of every set
    assert all(project.get_image(i).split is None for i in (6, 7, 8))
    assert all(project.get_image(i).split is not None for i in range(1, 6))


def test_hash_buckets_are_stable_and_follow_the_ratios():
    ratios = SplitRatios()  # 70 / 10 / 20
    ids = range(1, 2001)
    first = [split_for(i, ratios, 42) for i in ids]
    assert first == [split_for(i, ratios, 42) for i in ids]
    share = {s: first.count(s) / len(first) for s in ("train", "valid", "test")}
    assert abs(share["train"] - 0.7) < 0.03 and abs(share["test"] - 0.2) < 0.03
    assert 0 <= bucket(42, 1) < 1 and bucket(42, 1) != bucket(43, 1)
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(train=0.5, valid=0.1, test=0.1)


def test_old_projects_lose_the_split_of_unlabeled_photos_on_first_open(tmp_path):
    """Indexes written before this change gave every photo split="train" on
    arrival (and re-splits put unlabeled photos in valid/test). Opening such
    a project keeps the labeled photos' sets and clears the rest, once."""
    project = _bare_project(tmp_path, n=4)
    _label(project, 1)
    _label(project, 2)
    path = project.root / "images.json"
    old = json.loads(path.read_text())
    old.pop("version", None)
    by_id = {r["id"]: r for r in old["images"]}
    by_id[1]["split"] = "valid"   # labeled: kept
    by_id[3]["split"] = "train"   # unlabeled: cleared
    by_id[4]["split"] = "test"    # unlabeled, even in test: cleared
    path.write_text(json.dumps(old))
    reopened = open_project(project.root)
    splits = {i.id: i.split for i in reopened.list_images()}
    assert splits[1] == "valid" and splits[2] is not None
    assert splits[3] is None and splits[4] is None
    assert json.loads(path.read_text())["version"] >= 2  # migrated on disk, once


def test_resplit_rejects_bad_ratios(project):
    with pytest.raises(ProjectError, match="sum to 1.0"):
        resplit(project, train=0.9, valid=0.9, test=0.1)


def test_resplit_rejects_empty_project(tmp_path):
    project = create_project(tmp_path / "empty_proj")
    with pytest.raises(ProjectError, match="no images"):
        resplit(project)


def test_resplit_uses_attributes_not_symlinks(project):
    # R7: no symlinks anywhere in split handling
    resplit(project, seed=3)
    links = [p for p in project.root.rglob("*") if p.is_symlink()]
    assert links == []
