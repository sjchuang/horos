"""E10-T8: a round trains on every labeled image only, holds out test and
valid shares that grow with the labels by stable hashing, drops pending
pre-labels from the snapshot, and moves the round on when the run finishes."""

from __future__ import annotations

import json
import time

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeDetector, FakeEmbedder
from helpers.runs import FAKE, ensure_worker_can_import_helpers

from horos.api.loop import (
    close_round,
    get_round,
    loop_status,
    round_training_status,
    select_round,
    train_readiness,
    train_round,
)
from horos.api.project import create_project
from horos.api.train import training_status
from horos.core.dataset import Annotation, Category
from horos.errors import ProjectError


def _project(tmp_path, *, total=30, labeled=24, pallet_every=4):
    """`total` images; the first `labeled` get a confirmed box ('pallet' on
    every `pallet_every`-th, 'box' otherwise). One labeled and one unlabeled
    image also carry a pending pre-label that must never reach a snapshot."""
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box"), Category(id=2, name="pallet")])
    colours = list(COLOURS.values())
    for n in range(1, total + 1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, colours[n % len(colours)])
        project.add_image(path, width=64 + n, height=48)
    for image_id in range(1, labeled + 1):
        cat = 2 if image_id % pallet_every == 0 else 1
        anns = [Annotation(id=1, image_id=image_id, category_id=cat, bbox=(10, 10, 30, 20))]
        if image_id == 1:
            anns.append(Annotation(id=2, image_id=1, category_id=1, bbox=(1, 1, 5, 5),
                                   source="auto", status="pending", score=0.4))
        project.save_annotations(image_id, anns, expected_version=0)
    project.save_annotations(
        total, [Annotation(id=1, image_id=total, category_id=1, bbox=(2, 2, 6, 6),
                           source="auto", status="pending", score=0.3)], expected_version=0,
    )
    return project


def _open_round(project, count=2):
    return select_round(project, count=count, embedder=FakeEmbedder(),
                        embedding_model="fake-embedder", detector=FakeDetector(),
                        preannotate=False)


def _wait(project, run_id, timeout=90):
    deadline = time.monotonic() + timeout
    while training_status(project, run_id).run.state in ("queued", "pending", "running"):
        assert time.monotonic() < deadline, "fake training did not finish"
        time.sleep(0.2)
    return training_status(project, run_id).run


def _snapshot_counts(project, run_id):
    base = project.root / "runs" / run_id / "dataset"
    images = annotations = pending = 0
    for split in ("train", "valid", "test"):
        path = base / split / "_annotations.coco.json"
        if not path.is_file():
            continue
        body = json.loads(path.read_text("utf-8"))
        images += len(body["images"])
        annotations += len(body["annotations"])
        fakes = ([1, 1, 5, 5], [2, 2, 6, 6])
        pending += sum(1 for a in body["annotations"] if a.get("bbox") in fakes)
    return images, annotations, pending


# ---------------------------------------------------------------- readiness


def test_readiness_names_every_blocker(tmp_path):
    project = _project(tmp_path, total=12, labeled=10)
    ready = train_readiness(project)
    assert not ready.ready and ready.labeled_images == 10
    assert any("needs at least 20" in r for r in ready.reasons)
    assert any("class 'pallet' has 2 instance(s)" in r for r in ready.reasons)
    assert ready.instances == {"box": 8, "pallet": 2}

    project = _project(tmp_path / "b", total=30, labeled=24)
    ready = train_readiness(project)
    assert ready.ready and ready.reasons == [] and ready.validation_images >= 1
    assert ready.instances == {"box": 18, "pallet": 6}


def test_training_refuses_until_ready_and_outside_labeling(tmp_path):
    project = _project(tmp_path, total=12, labeled=10)
    record = _open_round(project)
    with pytest.raises(ProjectError, match="Not ready to train: 10 labeled"):
        train_round(project, record.number, entrypoint_override=FAKE, epochs=1)
    close_round(project, record.number)
    with pytest.raises(ProjectError, match="is 'closed'; training starts from 'labeling'"):
        train_round(project, record.number, entrypoint_override=FAKE, epochs=1)


# ----------------------------------------------------------------- training


def test_round_trains_on_labeled_images_and_holds_out_test_and_valid(tmp_path):
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    record = _open_round(project, count=3)
    labeled_before = loop_status(project).labeled_images

    trained = train_round(project, record.number, entrypoint_override=FAKE, epochs=1)
    assert trained.state == "training" and trained.train_run_id
    hold = trained.training["holdout"]
    assert hold["ratios"] == {"train": 0.7, "valid": 0.1, "test": 0.2} and hold["seed"] == 42
    # ~20 % test, ~10 % valid of 24 labeled photos, at least one each, rest train
    assert 1 <= hold["test_images"] <= 9 and 1 <= hold["validation_images"] <= 6
    assert hold["train_images"] + hold["validation_images"] + hold["test_images"] == 24
    assert hold["newly_held_out"] == 0  # every photo joined its set when it was labeled
    by_id = {r.id: r for r in project.list_images()}
    assert all(by_id[i].split is None for i in range(25, 31))  # unlabeled: in no set
    assert loop_status(project).test_images == hold["test_images"]

    # the snapshot: 24 labeled images, 24 confirmed boxes, no pending pre-label;
    # the trainer sees train + valid, the test split is exported but never trained on
    images, annotations, pending = _snapshot_counts(project, trained.train_run_id)
    assert (images, annotations, pending) == (24, 24, 0)

    run = _wait(project, trained.train_run_id)
    assert run.state == "completed"
    assert run.dataset_images == 24
    assert run.dataset_splits == {"train": hold["train_images"], "valid": hold["validation_images"],
                                  "test": hold["test_images"]}

    status = round_training_status(project, record.number)
    assert status.round.state == "reviewing"
    assert status.round.metrics  # the fake backend reports a loss
    assert loop_status(project).current.state == "reviewing"
    assert loop_status(project).labeled_images == labeled_before
    assert loop_status(project).has_model


def test_held_out_sets_grow_with_the_labels_and_never_lose_a_photo(tmp_path):
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    first = _open_round(project, count=2)
    first = train_round(project, first.number, entrypoint_override=FAKE, epochs=1)
    _wait(project, first.train_run_id)
    held_before = {r.id: r.split for r in project.list_images() if r.split in ("valid", "test")}
    close_round(project, first.number)

    # 40 more labels arrive; the held-out sets must follow at their share
    for image_id in range(25, 31):
        stored = project.load_annotations(image_id)
        project.save_annotations(
            image_id, [Annotation(id=1, image_id=image_id, category_id=1, bbox=(5, 5, 20, 20))],
            expected_version=stored.version,
        )
    for n in range(31, 73):
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, (200, 30, 30))
        rec = project.add_image(path, width=64 + n, height=48)
        if n <= 70:  # two photos stay unlabeled so the next round has something to pick
            project.save_annotations(
                rec.id, [Annotation(id=1, image_id=rec.id, category_id=1, bbox=(5, 5, 20, 20))],
                expected_version=0,
            )
    second = _open_round(project, count=1)
    second = train_round(project, second.number, entrypoint_override=FAKE, epochs=1)
    hold = second.training["holdout"]
    splits_now = {r.id: r.split for r in project.list_images()}
    assert all(splits_now[i] == s for i, s in held_before.items())  # nothing moved back
    assert hold["test_images"] > sum(1 for s in held_before.values() if s == "test")
    total = hold["train_images"] + hold["validation_images"] + hold["test_images"]
    assert total == 70 and 8 <= hold["test_images"] <= 21  # ≈ 20 % of 70
    _wait(project, second.train_run_id)
    assert get_round(project, second.number).state == "reviewing"


def test_rounds_continue_from_the_previous_run_of_the_same_model(tmp_path):
    """E10-T8 continuous training: the first round starts fresh (nothing to
    continue from); the next warm-starts from round 1's best checkpoint —
    recorded on the round and passed to the run as init_from — and derives
    fewer epochs. "fresh" in the settings (or warm_start=False) opts out."""
    from horos.api.loop import update_loop_settings

    get_run = lambda project, run_id: training_status(project, run_id).run  # noqa: E731

    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    first = _open_round(project, count=2)
    first = train_round(project, first.number, entrypoint_override=FAKE)
    assert first.training["init_from"] is None and "fresh start" in first.training["init_reason"]
    run1 = _wait(project, first.train_run_id)
    assert run1.state == "completed" and run1.checkpoint
    fresh_epochs = next(h.value for h in run1.hparams if h.name == "epochs")
    close_round(project, first.number)

    second = _open_round(project, count=2)
    second = train_round(project, second.number, entrypoint_override=FAKE)
    assert second.training["init_from"] == first.train_run_id
    assert "continues from run" in second.training["init_reason"]
    run2 = get_run(project, second.train_run_id)
    assert run2.config["init_from"] == run1.checkpoint
    warm_epochs = next(h for h in run2.hparams if h.name == "epochs")
    assert warm_epochs.value == max(5, round(fresh_epochs * 0.5))
    assert "continuing from an earlier run" in warm_epochs.reason
    _wait(project, second.train_run_id)
    close_round(project, second.number)

    # opting out: the settings say fresh, or the call says so
    update_loop_settings(project, training="fresh")
    third = _open_round(project, count=2)
    third = train_round(project, third.number, entrypoint_override=FAKE, epochs=1)
    assert third.training["init_from"] is None
    assert third.training["init_reason"] == "fresh start by choice"
    assert get_run(project, third.train_run_id).config["init_from"] is None
    _wait(project, third.train_run_id)
    close_round(project, third.number)
    fourth = _open_round(project, count=2)
    fourth = train_round(project, fourth.number, entrypoint_override=FAKE, epochs=1,
                         warm_start=True)
    assert fourth.training["init_from"] == third.train_run_id  # newest completed run of the model
    _wait(project, fourth.train_run_id)


def test_failed_training_returns_the_round_to_labeling(tmp_path):
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    record = _open_round(project)
    trained = train_round(project, record.number, entrypoint_override=FAKE, epochs=1,
                          extra={"fail": True})
    run = _wait(project, trained.train_run_id)
    assert run.state == "failed"
    after = get_round(project, record.number)
    assert after.state == "labeling"
    assert "simulated failure" in after.training["error"]
    # the user can fix things and train again from the same round
    again = train_round(project, record.number, entrypoint_override=FAKE, epochs=1)
    assert again.state == "training" and again.train_run_id != trained.train_run_id
    _wait(project, again.train_run_id)
    assert get_round(project, record.number).state == "reviewing"


def test_loop_picks_segmentation_model_when_labels_are_polygons(tmp_path):
    """E10-T18: polygons on most confirmed labels → RF-DETR-Seg Nano, boxes → RF-DETR Nano."""
    from horos.api.loop import default_model_for

    project = _project(tmp_path)
    labeled = {r.id for r in project.list_images() if r.id <= 24}
    assert default_model_for(project, labeled)[0] == "rfdetr-nano"
    for image_id in range(1, 15):  # 14 of 24 become polygons
        stored = project.load_annotations(image_id)
        anns = [a.model_copy(update={"segmentation": [[10, 10, 40, 10, 40, 30, 10, 30]]})
                for a in stored.annotations]
        project.save_annotations(image_id, anns, expected_version=stored.version)
    model, reason = default_model_for(project, labeled)
    assert model == "rfdetr-seg-nano" and "polygons" in reason

    ensure_worker_can_import_helpers()
    record = _open_round(project, count=2)
    trained = train_round(project, record.number, entrypoint_override=FAKE, epochs=1)
    assert trained.training["model"] == "rfdetr-seg-nano"
    assert "instance segmentation" in trained.training["model_reason"]
    _wait(project, trained.train_run_id)


def test_readiness_offers_to_train_without_the_short_classes(tmp_path):
    """Only the per-class minimum blocks: readiness names the short classes
    and says training without them would work (E10-T8)."""
    project = _project(tmp_path, total=40, labeled=30, pallet_every=10)
    ready = train_readiness(project)
    assert not ready.ready and ready.short_classes == ["pallet"]
    assert ready.instances == {"box": 27, "pallet": 3}
    assert ready.ready_without_short and ready.labeled_without_short == 27

    # too few labeled photos overall: dropping classes does not help
    few = _project(tmp_path / "b", total=12, labeled=10)
    assert few.root and not train_readiness(few).ready_without_short


def test_round_can_train_without_the_short_classes(tmp_path):
    ensure_worker_can_import_helpers()
    project = _project(tmp_path, total=40, labeled=30, pallet_every=10)
    record = _open_round(project, count=3)
    with pytest.raises(ProjectError, match="class 'pallet' has 3"):
        train_round(project, record.number, entrypoint_override=FAKE, epochs=1)

    trained = train_round(
        project, record.number, entrypoint_override=FAKE, epochs=1, ignore_short_classes=True
    )
    assert trained.training["ignored_classes"] == ["pallet"]
    assert "without pallet" in trained.training["ignored_reason"]
    # the snapshot keeps the 27 box photos and their 27 boxes; pallet is gone
    images, annotations, pending = _snapshot_counts(project, trained.train_run_id)
    assert (images, annotations, pending) == (27, 27, 0)
    assert _wait(project, trained.train_run_id).state == "completed"

    few = _project(tmp_path / "b", total=12, labeled=10)
    rec = _open_round(few)
    with pytest.raises(ProjectError, match="even without the short classes: 10 labeled"):
        train_round(few, rec.number, entrypoint_override=FAKE, epochs=1, ignore_short_classes=True)
