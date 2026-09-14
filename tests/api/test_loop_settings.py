"""E10-T19: the loop's standing choices — model, suggestions on/off, box or
polygon shapes — persist per project and steer selection and training."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeDetector, FakeEmbedder, FakePromptableSegmenter
from helpers.runs import FAKE, ensure_worker_can_import_helpers

from horos.api.loop import (
    get_loop_settings,
    loop_status,
    select_round,
    train_round,
    update_loop_settings,
)
from horos.api.project import create_project, open_project
from horos.core.dataset import Annotation, Category
from horos.errors import ProjectError, UnknownModelError


def _project(tmp_path, total=30, labeled=24):
    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box"), Category(id=2, name="pallet")])
    colours = list(COLOURS.values())
    for n in range(1, total + 1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, colours[n % len(colours)])
        project.add_image(path, width=64 + n, height=48)
    for image_id in range(1, labeled + 1):
        cat = 2 if image_id % 4 == 0 else 1
        project.save_annotations(
            image_id, [Annotation(id=1, image_id=image_id, category_id=cat, bbox=(10, 10, 30, 20))],
            expected_version=0,
        )
    return project


def _fakes():
    return dict(embedder=FakeEmbedder(), embedding_model="fake-embedder", detector=FakeDetector())


def _pending(project, image_id):
    return [a for a in project.load_annotations(image_id).annotations if a.status == "pending"]


def test_defaults_persist_and_validate(tmp_path):
    project = _project(tmp_path)
    s = get_loop_settings(project)
    assert (s.model, s.preannotate, s.shapes, s.refiner) == (None, True, "auto", "sam2.1-tiny")
    assert s.scan_factor == 100  # the scorer looks at 100× the round size by default
    assert s.training == "continue"  # rounds warm-start from the previous run by default
    assert s.balance is True  # rounds lean towards under-labeled classes
    assert loop_status(project).settings == s
    assert update_loop_settings(project, scan_factor=0).scan_factor is None  # 0 = no cap
    assert update_loop_settings(project, scan_factor=50).scan_factor == 50
    # a loop.json from the absolute-cap days loads, the old key is dropped
    import json

    path = project.root / "loop.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "score_limit": 2000}))
    assert get_loop_settings(project).scan_factor == 50

    updated = update_loop_settings(project, model="rfdetr-seg-small", shapes="polygon")
    assert updated.model == "rfdetr-seg-small" and updated.shapes == "polygon"
    assert updated.preannotate is True  # untouched by a partial update
    assert get_loop_settings(open_project(project.root)) == updated  # on disk
    assert (project.root / "loop.json").is_file()

    with pytest.raises(UnknownModelError):
        update_loop_settings(project, model="yolo-agpl")
    with pytest.raises(ProjectError, match="cannot be trained"):
        update_loop_settings(project, model="owlv2-base")
    with pytest.raises(ProjectError, match="Invalid loop settings"):
        update_loop_settings(project, shapes="circle")
    with pytest.raises(ProjectError, match="Unknown loop setting"):
        update_loop_settings(project, colour="blue")


def test_suggestions_off_means_no_pending_prelabels(tmp_path):
    project = _project(tmp_path)
    update_loop_settings(project, preannotate=False)
    record = select_round(project, count=3, **_fakes())
    assert record.preannotation == {}
    assert all(_pending(project, i) == [] for i in record.image_ids)
    assert any("suggestions are off" in n for n in record.selection.notes)
    # an explicit per-call override still wins
    from horos.api.loop import close_round

    close_round(project, record.number)
    record = select_round(project, count=3, preannotate=True, **_fakes())
    assert record.preannotation["images"] == 3


def test_polygon_shapes_turn_suggested_boxes_into_sam_polygons(tmp_path):
    project = _project(tmp_path)
    update_loop_settings(project, shapes="polygon")
    refiner = FakePromptableSegmenter()
    record = select_round(project, count=4, refiner=refiner, **_fakes())
    assert record.preannotation["shapes"] == "polygon"
    suggested = [a for i in record.image_ids for a in _pending(project, i)]
    assert suggested, "the fake detector suggests boxes on red/green/blue photos"
    assert all(a.segmentation and a.status == "pending" and a.source == "auto" for a in suggested)
    assert refiner.segment_calls == len(suggested)


def test_box_shapes_keep_suggestions_as_boxes(tmp_path):
    project = _project(tmp_path)
    update_loop_settings(project, shapes="box")
    record = select_round(project, count=4, **_fakes())
    suggested = [a for i in record.image_ids for a in _pending(project, i)]
    assert suggested and all(not a.segmentation for a in suggested)


def test_training_uses_the_settings_model_before_the_automatic_choice(tmp_path):
    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    update_loop_settings(project, model="rfdetr-seg-medium")
    record = select_round(project, count=2, preannotate=False, **_fakes())
    trained = train_round(project, record.number, entrypoint_override=FAKE, epochs=1)
    assert trained.training["model"] == "rfdetr-seg-medium"
    assert trained.training["model_reason"] == "from the loop settings"
    import time

    from horos.api.train import training_status

    deadline = time.monotonic() + 90
    active = ("queued", "pending", "running")
    while training_status(project, trained.train_run_id).run.state in active:
        assert time.monotonic() < deadline
        time.sleep(0.2)


def test_segmentation_model_setting_turns_box_suggestions_into_polygons(tmp_path):
    """Segm chosen + Shapes auto: a box scorer's pseudo-labels are refined to
    polygons — the user saw boxes from a detection run in a Segm loop."""
    project = _project(tmp_path)
    update_loop_settings(project, model="rfdetr-seg-nano")  # shapes stays "auto"
    refiner = FakePromptableSegmenter()
    record = select_round(project, count=4, refiner=refiner, **_fakes())
    assert record.preannotation["shapes"] == "polygon"
    suggested = [a for i in record.image_ids for a in _pending(project, i)]
    assert suggested and all(a.segmentation for a in suggested)
    # and a detection loop keeps boxes
    from horos.api.loop import close_round

    close_round(project, record.number)
    update_loop_settings(project, model="rfdetr-nano")
    record = select_round(project, count=4, **_fakes())
    assert record.preannotation["shapes"] == "auto"
    assert all(not a.segmentation for i in record.image_ids for a in _pending(project, i))


def test_scorer_prefers_the_newest_run_of_the_configured_model(tmp_path):
    from helpers.experiments import train_fake

    from horos.api.loop import _latest_completed_run

    ensure_worker_can_import_helpers()
    project = _project(tmp_path)
    project.update_image_splits({i: "valid" for i in range(1, 6)})  # a run needs a valid split
    import time

    det = train_fake(project, epochs=1, model="rfdetr-nano")
    time.sleep(1.05)
    seg = train_fake(project, epochs=1, model="rfdetr-seg-nano")
    time.sleep(1.05)
    det2 = train_fake(project, epochs=1, model="rfdetr-nano")
    assert _latest_completed_run(project).run_id == det2.run_id
    assert _latest_completed_run(project, "rfdetr-seg-nano").run_id == seg.run_id
    assert _latest_completed_run(project, "rfdetr-seg-small").run_id == det2.run_id  # none → newest
    assert det.run_id != seg.run_id
