"""E10-T7: a round's images are pre-labeled by the current scorer — the
trained run when one exists, OWLv2 zero-shot before — as pending auto
annotations with scores, so labeling starts from corrections (E10-S4)."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.fake_backend import COLOURS, FakeDetector, FakeEmbedder

from horos.api.loop import (
    close_round,
    get_round,
    preannotate_round,
    select_round,
    select_round_events,
)
from horos.api.project import create_project
from horos.core.dataset import Annotation, Category
from horos.errors import BackendError, ProjectError


def _project(tmp_path, layout, *, labeled=None, categories=True):
    project = create_project(tmp_path / "proj")
    if categories:
        project.set_categories([Category(id=1, name="box"), Category(id=2, name="pallet")])
    for n, colour in enumerate(layout, start=1):
        path = make_image(tmp_path / "src" / f"{n}.png", 64 + n, 48, COLOURS[colour])
        project.add_image(path, width=64 + n, height=48)
    for image_id, name in (labeled or {}).items():
        cat = next(c for c in project.categories if c.name == name)
        project.save_annotations(
            image_id,
            [Annotation(id=1, image_id=image_id, category_id=cat.id, bbox=(10, 10, 30, 20))],
            expected_version=0,
        )
    return project


def _pending(project, image_id):
    return [a for a in project.load_annotations(image_id).annotations if a.status == "pending"]


def _select(project, **kw):
    kw.setdefault("embedder", FakeEmbedder())
    kw.setdefault("embedding_model", "fake-embedder")
    return select_round(project, **kw)


def test_pal_round_prelabels_its_picks_without_a_second_inference_pass(tmp_path):
    layout = ["red"] * 3 + ["green"] * 4 + ["grey"] * 2
    project = _project(tmp_path, layout, labeled={1: "box"})
    detector = FakeDetector()
    record = _select(project, count=4, detector=detector, detector_label="fake-run")

    assert record.preannotation["scorer"] == "fake-run"
    assert record.preannotation["images"] == 4
    assert record.preannotation["threshold"] == 0.3
    # predictions from the PAL scoring pass were reused: 1 labeled + 8 pool images
    assert len(detector.seen) == 9
    written = 0
    for image_id in record.image_ids:
        for ann in _pending(project, image_id):
            assert ann.source == "auto" and ann.score is not None
            assert ann.category_id in {c.id for c in project.categories}
            written += 1
    assert written == record.preannotation["annotations"] > 0
    # the human label is untouched
    human = project.load_annotations(1).annotations
    assert len(human) == 1 and human[0].status == "confirmed"


def test_cold_start_round_is_prelabeled_by_the_zero_shot_stand_in(tmp_path):
    project = _project(tmp_path, ["red", "green", "blue", "grey"])
    detector = FakeDetector()
    record = _select(project, count=4, detector=detector, detector_label="owlv2-base")
    assert record.selection.strategy == "diversity"  # no labels → diversity picks
    assert record.preannotation["images"] == 4 and record.preannotation["annotations"] == 3
    assert len(detector.seen) == 4  # inference only for the picks
    assert _pending(project, 4) == []  # grey: the scorer saw nothing


def test_without_classes_nothing_is_prelabeled_and_the_round_says_so(tmp_path):
    project = _project(tmp_path, ["red", "green"], categories=False)
    record = _select(project, count=2, detector=FakeDetector())
    assert record.preannotation == {}
    assert any("no classes yet" in n for n in record.selection.notes)


def test_scorer_failure_does_not_lose_the_round(tmp_path):
    class Broken(FakeDetector):
        def infer_one(self, image, *, threshold=0.5):
            raise BackendError("weights missing", backend="fake-detector")

    project = _project(tmp_path, ["red", "green", "blue"])
    record = _select(project, count=2, detector=Broken())
    assert record.state == "labeling" and len(record.image_ids) == 2
    assert record.preannotation == {}
    assert any("pre-annotation skipped" in n and "weights missing" in n
               for n in record.selection.notes)


def test_preannotation_can_be_switched_off(tmp_path):
    project = _project(tmp_path, ["red", "green"])
    events = list(select_round_events(project, count=2, embedder=FakeEmbedder(),
                                      embedding_model="fake-embedder", detector=FakeDetector(),
                                      preannotate=False))
    assert events[-1].type == "completed"
    record = get_round(project, 1)
    assert record.preannotation == {} and all(_pending(project, i) == [] for i in record.image_ids)


def test_standalone_preannotate_replaces_pending_and_respects_human_work(tmp_path):
    project = _project(tmp_path, ["red", "green", "blue"])
    record = _select(project, count=3, detector=FakeDetector(), preannotate=False)
    # the annotator confirmed image 1 in the meantime
    project.save_annotations(
        1, [Annotation(id=1, image_id=1, category_id=1, bbox=(0, 0, 5, 5))], expected_version=0
    )
    detector = FakeDetector()
    updated = preannotate_round(project, record.number, detector=detector, detector_label="v2")
    assert updated.preannotation["scorer"] == "v2"
    assert updated.preannotation["images"] == 2  # image 1 skipped: confirmed
    assert str(project.image_path(project.get_image(1))) not in detector.seen
    assert _pending(project, 2) and _pending(project, 3)
    assert project.load_annotations(1).annotations[0].status == "confirmed"

    # running again replaces the pending set instead of piling up
    preannotate_round(project, record.number, detector=FakeDetector())
    assert len(_pending(project, 2)) == 1

    close_round(project, record.number)
    with pytest.raises(ProjectError, match="closed"):
        preannotate_round(project, record.number, detector=FakeDetector())
