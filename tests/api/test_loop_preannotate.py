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


def test_one_pseudo_label_per_object_even_when_the_model_stacks_queries(tmp_path):
    """RF-DETR has no NMS: a young model answers one object with several
    queries of the same class, and the annotator saw the class stacked on
    it several times. Per-class NMS keeps the best box; a different class on
    the same spot is a real ambiguity and stays."""
    from horos.backends.base import ImagePrediction, PredictedInstance

    class Stacking(FakeDetector):
        def infer_one(self, image, *, threshold=0.5):
            self.seen.append(str(image))
            mk = lambda name, box, score: PredictedInstance(  # noqa: E731
                bbox=box, score=score, category_id=0, category_name=name)
            inst = [mk("box", (10, 10, 30, 20), 0.9), mk("box", (11, 11, 29, 19), 0.8),
                    mk("box", (12, 9, 30, 21), 0.6), mk("pallet", (10, 10, 30, 20), 0.7),
                    mk("box", (40, 5, 20, 20), 0.5)]
            return ImagePrediction(image=str(image), width=64, height=48,
                                   instances=[i for i in inst if i.score >= threshold],
                                   candidates=inst)

    project = _project(tmp_path, ["red"] * 4, labeled={1: "box"})
    record = _select(project, count=2, detector=Stacking(), detector_label="fake-run")
    for image_id in record.image_ids:
        pending = _pending(project, image_id)
        by_class = {}
        for a in pending:
            by_class.setdefault(a.category_id, []).append(a)
        names = {c.id: c.name for c in project.categories}
        assert sorted(names[c] for c in by_class) == ["box", "pallet"]
        box_id = next(c for c in by_class if names[c] == "box")
        boxes = sorted(by_class[box_id], key=lambda a: -a.score)
        # the best of the stack plus the other object
        assert len(boxes) == 2 and boxes[0].score == pytest.approx(0.9)
    # two stacked queries dropped on each of the 2 photos
    assert record.preannotation["duplicates_dropped"] == 2 * 2
    assert record.preannotation["nms_iou"] == 0.5


def test_pseudo_labels_follow_a_renamed_class(tmp_path):
    """The model learned "box"; the user renamed the class to "Box". Its
    pseudo-labels must land on "Box" — no resurrected "box" class."""
    from horos.api.labels import update_category

    project = _project(tmp_path, ["red"] * 4, labeled={1: "box"})
    box = next(c for c in project.categories if c.name == "box")
    update_category(project, box.id, name="Box")
    record = _select(project, count=2, detector=FakeDetector(), detector_label="fake-run")
    names = {c.name for c in project.categories}
    assert "Box" in names and "box" not in names
    written = [a for i in record.image_ids for a in _pending(project, i)]
    assert written and all(a.category_id == box.id for a in written)


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


def test_image_predictions_use_the_loop_scorer_and_current_class_names(tmp_path, monkeypatch):
    """SAM-T4 class suggestion: the annotator asks what the loop's scorer sees
    on one photo; nothing is written, names go through the aliases."""
    from horos.api.labels import update_category
    from horos.api.loop import image_predictions

    project = _project(tmp_path, ["red", "blue", "grey"])
    detector = FakeDetector()
    monkeypatch.setattr("horos.api.autolabel._cached_backend", lambda key, device=None: detector)
    update_category(project, 1, name="Box")  # the model still says "box"

    result = image_predictions(project, 1)
    assert result.kind == "zero_shot" and result.model == "owlv2-base"
    assert [(d.label, d.score) for d in result.detections] == [("Box", 0.95)]
    assert result.detections[0].bbox == (10.0, 10.0, 30.0, 20.0)
    assert _pending(project, 1) == []  # a suggestion writes nothing
    assert image_predictions(project, 3).detections == []  # grey: the fake sees nothing
    assert image_predictions(project, 2, threshold=0.95).detections == []  # pallet 0.9 filtered

    with pytest.raises(ProjectError, match="No image with id 99"):
        image_predictions(project, 99)
    with pytest.raises(ProjectError, match="threshold must be"):
        image_predictions(project, 1, threshold=2)
