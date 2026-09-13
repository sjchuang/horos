"""E10-T11: geometry a model produced is always stored as a pending auto
pre-label with a score — never as confirmed human work. Covers the three
writers: autolabel (OWLv2), SAM boxes-to-polygons, and the annotator's
interactive segment candidate (which writes nothing on its own)."""

from __future__ import annotations

import pytest
from helpers.data import write_sample_coco_dir
from helpers.fake_backend import FakePromptableSegmenter

from horos.api.autolabel import _write_pending
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.segment import SegmentRequest, _reset_segmenters, boxes_to_polygons, segment_image
from horos.core.dataset import Annotation


@pytest.fixture
def project(tmp_path):
    _reset_segmenters()
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    yield proj
    _reset_segmenters()


def _confirmed_manual(project, image_id):
    return [
        a for a in project.load_annotations(image_id).annotations
        if a.status == "confirmed" and a.source == "manual"
    ]


def test_boxes_to_polygons_demotes_human_boxes_to_pending_auto(project):
    before = _confirmed_manual(project, 1)
    assert len(before) == 2 and all(not a.segmentation for a in before)
    fake = FakePromptableSegmenter()

    result = boxes_to_polygons(project, 1, backend=fake)
    assert result.converted == 2
    stored = project.load_annotations(1).annotations
    assert _confirmed_manual(project, 1) == []
    for original, new in zip(before, stored, strict=True):
        assert new.id == original.id and new.category_id == original.category_id
        assert new.segmentation and new.status == "pending" and new.source == "auto"
        assert new.score == pytest.approx(0.9)  # the fake segmenter's predicted IoU


def test_skipped_boxes_keep_their_human_status(project):
    class NoMask(FakePromptableSegmenter):
        def segment(self, embedding, prompt):
            from horos.backends.base import SegmentResult

            self.segment_calls += 1
            return SegmentResult(polygon=None, bbox=None, score=0.0, area=0)

    result = boxes_to_polygons(project, 1, backend=NoMask())
    assert (result.converted, result.skipped) == (0, 2)
    assert len(_confirmed_manual(project, 1)) == 2  # untouched


def test_autolabel_pre_labels_are_pending_auto_with_scores(project):
    cat = {c.name: c.id for c in project.categories}
    written = _write_pending(
        project, 2, [("forklift", (1.0, 1.0, 10.0, 10.0), 0.42)], cat, polygons=[[1, 1, 9, 1, 9, 9]]
    )
    assert written == 1
    pending = [a for a in project.load_annotations(2).annotations if a.status == "pending"]
    assert len(pending) == 1
    assert pending[0].source == "auto" and pending[0].score == pytest.approx(0.42)
    assert pending[0].segmentation == [[1.0, 1.0, 9.0, 1.0, 9.0, 9.0]]
    # the human box on the same image is untouched
    assert len(_confirmed_manual(project, 2)) == 1


def test_interactive_segment_candidate_writes_nothing(project):
    fake = FakePromptableSegmenter()
    before = project.load_annotations(1)
    candidate = segment_image(project, 1, SegmentRequest(points=[(5, 5)], labels=[1]),
                              backend=fake)
    assert candidate.polygon and candidate.score > 0
    after = project.load_annotations(1)
    assert after.version == before.version and after.annotations == before.annotations


def test_data_model_default_is_human_but_auto_needs_pending_review():
    human = Annotation(id=1, image_id=1, category_id=1, bbox=(0, 0, 1, 1))
    assert (human.source, human.status, human.score) == ("manual", "confirmed", None)
    auto = human.model_copy(update={"source": "auto", "status": "pending", "score": 0.5})
    assert auto.status == "pending"
