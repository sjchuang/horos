"""SAM-T6: box annotations are prompts — one image at a time, or the whole
project as a job — and become polygons in place, keeping id and class. The
polygon is machine-made, so the annotation becomes a pending auto pre-label
scored with SAM's predicted IoU (E10-T11). A box the segmenter cannot mask
stays a box, unchanged."""

from __future__ import annotations

import threading

import pytest
from helpers.data import write_sample_coco_dir
from helpers.fake_backend import FakePromptableSegmenter

from horos.api import jobs
from horos.api.annotate import get_annotations
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.segment import (
    _reset_segmenters,
    boxes_to_polygons,
    boxes_to_polygons_events,
    start_boxes_to_polygons,
)
from horos.core.dataset import Annotation
from horos.errors import HorosError, ProjectError


@pytest.fixture
def project(tmp_path):
    _reset_segmenters()
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    yield proj
    _reset_segmenters()


class NoMaskSegmenter(FakePromptableSegmenter):
    def segment(self, embedding, prompt):
        from horos.backends.base import SegmentResult

        self.segment_calls += 1
        return SegmentResult(polygon=None, bbox=None, score=0.0, area=0)


def _boxes(project):
    """(image_id, annotation) for every box-only annotation in the project."""
    out = []
    for record in project.list_images():
        for a in project.load_annotations(record.id).annotations:
            if not a.segmentation:
                out.append((record.id, a))
    return out


def test_boxes_become_polygons_in_place_with_one_embedding(project):
    fake = FakePromptableSegmenter()
    before = project.load_annotations(1)
    boxes = [a for a in before.annotations if not a.segmentation]
    assert len(boxes) >= 2

    result = boxes_to_polygons(project, 1, backend=fake)
    assert result.converted == len(boxes) and result.skipped == 0
    assert result.version == before.version + 1
    assert len(fake.embed_calls) == 1 and fake.segment_calls == len(boxes)  # one encoder run
    stored = project.load_annotations(1)
    assert stored.version == result.version
    by_id = {a.id: a for a in stored.annotations}
    for original in boxes:
        new = by_id[original.id]
        # the fake's mask is the prompt box itself: geometry as polygon, box unchanged
        x, y, w, h = original.bbox
        assert new.segmentation == [[x, y, x + w, y, x + w, y + h, x, y + h]]
        assert new.bbox == original.bbox
        assert new.category_id == original.category_id
        # machine geometry → pending auto pre-label with the segmenter's score (E10-T11)
        assert (new.status, new.source) == ("pending", "auto")
        assert new.score == pytest.approx(0.9)
    assert len(stored.annotations) == len(before.annotations)  # nothing dropped or added

    # nothing left to convert: no write, same version, no decoder call
    again = boxes_to_polygons(project, 1, backend=fake)
    assert again.converted == 0 and again.version == result.version
    assert fake.segment_calls == len(boxes)


def test_boxes_to_polygons_honours_the_point_cap(project):
    fake = FakePromptableSegmenter()
    result = boxes_to_polygons(project, 1, backend=fake, max_points=3)
    assert result.converted >= 1
    polygons = [a.segmentation[0] for a in get_annotations(project, 1).annotations
                if a.segmentation]
    assert polygons and all(len(p) == 6 for p in polygons)  # triangles, not the 4-corner boxes


def test_filters_by_category_ids_or_names_and_by_annotation_id(project):
    fake = FakePromptableSegmenter()
    stored = project.load_annotations(1)
    first = stored.annotations[0]
    name = next(c.name for c in project.categories if c.id == first.category_id)

    picked = boxes_to_polygons(project, 1, annotation_ids=[first.id], backend=fake)
    assert picked.converted == 1
    assert [a.id for a in picked.annotations if a.segmentation] == [first.id]

    others = [a for a in stored.annotations if a.id != first.id and not a.segmentation]
    by_name = boxes_to_polygons(project, 1, categories=[name], backend=fake)
    expect = sum(1 for a in others if a.category_id == first.category_id)
    assert by_name.converted == expect
    by_id = boxes_to_polygons(
        project, 1, categories=[a.category_id for a in others], backend=fake
    )
    assert by_id.converted == len(others) - expect
    with pytest.raises(ProjectError, match="Unknown category 'ghost'"):
        boxes_to_polygons(project, 1, categories=["ghost"], backend=fake)
    with pytest.raises(ProjectError, match="Unknown category id 999"):
        boxes_to_polygons(project, 1, categories=[999], backend=fake)


def test_pending_prelabels_can_be_left_alone_and_conflicts_surface(project):
    fake = FakePromptableSegmenter()
    stored = project.load_annotations(1)
    pending = Annotation(
        id=max(a.id for a in stored.annotations) + 1, image_id=1,
        category_id=stored.annotations[0].category_id, bbox=(2.0, 2.0, 10.0, 8.0),
        source="auto", status="pending", score=0.4,
    )
    saved = project.save_annotations(
        1, [*stored.annotations, pending], expected_version=stored.version
    )
    confirmed_only = boxes_to_polygons(project, 1, include_pending=False, backend=fake)
    assert confirmed_only.converted == len(stored.annotations)
    still_box = next(a for a in confirmed_only.annotations if a.id == pending.id)
    assert still_box.segmentation == [] and still_box.status == "pending"

    with_pending = boxes_to_polygons(project, 1, backend=fake)
    assert with_pending.converted == 1  # just the pending one now
    converted = next(a for a in with_pending.annotations if a.id == pending.id)
    assert converted.status == "pending" and converted.source == "auto"
    assert converted.score == pytest.approx(0.9) and converted.segmentation  # SAM's IoU

    # E2-T8: a stale version is a conflict, not a silent overwrite
    fresh = create_project(project.root.parent / "other")
    import_dataset(fresh, write_sample_coco_dir(project.root.parent / "coco2"))
    with pytest.raises(HorosError):
        boxes_to_polygons(fresh, 1, backend=fake, expected_version=saved.version + 41)


def test_a_box_without_a_mask_stays_a_box(project):
    fake = NoMaskSegmenter()
    before = project.load_annotations(1)
    boxes = [a for a in before.annotations if not a.segmentation]
    result = boxes_to_polygons(project, 1, backend=fake)
    assert result.converted == 0 and result.skipped == len(boxes)
    assert result.version == before.version  # nothing changed: nothing written
    assert project.load_annotations(1).annotations == before.annotations


def test_batch_events_cover_every_image_with_boxes_then_find_nothing(project):
    fake = FakePromptableSegmenter()
    boxes = _boxes(project)
    images = {image_id for image_id, _ in boxes}
    events = list(boxes_to_polygons_events(project, backend=fake))
    assert events[0].type == "started" and events[0].total == len(images)
    assert events[0].config["boxes"] == len(boxes)
    progress = [e for e in events if e.type == "progress"]
    assert len(progress) == len(images) and progress[-1].current == len(images)
    assert progress[0].phase == "boxes-to-polygons" and "converted" in progress[0].message
    done = events[-1]
    assert done.type == "completed"
    assert done.result == {"images": len(images), "converted": len(boxes), "skipped": 0}
    assert len(fake.embed_calls) == len(images)  # one encoder run per image
    assert _boxes(project) == []

    again = list(boxes_to_polygons_events(project, backend=fake))
    assert [e.type for e in again] == ["started", "warning", "completed"]
    assert again[-1].result["images"] == 0


def test_batch_filters_by_class_and_split_and_honours_cancel(project):
    fake = FakePromptableSegmenter()
    boxes = _boxes(project)
    cat = boxes[0][1].category_id
    name = next(c.name for c in project.categories if c.id == cat)
    events = list(boxes_to_polygons_events(project, categories=[name], backend=fake))
    assert events[-1].result["converted"] == sum(1 for _, a in boxes if a.category_id == cat)
    assert all(a.category_id != cat for _, a in _boxes(project))  # only that class touched
    assert events[0].config["categories"] == [cat]

    nothing = list(boxes_to_polygons_events(project, split="test", backend=fake))
    assert nothing[-1].result["images"] == 0 or all(
        r.split == "test" for r in project.list_images()
        if any(i == r.id for i, _ in _boxes(project))
    )

    cancel = threading.Event()
    cancel.set()
    cancelled = list(boxes_to_polygons_events(project, backend=fake, cancel=cancel))
    assert cancelled[-1].type == "completed" and cancelled[-1].result.get("cancelled") is True
    assert cancelled[-1].result["images"] == 0

    bad = list(boxes_to_polygons_events(project, categories=["ghost"], backend=fake))
    assert bad[-1].type == "failed" and "Unknown category" in bad[-1].message


def test_batch_runs_as_a_job(project):
    fake = FakePromptableSegmenter()
    with pytest.raises(ProjectError, match="Unknown category"):
        start_boxes_to_polygons(project, categories=["ghost"], backend=fake)
    with pytest.raises(ProjectError, match="split must be"):
        start_boxes_to_polygons(project, split="dev", backend=fake)
    job_id = start_boxes_to_polygons(project, backend=fake)
    import time

    deadline = time.monotonic() + 10
    while jobs.job_status(project, job_id).state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    status = jobs.job_status(project, job_id)
    assert status.state == "completed"
    assert status.events[-1]["result"]["converted"] > 0
    assert _boxes(project) == []
