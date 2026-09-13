"""SAM-T2: clicks and boxes become candidate shapes; nothing is written until
the annotator saves through the ordinary annotation path."""

from __future__ import annotations

import pytest
from helpers.data import write_sample_coco_dir
from helpers.fake_backend import FakePromptableSegmenter

from horos.api.annotate import get_annotations, save_annotations
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.segment import SegmentRequest, _reset_segmenters, segment_image
from horos.core.dataset import Annotation
from horos.errors import ProjectError


@pytest.fixture
def project(tmp_path):
    _reset_segmenters()
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    yield proj
    _reset_segmenters()


def test_click_yields_a_polygon_candidate_and_writes_nothing(project):
    fake = FakePromptableSegmenter()
    before = get_annotations(project, 1)
    cand = segment_image(project, 1, SegmentRequest(points=[(30, 20)], labels=[1]), backend=fake)
    assert cand.image_id == 1 and cand.model == "sam2.1-tiny"
    assert cand.shape_type == "polygon" and len(cand.points) == 4
    assert cand.bbox == (20, 10, 20, 20) and cand.score == pytest.approx(0.9)
    assert cand.polygon == [20, 10, 40, 10, 40, 30, 20, 30]
    after = get_annotations(project, 1)
    assert after.version == before.version and after.annotations == before.annotations


def test_max_points_caps_the_polygon_detail(project):
    """SAM outlines can be far more detailed than a label needs: the request
    caps the control points and the candidate comes back simplified."""
    fake = FakePromptableSegmenter()
    full = segment_image(project, 1, SegmentRequest(points=[(30, 20)], labels=[1]), backend=fake)
    assert len(full.points) == 4
    capped = segment_image(
        project, 1, SegmentRequest(points=[(30, 20)], labels=[1], max_points=3), backend=fake
    )
    assert capped.shape_type == "polygon" and len(capped.points) == 3
    assert len(capped.polygon) == 6 and capped.bbox == full.bbox
    # boxes are boxes whatever the cap
    box = segment_image(
        project, 1, SegmentRequest(points=[(30, 20)], labels=[1], output="bbox", max_points=3),
        backend=fake,
    )
    assert box.shape_type == "rectangle" and len(box.points) == 2
    with pytest.raises(ValueError):
        SegmentRequest(points=[(30, 20)], labels=[1], max_points=2)


def test_negative_point_and_box_prompts(project):
    fake = FakePromptableSegmenter()
    boxed = segment_image(project, 1, SegmentRequest(box=(4, 4, 16, 12)), backend=fake)
    assert boxed.bbox == (4, 4, 16, 12)
    trimmed = segment_image(
        project, 1, SegmentRequest(points=[(30, 20), (30, 25)], labels=[1, 0]), backend=fake
    )
    assert trimmed.bbox[3] == 10  # the fake shaves the bottom half on a negative inside
    assert trimmed.score == pytest.approx(0.8)


def test_bbox_output_returns_rectangle_corners(project):
    fake = FakePromptableSegmenter()
    cand = segment_image(
        project, 1, SegmentRequest(points=[(30, 20)], labels=[1], output="bbox"), backend=fake
    )
    assert cand.shape_type == "rectangle" and cand.points == [[20, 10], [40, 30]]
    assert cand.polygon  # the polygon still rides along for a client that wants it


def test_candidate_is_accepted_through_the_ordinary_save(project):
    fake = FakePromptableSegmenter()
    cand = segment_image(project, 1, SegmentRequest(points=[(30, 20)], labels=[1]), backend=fake)
    view = get_annotations(project, 1)
    forklift = next(c for c in project.categories if c.name == "forklift")
    saved = save_annotations(
        project, 1,
        [*view.annotations, Annotation(id=0, image_id=1, category_id=forklift.id,
                                        bbox=cand.bbox, segmentation=[cand.polygon])],
        expected_version=view.version,
    )
    assert saved.version == view.version + 1
    assert saved.annotations[-1].segmentation == [cand.polygon]


def test_validation_is_explicit(project):
    fake = FakePromptableSegmenter()
    with pytest.raises(ProjectError, match="at least one point or a box"):
        segment_image(project, 1, SegmentRequest(), backend=fake)
    with pytest.raises(ProjectError, match="differ in length"):
        segment_image(project, 1, SegmentRequest(points=[(1, 1)], labels=[1, 0]), backend=fake)
    with pytest.raises(ProjectError, match="outside the 64×48 image"):
        segment_image(project, 1, SegmentRequest(points=[(999, 1)], labels=[1]), backend=fake)
    with pytest.raises(ProjectError, match="No image with id"):
        segment_image(project, 999, SegmentRequest(points=[(1, 1)], labels=[1]), backend=fake)
    assert fake.embed_calls == []  # every refusal happened before the encoder ran


def test_empty_mask_is_reported_not_faked(project):
    fake = FakePromptableSegmenter()
    # only a negative point: the fake returns no mask
    cand = segment_image(project, 1, SegmentRequest(points=[(30, 20)], labels=[0]), backend=fake)
    assert cand.points == [] and cand.bbox is None and cand.polygon is None and cand.area == 0


def test_non_segmenter_model_is_refused(project, monkeypatch):
    from helpers.fake_backend import FakeBackend

    import horos.backends

    monkeypatch.setattr(horos.backends, "get_backend", lambda key, **kw: FakeBackend(None))
    with pytest.raises(ProjectError, match="not an interactive segmenter"):
        segment_image(project, 1, SegmentRequest(points=[(1, 1)], labels=[1], model="rfdetr-nano"))
