"""E6-T6: overlays rendered server-side with Pillow — the same drawing serves
the Python API, the CLI and the Web API."""

from __future__ import annotations

import pytest
from helpers.data import make_image
from helpers.runs import completed_fake_run
from PIL import Image

from horos.api.error_analysis import ImageErrorItem, ImageErrors
from horos.api.evaluate import _write_detections
from horos.api.visualize import (
    ERROR_COLORS,
    OverlayBox,
    error_overlay_boxes,
    prediction_overlay_boxes,
    render_error_overlay,
    render_overlay,
    render_prediction_overlay,
    to_png_bytes,
)
from horos.backends.base import ImagePrediction, PredictedInstance
from horos.errors import ProjectError

BLACK = (0, 0, 0)
RED = (255, 0, 0)


@pytest.fixture
def blank(tmp_path):
    return make_image(tmp_path / "blank.png", 200, 150, BLACK)


def _px(image, x, y):
    return image.getpixel((x, y))[:3]


# -------------------------------------------------------------- render_overlay


def test_solid_box_paints_its_edges_and_leaves_the_inside_alone(blank):
    image = render_overlay(blank, [OverlayBox(bbox=(20, 30, 60, 40), color="#ff0000", width=1)])
    assert image.size == (200, 150)
    assert _px(image, 20, 50) == RED  # left edge
    assert _px(image, 80, 50) == RED  # right edge
    assert _px(image, 50, 30) == RED  # top edge
    assert _px(image, 50, 70) == RED  # bottom edge
    assert _px(image, 50, 50) == BLACK  # inside untouched
    assert _px(image, 150, 120) == BLACK  # outside untouched


def test_dashed_box_paints_some_edge_pixels_but_not_all(blank):
    image = render_overlay(
        blank, [OverlayBox(bbox=(20, 30, 100, 40), color="#ff0000", style="dashed", width=1)]
    )
    top_edge = [_px(image, x, 30) for x in range(20, 120)]
    assert RED in top_edge and BLACK in top_edge


def test_label_is_drawn_above_the_box(blank):
    image = render_overlay(
        blank, [OverlayBox(bbox=(20, 60, 60, 40), color="#ff0000", label="forklift 0.91")]
    )
    # the label background sits just above the box's top edge
    above = [_px(image, x, 55) for x in range(21, 40)]
    assert RED in above


def test_out_path_writes_a_png_and_creates_parent_directories(blank, tmp_path):
    target = tmp_path / "nested" / "dir" / "out.png"
    render_overlay(blank, [OverlayBox(bbox=(0, 0, 10, 10), color="#00ff00")], out=target)
    with Image.open(target) as saved:
        assert saved.size == (200, 150) and saved.format == "PNG"


def test_pil_image_input_is_not_modified_in_place(blank):
    with Image.open(blank) as source:
        source = source.convert("RGB")
        image = render_overlay(source, [OverlayBox(bbox=(0, 0, 50, 50), color="#ff0000")])
        assert _px(source, 0, 25) == BLACK and _px(image, 0, 25) == RED


def test_missing_image_and_bad_colour_are_horos_errors(tmp_path, blank):
    with pytest.raises(ProjectError, match="not found"):
        render_overlay(tmp_path / "nope.png", [])
    with pytest.raises(ProjectError, match="#rrggbb"):
        render_overlay(blank, [OverlayBox(bbox=(0, 0, 1, 1), color="red")])


def test_to_png_bytes_round_trips():
    image = Image.new("RGB", (8, 6), RED)
    with Image.open(__import__("io").BytesIO(to_png_bytes(image))) as decoded:
        assert decoded.size == (8, 6) and decoded.format == "PNG"


# ----------------------------------------------------------- box builders


def test_error_boxes_follow_the_colour_code():
    errors = ImageErrors(
        image_id=1, file_name="a.png", width=100, height=100,
        errors=3, tp=1, fn=1, fp=1, confused=1, missed_area=0.1,
        items=[
            ImageErrorItem(kind="tp", bbox=(0, 0, 10, 10), gt_name="a", pred_name="a",
                           score=0.9, iou=1.0, gt_bbox=(0, 0, 10, 10)),
            ImageErrorItem(kind="confused", bbox=(20, 0, 10, 10), gt_name="a",
                           pred_name="b", score=0.8, iou=0.9, gt_bbox=(21, 0, 10, 10)),
            ImageErrorItem(kind="fp", bbox=(40, 0, 10, 10), pred_name="b", score=0.7),
            ImageErrorItem(kind="fn", bbox=(60, 0, 10, 10), gt_name="a"),
        ],
    )
    boxes = error_overlay_boxes(errors)
    by_color = {}
    for box in boxes:
        by_color.setdefault(box.color, []).append(box)
    # two matched pairs -> two thin green ground-truth boxes
    assert len(by_color[ERROR_COLORS["gt"]]) == 2
    assert all(b.width == 1 and b.label == "" for b in by_color[ERROR_COLORS["gt"]])
    (tp,) = by_color[ERROR_COLORS["tp"]]
    assert tp.label == "a 0.90" and tp.style == "solid"
    (confused,) = by_color[ERROR_COLORS["confused"]]
    assert confused.label == "b 0.80 (gt: a)"
    (fp,) = by_color[ERROR_COLORS["fp"]]
    assert fp.label == "false b 0.70"
    (fn,) = by_color[ERROR_COLORS["fn"]]
    assert fn.label == "missed a" and fn.style == "dashed"


def test_prediction_boxes_filter_by_threshold_and_colour_per_class():
    prediction = ImagePrediction(
        image="x.png",
        instances=[
            PredictedInstance(bbox=(0, 0, 10, 10), score=0.9, category_id=0,
                              category_name="forklift"),
            PredictedInstance(bbox=(20, 0, 10, 10), score=0.4, category_id=1,
                              category_name="pallet"),
            PredictedInstance(bbox=(40, 0, 10, 10), score=0.8, category_id=0,
                              category_name="forklift"),
        ],
    )
    boxes = prediction_overlay_boxes(prediction, threshold=0.5, colors={"forklift": "#123456"})
    assert [b.label for b in boxes] == ["forklift 0.90", "forklift 0.80"]
    assert {b.color for b in boxes} == {"#123456"}
    everything = prediction_overlay_boxes(prediction)
    assert len(everything) == 3
    assert len({b.color for b in everything}) == 2  # one colour per class


def test_error_boxes_carry_the_item_polygons():
    from horos.api.error_analysis import ImageErrorItem, ImageErrors
    from horos.api.visualize import error_overlay_boxes

    ring, part = [1.0, 1.0, 5.0, 1.0, 5.0, 5.0], [6.0, 6.0, 9.0, 6.0, 9.0, 9.0]
    errors = ImageErrors(
        image_id=1, file_name="a.png", width=10, height=10, errors=1, tp=0, fn=1, fp=0,
        confused=0, missed_area=0.1,
        items=[ImageErrorItem(kind="fn", bbox=(1, 1, 8, 8), gt_name="a",
                              segmentation=[ring, part])],
    )
    (box,) = error_overlay_boxes(errors)
    assert box.polygon == ring and box.more_polygons == [part]


def test_render_prediction_overlay_writes_the_file(blank, tmp_path):
    prediction = ImagePrediction(
        image=str(blank),
        instances=[PredictedInstance(bbox=(10, 10, 50, 50), score=0.9, category_id=0)],
    )
    out = tmp_path / "pred.png"
    image = render_prediction_overlay(blank, prediction, out=out)
    assert out.is_file() and image.size == (200, 150)
    assert _px(image, 10, 35) != BLACK


# ------------------------------------------------------- project entry point


def test_error_overlay_of_an_evaluated_image(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    # sample split "valid" holds c.png (32x32) with one pallet box at (2,2,12,10)
    _write_detections(project, run.run_id, "valid", [])
    image = render_error_overlay(project, run.run_id, "valid", 3)
    assert image.size == (32, 32)
    # the missed box is drawn dashed in the miss colour along its top edge
    orange = tuple(int(ERROR_COLORS["fn"][i:i + 2], 16) for i in (1, 3, 5))
    assert orange in [_px(image, x, 2) for x in range(2, 14)]

    out = tmp_path / "c.overlay.png"
    render_error_overlay(project, run.run_id, "valid", 3, out=out)
    assert out.is_file()


def test_error_overlay_refuses_an_image_outside_the_split(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    _write_detections(project, run.run_id, "valid", [])
    with pytest.raises(ProjectError, match="not part of"):
        render_error_overlay(project, run.run_id, "valid", 999)


def test_polygon_instances_are_drawn_as_outlines_not_boxes(blank):
    """A segmentation model's instance carries a polygon: the overlay draws
    its outline with a translucent fill and leaves the bbox corners alone."""
    from horos.api.visualize import prediction_overlay_boxes

    diamond = [40.0, 10.0, 70.0, 40.0, 40.0, 70.0, 10.0, 40.0]  # inside bbox (10,10,60,60)
    prediction = ImagePrediction(
        image="x", width=80, height=80,
        instances=[PredictedInstance(bbox=(10, 10, 60, 60), score=0.9, category_id=0,
                                     category_name="box", segmentation=[diamond])],
    )
    boxes = prediction_overlay_boxes(prediction, colors={"box": "#ff0000"})
    assert boxes[0].polygon == diamond
    out = render_overlay(blank, boxes)  # `blank` is black
    px = out.load()
    assert px[40, 10][0] > 200          # a polygon vertex is painted red
    assert px[70, 20] == (0, 0, 0)      # the bbox's right edge is untouched: no rectangle
    inside = px[40, 40]
    assert 30 < inside[0] < 200 and inside[1] == 0  # translucent red fill, not the outline
