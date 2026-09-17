"""Result visualization — overlays rendered server-side with Pillow (E6-T6).

The WebUI can draw boxes on a canvas by itself, but the Python API and the CLI
need image files: a report attachment, a folder of the worst cases to hand to
the annotator, a sanity check without a browser. Rendering here means one
implementation serves all three layers (R2); the Web API streams the same PNG.

Colour code for error overlays (confirmed design):

    ground truth of a matched pair  green, thin
    true positive (prediction)      blue
    false positive (prediction)     red
    miss (ground truth)             orange, dashed
    confusion (prediction)          purple, label shows "pred (gt: name)"
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from horos.api.error_analysis import ImageErrors, image_errors
from horos.api.evaluate import eval_ground_truth
from horos.api.manifest import capability
from horos.core.project import Project
from horos.errors import ProjectError

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from horos.backends.base import ImagePrediction

__all__ = [
    "ERROR_COLORS",
    "OverlayBox",
    "error_overlay_boxes",
    "prediction_overlay_boxes",
    "render_error_overlay",
    "render_overlay",
    "render_prediction_overlay",
    "to_png_bytes",
]

ERROR_COLORS: dict[str, str] = {
    "gt": "#51cf66",
    "tp": "#4dabf7",
    "fp": "#ff6b6b",
    "fn": "#ffa94d",
    "confused": "#cc5de8",
}

_LABEL_BG = "#101418"


class OverlayBox(BaseModel):
    bbox: tuple[float, float, float, float]  # COCO xywh, image pixels
    color: str  # "#rrggbb"
    label: str = ""
    style: Literal["solid", "dashed"] = "solid"
    width: int | None = None  # default scales with the image
    #: flat [x1, y1, x2, y2, ...] outline of a segmentation instance; drawn
    #: (with a translucent fill) instead of the rectangle when present
    polygon: list[float] | None = None
    #: further rings of a multi-part object (an occluded object in pieces)
    more_polygons: list[list[float]] = Field(default_factory=list)


# ------------------------------------------------------------------ drawing


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ProjectError(f"Overlay colours must be #rrggbb, got '{color}'")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def _dashed_rectangle(draw, xyxy, color, width: int, dash: int) -> None:
    x0, y0, x1, y1 = xyxy
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (ax, ay), (bx, by) in edges:
        length = max(abs(bx - ax), abs(by - ay))
        if length <= 0:
            continue
        steps = int(length // dash)
        for k in range(0, steps + 1, 2):
            t0 = k * dash / length
            t1 = min((k + 1) * dash, length) / length
            draw.line(
                [
                    (ax + (bx - ax) * t0, ay + (by - ay) * t0),
                    (ax + (bx - ax) * t1, ay + (by - ay) * t1),
                ],
                fill=color,
                width=width,
            )


@capability(
    "visualize.overlay",
    summary="Draw colour-coded boxes on an image (the primitive behind every "
    "overlay horos renders)",
    not_web_because="A drawing primitive over a local image; the Web API serves "
    "the rendered overlay of each evaluated image instead (evaluate.overlay).",
    not_cli_because="Reached through 'horos infer --overlay-dir' and "
    "'horos analyze --overlays'.",
)
def render_overlay(
    image: Path | str | PILImage,
    boxes: list[OverlayBox],
    *,
    out: Path | str | None = None,
) -> PILImage:
    """Draw `boxes` on a copy of `image`; write it to `out` when given."""
    from PIL import Image, ImageDraw

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise ProjectError(f"Image not found: {path}")
        with Image.open(path) as opened:
            canvas = opened.convert("RGB")
    else:
        canvas = image.convert("RGB")

    # line width and font scale with the image so labels stay legible on
    # 4K frames and do not swallow 64-px thumbnails
    base = max(1, round(max(canvas.size) / 400))
    font_size = max(10, base * 6)
    font = _font(font_size)
    # masks first, as one translucent layer, so outlines and labels stay crisp
    polygons = [b for b in boxes if b.polygon and len(b.polygon) >= 6]
    if polygons:
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        for box in polygons:
            for ring in (box.polygon, *box.more_polygons):
                pts = list(zip(ring[0::2], ring[1::2], strict=True))
                layer_draw.polygon(pts, fill=(*_hex_to_rgb(box.color), 70))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        x, y, w, h = box.bbox
        xyxy = (x, y, x + w, y + h)
        color = _hex_to_rgb(box.color)
        width = box.width or base
        if box.polygon and len(box.polygon) >= 6:
            for ring in (box.polygon, *box.more_polygons):
                pts = list(zip(ring[0::2], ring[1::2], strict=True))
                draw.line([*pts, pts[0]], fill=color, width=width, joint="curve")
        elif box.style == "dashed":
            _dashed_rectangle(draw, xyxy, color, width, dash=max(4, base * 3))
        else:
            draw.rectangle(xyxy, outline=color, width=width)
        if box.label:
            left, top, right, bottom = draw.textbbox((0, 0), box.label, font=font)
            tw, th = right - left, bottom - top
            pad = max(1, base)
            ly = y - th - 2 * pad
            if ly < 0:
                ly = y  # label inside the box when there is no room above
            draw.rectangle((x, ly, x + tw + 2 * pad, ly + th + 2 * pad), fill=color)
            draw.text((x + pad - left, ly + pad - top), box.label, fill=_LABEL_BG, font=font)

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
    return canvas


def to_png_bytes(image: PILImage) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------ box builders


def error_overlay_boxes(errors: ImageErrors) -> list[OverlayBox]:
    """The colour-coded boxes of one image's matching breakdown."""
    boxes: list[OverlayBox] = []
    for item in errors.items:
        if item.gt_bbox is not None:  # matched pair: show what it matched
            boxes.append(OverlayBox(bbox=item.gt_bbox, color=ERROR_COLORS["gt"], width=1))
    for item in errors.items:
        if item.kind == "fn":
            label = f"missed {item.gt_name}"
        elif item.kind == "fp":
            label = f"false {item.pred_name} {item.score:.2f}"
        elif item.kind == "confused":
            label = f"{item.pred_name} {item.score:.2f} (gt: {item.gt_name})"
        else:
            label = f"{item.pred_name} {item.score:.2f}"
        seg = item.segmentation or []
        boxes.append(
            OverlayBox(
                bbox=item.bbox,
                color=ERROR_COLORS[item.kind],
                label=label,
                style="dashed" if item.kind == "fn" else "solid",
                polygon=seg[0] if seg else None,
                more_polygons=list(seg[1:]),
            )
        )
    return boxes


def prediction_overlay_boxes(
    prediction: ImagePrediction,
    *,
    threshold: float = 0.0,
    colors: dict[str, str] | None = None,
) -> list[OverlayBox]:
    """Plain prediction overlay: one box per instance above `threshold`,
    coloured per class (`colors` maps class name to #rrggbb; unknown classes
    get a stable palette colour)."""
    palette = ["#4dabf7", "#51cf66", "#ffa94d", "#cc5de8", "#ff6b6b", "#fcc419", "#20c997"]
    colors = dict(colors or {})
    boxes: list[OverlayBox] = []
    for inst in prediction.instances:
        if inst.score < threshold:
            continue
        name = inst.category_name or str(inst.category_id)
        if name not in colors:
            colors[name] = palette[len(colors) % len(palette)]
        boxes.append(
            OverlayBox(
                bbox=inst.bbox, color=colors[name], label=f"{name} {inst.score:.2f}",
                polygon=inst.segmentation[0] if inst.segmentation else None,
                more_polygons=list(inst.segmentation[1:]) if inst.segmentation else [],
            )
        )
    return boxes


@capability(
    "visualize.prediction",
    summary="Render an image with a model's predictions drawn on it",
    not_web_because="The evaluate page draws predictions on a canvas from the "
    "JSON of infer.image; a server-rendered PNG would only duplicate it.",
    cli="infer",
)
def render_prediction_overlay(
    image: Path | str,
    prediction: ImagePrediction,
    *,
    threshold: float = 0.0,
    colors: dict[str, str] | None = None,
    out: Path | str | None = None,
) -> PILImage:
    boxes = prediction_overlay_boxes(prediction, threshold=threshold, colors=colors)
    return render_overlay(image, boxes, out=out)


# ------------------------------------------------------- project entry point


@capability(
    "evaluate.overlay",
    summary="Render one evaluated image with its ground truth, misses, false "
    "positives and confusions colour-coded",
    web_route="/api/v1/train/runs/<run_id>/eval/<split>/images/<int:image_id>/overlay.png",
    web_methods=("GET",),
    cli="analyze",
)
def render_error_overlay(
    project: Project,
    run_id: str,
    split: str,
    image_id: int,
    *,
    threshold: float = 0.5,
    iou: float = 0.5,
    out: Path | str | None = None,
) -> PILImage:
    errors = image_errors(project, run_id, split, image_id, threshold=threshold, iou=iou)
    # the photo as the evaluation saw it: the project's copy for current
    # labels, the run's snapshot copy otherwise
    _, _, image_path_of = eval_ground_truth(project, run_id, split)
    image_path = image_path_of(errors.file_name)
    return render_overlay(image_path, error_overlay_boxes(errors), out=out)
