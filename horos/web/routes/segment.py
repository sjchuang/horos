"""Interactive segmentation routes (SAM-T3). Thin by rule (R2)."""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

import horos.api as api
from horos.api.segment import DEFAULT_SEGMENTER, SegmentRequest
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("segment", __name__, url_prefix="/api/v1")


@bp.post("/images/<int:image_id>/segment")
def segment_image(image_id: int):
    body = request.get_json(silent=True) or {}
    try:
        spec = SegmentRequest.model_validate(body)
    except ValidationError as exc:
        raise ProjectError(f"Invalid segment request: {exc}") from exc
    return jsonify(api.segment_image(_project(), image_id, spec).model_dump())


@bp.post("/images/<int:image_id>/segment/prefetch")
def prefetch_embedding(image_id: int):
    body = request.get_json(silent=True) or {}
    model = body.get("model", DEFAULT_SEGMENTER)
    if not isinstance(model, str) or not model:
        raise ProjectError("'model' must be a model key")
    return jsonify(api.prefetch_embedding(_project(), image_id, model=model).model_dump())


def _model(body: dict) -> str:
    model = body.get("model", DEFAULT_SEGMENTER)
    if not isinstance(model, str) or not model:
        raise ProjectError("'model' must be a model key")
    return model


def _max_points(body: dict) -> int | None:
    value = body.get("max_points")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 3:
        raise ProjectError("'max_points' must be an integer of at least 3")
    return value


def _categories(body: dict):
    categories = body.get("categories")
    if categories is None:
        return None
    if not isinstance(categories, list) or not all(isinstance(c, int | str) for c in categories):
        raise ProjectError("'categories' must be a list of category ids or names")
    return categories


@bp.post("/images/<int:image_id>/segment/boxes")
def boxes_to_polygons(image_id: int):
    body = request.get_json(silent=True) or {}
    ids = body.get("annotation_ids")
    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, int) for i in ids)):
        raise ProjectError("'annotation_ids' must be a list of annotation ids")
    version = body.get("expected_version")
    if version is not None and not isinstance(version, int):
        raise ProjectError("'expected_version' must be an integer")
    result = api.boxes_to_polygons(
        _project(), image_id, annotation_ids=ids, categories=_categories(body),
        include_pending=bool(body.get("include_pending", True)), model=_model(body),
        max_points=_max_points(body), expected_version=version,
    )
    return jsonify(result.model_dump())


@bp.post("/segment/boxes")
def start_boxes_to_polygons():
    body = request.get_json(silent=True) or {}
    job_id = api.start_boxes_to_polygons(
        _project(), categories=_categories(body), split=body.get("split") or None,
        include_pending=bool(body.get("include_pending", True)), model=_model(body),
        max_points=_max_points(body),
    )
    return jsonify({"job_id": job_id}), 202
