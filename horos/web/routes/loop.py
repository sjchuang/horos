"""Active-learning loop routes (E10-T12). Thin by rule (R2): validate the
request, call horos.api, serialise the result."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import horos.api as api
from horos.api.embeddings import DEFAULT_EMBEDDING_MODEL
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("loop", __name__, url_prefix="/api/v1/loop")


def _model(source: dict, default: str) -> str:
    model = source.get("model", default)
    if not isinstance(model, str) or not model:
        raise ProjectError("'model' must be a model key")
    return model


@bp.get("/embeddings")
def embedding_status():
    model = _model(request.args, DEFAULT_EMBEDDING_MODEL)
    return jsonify(api.embedding_status(_project(), model).model_dump())


@bp.post("/embeddings")
def start_embedding_job():
    body = request.get_json(silent=True) or {}
    job_id = api.start_embedding_job(
        _project(), _model(body, DEFAULT_EMBEDDING_MODEL), device=body.get("device") or None
    )
    return jsonify({"job_id": job_id}), 202


@bp.get("")
def loop_status():
    return jsonify(api.loop_status(_project()).model_dump())


@bp.get("/rounds/<int:number>")
def get_round(number: int):
    return jsonify(api.get_round(_project(), number).model_dump())


def _count(body: dict) -> tuple[int | None, float | None]:
    count, percent = body.get("count"), body.get("percent")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
        raise ProjectError("'count' must be an integer")
    if percent is not None and (isinstance(percent, bool) or not isinstance(percent, int | float)):
        raise ProjectError("'percent' must be a number")
    return count, percent


@bp.post("/rounds")
def start_round_job():
    body = request.get_json(silent=True) or {}
    count, percent = _count(body)
    strategy = body.get("strategy", "auto")
    if strategy not in ("auto", "pal", "diversity", "random"):
        raise ProjectError("'strategy' must be auto, pal, diversity or random")
    job_id = api.start_round_job(
        _project(), count=count, percent=percent, strategy=strategy,
        embedding_model=_model(body, DEFAULT_EMBEDDING_MODEL), device=body.get("device") or None,
    )
    return jsonify({"job_id": job_id}), 202


@bp.post("/rounds/<int:number>/close")
def close_round(number: int):
    return jsonify(api.close_round(_project(), number).model_dump())


@bp.post("/rounds/<int:number>/preannotate")
def start_preannotate_job(number: int):
    body = request.get_json(silent=True) or {}
    job_id = api.start_preannotate_job(_project(), number, device=body.get("device") or None)
    return jsonify({"job_id": job_id}), 202


@bp.get("/readiness")
def train_readiness():
    return jsonify(api.train_readiness(_project()).model_dump())


@bp.post("/rounds/<int:number>/train")
def train_round(number: int):
    body = request.get_json(silent=True) or {}
    kwargs = {}
    for key in ("epochs", "batch_size", "resolution"):
        value = body.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProjectError(f"'{key}' must be a positive integer")
            kwargs[key] = value
    model = body.get("model")  # None → the loop picks detection or segmentation itself
    if model is not None and (not isinstance(model, str) or not model):
        raise ProjectError("'model' must be a model key")
    extra = body.get("extra") or {}
    if not isinstance(extra, dict):
        raise ProjectError("'extra' must be an object")
    record = api.train_round(
        _project(), number, model=model, device=body.get("device") or None, extra=extra,
        ignore_short_classes=bool(body.get("ignore_short_classes", False)), **kwargs
    )
    return jsonify(record.model_dump()), 202


@bp.get("/rounds/<int:number>/training")
def round_training_status(number: int):
    after = request.args.get("after", default=0, type=int)
    return jsonify(api.round_training_status(_project(), number, after=after).model_dump())


@bp.get("/history")
def loop_history():
    return jsonify([row.model_dump() for row in api.loop_history(_project())])


@bp.post("/rounds/<int:number>/assign")
def assign_round(number: int):
    body = request.get_json(silent=True) or {}
    annotators = body.get("annotators")
    if not isinstance(annotators, list) or not all(isinstance(a, str) for a in annotators):
        raise ProjectError("'annotators' must be a list of names")
    record = api.assign_round(
        _project(), number, annotators, reassign=bool(body.get("reassign", False))
    )
    return jsonify(record.model_dump())


@bp.get("/rounds/<int:number>/queue")
def round_queue(number: int):
    items = api.round_queue(
        _project(), number,
        annotator=request.args.get("annotator") or None,
        session_id=request.args.get("session") or None,
    )
    return jsonify([item.model_dump() for item in items])


def _ids(body: dict) -> list[int]:
    ids = body.get("image_ids")
    if not isinstance(ids, list) or not ids or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in ids
    ):
        raise ProjectError("'image_ids' must be a non-empty list of image ids")
    return ids


images_bp = Blueprint("loop_images", __name__, url_prefix="/api/v1/images")


@images_bp.get("/clusters")
def image_clusters():
    k = request.args.get("k", type=int)
    if request.args.get("k") and k is None:
        raise ProjectError("'k' must be an integer")
    result = api.cluster_images(
        _project(), k=k,
        model=_model(request.args, DEFAULT_EMBEDDING_MODEL),
        samples=request.args.get("samples", default=6, type=int),
        seed=request.args.get("seed", default=0, type=int),
    )
    return jsonify(result.model_dump())


@images_bp.get("/<int:image_id>/predictions")
def image_predictions(image_id: int):
    threshold = request.args.get("threshold", default=0.1, type=float)
    result = api.image_predictions(
        _project(), image_id, threshold=threshold, device=request.args.get("device") or None
    )
    return jsonify(result.model_dump())


@images_bp.get("/<int:image_id>/similar")
def similar_images(image_id: int):
    threshold = request.args.get("threshold", default=0.8, type=float)
    limit = request.args.get("limit", default=48, type=int)
    items = api.similar_images(
        _project(), image_id, threshold=threshold, limit=limit,
        model=_model(request.args, DEFAULT_EMBEDDING_MODEL),
        include_labeled=request.args.get("include_labeled") == "1",
    )
    return jsonify([item.model_dump() for item in items])


@images_bp.post("/skip")
def skip_images():
    body = request.get_json(silent=True) or {}
    note = body.get("note", "")
    if not isinstance(note, str):
        raise ProjectError("'note' must be a string")
    return jsonify(api.skip_images(_project(), _ids(body), note=note).model_dump())


@images_bp.post("/restore")
def restore_images():
    body = request.get_json(silent=True) or {}
    return jsonify(api.restore_images(_project(), _ids(body)).model_dump())


@bp.post("/rounds/<int:number>/refill")
def refill_round(number: int):
    body = request.get_json(silent=True) or {}
    record = api.refill_round(_project(), number, device=body.get("device") or None)
    return jsonify(record.model_dump())


@bp.get("/settings")
def get_loop_settings():
    return jsonify(api.get_loop_settings(_project()).model_dump())


@bp.put("/settings")
def update_loop_settings():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        raise ProjectError("Send an object with the settings to change")
    return jsonify(api.update_loop_settings(_project(), **body).model_dump())


@bp.get("/advice")
def loop_advice():
    return jsonify(api.loop_advice(_project()).model_dump())


@bp.post("/rounds/<int:number>/evaluate")
def evaluate_round_splits(number: int):
    body = request.get_json(silent=True) or {}
    record = api.evaluate_round_splits(_project(), number, device=body.get("device") or None)
    return jsonify(record.model_dump())
