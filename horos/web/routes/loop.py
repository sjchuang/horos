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
    model = body.get("model", "rfdetr-nano")
    if not isinstance(model, str) or not model:
        raise ProjectError("'model' must be a model key")
    extra = body.get("extra") or {}
    if not isinstance(extra, dict):
        raise ProjectError("'extra' must be an object")
    record = api.train_round(
        _project(), number, model=model, device=body.get("device") or None, extra=extra, **kwargs
    )
    return jsonify(record.model_dump()), 202


@bp.get("/rounds/<int:number>/training")
def round_training_status(number: int):
    after = request.args.get("after", default=0, type=int)
    return jsonify(api.round_training_status(_project(), number, after=after).model_dump())
