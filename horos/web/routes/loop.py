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
