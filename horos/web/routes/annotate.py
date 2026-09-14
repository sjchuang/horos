"""Annotation routes (E2-T9). Thin by rule (R2): validate params, call horos.api."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request, send_file

import horos.api as api
from horos.errors import ProjectError

bp = Blueprint("annotate", __name__, url_prefix="/api/v1")


def _project():
    root = current_app.config.get("HOROS_PROJECT_ROOT")
    if not root:
        raise ProjectError(
            "This server is not bound to a project. Start it with "
            "'horos ui <dir>' or POST /api/v1/projects first."
        )
    return api.open_project(root)


def _body() -> dict:
    return request.get_json(silent=True) or {}


@bp.get("/queue")
def queue():
    items = api.image_queue(
        _project(),
        mode=request.args.get("mode", "unannotated_first"),
        split=request.args.get("split") or None,
        category_id=request.args.get("category_id", type=int),
        session_id=request.args.get("session") or None,
    )
    return jsonify([i.model_dump() for i in items])


@bp.get("/progress")
def progress():
    return jsonify(api.annotation_progress(_project()).model_dump())


@bp.get("/images/<int:image_id>/annotations")
def get_annotations(image_id: int):
    return jsonify(api.get_annotations(_project(), image_id).model_dump())


@bp.put("/images/<int:image_id>/annotations")
def put_annotations(image_id: int):
    body = _body()
    if "expected_version" not in body:
        raise ProjectError("Request body must include 'expected_version'")
    view = api.save_annotations(
        _project(),
        image_id,
        body.get("annotations", []),
        expected_version=int(body["expected_version"]),
    )
    return jsonify(view.model_dump())


@bp.get("/images/<int:image_id>/file")
def image_file(image_id: int):
    return send_file(api.image_file_path(_project(), image_id))


@bp.post("/images/<int:image_id>/claim")
def claim(image_id: int):
    body = _body()
    session = body.get("session")
    if not session:
        raise ProjectError("Request body must include 'session'")
    result = api.claim_image(_project(), image_id, session)
    return jsonify(result.model_dump()), 200 if result.granted else 409


@bp.delete("/images/<int:image_id>/claim")
def release(image_id: int):
    body = _body()
    session = body.get("session")
    if not session:
        raise ProjectError("Request body must include 'session'")
    return jsonify({"released": api.release_claim(_project(), image_id, session)})


@bp.post("/categories")
def add_category():
    body = _body()
    if not body.get("name"):
        raise ProjectError("Request body must include 'name'")
    category = api.add_category(_project(), body["name"], color=body.get("color"))
    return jsonify(category.model_dump()), 201


@bp.post("/categories/merge")
def merge_categories():
    body = _body()
    sources, target = body.get("sources"), body.get("target")
    if not isinstance(sources, list) or not all(isinstance(v, int) for v in sources):
        raise ProjectError("Request body must include 'sources': a list of category ids")
    if not isinstance(target, int):
        raise ProjectError("Request body must include 'target': the category id to keep")
    return jsonify(api.merge_categories(_project(), sources, target).model_dump())


@bp.patch("/categories/<int:category_id>")
def update_category(category_id: int):
    body = _body()
    category = api.update_category(
        _project(), category_id, name=body.get("name"), color=body.get("color")
    )
    return jsonify(category.model_dump())


@bp.delete("/categories/<int:category_id>")
def delete_category(category_id: int):
    deleted = api.delete_category(
        _project(), category_id, force=bool(_body().get("force", False))
    )
    return jsonify({"deleted_annotations": deleted})


@bp.post("/categories/<int:category_id>/delete")
def start_delete_category_job(category_id: int):
    job_id = api.start_delete_category_job(
        _project(), category_id, force=bool(_body().get("force", False))
    )
    return jsonify({"job_id": job_id}), 202
