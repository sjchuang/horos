"""Export routes (E8-T8). Thin by rule (R2): validate params, call horos.api."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

import horos.api as api
from horos.api.export import export_file_path
from horos.web.routes.autolabel import _project

bp = Blueprint("export", __name__, url_prefix="/api/v1")

#: artifacts a browser can render in a tab; everything else only downloads
_VIEWABLE = {".png", ".pdf", ".jpg", ".jpeg", ".svg"}


def _urls(run_id: str, name: str) -> dict:
    base = f"/api/v1/train/runs/{run_id}/exports/{name}"
    urls = {"download_url": base}
    if Path(name).suffix.lower() in _VIEWABLE:
        urls["view_url"] = base + "?inline=1"
    return urls


@bp.post("/train/runs/<run_id>/export/report")
def export_report(run_id: str):
    body = request.get_json(silent=True) or {}
    path = api.export_training_report(_project(), run_id, format=body.get("format", "png"))
    return jsonify({"name": path.name, "path": str(path), **_urls(run_id, path.name)})


@bp.post("/train/runs/<run_id>/export/evaluation")
def export_evaluation(run_id: str):
    body = request.get_json(silent=True) or {}
    path = api.export_evaluation_chart(
        _project(), run_id, body.get("split", "test"),
        threshold=float(body.get("threshold", 0.5)),
        iou=float(body.get("iou", 0.5)),
        format=body.get("format", "png"),
    )
    return jsonify({"name": path.name, "path": str(path), **_urls(run_id, path.name)})


@bp.post("/train/runs/<run_id>/export/model")
def export_model(run_id: str):
    body = request.get_json(silent=True) or {}
    job_id = api.start_model_export(
        _project(), run_id, format=body.get("format", "onnx"), options=body.get("options") or {}
    )
    return jsonify({"job_id": job_id}), 202


@bp.get("/train/runs/<run_id>/exports")
def list_exports(run_id: str):
    items = api.list_exports(_project(), run_id)
    return jsonify([a.model_dump() | _urls(run_id, a.name) for a in items])


@bp.get("/train/runs/<run_id>/exports/<name>")
def download_export(run_id: str, name: str):
    # export_file_path refuses anything outside <run>/exports/. ?inline=1 lets
    # the browser render viewable reports (PNG/PDF) in a tab instead of saving.
    path = export_file_path(_project(), run_id, name)
    inline = request.args.get("inline", "0") in ("1", "true") and path.suffix.lower() in _VIEWABLE
    return send_file(path, as_attachment=not inline)
