"""Evaluation routes (E6-T9). Thin by rule (R2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, request

import horos.api as api
from horos.errors import ProjectError
from horos.web.routes.autolabel import _project

bp = Blueprint("evaluate", __name__, url_prefix="/api/v1")


@bp.post("/train/runs/<run_id>/infer")
def infer_image(run_id: str):
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ProjectError("Attach an image as multipart field 'file'.")
    threshold = request.form.get("threshold", 0.05, type=float)
    suffix = Path(upload.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        upload.save(handle)
        temp_path = Path(handle.name)
    try:
        prediction = api.infer_image(
            _project(), run_id, temp_path, threshold=threshold
        )
        payload = prediction.model_dump()
        payload["image"] = upload.filename  # never leak the server temp path
        return jsonify(payload)
    finally:
        temp_path.unlink(missing_ok=True)


@bp.post("/train/runs/<run_id>/media")
def start_media_inference(run_id: str):
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ProjectError("Attach a photo, GIF, or video as multipart field 'file'.")
    suffix = Path(upload.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        upload.save(handle)
        temp_path = Path(handle.name)
    try:
        media_id, job_id = api.start_media_inference(
            _project(), run_id, temp_path, source_name=upload.filename
        )
        return jsonify({"media_id": media_id, "job_id": job_id}), 202
    finally:
        # the API keeps its own copy inside the media directory
        temp_path.unlink(missing_ok=True)


@bp.get("/train/runs/<run_id>/media")
def list_media(run_id: str):
    return jsonify([m.model_dump() for m in api.list_media(_project(), run_id)])


@bp.get("/train/runs/<run_id>/media/<media_id>")
def get_media(run_id: str, media_id: str):
    return jsonify(api.get_media(_project(), run_id, media_id).model_dump())


@bp.delete("/train/runs/<run_id>/media/<media_id>")
def delete_media(run_id: str, media_id: str):
    return jsonify({"deleted": api.delete_media(_project(), run_id, media_id)})


@bp.get("/train/runs/<run_id>/media/<media_id>/frames/<frame_name>")
def get_media_frame(run_id: str, media_id: str, frame_name: str):
    from flask import send_from_directory

    from horos.api.media import media_dir

    frames = media_dir(_project(), run_id, media_id) / "frames"
    # send_from_directory refuses path escapes (../) on its own
    return send_from_directory(frames, frame_name, max_age=3600)


@bp.post("/train/runs/<run_id>/evaluate")
def start_evaluation(run_id: str):
    body = request.get_json(silent=True) or {}
    job_id = api.start_evaluation(
        _project(), run_id,
        split=body.get("split", "test"),
        labels=body.get("labels", api.DEFAULT_LABELS),
    )
    return jsonify({"job_id": job_id}), 202


@bp.get("/train/runs/<run_id>/eval/<split>")
def get_eval_report(run_id: str, split: str):
    return jsonify(api.get_eval_report(_project(), run_id, split).model_dump())


def _analysis_args() -> dict:
    return {
        "threshold": request.args.get("threshold", 0.5, type=float),
        "iou": request.args.get("iou", 0.5, type=float),
    }


@bp.get("/train/runs/<run_id>/eval/<split>/errors")
def analyze_errors(run_id: str, split: str):
    analysis = api.analyze_errors(_project(), run_id, split, **_analysis_args())
    return jsonify(analysis.model_dump())


@bp.get("/train/runs/<run_id>/eval/<split>/threshold")
def suggest_threshold(run_id: str, split: str):
    advice = api.suggest_threshold(
        _project(), run_id, split,
        iou=request.args.get("iou", 0.5, type=float),
        beta=request.args.get("beta", 1.0, type=float),
    )
    return jsonify(advice.model_dump())


@bp.get("/train/runs/<run_id>/eval/<split>/worst")
def worst_cases(run_id: str, split: str):
    report = api.worst_cases(
        _project(), run_id, split,
        top_k=request.args.get("top", 50, type=int),
        **_analysis_args(),
    )
    return jsonify(report.model_dump())


@bp.get("/train/runs/<run_id>/eval/<split>/images/<int:image_id>/overlay.png")
def error_overlay(run_id: str, split: str, image_id: int):
    import io

    from flask import send_file

    from horos.api.visualize import to_png_bytes

    image = api.render_error_overlay(
        _project(), run_id, split, image_id, **_analysis_args()
    )
    return send_file(io.BytesIO(to_png_bytes(image)), mimetype="image/png", max_age=0)
