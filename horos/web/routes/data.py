"""Dataset routes (E1-T9). Thin by rule (R2): validate params, call horos.api."""

from __future__ import annotations

import json

from flask import Blueprint, current_app, jsonify, request

import horos.api as api
from horos.errors import ProjectError

bp = Blueprint("data", __name__, url_prefix="/api/v1")


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


@bp.post("/projects")
def create_project():
    body = _body()
    path = body.get("path")
    if not path:
        raise ProjectError("Request body must include 'path'")
    project = api.create_project(path, name=body.get("name"))
    if not current_app.config.get("HOROS_PROJECT_ROOT"):
        current_app.config["HOROS_PROJECT_ROOT"] = str(project.root)
    return jsonify({"root": str(project.root), "name": project.manifest.name}), 201


@bp.get("/project")
def project_summary():
    project = _project()
    return jsonify(
        {
            "root": str(project.root),
            "name": project.manifest.name,
            "categories": [c.model_dump() for c in project.categories],
            "split_ratios": project.split_ratios.model_dump(),
            "split_seed": project.manifest.split_seed,
            "num_images": len(project.list_images()),
        }
    )


def _class_names(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        names = json.loads(raw)
    except ValueError as exc:
        raise ProjectError(f"'class_names' is not valid JSON: {exc}") from exc
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ProjectError("'class_names' must be a JSON array of strings")
    return names


@bp.post("/dataset/import")
def import_dataset():
    body = _body()
    source = body.get("path")
    if not source:
        raise ProjectError("Request body must include 'path' (dataset directory)")
    summary = api.import_dataset(
        _project(),
        source,
        format=body.get("format"),
        copy_images=bool(body.get("copy_images", True)),
        on_conflict=body.get("on_conflict", "ask"),
        on_annotations=body.get("on_annotations", "ask"),
        class_names=body.get("class_names"),
    )
    return jsonify(summary.model_dump())


@bp.post("/dataset/upload")
def upload_dataset():
    """Stage the upload and start its import job. One 'file' field holding a
    zip stages a dataset; one or more 'file' fields holding photos stage
    loose photos (E1-T11). The page polls /jobs/<job_id>; the completed
    event's result is the ImportSummary, a failed event with
    details.retryable=true is answered by POST /dataset/upload/<id>/import."""
    uploads = [f for f in request.files.getlist("file") if f.filename]
    if not uploads:
        raise ProjectError("Attach the dataset zip or the photos as multipart field 'file'")
    class_names = _class_names(request.form.get("class_names"))
    project = _project()
    if len(uploads) == 1 and uploads[0].filename.lower().endswith(".zip"):
        staged = api.stage_upload(project, uploads[0].stream, file_name=uploads[0].filename)
    else:
        staged = api.stage_photos(project, [(f.filename, f.stream) for f in uploads])
    job_id = api.start_upload_import(
        project,
        staged.upload_id,
        on_conflict=request.form.get("on_conflict", "ask"),
        on_annotations=request.form.get("on_annotations", "ask"),
        class_names=class_names,
        # the UI shows an editable class-name dialog instead of placeholders
        require_class_names=True,
    )
    return jsonify(staged.model_dump() | {"job_id": job_id}), 202


@bp.post("/dataset/upload/<upload_id>/import")
def import_upload(upload_id: str):
    body = _body()
    job_id = api.start_upload_import(
        _project(),
        upload_id,
        on_conflict=body.get("on_conflict", "ask"),
        on_annotations=body.get("on_annotations", "ask"),
        class_names=body.get("class_names"),
        require_class_names=True,
    )
    return jsonify({"upload_id": upload_id, "job_id": job_id}), 202


@bp.delete("/dataset/upload/<upload_id>")
def discard_upload(upload_id: str):
    return jsonify({"discarded": api.discard_upload(_project(), upload_id)})


@bp.post("/dataset/export")
def export_dataset():
    body = _body()
    out_dir = body.get("out_dir")
    if not out_dir:
        raise ProjectError("Request body must include 'out_dir'")
    written = api.export_dataset(
        _project(), out_dir, format=body.get("format", "coco")
    )
    return jsonify({"path": str(written)})


@bp.get("/dataset/validation")
def validation():
    report = api.validate_project(_project())
    return jsonify(report.model_dump() | {"ok": report.ok, "counts": report.counts()})


@bp.post("/dataset/validation/fix")
def validation_fix():
    result = api.fix_validation_issues(_project())
    return jsonify(result.model_dump() | {"ok": result.report.ok})


@bp.get("/dataset/stats")
def stats():
    raw = request.args.get("categories")
    categories = [n for n in raw.split(",") if n] if raw is not None else None
    include_background = request.args.get("include_background", "0").lower() in ("1", "true")
    return jsonify(
        api.dataset_stats(
            _project(), categories=categories, include_background=include_background
        ).model_dump()
    )


@bp.post("/dataset/split")
def split():
    body = _body()
    opt = lambda key, cast: None if body.get(key) is None else cast(body[key])  # noqa: E731
    counts = api.resplit(
        _project(),
        train=opt("train", float), valid=opt("valid", float), test=opt("test", float),
        seed=opt("seed", int), reshuffle=bool(body.get("reshuffle", False)),
    )
    return jsonify(counts)


@bp.get("/images")
def images():
    return jsonify([i.model_dump() for i in api.list_images(_project())])


@bp.delete("/dataset")
def dataset_clear():
    body = _body()
    confirm = body.get("confirm")
    if not isinstance(confirm, str):
        raise ProjectError("Request body must include 'confirm': the project name")
    summary = api.clear_dataset(
        _project(),
        confirm=confirm,
        keep_categories=bool(body.get("keep_categories", True)),
        session_id=body.get("session"),
    )
    return jsonify(summary.model_dump())


@bp.post("/images/delete")
def images_delete():
    body = _body()
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ProjectError("Request body must include a non-empty 'ids' list")
    summary = api.delete_images(
        _project(), [int(i) for i in ids], session_id=body.get("session")
    )
    return jsonify(summary.model_dump())
