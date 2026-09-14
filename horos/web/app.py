"""Flask app factory with the unified error format (E9-T3)."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify

from horos.errors import (
    AnnotationConflictError,
    CategoryInUseError,
    ClassNamesRequiredError,
    HorosError,
    ImportConflictError,
    LicenseError,
    ProjectError,
    UnknownModelError,
    UnsupportedPlatformError,
)

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: list[tuple[type[HorosError], int]] = [
    (AnnotationConflictError, 409),
    (CategoryInUseError, 409),  # a question for the UI, not a bad request
    (ImportConflictError, 409),
    (ClassNamesRequiredError, 422),
    (LicenseError, 403),
    (UnknownModelError, 404),
    (UnsupportedPlatformError, 409),
    (ProjectError, 400),
]


def _status_for(exc: HorosError) -> int:
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return status
    return 400


def error_payload(code: str, message: str, details: dict | None = None) -> dict:
    """The one error shape every /api response uses: {"error": {code, message}}.

    `details` carries optional structured payload (e.g. conflict file lists)
    so the UI can act on the error, not just display it."""
    payload: dict = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return payload


def create_app(project_root: str | Path | None = None) -> Flask:
    """Build the Web API app, optionally bound to a project directory.

    The UI blueprint is registered too — the same server serves both, but the
    UI talks to the API over HTTP only (R2).
    """
    # static_folder=None: the app's default /static route would otherwise
    # shadow the UI blueprint's /static (horos/ui/static, favicon etc.)
    app = Flask("horos.web", static_folder=None)
    # resolve() matters: `horos ui my-project` hands a RELATIVE path here, and
    # Flask file helpers (send_from_directory/send_file) resolve relative
    # paths against app.root_path — the horos.web package dir — not the cwd,
    # which turned every served file into a silent 404
    app.config["HOROS_PROJECT_ROOT"] = (
        str(Path(project_root).resolve()) if project_root else None
    )
    # uploads capped at 2 GB; datasets bigger than that should be imported by path
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3
    # horos ui is a local tool: without this, Jinja caches templates per process
    # and UI changes only appear after a server restart — endlessly confusing
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    from horos.web.routes.annotate import bp as annotate_bp
    from horos.web.routes.autolabel import bp as autolabel_bp
    from horos.web.routes.data import bp as data_bp
    from horos.web.routes.evaluate import bp as evaluate_bp
    from horos.web.routes.experiment import bp as experiment_bp
    from horos.web.routes.export import bp as export_bp
    from horos.web.routes.loop import bp as loop_bp
    from horos.web.routes.loop import images_bp as loop_images_bp
    from horos.web.routes.meta import bp as meta_bp
    from horos.web.routes.segment import bp as segment_bp
    from horos.web.routes.serve import bp as serve_bp
    from horos.web.routes.train import bp as train_bp

    app.register_blueprint(annotate_bp)
    app.register_blueprint(autolabel_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(evaluate_bp)
    app.register_blueprint(experiment_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(loop_bp)
    app.register_blueprint(loop_images_bp)
    app.register_blueprint(meta_bp)
    app.register_blueprint(segment_bp)
    app.register_blueprint(serve_bp)
    app.register_blueprint(train_bp)

    from horos.ui import bp as ui_bp

    app.register_blueprint(ui_bp)

    @app.errorhandler(HorosError)
    def handle_horos_error(exc: HorosError):
        logger.info("request failed: %s: %s", exc.code, exc)
        return jsonify(error_payload(exc.code, str(exc), exc.details)), _status_for(exc)

    @app.errorhandler(404)
    def handle_not_found(exc):
        return jsonify(error_payload("not_found", "No such endpoint or resource")), 404

    @app.errorhandler(405)
    def handle_bad_method(exc):
        return jsonify(error_payload("method_not_allowed", str(exc))), 405

    @app.errorhandler(400)
    def handle_bad_request(exc):
        return jsonify(error_payload("bad_request", str(exc))), 400

    return app
