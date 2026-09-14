"""horos WebUI blueprint.

R2 / E4-T8: this package must not import horos core modules (horos.core,
horos.api, horos.backends). It renders templates; all data flows through the
Web API via fetch() in the browser.
"""

from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint(
    "ui",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/loop")
def loop():
    """The active-learning loop: select → label → train → review (E10-T14).
    Its Label step embeds the annotator as /annotate?embed=1&round=<n>."""
    return render_template("loop.html")


@bp.get("/annotate")
def annotate():
    """The annotator: every photo, the canvas, classes, auto-label. Also
    the loop's embedded Label step (?embed=1&round=<n>); the old ?canvas=1
    still lands here."""
    return render_template("canvas.html")


@bp.get("/train")
def train():
    return render_template("train.html")


@bp.get("/evaluate")
def evaluate():
    return render_template("evaluate.html")


@bp.get("/experiments")
def experiments():
    return render_template("experiments.html")


@bp.get("/lab")
def lab():
    return render_template("lab.html")
