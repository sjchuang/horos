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
    return render_template("loop.html")


@bp.get("/annotate")
def annotate():
    return render_template("annotate.html")


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
