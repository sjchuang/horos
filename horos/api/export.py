"""Export a training run: reports (charts) and deployable models (E8).

Reports render synchronously (a few seconds) into <run>/exports/. Model export
runs as a background job: the backend writes the artifact, horos verifies it
against the original weights where it can (E8-T5), writes model_card.json
next to it (E8-T4, R3: the license travels with the weights) and zips the
bundle for download. TensorRT is gated by the platform capability list and by
the presence of the tensorrt package — never a silent fallback (E8-T2/T6).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import zipfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

import horos
from horos.api import jobs
from horos.api.manifest import capability
from horos.api.report import (
    REPORT_FORMATS,
    TrainingReport,
    build_training_report,
    render_report,
)
from horos.core.project import Project
from horos.errors import HorosError, ProjectError

if TYPE_CHECKING:
    from horos.backends.base import Event

logger = logging.getLogger(__name__)

__all__ = [
    "ExportArtifact",
    "ModelCard",
    "TrainingReport",
    "export_training_report",
    "export_evaluation_chart",
    "start_model_export",
    "model_export_events",
    "list_exports",
    "export_file_path",
]

ModelFormat = Literal["pytorch", "onnx", "tensorrt", "tflite"]
MODEL_FORMATS: tuple[str, ...] = ("pytorch", "onnx", "tensorrt", "tflite")
EXPORTS_DIR = "exports"
MODEL_CARD_NAME = "model_card.json"
JOB_KIND = "export"
#: images from the run's valid split used for the parity check
PARITY_IMAGES = 3
#: largest allowed score difference between matched detections (see the
#: backend's export_parity for the matching rule)
PARITY_TOLERANCE = 0.02
#: int8 (E8-T3b): how far the quantised variant's matched detections may
#: drift — int8 weights move scores more than a float16 cast, so the bar is
#: deliberately lower than the primary artifact's
INT8_PARITY_TOLERANCE = 0.1

TENSORRT_PORTABILITY_WARNING = (
    "A TensorRT engine is compiled for THIS machine's GPU architecture and "
    "TensorRT version. It will not load on another GPU model or TensorRT "
    "release — export again on each deployment device."
)


class ExportArtifact(BaseModel):
    name: str  # file name under <run>/exports/
    kind: Literal["report", "model"]
    format: str
    size_bytes: int
    created_at: str


class ModelCard(BaseModel):
    """What ships next to every exported model (E8-S5)."""

    horos_version: str
    created_at: str
    run_id: str
    model: str
    display_name: str
    code_license: str
    weights_license: str
    license_url: str
    format: str
    artifact: str
    files: list[str]
    classes: list[str]
    input: dict[str, Any] = Field(default_factory=dict)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    dataset: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    parity: dict[str, Any] = Field(default_factory=dict)
    #: the confidence to run this artifact at, from the evaluation's F-score
    #: sweep (E6-T10), with what it was derived from and the per-class figures.
    #: Empty `confidence` when the run has no evaluation to derive it from —
    #: `note` then says how to get one
    threshold: dict[str, Any] = Field(default_factory=dict)
    portability: str = ""
    #: extra precision / quantisation files in the bundle, e.g. TFLite
    #: {"float16": {"artifact", "method"}, "int8": {"artifact", "method",
    #: "weights", "activations", "input_layout", "parity"}}
    variants: dict[str, Any] = Field(default_factory=dict)


def exports_dir(project: Project, run_id: str) -> Path:
    from horos.api.train import _run_dir

    return _run_dir(project, run_id) / EXPORTS_DIR


# ------------------------------------------------------------------ reports


@capability(
    "export.report",
    summary="Render a run's training report as a 16:9 PNG dashboard, PDF, or Excel workbook",
    web_route="/api/v1/train/runs/<run_id>/export/report",
    web_methods=("POST",),
    cli="report",
)
def export_training_report(
    project: Project,
    run_id: str,
    *,
    format: str = "png",
    out_path: Path | str | None = None,
) -> Path:
    """Write the report to <run>/exports/training_report.<format> (or
    `out_path`) and return the path. Works for any run that has recorded
    events — a failed run's report shows where it failed."""
    if format not in REPORT_FORMATS:
        raise ProjectError(
            f"Unsupported report format '{format}' ({'|'.join(REPORT_FORMATS)})"
        )
    report = build_training_report(project, run_id)
    target = (
        Path(out_path)
        if out_path is not None
        else exports_dir(project, run_id) / f"training_report.{format}"
    )
    return render_report(report, format, target)


#: the evaluation sheet renders straight from matplotlib, so PNG and PDF only
EVAL_CHART_FORMATS: tuple[str, ...] = ("png", "pdf")


@capability(
    "export.evaluation_chart",
    summary="Render a run's evaluation as one sheet: the confusion matrix beside "
    "the per-class performance table",
    web_route="/api/v1/train/runs/<run_id>/export/evaluation",
    web_methods=("POST",),
    cli="report",
)
def export_evaluation_chart(
    project: Project,
    run_id: str,
    split: str = "test",
    *,
    threshold: float = 0.5,
    iou: float = 0.5,
    format: str = "png",
    out_path: Path | str | None = None,
) -> Path:
    """Write the evaluation sheet to <run>/exports/ (or `out_path`).

    Everything on it comes from the same error analysis the evaluate page
    draws at that threshold and IoU, so the exported sheet and the page cannot
    disagree. Needs a prior evaluation of `split` — the analysis re-matches its
    persisted detections."""
    from horos.api.error_analysis import analyze_errors
    from horos.api.evaluate import get_eval_report
    from horos.api.report import evaluation_figure
    from horos.api.threshold import suggest_threshold
    from horos.api.train import read_record

    if format not in EVAL_CHART_FORMATS:
        raise ProjectError(
            f"Unsupported evaluation chart format '{format}' "
            f"({'|'.join(EVAL_CHART_FORMATS)})"
        )
    analysis = analyze_errors(project, run_id, split, threshold=threshold, iou=iou)
    try:
        report = get_eval_report(project, run_id, split)
    except HorosError:  # analysed from detections an older evaluation left
        report = None
    try:
        advice = suggest_threshold(project, run_id, split, iou=iou)
    except HorosError:  # pragma: no cover — the analysis above just read them
        advice = None
    try:
        model = read_record(_run_dir_of(project, run_id)).model
    except (HorosError, OSError):
        model = ""
    figure = evaluation_figure(
        analysis=analysis, eval_report=report, advice=advice, model=model
    )
    target = (
        Path(out_path) if out_path is not None
        else exports_dir(project, run_id) / f"evaluation_{split}.{format}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, facecolor="white")
    return target


def _run_dir_of(project: Project, run_id: str) -> Path:
    from horos.api.train import _run_dir

    return _run_dir(project, run_id)


# ------------------------------------------------------------------ models


#: splits to look for a threshold in, best first — an operating point belongs
#: on held-out data, and test is the honest one
_THRESHOLD_SPLITS = ("test", "valid")


def _suggested_threshold(project: Project, run_id: str) -> dict[str, Any]:
    """The confidence to deploy at, for the model card (E8-T4).

    It comes from the same sweep the evaluate page shows (E6-T10), so the
    number shipped next to the model is the number the user was looking at.
    Derived from an evaluation this run already has: nothing is inferred or
    re-run here, and a run that was never evaluated ships the reason instead
    of a made-up value."""
    from horos.api.evaluate import get_eval_report
    from horos.api.threshold import suggest_threshold

    for split in _THRESHOLD_SPLITS:
        try:
            advice = suggest_threshold(project, run_id, split)
        except (HorosError, OSError):
            continue  # this split was never evaluated
        if not advice.confident:
            why = advice.reason[0].lower() + advice.reason[1:]
            return {"confidence": None, "note": f"Evaluated on '{split}', but {why}"}
        try:
            report = get_eval_report(project, run_id, split)
        except HorosError:  # pragma: no cover — the sweep just read that split
            report = None
        return {
            "confidence": advice.recommended,
            "reason": advice.reason,
            "metric": f"F{advice.beta:g}",
            "precision": round(advice.best.precision, 4),
            "recall": round(advice.best.recall, 4),
            "f_score": round(advice.best.f_score, 4),
            #: every threshold in this range scores within 1 % of the peak
            "plateau": list(advice.plateau),
            "derived_from": {
                "split": split,
                "iou": advice.iou,
                # which labels that evaluation scored (E6-T13)
                "labels": report.labels if report else None,
                "evaluated_at": report.created_at if report else None,
                "images": report.num_images if report else None,
                "instances": report.num_instances if report else None,
            },
            "per_class": [
                {
                    "name": c.name,
                    "confidence": c.recommended,
                    "precision": round(c.precision, 4),
                    "recall": round(c.recall, 4),
                    "instances": c.instances,
                    # too little ground truth for this one to mean much
                    "enough_data": c.enough_data,
                }
                for c in advice.per_class
            ],
            "notes": advice.notes,
        }
    return {
        "confidence": None,
        "note": "No evaluation to derive one from — run 'horos evaluate' on this "
                "run's test or valid split and export again.",
    }


def _dataset_fingerprint(run_dir: Path) -> dict[str, Any]:
    """The run's dataset fingerprint for the model card (E7-T2): the one
    recorded at enqueue time, or — for runs older than that field — computed
    from the snapshot with the same canonical method."""
    from horos.api.train import read_record
    from horos.core.fingerprint import METHOD, fingerprint_snapshot

    fingerprint = None
    if (run_dir / "run.json").is_file():
        fingerprint = read_record(run_dir).dataset_fingerprint
    if fingerprint is None:
        fingerprint = fingerprint_snapshot(run_dir / "dataset")
    return {
        "fingerprint": fingerprint.digest if fingerprint else None,
        "split_fingerprints": fingerprint.splits if fingerprint else {},
        "method": fingerprint.method if fingerprint else METHOD,
    }


def _tensorrt_available() -> bool:
    try:
        return importlib.util.find_spec("tensorrt") is not None
    except (ImportError, ValueError):
        return False


def _tflite_available() -> bool:
    from horos.backends.convert.tflite import toolchain_available

    return toolchain_available()


def _check_format(format: str) -> None:
    if format not in MODEL_FORMATS:
        raise ProjectError(
            f"Unsupported model format '{format}' ({'|'.join(MODEL_FORMATS)})"
        )
    if format == "tensorrt":
        from horos.api.system import ensure_supported

        ensure_supported("export_tensorrt")  # macOS: refused, never a CPU fallback
        if not _tensorrt_available():
            raise ProjectError(
                "TensorRT export needs NVIDIA's 'tensorrt' Python package on this "
                "machine. Run 'horos install --tensorrt' to add the wheels matching "
                "this GPU's CUDA version (NVIDIA TensorRT license — installed only on "
                "your request); on Jetson use JetPack's tensorrt via a "
                "--system-site-packages venv. Then retry."
            )
    if format == "tflite":
        from horos.api.system import ensure_supported

        ensure_supported("export_tflite")
        if not _tflite_available():
            raise ProjectError(
                "TFLite export needs the conversion toolchain (onnx2tf + tensorflow, "
                "~600 MB, Apache 2.0 / MIT). Run 'horos install --tflite' to add it, "
                "then retry."
            )


def _parity_images(run_dir: Path, limit: int) -> list[Path]:
    for split in ("valid", "test", "train"):
        gt_path = run_dir / "dataset" / split / "_annotations.coco.json"
        if not gt_path.is_file():
            continue
        try:
            gt = json.loads(gt_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        images = [gt_path.parent / img["file_name"] for img in gt.get("images", [])]
        images = [p for p in images if p.is_file()]
        if images:
            return images[:limit]
    return []


def model_export_events(
    project: Project,
    run_id: str,
    *,
    format: str = "onnx",
    options: dict[str, Any] | None = None,
) -> Iterator[Event]:
    """R4 stream: started → backend progress → parity → model card → bundle →
    completed(result={format, artifact, bundle, model_card, parity})."""
    from horos.api.evaluate import _load_run_backend
    from horos.api.train import _run_dir
    from horos.backends.base import (
        ExportSpec,
        ProgressUpdated,
        RunCompleted,
        RunFailed,
        RunStarted,
        WarningRaised,
    )
    from horos.core.registry import get_model_info
    from horos.errors import UnknownModelError

    options = dict(options or {})
    _check_format(format)
    backend, record = _load_run_backend(project, run_id)  # completed runs only
    run_dir = _run_dir(project, run_id)
    out_dir = exports_dir(project, run_id) / format
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    yield RunStarted(config={"run_id": run_id, "format": format, "options": options})
    if format == "tensorrt":
        yield WarningRaised(message=TENSORRT_PORTABILITY_WARNING)

    spec = ExportSpec(format=format, output_dir=out_dir, options=options)  # type: ignore[arg-type]
    result: dict[str, Any] | None = None
    try:
        for event in backend.export(Path(record.checkpoint), spec):
            if event.type == "completed":
                result = dict(event.result)
            elif event.type == "failed":
                yield event
                return
            elif event.type != "started":
                yield event
        if result is None or not result.get("artifact"):
            raise ProjectError("The backend finished without reporting an artifact")
        artifact = Path(result["artifact"])
        if not artifact.is_file():
            raise ProjectError(f"Exported artifact is missing: {artifact}")

        # class names travel with every bundle, whatever the backend wrote
        classes = list(record.dataset_classes)
        names_path = out_dir / "class_names.txt"
        if not names_path.is_file() and classes:
            names_path.write_text("\n".join(classes) + "\n", "utf-8")

        # parity against the original weights (E8-T5) — the backend knows the
        # framework; horos supplies real images from the run's own split
        yield ProgressUpdated(current=0, total=None, phase="verifying against the original weights")
        parity: dict[str, Any]
        images = _parity_images(run_dir, int(options.get("parity_images", PARITY_IMAGES)))
        tolerance = float(options.get("parity_tolerance", PARITY_TOLERANCE))
        try:
            check = backend.export_parity(artifact, spec, images, tolerance=tolerance)
        except HorosError as exc:
            check = {"status": "error", "message": str(exc)}
        if check is None:
            parity = {"status": "not_available",
                      "message": f"no parity check for {format} in the {backend.family} backend"}
        else:
            parity = {"status": "passed" if check.get("passed", True) else "failed", **check}
            if parity["status"] == "failed":
                detail = check.get("message") or (
                    f"{check.get('unmatched_detections', 0)} unmatched detection(s), "
                    f"max score difference {check.get('max_score_diff')} (tolerance {tolerance})"
                )
                yield WarningRaised(
                    message=f"Parity check FAILED: {detail} — the exported model does not "
                    f"reproduce the original weights' detections"
                )

        # precision / quantisation variants (E8-T3b): each is verified like the
        # primary artifact; the float32 model stays what the card points at
        variants: dict[str, Any] = {}
        for name, meta in (result.get("variants") or {}).items():
            meta = dict(meta) if isinstance(meta, dict) else {"artifact": str(meta)}
            vpath = Path(meta.pop("artifact"))
            entry: dict[str, Any] = {"artifact": vpath.name, **meta}
            if name == "int8":
                yield ProgressUpdated(
                    current=0, total=None,
                    phase="verifying the int8 model against the original weights",
                )
                int8_tolerance = float(options.get("int8_parity_tolerance", INT8_PARITY_TOLERANCE))
                try:
                    vcheck = backend.export_parity(vpath, spec, images, tolerance=int8_tolerance)
                except HorosError as exc:
                    vcheck = {"status": "error", "message": str(exc)}
                if vcheck is not None:
                    entry["parity"] = {
                        "status": "passed" if vcheck.get("passed", True) else "failed", **vcheck,
                    }
                    if entry["parity"]["status"] == "failed":
                        unmatched = vcheck.get("unmatched_detections", 0)
                        yield WarningRaised(
                            message=f"int8 parity check FAILED ({unmatched} unmatched "
                                    f"detection(s), max score difference "
                                    f"{vcheck.get('max_score_diff')} > {int8_tolerance}) — the "
                                    f"int8 file is kept for inspection; the float32 model "
                                    f"remains the bundle's artifact"
                        )
            variants[name] = entry
        # model card (E8-T4)
        yield ProgressUpdated(current=0, total=None, phase="writing model card")
        try:
            info = get_model_info(record.model)
            display, code_lic, weights_lic, url = (
                info.display_name, info.code_license, info.weights_license, info.license_url,
            )
            resolution = info.input_resolution
        except UnknownModelError:
            display, code_lic, weights_lic, url = record.model, "unknown", "unknown", ""
            resolution = None
        hparams = {h.name: h.value for h in record.hparams}
        if hparams.get("resolution"):
            resolution = hparams["resolution"]
        report = build_training_report(project, run_id)
        input_spec = result.get("input") or {
            "name": "input", "dtype": "float32",
            "shape": [1, 3, resolution, resolution] if resolution else None,
            "layout": "NCHW, RGB, scaled to [0,1] then ImageNet mean/std normalised",
        }
        files = sorted(
            p.name for p in out_dir.iterdir() if p.is_file() and p.name != MODEL_CARD_NAME
        )
        card = ModelCard(
            horos_version=horos.__version__,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            run_id=run_id,
            model=record.model,
            display_name=display,
            code_license=code_lic,
            weights_license=weights_lic,
            license_url=url,
            format=format,
            artifact=artifact.name,
            files=files,
            classes=classes,
            input=input_spec,
            outputs=list(result.get("outputs") or []),
            dataset={
                "images": record.dataset_images,
                "splits": record.dataset_splits,
                "classes": classes,
                **_dataset_fingerprint(run_dir),
            },
            metrics=report.final_metrics,
            hyperparameters=hparams,
            parity=parity,
            threshold=_suggested_threshold(project, run_id),
            portability=TENSORRT_PORTABILITY_WARNING if format == "tensorrt" else
            "Portable: this artifact runs on any machine with the matching runtime.",
            variants=variants,
        )
        (out_dir / MODEL_CARD_NAME).write_text(card.model_dump_json(indent=2), "utf-8")

        # bundle
        yield ProgressUpdated(current=0, total=None, phase="bundling")
        bundle = exports_dir(project, run_id) / f"{run_id}_{format}.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out_dir.iterdir()):
                if path.is_file():
                    zf.write(path, path.name)
    except HorosError as exc:
        yield RunFailed(error_code=exc.code, message=str(exc), details=exc.details or {})
        return
    except Exception as exc:  # noqa: BLE001 — R4: the stream terminates itself
        logger.exception("model export of run %s failed", run_id)
        yield RunFailed(error_code="export_error", message=f"{type(exc).__name__}: {exc}")
        return

    yield RunCompleted(
        result={
            "format": format,
            "artifact": str(artifact),
            "files": files,
            "bundle": bundle.name,
            "model_card": card.model_dump(),
            "parity": parity,
            "variants": variants,
        }
    )


@capability(
    "export.model",
    summary="Export a completed run's model (PyTorch weights, ONNX, TensorRT) as a job",
    web_route="/api/v1/train/runs/<run_id>/export/model",
    web_methods=("POST",),
    cli="export-model",
)
def start_model_export(
    project: Project,
    run_id: str,
    *,
    format: str = "onnx",
    options: dict[str, Any] | None = None,
) -> str:
    """Start the export job; returns the job id (poll jobs.status). Format,
    platform and dependency problems raise here, synchronously."""
    from horos.api.evaluate import _load_run_backend

    _check_format(format)
    _load_run_backend(project, run_id)  # fail now for non-completed runs
    return jobs.start_job(
        project,
        JOB_KIND,
        lambda cancel: model_export_events(project, run_id, format=format, options=options),
    )


# ------------------------------------------------------------------ listing


def _artifact(path: Path) -> ExportArtifact | None:
    if not path.is_file():
        return None
    if path.name.startswith("training_report."):
        kind, fmt = "report", path.suffix.lstrip(".")
    elif path.suffix == ".zip":
        kind, fmt = "model", path.stem.rsplit("_", 1)[-1]
    else:
        return None
    stat = path.stat()
    return ExportArtifact(
        name=path.name,
        kind=kind,  # type: ignore[arg-type]
        format=fmt,
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
    )


@capability(
    "export.list",
    summary="List a run's exported reports and model bundles",
    web_route="/api/v1/train/runs/<run_id>/exports",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Exports land in <project>/runs/<run_id>/exports/ — ls shows them.",
)
def list_exports(project: Project, run_id: str) -> list[ExportArtifact]:
    root = exports_dir(project, run_id)
    if not root.is_dir():
        return []
    items = [a for a in (_artifact(p) for p in root.iterdir()) if a is not None]
    return sorted(items, key=lambda a: a.created_at, reverse=True)


def export_file_path(project: Project, run_id: str, name: str) -> Path:
    """The on-disk file behind a download; refuses anything outside exports/."""
    root = exports_dir(project, run_id).resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise ProjectError(f"No export named '{name}' for run {run_id}")
    return path
