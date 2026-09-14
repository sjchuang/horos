"""Evaluation and inference over trained runs (E6).

Design decision (confirmed): the model source is always a run id — inference
and evaluation load that run's best checkpoint and model settings, and results
are written back into the run directory so experiment comparison (E7) can read
them with full lineage. Evaluation reads the run's OWN dataset export
(`runs/<id>/dataset/<split>`), not the project's current data: the snapshot
the model trained against is the only comparable ground truth.

Metrics come from pycocotools (the reference implementation, confirmed),
imported lazily like every heavy dependency (R1b) — an annotation-only install
gets a clear error, not an import crash at startup.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from horos.api import jobs
from horos.api.manifest import capability
from horos.api.train import TrainRunConfig, _run_dir, read_record
from horos.backends.base import (
    ImagePrediction,
    MetricsUpdated,
    ProgressUpdated,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from horos.core.project import Project
from horos.errors import ProjectError

if TYPE_CHECKING:
    from horos.backends.base import ModelBackend

logger = logging.getLogger(__name__)

__all__ = [
    "ClassEval",
    "EvalReport",
    "infer_image",
    "evaluation_events",
    "evaluate_run",
    "start_evaluation",
    "get_eval_report",
    "load_detections",
]

#: evaluation needs low-confidence detections; COCO AP integrates over them
_EVAL_THRESHOLD = 0.001


class ClassEval(BaseModel):
    category_id: int
    name: str
    instances: int
    ap: float  # AP@[.50:.95]
    ap50: float
    #: precision at the 101 standard recall points, IoU=0.5 (PR curve)
    pr_curve_50: list[float] = Field(default_factory=list)


class EvalReport(BaseModel):
    run_id: str
    split: str
    created_at: str
    num_images: int
    num_instances: int
    map_5095: float
    map_50: float
    map_75: float
    mar_100: float
    per_class: list[ClassEval] = Field(default_factory=list)


# ------------------------------------------------------------- run binding


_BACKENDS: dict[tuple[str, str, str | None], ModelBackend] = {}
_BACKENDS_LOCK = threading.Lock()


def _load_run_backend(project: Project, run_id: str, device: str | None = None):
    """The run's best checkpoint, loaded through its own model settings."""
    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    if record.state != "completed" or not record.checkpoint:
        raise ProjectError(
            f"Run {run_id} is '{record.state}' and has no usable checkpoint — "
            f"only completed runs can be evaluated or used for inference."
        )
    checkpoint = Path(record.checkpoint)
    if not checkpoint.is_file():
        raise ProjectError(f"Checkpoint of run {run_id} is missing: {checkpoint}")

    key = (str(project.root), run_id, device)
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(key)
    if backend is None:
        config = TrainRunConfig.model_validate_json(
            (run_dir / "config.json").read_text("utf-8")
        )
        if config.entrypoint_override:
            # testing hook — resolved exactly like the training worker does
            import importlib

            module_name, _, class_name = config.entrypoint_override.partition(":")
            backend_cls = getattr(importlib.import_module(module_name), class_name)
            backend = backend_cls(None, device=device, checkpoint=checkpoint)
        else:
            from horos.backends import get_backend

            backend = get_backend(record.model, device=device, checkpoint=checkpoint)
        with _BACKENDS_LOCK:
            _BACKENDS[key] = backend
    return backend, record


def _reset_backend_cache() -> None:  # tests only
    with _BACKENDS_LOCK:
        _BACKENDS.clear()


def _split_gt(project: Project, run_id: str, split: str) -> tuple[Path, dict]:
    gt_path = _run_dir(project, run_id) / "dataset" / split / "_annotations.coco.json"
    if not gt_path.is_file():
        raise ProjectError(
            f"Run {run_id} has no '{split}' split in its dataset snapshot. "
            f"Available splits live under runs/{run_id}/dataset/."
        )
    gt = json.loads(gt_path.read_text("utf-8"))
    if not gt.get("images"):
        raise ProjectError(
            f"The '{split}' split of run {run_id} contains no images."
        )
    return gt_path, gt


# ---------------------------------------------------------------- inference


@capability(
    "infer.image",
    summary="Run a trained run's model on one image",
    web_route="/api/v1/train/runs/<run_id>/infer",
    web_methods=("POST",),
    cli="infer",
)
def infer_image(
    project: Project,
    run_id: str,
    image: Path | str,
    *,
    threshold: float = 0.5,
    device: str | None = None,
) -> ImagePrediction:
    """Single-image inference with the run's best checkpoint (E6-S1)."""
    backend, _ = _load_run_backend(project, run_id, device)
    image = Path(image)
    if not image.is_file():
        raise ProjectError(f"No such image file: {image}")
    from horos.api.labels import resolve_prediction_names

    # a class renamed since this run trained is reported under its current name
    return resolve_prediction_names(project, backend.infer_one(image, threshold=threshold))


# --------------------------------------------------------------- evaluation


def evaluation_events(
    project: Project,
    run_id: str,
    *,
    split: str = "test",
    device: str | None = None,
    cancel: threading.Event | None = None,
) -> Any:
    """R4 event stream: inference over the run's split snapshot, then COCO
    metrics. RunCompleted carries the report; it is also persisted under
    `runs/<id>/eval/<split>.json`.

    The raw low-threshold detections are persisted next to it as
    `<split>.detections.json` (confirmed design, E6-T4): error analysis and
    worst-case mining re-match them against the ground truth at whatever
    operating threshold the user asks for, without re-running inference."""
    backend, record = _load_run_backend(project, run_id, device)
    gt_path, gt = _split_gt(project, run_id, split)

    def stream():
        images = gt["images"]
        # Predictions identify classes by NAME (backends emit their own label
        # indices — rfdetr's are 0-based and unrelated to COCO category ids);
        # map names onto this split's category ids. A prediction whose class
        # is not in the ground truth stays under an id no gt category uses,
        # so it can never be scored as a match by accident.
        id_by_name = {c["name"]: c["id"] for c in gt.get("categories", [])}
        unmatched_id = min(id_by_name.values(), default=1) - 1

        yield RunStarted(
            run_id=run_id, total=len(images), config={"split": split}
        )
        detections: list[dict] = []
        try:
            for index, info in enumerate(images):
                if cancel is not None and cancel.is_set():
                    yield RunCompleted(run_id=run_id, result={"cancelled": True})
                    return
                image_path = gt_path.parent / info["file_name"]
                prediction = backend.infer_one(
                    image_path, threshold=_EVAL_THRESHOLD
                )
                for inst in prediction.instances:
                    if inst.category_name is not None:
                        category_id = id_by_name.get(inst.category_name, unmatched_id)
                    else:  # backend without names: ids are trusted as-is
                        category_id = inst.category_id
                    detections.append(
                        {
                            "image_id": info["id"],
                            "category_id": category_id,
                            "bbox": list(inst.bbox),
                            "score": inst.score,
                        }
                    )
                yield ProgressUpdated(
                    run_id=run_id,
                    current=index + 1,
                    total=len(images),
                    phase="inference",
                )
            # persisted before the metrics: a missing pycocotools must not
            # cost the user the inference pass they just waited for
            _write_detections(project, run_id, split, detections)
            report = _compute_metrics(gt_path, gt, detections, run_id, split)
        except Exception as exc:  # noqa: BLE001 — R4: the stream reports itself
            logger.exception("evaluation of run %s failed", run_id)
            yield RunFailed(
                run_id=run_id,
                error_code=getattr(exc, "code", "backend_error"),
                message=str(exc),
            )
            return
        eval_dir = _eval_dir(project, run_id)
        (eval_dir / f"{split}.json").write_text(
            report.model_dump_json(indent=2), "utf-8"
        )
        yield MetricsUpdated(
            run_id=run_id,
            step=0,
            metrics={
                "mAP@[.5:.95]": report.map_5095,
                "mAP@50": report.map_50,
                "mAP@75": report.map_75,
                "mAR@100": report.mar_100,
            },
        )
        yield RunCompleted(run_id=run_id, result=report.model_dump(mode="json"))

    return stream()


def evaluate_run(
    project: Project,
    run_id: str,
    *,
    split: str = "test",
    device: str | None = None,
) -> EvalReport:
    """Synchronous convenience: consume the event stream, return the report."""
    for event in evaluation_events(project, run_id, split=split, device=device):
        if event.type == "failed":
            raise ProjectError(f"Evaluation failed: {event.message}")
        if event.type == "completed":
            if event.result.get("cancelled"):
                raise ProjectError("Evaluation was cancelled")
            return EvalReport.model_validate(event.result)
    raise ProjectError("Evaluation ended without a result")


@capability(
    "evaluate.start",
    summary="Evaluate a run on its held-out split (COCO metrics, as a job)",
    web_route="/api/v1/train/runs/<run_id>/evaluate",
    web_methods=("POST",),
    cli="evaluate",
)
def start_evaluation(
    project: Project,
    run_id: str,
    *,
    split: str = "test",
    device: str | None = None,
) -> str:
    """Background evaluation via the shared job machinery; poll /jobs/<id>."""
    # validate before the job starts so the caller gets errors synchronously
    _load_run_backend(project, run_id, device)
    _split_gt(project, run_id, split)
    return jobs.start_job(
        project,
        "evaluate",
        lambda cancel: evaluation_events(
            project, run_id, split=split, device=device, cancel=cancel
        ),
    )


@capability(
    "evaluate.report",
    summary="Read the persisted evaluation report of a run",
    web_route="/api/v1/train/runs/<run_id>/eval/<split>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos evaluate' prints the report when it finishes.",
)
def get_eval_report(project: Project, run_id: str, split: str) -> EvalReport:
    path = _run_dir(project, run_id) / "eval" / f"{split}.json"
    if not path.is_file():
        raise ProjectError(
            f"Run {run_id} has no persisted evaluation for split '{split}' — "
            f"run an evaluation first."
        )
    return EvalReport.model_validate_json(path.read_text("utf-8"))


# ------------------------------------------------------- raw detections


def _eval_dir(project: Project, run_id: str) -> Path:
    eval_dir = _run_dir(project, run_id) / "eval"
    eval_dir.mkdir(exist_ok=True)
    return eval_dir


def _detections_path(project: Project, run_id: str, split: str) -> Path:
    return _run_dir(project, run_id) / "eval" / f"{split}.detections.json"


def _write_detections(
    project: Project, run_id: str, split: str, detections: list[dict]
) -> Path:
    """COCO-results-style list (image_id, category_id, bbox xywh, score),
    category ids already mapped onto the split's ground-truth ids."""
    path = _eval_dir(project, run_id) / f"{split}.detections.json"
    payload = {
        "run_id": run_id,
        "split": split,
        "threshold": _EVAL_THRESHOLD,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
    }
    path.write_text(json.dumps(payload), "utf-8")
    return path


def load_detections(project: Project, run_id: str, split: str) -> list[dict]:
    """The raw detections persisted by the last evaluation of this split."""
    path = _detections_path(project, run_id, split)
    if not path.is_file():
        raise ProjectError(
            f"Run {run_id} has no persisted detections for split '{split}' — "
            f"run an evaluation first (evaluations made before detections were "
            f"persisted need to be re-run once)."
        )
    payload = json.loads(path.read_text("utf-8"))
    return list(payload.get("detections", []))


# ------------------------------------------------------------- COCO metrics


def _compute_metrics(
    gt_path: Path,
    gt: dict,
    detections: list[dict],
    run_id: str,
    split: str,
) -> EvalReport:
    """pycocotools COCOeval over one split (E6-T3). The library prints its own
    progress to stdout; that would corrupt the CLI's JSONL stream, so all of
    its output is swallowed here."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ProjectError(
            "Evaluation needs pycocotools, which is not installed in this "
            "environment. Run 'horos doctor --fix' to complete the training "
            "stack."
        ) from exc
    import numpy as np

    names = {c["id"]: c["name"] for c in gt.get("categories", [])}
    counts: dict[int, int] = {}
    for ann in gt.get("annotations", []):
        counts[ann["category_id"]] = counts.get(ann["category_id"], 0) + 1

    base = dict(
        run_id=run_id,
        split=split,
        created_at=datetime.now(timezone.utc).isoformat(),
        num_images=len(gt.get("images", [])),
        num_instances=len(gt.get("annotations", [])),
    )
    if not detections:
        return EvalReport(
            **base,
            map_5095=0.0, map_50=0.0, map_75=0.0, mar_100=0.0,
            per_class=[
                ClassEval(category_id=cid, name=name,
                          instances=counts.get(cid, 0), ap=0.0, ap50=0.0)
                for cid, name in names.items()
            ],
        )

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # [mAP, mAP50, mAP75, s, m, l, AR1, AR10, AR100, ...]
    # precision tensor: [iou_thresholds, recall(101), classes, areas, max_dets]
    precision = coco_eval.eval["precision"]

    def _mean_valid(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else 0.0

    per_class = []
    for k, cat_id in enumerate(coco_eval.params.catIds):
        curve = precision[0, :, k, 0, -1]  # IoU=.5, area=all, top maxDets
        per_class.append(
            ClassEval(
                category_id=int(cat_id),
                name=names.get(int(cat_id), str(cat_id)),
                instances=counts.get(int(cat_id), 0),
                ap=_mean_valid(precision[:, :, k, 0, -1]),
                ap50=_mean_valid(curve),
                pr_curve_50=[float(max(v, 0.0)) for v in curve],
            )
        )
    return EvalReport(
        **base,
        map_5095=max(float(stats[0]), 0.0),
        map_50=max(float(stats[1]), 0.0),
        map_75=max(float(stats[2]), 0.0),
        mar_100=max(float(stats[8]), 0.0),
        per_class=per_class,
    )
