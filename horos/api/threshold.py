"""Suggested operating confidence threshold of an evaluated run (E6-T10).

mAP integrates over every threshold, so it never says which one to deploy at.
This module does: it re-matches a run's persisted low-threshold detections
(`horos/api/evaluate.py`) against the ground truth over a grid of thresholds
and reports where the F-score peaks, together with the whole sweep so the UI
can draw the curve (E6-S3).

Exactly one matching pass
-------------------------
`error_analysis.match_image` visits predictions from the most to the least
confident, so raising the threshold only ever removes a *suffix* of the visit
order: every kept prediction is matched exactly as it was, and the prefix's
decisions cannot change. One pass over all detections therefore yields every
threshold's counts, and the sweep is a walk down the sorted scores instead of
one matching pass per grid point. The test asserts the two agree.

What is recommended
-------------------
F-beta peaks over a plateau far more often than at a sharp point, so the peak
alone is a fragile recommendation. horos takes the contiguous run of
thresholds whose F-beta is within `PLATEAU_TOLERANCE` of the peak and
recommends the grid point nearest that run's middle — the value least
sensitive to a small shift in the data. Per class the same rule runs on that
class's own F-beta, and classes with fewer than `MIN_INSTANCES` ground-truth
boxes are marked as too thin to advise on.

`beta` is the recall/precision trade the caller wants: 1 balances them, 2
weights recall (fewer misses), 0.5 weights precision (fewer false alarms).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from horos.api.error_analysis import _class_names, _validate_fractions, match_image
from horos.api.evaluate import eval_ground_truth, load_detections
from horos.api.manifest import capability
from horos.core.project import Project
from horos.errors import ProjectError

__all__ = [
    "ClassThreshold",
    "ThresholdAdvice",
    "ThresholdPoint",
    "suggest_threshold",
    "sweep_detections",
]

#: the sweep grid — the page's slider steps in the same 0.01
GRID_LO = 0.05
GRID_HI = 0.95
GRID_STEP = 0.01
#: F-beta within this much of the peak (relative) counts as the same plateau
PLATEAU_TOLERANCE = 0.01
#: fewer ground-truth boxes than this and a per-class threshold is noise
MIN_INSTANCES = 10
#: the threshold the rest of horos defaults to, reported for comparison
BASELINE = 0.5


class ThresholdPoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    f_score: float  # F-beta with the requested beta
    tp: int
    fp: int  # predictions matching no ground-truth box at all
    fn: int  # ground-truth boxes left undetected
    confused: int  # ground truth found but labeled as another class


class ClassThreshold(BaseModel):
    category_id: int
    name: str
    instances: int
    recommended: float
    precision: float
    recall: float
    f_score: float
    #: enough ground truth for the number to mean anything
    enough_data: bool


class ThresholdAdvice(BaseModel):
    run_id: str
    split: str
    iou: float
    beta: float
    #: the threshold horos suggests operating at
    recommended: float
    #: metrics at `recommended`
    best: ThresholdPoint
    #: metrics at the 0.5 default, so a caller can show what changes
    baseline: ThresholdPoint
    #: the contiguous range of thresholds that score as well as the peak
    plateau: tuple[float, float]
    #: the whole sweep, lowest threshold first — the curve to draw
    points: list[ThresholdPoint] = Field(default_factory=list)
    per_class: list[ClassThreshold] = Field(default_factory=list)
    #: False when the split has no ground truth or the run detects nothing:
    #: `recommended` is then the plain 0.5 default, not a measurement
    confident: bool
    reason: str
    notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ the sweep


def _f_score(tp: int, fp: int, fn: int, beta: float) -> tuple[float, float, float]:
    """(precision, recall, F-beta). `fp` counts every wrong prediction,
    `fn` every missed ground-truth box."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    f = (1 + b2) * precision * recall / denom if denom else 0.0
    return precision, recall, f


def _grid() -> list[float]:
    steps = int(round((GRID_HI - GRID_LO) / GRID_STEP))
    return [round(GRID_LO + i * GRID_STEP, 4) for i in range(steps + 1)]


def _plateau(values: list[float]) -> tuple[int, int, int]:
    """(lo, hi, pick) over a uniform grid: the contiguous run of indices around
    the highest value that stays within `PLATEAU_TOLERANCE` of it, and the
    index to recommend — its middle. F-beta against threshold is normally
    unimodal, and taking only the run containing the peak keeps a second,
    unrelated bump from widening the answer. The middle is taken on the index,
    not on the value: the grid is uniform so it is the same point, without a
    floating-point tie deciding the answer, and an even-length plateau rounds
    down, towards recall."""
    peak_index = max(range(len(values)), key=lambda i: values[i])
    floor = values[peak_index] * (1.0 - PLATEAU_TOLERANCE)
    lo = hi = peak_index
    while lo > 0 and values[lo - 1] >= floor:
        lo -= 1
    while hi < len(values) - 1 and values[hi + 1] >= floor:
        hi += 1
    return lo, hi, (lo + hi) // 2


def _collect(
    gt: dict, detections: list[dict], *, iou: float
) -> tuple[list[tuple[float, str, str, str]], Counter, dict[int, str]]:
    """One matching pass: every prediction as (score, kind, gt name, pred
    name), plus the ground-truth count per class name."""
    names = _class_names(gt, detections)
    images = {int(info["id"]): info for info in gt.get("images", [])}
    gt_by_image: dict[int, list[dict]] = {}
    instances: Counter = Counter()
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in images:  # analyze_detections ignores these too
            continue
        gt_by_image.setdefault(image_id, []).append(ann)
        instances[names[int(ann["category_id"])]] += 1
    det_by_image: dict[int, list[dict]] = {}
    for det in detections:
        det_by_image.setdefault(int(det["image_id"]), []).append(det)

    predictions: list[tuple[float, str, str, str]] = []
    for image_id in images:
        for item in match_image(
            gt_by_image.get(image_id, []),
            det_by_image.get(image_id, []),
            iou_threshold=iou,
            names=names,
        ):
            if item.kind == "fn":  # a miss is the absence of a prediction
                continue
            predictions.append(
                (float(item.score or 0.0), item.kind, item.gt_name or "", item.pred_name or "")
            )
    return predictions, instances, names


def sweep_detections(
    gt: dict,
    detections: list[dict],
    *,
    iou: float = 0.5,
    beta: float = 1.0,
    run_id: str = "",
    split: str = "",
) -> ThresholdAdvice:
    """Threshold advice from COCO-style ground truth and detections. Pure
    function — the project-bound entry point below only loads the inputs."""
    # the grid is ours; iou is the only fraction the caller chooses
    _validate_fractions(BASELINE, iou)
    if beta <= 0:
        raise ProjectError(f"beta must be greater than 0, got {beta}")

    predictions, instances, names = _collect(gt, detections, iou=iou)
    id_of = {name: cid for cid, name in names.items()}
    total_gt = sum(instances.values())
    grid = _grid()

    # walk the grid from the top down, adding predictions as they come into
    # range: the counts at each threshold without re-matching anything
    order = sorted(predictions, key=lambda p: -p[0])
    tp: Counter = Counter()
    fp: Counter = Counter()
    confused_as: Counter = Counter()
    confused_from: Counter = Counter()
    cursor = 0
    points: list[ThresholdPoint] = []
    # per class the same three numbers at every grid point, so each class can
    # be given the plateau treatment on its own curve
    curves: dict[str, list[tuple[float, float, float]]] = {name: [] for name in instances}
    for threshold in reversed(grid):
        while cursor < len(order) and order[cursor][0] >= threshold:
            score, kind, gt_name, pred_name = order[cursor]
            if kind == "tp":
                tp[pred_name] += 1
            elif kind == "fp":
                fp[pred_name] += 1
            else:  # found, but called something else
                confused_as[gt_name] += 1
                confused_from[pred_name] += 1
            cursor += 1
        hits = sum(tp.values())
        wrong = sum(fp.values()) + sum(confused_from.values())
        missed = total_gt - hits - sum(confused_as.values())
        precision, recall, f = _f_score(hits, wrong, missed + sum(confused_as.values()), beta)
        points.append(
            ThresholdPoint(
                threshold=threshold, precision=precision, recall=recall, f_score=f,
                tp=hits, fp=sum(fp.values()), fn=missed, confused=sum(confused_as.values()),
            )
        )
        for name, count in instances.items():
            c_tp = tp[name]
            c_wrong = fp[name] + confused_from[name]
            c_missed = count - c_tp - confused_as[name]
            c_p, c_r, c_f = _f_score(c_tp, c_wrong, c_missed + confused_as[name], beta)
            curves[name].append((c_f, c_p, c_r))
    points.reverse()
    for curve in curves.values():
        curve.reverse()

    scores = [p.f_score for p in points]
    peak = max(scores)
    baseline = min(points, key=lambda p: abs(p.threshold - BASELINE))
    notes: list[str] = []
    if split == "test":
        notes.append(
            "This is the test split. Picking a threshold on it means the split is no "
            "longer untouched — valid is the safer one to tune on."
        )

    if not total_gt or not predictions or peak <= 0.0:
        why = (
            f"the {split or 'chosen'} split has no ground truth" if not total_gt
            else "this run detects nothing above the evaluation floor" if not predictions
            else "no threshold detects anything correctly"
        )
        return ThresholdAdvice(
            run_id=run_id, split=split, iou=iou, beta=beta,
            recommended=BASELINE, best=baseline, baseline=baseline,
            plateau=(BASELINE, BASELINE), points=points, per_class=[],
            confident=False,
            reason=f"No threshold to suggest — {why}. Leaving the 0.50 default in place.",
            notes=notes,
        )

    lo, hi, pick = _plateau(scores)
    best = points[pick]

    per_class = []
    for name, count in sorted(instances.items(), key=lambda kv: (-kv[1], kv[0])):
        curve = curves[name]
        _, _, c_pick = _plateau([f for f, _, _ in curve])
        c_f, c_p, c_r = curve[c_pick]
        per_class.append(
            ClassThreshold(
                category_id=id_of.get(name, -1), name=name, instances=count,
                recommended=grid[c_pick], precision=c_p, recall=c_r, f_score=c_f,
                enough_data=count >= MIN_INSTANCES,
            )
        )
    thin = [c.name for c in per_class if not c.enough_data]
    if thin:
        notes.append(
            f"Too little ground truth to advise per class on {', '.join(thin)} "
            f"(under {MIN_INSTANCES} boxes each)."
        )

    label = "F1" if beta == 1.0 else f"F{beta:g}"
    flat = "" if lo == hi else (
        f", and anything from {points[lo].threshold:.2f} to {points[hi].threshold:.2f} "
        f"scores as well"
    )
    reason = (
        f"{label} peaks at {peak:.3f}{flat}. At {best.threshold:.2f} the run keeps "
        f"{best.precision:.0%} precision and {best.recall:.0%} recall"
    )
    if abs(best.threshold - baseline.threshold) < GRID_STEP / 2:
        reason += f", which is the {BASELINE:.2f} default."
    elif points[lo].threshold <= baseline.threshold <= points[hi].threshold:
        # the default is on the plateau: say there is nothing to do
        reason += f". The {BASELINE:.2f} default scores just as well, so nothing needs changing."
    else:
        reason += (
            f"; the {BASELINE:.2f} default scores {baseline.f_score:.3f} "
            f"({baseline.precision:.0%} / {baseline.recall:.0%})."
        )
    return ThresholdAdvice(
        run_id=run_id, split=split, iou=iou, beta=beta,
        recommended=best.threshold, best=best, baseline=baseline,
        plateau=(points[lo].threshold, points[hi].threshold),
        points=points, per_class=per_class, confident=True,
        reason=reason, notes=notes,
    )


# ------------------------------------------------------- project entry point


@capability(
    "evaluate.threshold",
    summary="Suggested operating confidence threshold of an evaluated run, with the "
    "precision / recall / F-score sweep it was chosen from",
    web_route="/api/v1/train/runs/<run_id>/eval/<split>/threshold",
    web_methods=("GET",),
    cli="analyze",
)
def suggest_threshold(
    project: Project,
    run_id: str,
    split: str = "test",
    *,
    iou: float = 0.5,
    beta: float = 1.0,
) -> ThresholdAdvice:
    """Where to set the confidence threshold for this run, from the detections
    its last evaluation of `split` persisted. `beta` above 1 weights recall,
    below 1 weights precision."""
    gt, _, _ = eval_ground_truth(project, run_id, split)
    detections: list[dict[str, Any]] = load_detections(project, run_id, split)
    return sweep_detections(
        gt, detections, iou=iou, beta=beta, run_id=run_id, split=split
    )
