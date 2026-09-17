"""Error analysis over a run's persisted evaluation detections (E6-T4, E6-T5).

mAP says how good a model is; this module says *where it is wrong*. The
evaluation pass (`horos/api/evaluate.py`) persists its raw low-threshold
detections, and everything here re-matches them against the ground truth at a
user-chosen operating threshold — pure Python, milliseconds, no inference — so
the WebUI can move a slider and watch the confusion matrix change (E6-S3).

Matching (confirmed design): class-agnostic greedy IoU matching. Predictions
are visited from the most to the least confident; each takes the still
unmatched ground-truth box with the highest IoU above the IoU threshold,
preferring a same-class box when one qualifies. Then the classes are compared:

- same class            -> true positive
- different class       -> confusion (the pair is what E6-S5 asks for)
- ground truth unmatched -> miss (false negative)
- prediction unmatched  -> false positive

pycocotools matches per class, so it cannot see cross-class confusion; that is
why this is not built on `COCOeval.evalImgs`.

Worst-case mining ranks images by their error count (misses + false positives
+ confusions), ties broken by the relative area of the missed ground truth —
a big missed object is worse than a small one (confirmed design).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from horos.api.evaluate import eval_ground_truth, load_detections
from horos.api.manifest import capability
from horos.core.project import Project
from horos.errors import ProjectError

__all__ = [
    "BACKGROUND",
    "ClassErrors",
    "ConfusionPair",
    "ErrorAnalysis",
    "ImageErrorItem",
    "ImageErrors",
    "WorstCases",
    "analyze_detections",
    "analyze_errors",
    "image_errors",
    "worst_cases",
]

#: name of the pseudo-class used for misses (row) and false positives (column)
BACKGROUND = "background"

ErrorKind = Literal["tp", "fp", "fn", "confused"]


class ClassErrors(BaseModel):
    category_id: int
    name: str
    instances: int  # ground-truth boxes of this class
    tp: int
    fn: int  # ground truth of this class left unmatched (missed)
    fp: int  # predictions of this class matching no ground truth at all
    confused_as: int  # ground truth of this class matched by another class
    confused_from: int  # predictions of this class that landed on another class
    recall: float
    precision: float


class ConfusionPair(BaseModel):
    gt_name: str
    pred_name: str
    count: int


class ErrorAnalysis(BaseModel):
    run_id: str
    split: str
    threshold: float
    iou: float
    num_images: int
    #: class names in matrix order (background is appended last)
    classes: list[str]
    #: rows = ground truth, columns = prediction; the last row is background
    #: (false positives), the last column is background (misses)
    matrix: list[list[int]]
    per_class: list[ClassErrors]
    #: cross-class confusions, most frequent first
    confused_pairs: list[ConfusionPair]
    tp: int
    fp: int
    fn: int
    confused: int


class ImageErrorItem(BaseModel):
    kind: ErrorKind
    #: the prediction box for tp/fp/confused, the ground-truth box for fn
    bbox: tuple[float, float, float, float]
    gt_name: str | None = None
    pred_name: str | None = None
    score: float | None = None
    iou: float | None = None
    #: the ground-truth box the prediction matched (tp/confused)
    gt_bbox: tuple[float, float, float, float] | None = None
    #: polygon rings of the drawn object — the prediction's mask for
    #: tp/fp/confused, the ground truth's for fn — so the overlay of a
    #: segmentation run shows masks, not just their boxes (E6-T6)
    segmentation: list[list[float]] | None = None


class ImageErrors(BaseModel):
    image_id: int
    file_name: str
    width: int
    height: int
    errors: int  # fn + fp + confused
    tp: int
    fn: int
    fp: int
    confused: int
    #: summed area of the missed ground truth, relative to the image area
    missed_area: float
    items: list[ImageErrorItem] = Field(default_factory=list)


class WorstCases(BaseModel):
    run_id: str
    split: str
    threshold: float
    iou: float
    top_k: int
    total_images: int
    images_with_errors: int
    images: list[ImageErrors]


# ------------------------------------------------------------------ geometry


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two COCO xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------ matching


def match_image(
    gt_boxes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float,
    names: dict[int, str],
) -> list[ImageErrorItem]:
    """Class-agnostic greedy matching of one image (see module docstring).

    `gt_boxes` and `predictions` are COCO-style dicts (`bbox`, `category_id`,
    predictions also `score`). Returns one item per prediction and one per
    missed ground-truth box."""
    order = sorted(range(len(predictions)), key=lambda i: -predictions[i]["score"])
    taken = [False] * len(gt_boxes)

    def rings(record: dict[str, Any]) -> list[list[float]] | None:
        # COCO polygons only; RLE masks (dict) have no outline to draw
        seg = record.get("segmentation")
        if not isinstance(seg, list):
            return None
        out = [
            [float(v) for v in ring] for ring in seg if isinstance(ring, list) and len(ring) >= 6
        ]
        return out or None

    items: list[ImageErrorItem] = []
    for pi in order:
        pred = predictions[pi]
        pred_cat = pred["category_id"]
        best_same: tuple[float, int] | None = None
        best_any: tuple[float, int] | None = None
        for gi, gt in enumerate(gt_boxes):
            if taken[gi]:
                continue
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou < iou_threshold:
                continue
            if gt["category_id"] == pred_cat and (best_same is None or iou > best_same[0]):
                best_same = (iou, gi)
            if best_any is None or iou > best_any[0]:
                best_any = (iou, gi)
        match = best_same or best_any
        if match is None:
            items.append(
                ImageErrorItem(
                    kind="fp",
                    bbox=tuple(pred["bbox"]),
                    pred_name=names[pred_cat],
                    score=pred["score"],
                    segmentation=rings(pred),
                )
            )
            continue
        iou, gi = match
        taken[gi] = True
        gt = gt_boxes[gi]
        items.append(
            ImageErrorItem(
                kind="tp" if gt["category_id"] == pred_cat else "confused",
                bbox=tuple(pred["bbox"]),
                gt_name=names[gt["category_id"]],
                pred_name=names[pred_cat],
                score=pred["score"],
                iou=iou,
                gt_bbox=tuple(gt["bbox"]),
                segmentation=rings(pred),
            )
        )
    for gi, gt in enumerate(gt_boxes):
        if not taken[gi]:
            items.append(
                ImageErrorItem(
                    kind="fn", bbox=tuple(gt["bbox"]), gt_name=names[gt["category_id"]],
                    segmentation=rings(gt),
                )
            )
    return items


def _class_names(gt: dict, detections: Iterable[dict]) -> dict[int, str]:
    """Ground-truth categories first (in id order); a prediction whose id is
    unknown to the split still gets a stable, visibly foreign name instead of
    crashing the analysis."""
    categories = sorted(gt.get("categories", []), key=lambda c: c["id"])
    names = {int(c["id"]): c["name"] for c in categories}
    for det in detections:
        cid = int(det["category_id"])
        if cid not in names:
            names[cid] = f"unknown:{cid}"
    return names


def analyze_detections(
    gt: dict,
    detections: list[dict],
    *,
    threshold: float,
    iou: float,
    run_id: str = "",
    split: str = "",
) -> tuple[ErrorAnalysis, list[ImageErrors]]:
    """One matching pass over a split: the aggregate analysis and the per-image
    breakdown it was summed from. Pure function — the project-bound entry
    points below only load the inputs."""
    _validate_fractions(threshold, iou)
    kept = [d for d in detections if d["score"] >= threshold]
    names = _class_names(gt, kept)
    class_ids = list(names)
    index = {cid: k for k, cid in enumerate(class_ids)}
    name_index = {names[cid]: k for cid, k in index.items()}
    size = len(class_ids) + 1  # + background
    matrix = [[0] * size for _ in range(size)]
    bg = size - 1

    gt_by_image: dict[int, list[dict]] = {}
    for ann in gt.get("annotations", []):
        gt_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    det_by_image: dict[int, list[dict]] = {}
    for det in kept:
        det_by_image.setdefault(int(det["image_id"]), []).append(det)

    per_image: list[ImageErrors] = []
    pair_counts: dict[tuple[str, str], int] = {}
    for info in gt.get("images", []):
        image_id = int(info["id"])
        items = match_image(
            gt_by_image.get(image_id, []),
            det_by_image.get(image_id, []),
            iou_threshold=iou,
            names=names,
        )
        counts = {"tp": 0, "fp": 0, "fn": 0, "confused": 0}
        missed_area = 0.0
        image_area = float(info.get("width", 0) * info.get("height", 0)) or 1.0
        for item in items:
            counts[item.kind] += 1
            if item.kind == "fn":
                matrix[name_index[item.gt_name]][bg] += 1
                missed_area += item.bbox[2] * item.bbox[3] / image_area
            elif item.kind == "fp":
                matrix[bg][name_index[item.pred_name]] += 1
            else:
                matrix[name_index[item.gt_name]][name_index[item.pred_name]] += 1
                if item.kind == "confused":
                    key = (item.gt_name, item.pred_name)
                    pair_counts[key] = pair_counts.get(key, 0) + 1
        per_image.append(
            ImageErrors(
                image_id=image_id,
                file_name=info["file_name"],
                width=int(info.get("width", 0)),
                height=int(info.get("height", 0)),
                errors=counts["fn"] + counts["fp"] + counts["confused"],
                missed_area=round(missed_area, 6),
                items=items,
                **counts,
            )
        )

    per_class: list[ClassErrors] = []
    for cid in class_ids:
        k = index[cid]
        tp = matrix[k][k]
        fn = matrix[k][bg]
        fp = matrix[bg][k]
        confused_as = sum(matrix[k][j] for j in range(size - 1) if j != k)
        confused_from = sum(matrix[i][k] for i in range(size - 1) if i != k)
        instances = tp + fn + confused_as
        predicted = tp + fp + confused_from
        per_class.append(
            ClassErrors(
                category_id=cid,
                name=names[cid],
                instances=instances,
                tp=tp,
                fn=fn,
                fp=fp,
                confused_as=confused_as,
                confused_from=confused_from,
                recall=tp / instances if instances else 0.0,
                precision=tp / predicted if predicted else 0.0,
            )
        )
    analysis = ErrorAnalysis(
        run_id=run_id,
        split=split,
        threshold=threshold,
        iou=iou,
        num_images=len(per_image),
        classes=[names[cid] for cid in class_ids] + [BACKGROUND],
        matrix=matrix,
        per_class=per_class,
        confused_pairs=sorted(
            (
                ConfusionPair(gt_name=g, pred_name=p, count=n)
                for (g, p), n in pair_counts.items()
            ),
            key=lambda pair: (-pair.count, pair.gt_name, pair.pred_name),
        ),
        tp=sum(c.tp for c in per_class),
        fp=sum(c.fp for c in per_class),
        fn=sum(c.fn for c in per_class),
        confused=sum(c.confused_as for c in per_class),
    )
    return analysis, per_image


def rank_worst(images: Iterable[ImageErrors]) -> list[ImageErrors]:
    """Most errors first; ties broken by the missed relative area, then by
    image id so the order is stable."""
    return sorted(
        (img for img in images if img.errors > 0),
        key=lambda img: (-img.errors, -img.missed_area, img.image_id),
    )


def _validate_fractions(threshold: float, iou: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ProjectError(f"threshold must be within [0, 1], got {threshold}")
    if not 0.0 < iou <= 1.0:
        raise ProjectError(f"iou must be within (0, 1], got {iou}")


# ------------------------------------------------------- project entry points


def _inputs(project: Project, run_id: str, split: str) -> tuple[dict, list[dict]]:
    # the ground truth the last evaluation scored, not whatever the project
    # holds now: these detections were matched against those boxes
    gt, _, _ = eval_ground_truth(project, run_id, split)
    return gt, load_detections(project, run_id, split)


@capability(
    "evaluate.errors",
    summary="Confusion matrix and per-class miss / false-positive analysis of an "
    "evaluated run at an operating threshold",
    web_route="/api/v1/train/runs/<run_id>/eval/<split>/errors",
    web_methods=("GET",),
    cli="analyze",
)
def analyze_errors(
    project: Project,
    run_id: str,
    split: str = "test",
    *,
    threshold: float = 0.5,
    iou: float = 0.5,
) -> ErrorAnalysis:
    gt, detections = _inputs(project, run_id, split)
    analysis, _ = analyze_detections(
        gt, detections, threshold=threshold, iou=iou, run_id=run_id, split=split
    )
    return analysis


@capability(
    "evaluate.worst_cases",
    summary="The images an evaluated run gets most wrong, with every miss, "
    "false positive and confusion listed",
    web_route="/api/v1/train/runs/<run_id>/eval/<split>/worst",
    web_methods=("GET",),
    cli="analyze",
)
def worst_cases(
    project: Project,
    run_id: str,
    split: str = "test",
    *,
    threshold: float = 0.5,
    iou: float = 0.5,
    top_k: int = 50,
) -> WorstCases:
    if top_k < 1:
        raise ProjectError(f"top_k must be at least 1, got {top_k}")
    gt, detections = _inputs(project, run_id, split)
    _, per_image = analyze_detections(
        gt, detections, threshold=threshold, iou=iou, run_id=run_id, split=split
    )
    ranked = rank_worst(per_image)
    return WorstCases(
        run_id=run_id,
        split=split,
        threshold=threshold,
        iou=iou,
        top_k=top_k,
        total_images=len(per_image),
        images_with_errors=len(ranked),
        images=ranked[:top_k],
    )


def image_errors(
    project: Project,
    run_id: str,
    split: str,
    image_id: int,
    *,
    threshold: float = 0.5,
    iou: float = 0.5,
) -> ImageErrors:
    """The matching breakdown of one image of the split (feeds the overlay)."""
    gt, detections = _inputs(project, run_id, split)
    info = next((i for i in gt.get("images", []) if int(i["id"]) == image_id), None)
    if info is None:
        raise ProjectError(
            f"Image {image_id} is not part of the '{split}' split of run {run_id}."
        )
    one = {
        "images": [info],
        "categories": gt.get("categories", []),
        "annotations": [a for a in gt.get("annotations", []) if int(a["image_id"]) == image_id],
    }
    dets = [d for d in detections if int(d["image_id"]) == image_id]
    _, per_image = analyze_detections(one, dets, threshold=threshold, iou=iou)
    return per_image[0]
