"""The active-learning loop — round selection (E10-T6).

One round: select → label → train → review. This module owns the *select*
step and the round bookkeeping; labeling reuses the annotator, training
and review land in E10-T8/T9.

How a batch is chosen (strategy "auto", the only thing the UI ever sends):

- no labeled image yet → **diversity**: k-center greedy over DINOv2
  embeddings (E10-T4). The paper behind the model-based metric seeds with
  a random 2 %; the confirmed E10 design replaces that with similarity.
- labeled images exist → **pal** (E10-T5): the newest completed training
  run scores every unlabeled image; without a run, OWLv2 zero-shot with the
  project's class names as prompts stands in as the scorer (decision D4).
  PAL cannot reach images the scorer sees nothing in; the remainder of the
  budget is filled by diversity so the round is always the size asked for.
- the embedding model cannot load → **random**, and the round says so.

Pool = images with no confirmed annotation whose split is "train". Valid
and test images are never selected: the validation split is locked at the
first training (E10-T8) and its metrics must stay comparable across
rounds (E7-T2).
"""

from __future__ import annotations

import json
import logging
import random
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event as CancelEvent
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from horos.api.annotate import _load_claims
from horos.api.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_events, load_embeddings
from horos.api.jobs import RunningJob, running_job, start_job
from horos.api.manifest import capability
from horos.api.train import TrainRunConfig, TrainStatus, list_runs, start_training, training_status
from horos.core import pal
from horos.core.dataset import ImageRecord
from horos.core.fsutil import atomic_write_text
from horos.core.project import Project
from horos.core.registry import get_model_info
from horos.core.rounds import (
    LoopRound,
    PickedImage,
    SelectionRecord,
    SelectionStrategy,
    create_round,
    current_round,
    list_rounds,
    load_round,
    save_round,
)
from horos.core.selection import kcenter_greedy, random_picks, resolve_count
from horos.errors import BackendError, HorosError, ProjectError

if TYPE_CHECKING:
    from horos.backends.base import (
        Event,
        ImageEmbedder,
        ImagePrediction,
        ModelBackend,
        PromptableSegmenter,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "LoopStatus",
    "RoundSummary",
    "loop_status",
    "get_round",
    "select_round_events",
    "select_round",
    "start_round_job",
    "close_round",
    "preannotate_events",
    "preannotate_round",
    "start_preannotate_job",
    "TrainReadiness",
    "train_readiness",
    "train_round",
    "round_training_status",
    "loop_history",
    "assign_round",
    "RoundQueueItem",
    "round_queue",
    "SimilarImage",
    "similar_images",
    "ImageDetection",
    "ImagePredictions",
    "image_predictions",
    "SkipResult",
    "skip_images",
    "restore_images",
    "refill_round",
    "LoopSettings",
    "get_loop_settings",
    "update_loop_settings",
    "LoopAdvice",
    "loop_advice",
    "evaluate_round_splits",
]

Strategy = Literal["auto", "pal", "diversity", "random"]
#: a detection counts as "final" for PAL scoring at or above this confidence;
#: the backends report raw candidates below it (down to their own floor)
SCORE_THRESHOLD = 0.3
#: pseudo-labels of one class overlapping at least this much are one object:
#: RF-DETR has no NMS of its own, so a young model answers one object with
#: several queries and the annotator saw the same class stacked on it
PRELABEL_NMS_IOU = 0.5
#: background job kinds the loop page owns — LoopStatus.job reports the one in flight
LOOP_JOB_KINDS = ("loop-select", "loop-preannotate", "embeddings")
#: labeled images used to fit PAL's logistic classifiers per round — a cap
#: so a large labeled set does not cost a full inference pass every round
MAX_FIT_IMAGES = 300
#: photos per batched scorer call while scoring (progress is reported per chunk)
SCORE_CHUNK = 16
#: the scorer looks at this many times the round size (a seeded random
#: sample of the pool) — 20 photos × 100 = 2000 candidates; scoring 10 000
#: photos one by one took ~14 min, and PAL ranks a few thousand candidates
#: as well as it ranks them all
DEFAULT_SCAN_FACTOR = 100
ZERO_SHOT_SCORER = "owlv2-base"
#: pre-label confidence floor per scorer kind: a fine-tuned run is calibrated
#: on the project's classes, zero-shot OWLv2 scores run low (autolabel's
#: default is 0.1)
PRELABEL_THRESHOLD = {"run": SCORE_THRESHOLD, "zero_shot": 0.1}
#: training readiness (E10-T8): enough labeled images, and every class that
#: appears at all has enough instances for a validation split to mean anything
MIN_LABELED_IMAGES = 20
MIN_INSTANCES_PER_CLASS = 5
#: a class with fewer labels than this share of the mean class count (and
#: fewer than 2 × MIN_INSTANCES_PER_CLASS) is "rare": the model has seen too
#: little of it to propose it, so its next photos are found by look-alikes
RARE_CLASS_FACTOR = 0.25
#: at most this share of a round goes to rare-class look-alikes
LOOKALIKE_SHARE = 0.25
#: held-out shares of the labeled photos. Every newly labeled photo is put
#: into test / valid / train by a deterministic hash bucket, so both held-out
#: sets grow in proportion to the labels and a photo never changes split:
#: test is never trained on (the learning curve's honest line), valid is what
#: the trainer selects its checkpoint on


Shapes = Literal["auto", "box", "polygon"]
SETTINGS_FILE = "loop.json"


class LoopSettings(BaseModel):
    """What the user can decide once for the whole loop (E10-T19); stored in
    <root>/loop.json and applied to every round until changed."""

    #: training model key; None = the loop picks detection or segmentation
    #: from the labels (default_model_for)
    model: str | None = None
    #: write the scorer's suggestions on picked photos (pseudo-labels)
    preannotate: bool = True
    #: suggestion geometry: "auto" keeps whatever the scorer produced, "box"
    #: drops polygons to boxes, "polygon" turns boxes into SAM polygons
    shapes: Shapes = "auto"
    #: the segmenter used for shapes="polygon"
    refiner: str = "sam2.1-tiny"
    #: how many times the round size the scorer looks at (random sample when
    #: the pool is bigger); None = score the whole pool, however long it takes
    scan_factor: int | None = Field(default=DEFAULT_SCAN_FACTOR, ge=1)
    #: tilt each round towards under-labeled classes: PAL budgets weighted by
    #: inverse label frequency, and look-alikes of a rare class's few photos
    #: (by embedding) when the model cannot find it yet
    balance: bool = True
    #: "continue": each round starts from the previous run of the same model
    #: (weights kept, optimizer fresh, class head resized — new classes are
    #: fine) and trains fewer epochs; "fresh": from the published weights
    training: Literal["continue", "fresh"] = "continue"
    #: stop-when-reached value of the headline metric (mAP flavours are in
    #: [0, 1]); None = keep going until the pool is empty or gains flatten
    target: float | None = Field(default=None, gt=0.0)


def _settings_path(project: Project):
    return project.root / SETTINGS_FILE


@capability(
    "loop.settings",
    summary="The loop's standing choices: model, suggestions on/off, box or polygon shapes",
    web_route="/api/v1/loop/settings",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' prints them with the status.",
)
def get_loop_settings(project: Project) -> LoopSettings:
    path = _settings_path(project)
    if not path.is_file():
        return LoopSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("target") is not None and data["target"] <= 0:
            data["target"] = None  # written by earlier versions: a goal of 0 is no goal
        if isinstance(data, dict):
            data.pop("score_limit", None)  # an absolute cap, replaced by scan_factor
        return LoopSettings.model_validate(data)
    except (ValueError, TypeError) as exc:
        raise ProjectError(f"Corrupt loop settings at {path}: {exc}") from exc


@capability(
    "loop.settings.update",
    summary="Change the loop's standing choices (partial update)",
    web_route="/api/v1/loop/settings",
    web_methods=("PUT",),
    cli=None,
    not_cli_because="'horos loop select/train' take the same choices as flags.",
)
def update_loop_settings(project: Project, **changes) -> LoopSettings:
    current = get_loop_settings(project)
    unknown = set(changes) - set(LoopSettings.model_fields)
    if unknown:
        raise ProjectError(f"Unknown loop setting(s): {sorted(unknown)}")
    if "target" in changes and changes["target"] is not None and float(changes["target"]) <= 0:
        changes["target"] = None  # a goal of 0 would be met by any model: it means "no goal"
    if "scan_factor" in changes and changes["scan_factor"] is not None \
            and int(changes["scan_factor"]) <= 0:
        changes["scan_factor"] = None  # 0 = no cap: score every unlabeled photo
    try:
        updated = current.model_copy(update=changes)
        updated = LoopSettings.model_validate(updated.model_dump())
    except ValueError as exc:
        raise ProjectError(f"Invalid loop settings: {exc}") from exc
    if updated.model is not None:
        info = get_model_info(updated.model)  # unknown keys raise UnknownModelError
        if not info.trainable:
            raise ProjectError(f"Model '{updated.model}' cannot be trained; pick a trainable one")
    get_model_info(updated.refiner)
    atomic_write_text(_settings_path(project), updated.model_dump_json(indent=2))
    return updated


#: which of a run's best-checkpoint scores headlines a round, first match wins:
#: the validation mAP flavours RF-DETR reports (EMA first — the weights that
#: are actually saved), then generic names, then loss. For "loss" lower is
#: better, for every mAP flavour higher is
_METRIC_PREFERENCE = (
    # the loop's own post-training evaluation first: same code, same
    # threshold, same fixed validation set every round — comparable by design
    "eval/test/map_5095", "eval/test/map_50", "eval/valid/map_5095", "eval/valid/map_50",
    "val/ema_mAP_50_95", "val/mAP_50_95", "map_5095", "val/ema_mAP_50", "val/mAP_50",
    "map50", "map_50", "map", "loss",
)


class RoundSummary(BaseModel):
    """One row of the round history (E10-T9)."""

    number: int
    state: str
    strategy: str | None = None
    requested: int = 0
    picked: int = 0
    #: picks that now hold confirmed annotations
    labeled: int = 0
    #: picks the annotators skipped as unfit (E10-T16)
    skipped: int = 0
    labeled_before: int = 0
    #: labeled images gained during the round (so far, while it is open)
    labels_spent: int = 0
    train_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    #: the headline metric of this round's model and its change against the
    #: previous round that has one; `improved` accounts for loss vs mAP
    metric_key: str | None = None
    metric: float | None = None
    delta: float | None = None
    improved: bool | None = None
    #: mAP@50 of this round's model on each split of its snapshot, from the
    #: post-training evaluation (E10-S5 learning curve); missing = not run
    curve: dict[str, float] = Field(default_factory=dict)
    #: labeled photos when the round ended
    labeled_total: int = 0
    #: photos the round's model actually trained on (labeled minus the
    #: held-out test and valid sets) — the learning curve's x axis
    train_images: int | None = None
    #: the split evaluation is still running in the background
    evaluating: bool = False
    created_at: str


class LoopStatus(BaseModel):
    total_images: int
    labeled_images: int
    #: skipped by annotators as unfit for training (E10-T16)
    skipped_images: int = 0
    #: selectable images: unlabeled, not skipped, in the train split, not in the open round
    pool_size: int
    validation_images: int
    #: labeled photos held out as test — the model never trains on them
    test_images: int = 0
    categories: list[str]
    #: a completed training run exists — PAL will score with it
    has_model: bool
    #: what strategy "auto" would pick for the next round
    next_strategy: SelectionStrategy
    current: LoopRound | None = None
    #: the loop job in flight (selection, pre-annotation, embeddings) — a
    #: reloaded page reattaches to it instead of showing the Pick button
    job: RunningJob | None = None
    rounds: list[RoundSummary] = Field(default_factory=list)
    settings: LoopSettings = Field(default_factory=LoopSettings)


# --------------------------------------------------------------- helpers


def _labeled_ids(project: Project) -> set[int]:
    """Images with confirmed annotations that are not skipped — the training set."""
    out = set()
    for record in project.list_images():
        if record.excluded:
            continue
        if any(a.status == "confirmed" for a in project.load_annotations(record.id).annotations):
            out.add(record.id)
    return out


def _in_pool(record, labeled: set[int]) -> bool:
    # unlabeled photos are in no set; held-out photos are labeled, so never here
    return not record.excluded and record.id not in labeled


def _latest_completed_run(project: Project, model: str | None = None):
    """Newest completed run with weights; with `model`, the newest of THAT
    model when one exists (the loop scores and pseudo-labels with the model
    the user chose), else the newest of any model."""
    runs = [r for r in list_runs(project) if r.state == "completed" and r.checkpoint]
    if model is not None:
        same = [r for r in runs if r.model == model]
        if same:
            runs = same
    return max(runs, key=lambda r: r.created_at) if runs else None


def _wants_polygons(project: Project, labeled: set[int]) -> bool:
    """Shapes "auto" means polygons when the loop trains a segmentation model
    — chosen in the settings or implied by polygon labels — so pseudo-labels
    from a box scorer (a detection run, OWLv2) are refined with SAM instead of
    landing as boxes in a segmentation project."""
    settings = get_loop_settings(project)
    model = settings.model or default_model_for(project, labeled)[0]
    try:
        return get_model_info(model).task == "instance_segmentation"
    except HorosError:
        return False


def _auto_strategy(project: Project, labeled: set[int]) -> tuple[SelectionStrategy, str]:
    if not labeled:
        return "diversity", "no labeled images yet — spreading the first batch over the data"
    if _latest_completed_run(project) is not None:
        return "pal", "a trained model exists — scoring with it (PAL)"
    if project.categories:
        return "pal", (
            f"no trained model yet — {ZERO_SHOT_SCORER} zero-shot on the class names "
            f"stands in as the PAL scorer"
        )
    return "diversity", "labels exist but the project has no classes to prompt a scorer with"


def _headline(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    for key in _METRIC_PREFERENCE:
        if key in metrics:
            return key, float(metrics[key])
    return None, None


def _summary(
    project: Project,
    record: LoopRound,
    *,
    labeled_now: int,
    previous: tuple[str, float] | None,
) -> RoundSummary:
    labeled = skipped = 0
    if record.selection:
        by_id = {r.id: r for r in project.list_images()}
        for image_id in record.selection.image_ids:
            image = by_id.get(image_id)
            if image is None:
                continue
            if image.excluded:
                skipped += 1
                continue
            anns = project.load_annotations(image_id).annotations
            if any(a.status == "confirmed" for a in anns):
                labeled += 1
    after = record.labeled_after if record.labeled_after is not None else labeled_now
    key, value = _headline(record.metrics)
    delta = improved = None
    if key is not None and previous is not None and previous[0] == key:
        delta = value - previous[1]
        improved = delta < 0 if key == "loss" else delta > 0
    return RoundSummary(
        number=record.number,
        state=record.state,
        strategy=record.selection.strategy if record.selection else None,
        requested=record.selection.requested if record.selection else 0,
        picked=len(record.image_ids),
        labeled=labeled,
        skipped=skipped,
        labeled_before=record.labeled_before,
        labels_spent=max(0, after - record.labeled_before),
        train_run_id=record.train_run_id,
        metrics=record.metrics,
        metric_key=key,
        metric=value,
        delta=delta,
        improved=improved,
        curve={
            split: record.metrics[f"eval/{split}/map_50"]
            for split in ("train", "valid", "test") if f"eval/{split}/map_50" in record.metrics
        },
        labeled_total=after,
        train_images=(record.training.get("holdout") or {}).get("train_images"),
        evaluating=bool(record.training.get("evaluating")),
        created_at=record.created_at,
    )


@capability(
    "loop.history",
    summary="Round history: labels spent, headline metric and its change per round",
    web_route="/api/v1/loop/history",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' prints the same table.",
)
def loop_history(project: Project) -> list[RoundSummary]:
    labeled_now = len(_labeled_ids(project))
    out: list[RoundSummary] = []
    previous: tuple[str, float] | None = None
    for record in list_rounds(project):
        if record.state == "closed" and record.selection is None:
            continue  # a selection that was cancelled or interrupted: nothing happened
        record = _reconcile_round(project, record)
        row = _summary(project, record, labeled_now=labeled_now, previous=previous)
        out.append(row)
        if row.metric_key is not None:
            previous = (row.metric_key, row.metric)  # type: ignore[assignment]
    return out


# ------------------------------------------------------------------ status


@capability(
    "loop.status",
    summary="Where the active-learning loop stands: pool, labels, rounds, next strategy",
    web_route="/api/v1/loop",
    web_methods=("GET",),
    cli="loop",
)
def loop_status(project: Project) -> LoopStatus:
    images = project.list_images()
    labeled = _labeled_ids(project)
    active = current_round(project)
    if active is not None:
        active = _reconcile_round(project, active)
    reserved = set(active.image_ids) if active else set()
    pool = [r for r in images if _in_pool(r, labeled) and r.id not in reserved]
    strategy, _ = _auto_strategy(project, labeled)
    return LoopStatus(
        total_images=len(images),
        labeled_images=len(labeled),
        skipped_images=sum(1 for r in images if r.excluded),
        pool_size=len(pool),
        validation_images=sum(1 for r in images if r.split == "valid" and r.id in labeled),
        test_images=sum(1 for r in images if r.split == "test" and r.id in labeled),
        categories=[c.name for c in project.categories],
        has_model=_latest_completed_run(project) is not None,
        next_strategy=strategy,
        current=active,
        job=running_job(LOOP_JOB_KINDS),
        rounds=loop_history(project),
        settings=get_loop_settings(project),
    )


@capability(
    "loop.round",
    summary="One round's record: strategy, every pick with its score and reason",
    web_route="/api/v1/loop/rounds/<int:number>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' prints the round table; per-pick reasons are for the UI.",
)
def get_round(project: Project, number: int) -> LoopRound:
    return _reconcile_round(project, load_round(project, number))


@capability(
    "loop.close",
    summary="Close a round (from any state) so the next one can start",
    web_route="/api/v1/loop/rounds/<int:number>/close",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Rounds close themselves at review; abandoning one is a UI action.",
)
def close_round(project: Project, number: int) -> LoopRound:
    record = _reconcile_round(project, load_round(project, number))
    if record.state == "closed":
        return record
    # re-read under the lock: the split evaluation thread merges its metrics
    # into the same record
    with _ROUND_WRITE_LOCK:
        record = load_round(project, number)
        if record.state == "closed":
            return record
        if record.labeled_after is None:
            record = record.model_copy(update={"labeled_after": len(_labeled_ids(project))})
        return save_round(project, record.advance("closed"))


# --------------------------------------------------------------- selection


def _pal_detections(
    prediction: ImagePrediction,
    image_id: int,
    name_of: dict[int, str],
    *,
    threshold: float,
    resolve: Callable[[str], str] | None = None,
) -> list[pal.Detection]:
    """Backend prediction → PAL detections with support counts. `resolve`
    maps the model's class names to the project's current ones (a class
    renamed since the model learned it), so they match the labels."""
    final = [i for i in prediction.instances if i.score >= threshold]
    cand_boxes = [c.bbox for c in prediction.candidates]
    support = pal.support_counts([f.bbox for f in final], cand_boxes) if cand_boxes else None
    out = []
    for n, inst in enumerate(final):
        name = inst.category_name or name_of.get(inst.category_id)
        if name is None:
            continue
        if resolve is not None:
            name = resolve(name)
        out.append(
            pal.Detection(
                image_id=image_id,
                category=name,
                confidence=inst.score,
                support=None if support is None else support[n],
                class_probs=inst.class_probs,
            )
        )
    return out


def _scorer(project: Project, device: str | None):
    """(backend, label, name_of, kind) — the newest completed run's model
    ("run"), else OWLv2 prompted with the class names ("zero_shot")."""
    run = _latest_completed_run(project, get_loop_settings(project).model)
    if run is not None:
        from horos.api.evaluate import _load_run_backend

        backend, _ = _load_run_backend(project, run.run_id, device=device)
        return backend, run.run_id, {}, "run"
    if not project.categories:
        raise ProjectError(
            "No trained model and no classes to prompt a zero-shot scorer with — "
            "add the project's classes first."
        )
    from horos.api.autolabel import _cached_backend

    names = [c.name for c in project.categories]
    backend = _cached_backend(ZERO_SHOT_SCORER, device)  # loaded once per process
    backend.configure_prompts(names)  # type: ignore[attr-defined]
    return backend, ZERO_SHOT_SCORER, dict(enumerate(names)), "zero_shot"


def _preannotate_images(
    project: Project,
    image_ids: list[int],
    backend: ModelBackend,
    name_of: dict[int, str],
    *,
    threshold: float,
    cached: dict[int, ImagePrediction] | None = None,
    cancel: CancelEvent | None = None,
    phase: str = "pre-annotating",
    shapes: Shapes = "auto",
    refiner: PromptableSegmenter | None = None,
    refiner_model: str = "sam2.1-tiny",
    device: str | None = None,
) -> Iterator[Event]:
    """Write the scorer's detections on `image_ids` as pending auto pre-labels
    (E10-T7, E10-T11). Predictions in `cached` (from the PAL scoring pass)
    are reused, so a PAL round costs no second inference. Images that
    already hold confirmed annotations are left alone. Returns the summary
    dict via StopIteration.value; use `result = yield from ...`."""
    from horos.api.autolabel import _ensure_categories, _iou, _write_pending
    from horos.backends.base import ProgressUpdated

    by_id = {r.id: r for r in project.list_images()}
    images = annotations = duplicates = 0
    for n, image_id in enumerate(image_ids):
        if cancel is not None and cancel.is_set():
            raise _Cancelled()
        record = by_id.get(image_id)
        if record is None:
            continue
        stored = project.load_annotations(image_id)
        if any(a.status == "confirmed" for a in stored.annotations):
            continue  # a person already labeled it; never overwrite human work
        pred = (cached or {}).get(image_id)
        if pred is None:
            pred = backend.infer_one(project.image_path(record), threshold=threshold)
        detections, polygons = [], []
        # highest score first, then per-class NMS: one pseudo-label per object
        ranked = sorted((i for i in pred.instances if i.score >= threshold), key=lambda i: -i.score)
        for inst in ranked:
            name = inst.category_name or name_of.get(inst.category_id)
            if name is None:
                continue
            name = project.resolve_category_name(name)  # renamed since the model learned it?
            if any(c == name and _iou(b, inst.bbox) >= PRELABEL_NMS_IOU for c, b, _ in detections):
                duplicates += 1
                continue
            detections.append((name, inst.bbox, inst.score))
            # a segmentation scorer's polygon rides along unless boxes were asked for
            polygons.append(inst.segmentation[0] if inst.segmentation and shapes != "box" else None)
        cat_ids = _ensure_categories(project, {d[0] for d in detections}) if detections else {}
        annotations += _write_pending(project, image_id, detections, cat_ids, polygons)
        if shapes == "polygon" and detections:
            # boxes → SAM polygons, pending ones on this image only (E10-T19)
            from horos.api.segment import _segmenter, boxes_to_polygons

            stored = project.load_annotations(image_id)
            box_ids = [
                a.id for a in stored.annotations if a.status == "pending" and not a.segmentation
            ]
            if box_ids:
                refiner = refiner or _segmenter(refiner_model, device)
                boxes_to_polygons(project, image_id, annotation_ids=box_ids, include_pending=True,
                                  model=refiner_model, device=device, backend=refiner)
        images += 1
        yield ProgressUpdated(
            current=n + 1, total=len(image_ids), phase=phase,
            message=f"{record.file_name}: {len(detections)} pre-label(s)",
        )
    return {"images": images, "annotations": annotations, "threshold": threshold,
            "duplicates_dropped": duplicates, "nms_iou": PRELABEL_NMS_IOU}


def _rare_class_lookalikes(
    project: Project,
    picks: list[PickedImage],
    wanted: int,
    *,
    labeled_ids: list[int],
    covered: np.ndarray,
    pool_ids: list[int],
    pool_vecs: np.ndarray,
    notes: list[str],
) -> list[PickedImage]:
    """Class balance the scorer cannot give: a class with a handful of labels
    is one the model barely proposes, so PAL finds no candidates for it.
    Its next photos are the pool's nearest look-alikes (embedding cosine) of
    the few photos that already carry it. At most LOOKALIKE_SHARE of the
    round; they replace PAL's lowest-scored picks when the round is full."""
    by_class: dict[str, list[int]] = {}
    names = {c.id: c.name for c in project.categories}
    for image_id in labeled_ids:
        for a in project.load_annotations(image_id).annotations:
            if a.status == "confirmed" and a.category_id in names:
                by_class.setdefault(names[a.category_id], []).append(image_id)
    counts = {c: len(ids) for c, ids in by_class.items()}
    if len(counts) < 2:
        return picks
    mean = sum(counts.values()) / len(counts)
    rare = sorted(
        c for c, n in counts.items()
        if n < max(2 * MIN_INSTANCES_PER_CLASS, RARE_CLASS_FACTOR * mean)
    )
    if not rare:
        return picks
    total = max(1, round(wanted * LOOKALIKE_SHARE))
    row_of = {image_id: n for n, image_id in enumerate(labeled_ids)}
    taken = {p.image_id for p in picks}
    added: list[PickedImage] = []
    per_class = {c: 0 for c in rare}
    # round robin over the rare classes, best look-alike first
    ranked: dict[str, list[tuple[float, int]]] = {}
    for c in rare:
        rows = [row_of[i] for i in set(by_class[c]) if i in row_of]
        if not rows:
            continue
        sims = pool_vecs @ covered[rows].T  # cosine: vectors are L2-normalised
        best = sims.max(axis=1)
        ranked[c] = sorted(((float(best[n]), pool_ids[n]) for n in range(len(pool_ids))
                            if pool_ids[n] not in taken), key=lambda t: (-t[0], t[1]))
    while len(added) < total and ranked:
        progressed = False
        for c in list(ranked):
            while ranked[c] and ranked[c][0][1] in taken:
                ranked[c].pop(0)
            if not ranked[c]:
                del ranked[c]
                continue
            sim, image_id = ranked[c].pop(0)
            taken.add(image_id)
            n_photos = len(set(by_class[c]))
            added.append(PickedImage(
                image_id=image_id, score=sim,
                reason=(f"class balance: looks like the {n_photos} labeled photo(s) of rare class "
                        f"'{c}' ({counts[c]} label(s) vs {mean:.0f} per class on average) — "
                        f"cosine {sim:.2f}; the model has too few examples to propose it"),
            ))
            per_class[c] += 1
            progressed = True
            if len(added) >= total:
                break
        if not progressed:
            break
    if not added:
        return picks
    # make room: drop PAL's lowest-scored picks so the round still holds `wanted`
    keep = sorted(picks, key=lambda p: (-p.score, p.image_id))[: max(0, wanted - len(added))]
    notes.append(
        "class balance: " + ", ".join(
            f"{c} ({counts[c]} label(s)) +{per_class[c]} look-alike(s)"
            for c in rare if per_class[c]
        ) + f" — {len(added)} photo(s) chosen by similarity to the rare classes' photos"
    )
    return keep + added


def select_round_events(
    project: Project,
    *,
    count: int | None = None,
    percent: float | None = None,
    strategy: Strategy = "auto",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    embedder: ImageEmbedder | None = None,
    detector: ModelBackend | None = None,
    detector_label: str | None = None,
    cancel: CancelEvent | None = None,
    seed: int | None = None,
    preannotate: bool | None = None,
    shapes: Shapes | None = None,
    refiner: PromptableSegmenter | None = None,
    scan_factor: int | None = None,
    balance: bool | None = None,
) -> Iterator[Event]:
    """Open the next round and choose its images, as an R4 event stream.
    `preannotate` / `shapes` default to the loop settings (E10-T19).
    Ends with RunCompleted(result={"round", "picked", "strategy"}) and the
    round in state "labeling"; on failure the round is closed again so the
    loop is never stuck. `embedder`/`detector` are injectable for tests."""
    from horos.backends.base import ProgressUpdated, RunCompleted, RunFailed, RunStarted

    settings = get_loop_settings(project)
    if preannotate is None:
        preannotate = settings.preannotate
    if shapes is None:
        shapes = settings.shapes
    images = project.list_images()
    labeled = _labeled_ids(project)
    if shapes == "auto" and _wants_polygons(project, labeled):
        shapes = "polygon"
    pool = [r for r in images if _in_pool(r, labeled)]
    wanted = resolve_count(len(pool), count=count, percent=percent)
    if strategy == "auto":
        chosen, why = _auto_strategy(project, labeled)
    else:
        chosen, why = strategy, f"strategy '{strategy}' requested explicitly"
    if chosen == "pal" and not labeled:
        raise ProjectError(
            "PAL needs labeled images to learn from; label a first batch or use 'diversity'"
        )
    record = create_round(project, labeled_before=len(labeled))
    notes = [why]
    yield RunStarted(
        total=wanted,
        config={"round": record.number, "requested": wanted, "percent": percent,
                "strategy": chosen, "pool": len(pool), "embedding_model": embedding_model,
                "preannotate": preannotate, "shapes": shapes},
    )
    try:
        # ---- embeddings (diversity needs them; PAL uses them for RCSP)
        pool_ids = [r.id for r in pool]
        labeled_ids = sorted(labeled)
        pool_vecs = None
        covered = None
        try:
            for event in embedding_events(
                project, embedding_model, device=device, backend=embedder, cancel=cancel
            ):
                if event.type == "progress":
                    yield event
                elif event.type == "failed":
                    raise BackendError(event.message, backend="embeddings")
            pool_vecs = load_embeddings(project, pool_ids, embedding_model)
            covered = load_embeddings(project, labeled_ids, embedding_model)
        except BackendError as exc:
            notes.append(f"embedding model {embedding_model} unavailable: {exc}")
            pool_vecs = covered = None
        if cancel is not None and cancel.is_set():
            raise _Cancelled()

        picks: list[PickedImage] = []
        scorer_label: str | None = None
        scorer_kind = "run"
        backend = detector
        name_of: dict[int, str] = dict(enumerate(c.name for c in project.categories))
        if backend is not None:
            scorer_label = detector_label or getattr(backend, "family", "detector")
        pool_preds: dict[int, ImagePrediction] = {}
        if chosen == "pal":
            if backend is None:
                backend, scorer_label, name_of, scorer_kind = _scorer(project, device)
            by_id = {r.id: r for r in images}
            # labeled detections with true/false-positive flags → LIUS training set
            fit_ids = labeled_ids
            if len(fit_ids) > MAX_FIT_IMAGES:
                fit_ids = sorted(random.Random(seed).sample(fit_ids, MAX_FIT_IMAGES))
                notes.append(
                    f"PAL classifiers fitted on {MAX_FIT_IMAGES} of {len(labeled_ids)} "
                    f"labeled images"
                )
            labeled_dets: list[pal.Detection] = []
            # the scorer looks at a seeded sample of a big pool: PAL ranks a few
            # thousand candidates as well as all of them, in a fraction of the time
            # per-call override: None = the settings, 0 = no cap; the cap is a
            # multiple of the round size so it scales with what is being picked
            factor = (settings.scan_factor if scan_factor is None
                      else (scan_factor if scan_factor > 0 else None))
            limit = None if factor is None else max(wanted * factor, wanted)
            scored = pool
            if limit is not None and len(pool) > limit:
                scored = sorted(random.Random(seed).sample(pool, limit), key=lambda r: r.id)
                notes.append(
                    f"scored {limit} of {len(pool)} unlabeled photos ({factor}× the round, "
                    f"random sample, seed {seed}); raise the Scan setting to look at more"
                )
            total_steps = len(fit_ids) + len(scored)
            step = 0
            cats = {c.id: c.name for c in project.categories}
            # boxes only: the scorer counts candidates, polygons come later for the picks
            for start in range(0, len(fit_ids), SCORE_CHUNK):
                if cancel is not None and cancel.is_set():
                    raise _Cancelled()
                chunk = fit_ids[start:start + SCORE_CHUNK]
                preds = backend.infer_many(
                    [project.image_path(by_id[i]) for i in chunk],
                    threshold=SCORE_THRESHOLD, masks=False,
                )
                for image_id, pred in zip(chunk, preds, strict=True):
                    dets = _pal_detections(pred, image_id, name_of, threshold=SCORE_THRESHOLD,
                                           resolve=project.resolve_category_name)
                    final = [i for i in pred.instances if i.score >= SCORE_THRESHOLD]
                    boxes = [i.bbox for i in final
                             if (i.category_name or name_of.get(i.category_id))]
                    gt = [a for a in project.load_annotations(image_id).annotations
                          if a.status == "confirmed"]
                    flags = pal.match_true_positives(
                        boxes, [d.category for d in dets], [d.confidence for d in dets],
                        [a.bbox for a in gt], [cats.get(a.category_id, "") for a in gt],
                    )
                    labeled_dets.extend(
                        pal.Detection(
                            image_id=d.image_id, category=d.category, confidence=d.confidence,
                            support=d.support, class_probs=d.class_probs, true_positive=flag,
                        )
                        for d, flag in zip(dets, flags, strict=True)
                    )
                    step += 1
                yield ProgressUpdated(current=step, total=total_steps, phase="scoring labeled",
                                      message=f"{step} of {len(fit_ids)} labeled photos")
            unlabeled_dets: dict[int, list[pal.Detection]] = {}
            for start in range(0, len(scored), SCORE_CHUNK):
                if cancel is not None and cancel.is_set():
                    raise _Cancelled()
                chunk = scored[start:start + SCORE_CHUNK]
                preds = backend.infer_many(
                    [project.image_path(rec) for rec in chunk],
                    threshold=SCORE_THRESHOLD, masks=False,
                )
                for rec, pred in zip(chunk, preds, strict=True):
                    pool_preds[rec.id] = pred
                    unlabeled_dets[rec.id] = _pal_detections(
                        pred, rec.id, name_of, threshold=SCORE_THRESHOLD,
                        resolve=project.resolve_category_name,
                    )
                    step += 1
                found = sum(len(d) for d in unlabeled_dets.values())
                yield ProgressUpdated(current=step, total=total_steps, phase="scoring unlabeled",
                                      message=f"{step - len(fit_ids)} of {len(scored)} photos · "
                                              f"{found} detection(s)")
            embeddings = (
                {image_id: pool_vecs[n] for n, image_id in enumerate(pool_ids)}
                if pool_vecs is not None else None
            )
            balanced = settings.balance if balance is None else balance
            result = pal.select(unlabeled_dets, labeled_dets, wanted, embeddings=embeddings,
                                balance=pal.BALANCE if balanced else 0.0)
            notes.extend(result.notes)
            notes.append(
                "PAL class budgets: " + ", ".join(
                    f"{c}={b}" for c, b in sorted(result.budgets.items())
                )
                + (" (weighted by inverse label frequency: " + ", ".join(
                    f"{c}×{w:.1f}" for c, w in sorted(result.balance.items()) if abs(w - 1) > 0.05
                ) + ")" if balanced and any(abs(w - 1) > 0.05 for w in result.balance.values())
                   else "")
            )
            picks = [
                PickedImage(image_id=p.image_id, score=p.score, reason=p.reason)
                for p in result.picks
            ]
            if balanced and pool_vecs is not None and covered is not None:
                picks = _rare_class_lookalikes(
                    project, picks, wanted, labeled_ids=labeled_ids, covered=covered,
                    pool_ids=pool_ids, pool_vecs=pool_vecs, notes=notes,
                )
            if result.shortfall:
                notes.append(
                    f"{result.shortfall} image(s) added by diversity to fill the round"
                )
        # ---- diversity / random, and PAL's top-up
        remaining = wanted - len(picks)
        if remaining > 0:
            taken = {p.image_id for p in picks}
            rest_idx = [n for n, image_id in enumerate(pool_ids) if image_id not in taken]
            if chosen == "random":
                extra = random_picks(len(rest_idx), remaining, seed=seed)
            elif pool_vecs is not None and rest_idx:
                cover_parts = []
                if covered is not None and len(covered):
                    cover_parts.append(covered)
                if taken:
                    cover_parts.append(pool_vecs[[n for n in range(len(pool_ids))
                                                  if pool_ids[n] in taken]])
                cover = np.vstack(cover_parts) if cover_parts else None
                extra = kcenter_greedy(pool_vecs[rest_idx], remaining, covered=cover)
            else:
                if chosen == "diversity":
                    chosen = "random"
                    notes.append("no embeddings available — fell back to random selection")
                elif chosen == "pal":
                    notes.append("no embeddings available — the PAL top-up is random")
                extra = random_picks(len(rest_idx), remaining, seed=seed)
            for p in extra:
                picks.append(PickedImage(image_id=pool_ids[rest_idx[p.index]],
                                         score=p.score, reason=p.reason))
        yield ProgressUpdated(current=wanted, total=wanted, phase="selecting",
                              message=f"{len(picks)} image(s) chosen")
        # ---- pre-annotate the picks so labeling starts from corrections (E10-T7)
        preannotation: dict = {}
        if preannotate and project.categories:
            try:
                if backend is None:
                    backend, scorer_label, name_of, scorer_kind = _scorer(project, device)
                threshold = PRELABEL_THRESHOLD[scorer_kind]
                # a PAL pass already predicted at SCORE_THRESHOLD; reuse those
                # predictions when the pre-label floor is not lower — except for
                # polygon rounds, which were scored boxes-only and want the
                # segmentation model's own masks on the picks
                cached = (pool_preds if threshold >= SCORE_THRESHOLD and shapes != "polygon"
                          else None)
                summary = yield from _preannotate_images(
                    project, [p.image_id for p in picks], backend, name_of,
                    threshold=threshold, cached=cached, cancel=cancel, shapes=shapes,
                    refiner=refiner, refiner_model=settings.refiner, device=device,
                )
                preannotation = {"scorer": scorer_label, "shapes": shapes, **summary}
            except BackendError as exc:
                notes.append(f"pre-annotation skipped — scorer unavailable: {exc}")
        elif preannotate:
            notes.append("pre-annotation skipped — the project has no classes yet")
        else:
            notes.append("suggestions are off in the loop settings")
        record = record.model_copy(update={
            "selection": SelectionRecord(
                strategy=chosen, requested=wanted, requested_percent=percent,
                pool_size=len(pool),
                embedding_model=embedding_model if pool_vecs is not None else None,
                scorer=scorer_label, picks=picks, notes=notes,
            ),
            "preannotation": preannotation,
        }).advance("labeling")
        save_round(project, record)
        yield RunCompleted(result={"round": record.number, "picked": len(picks),
                                   "strategy": chosen, "cancelled": False})
    except _Cancelled:
        save_round(project, record.advance("closed"))
        yield RunCompleted(result={"round": record.number, "picked": 0, "cancelled": True})
    except Exception as exc:  # noqa: BLE001 — the stream must end with an event (R4)
        logger.exception("round selection failed")
        try:
            save_round(project, load_round(project, record.number).advance("closed"))
        except HorosError:
            pass
        yield RunFailed(error_code=getattr(exc, "code", "backend_error"), message=str(exc))


class _Cancelled(Exception):
    pass


# ---------------------------------------------------------------- training


class TrainReadiness(BaseModel):
    ready: bool
    labeled_images: int
    #: confirmed instances per class name
    instances: dict[str, int] = Field(default_factory=dict)
    #: labeled images currently held out as valid / test
    validation_images: int = 0
    test_images: int = 0
    #: what still blocks training; empty when ready
    reasons: list[str] = Field(default_factory=list)
    #: classes below the per-class minimum — the ones "train without them" drops
    short_classes: list[str] = Field(default_factory=list)
    #: would training be possible with the short classes left out of the run?
    ready_without_short: bool = False
    #: labeled images that still carry a kept class in that case
    labeled_without_short: int = 0


def _confirmed_instances(project: Project, labeled: set[int]) -> dict[str, int]:
    names = {c.id: c.name for c in project.categories}
    counts: dict[str, int] = {}
    for image_id in labeled:
        for a in project.load_annotations(image_id).annotations:
            if a.status == "confirmed":
                name = names.get(a.category_id, f"#{a.category_id}")
                counts[name] = counts.get(name, 0) + 1
    return counts


@capability(
    "loop.readiness",
    summary="Can the loop train now? Labeled images, instances per class, blockers",
    web_route="/api/v1/loop/readiness",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' includes readiness in its status output.",
)
def train_readiness(
    project: Project,
    *,
    min_images: int = MIN_LABELED_IMAGES,
    min_instances: int = MIN_INSTANCES_PER_CLASS,
) -> TrainReadiness:
    """Blockers for training now. `short_classes` are the classes under the
    per-class minimum; `ready_without_short` says whether dropping them from
    the run (TrainRunConfig.categories) would clear every blocker — the
    "train without them" option the user asked for (E10-T8)."""
    labeled = _labeled_ids(project)
    instances = _confirmed_instances(project, labeled)
    reasons: list[str] = []
    if len(labeled) < min_images:
        reasons.append(
            f"{len(labeled)} labeled image(s); training needs at least {min_images}"
        )
    if not instances:
        reasons.append("no confirmed annotations yet")
    short = sorted(name for name, count in instances.items() if count < min_instances)
    for name in short:
        reasons.append(
            f"class '{name}' has {instances[name]} instance(s); needs at least {min_instances}"
        )
    kept = {name for name in instances if name not in short}
    without = _images_with_classes(project, labeled, kept) if kept and short else set()
    ready_without = bool(short) and bool(kept) and len(without) >= min_images
    by_id = {r.id: r for r in project.list_images()}
    valid = sum(1 for i in labeled if by_id[i].split == "valid")
    test = sum(1 for i in labeled if by_id[i].split == "test")
    return TrainReadiness(
        ready=not reasons, labeled_images=len(labeled), instances=instances,
        validation_images=valid, test_images=test, reasons=reasons,
        short_classes=short, ready_without_short=ready_without,
        labeled_without_short=len(without),
    )


def _images_with_classes(project: Project, labeled: set[int], names: set[str]) -> set[int]:
    """Labeled images carrying at least one confirmed instance of `names`."""
    by_name = {c.name: c.id for c in project.categories}
    wanted = {by_name[n] for n in names if n in by_name}
    return {
        image_id for image_id in labeled
        if any(
            a.status == "confirmed" and a.category_id in wanted
            for a in project.load_annotations(image_id).annotations
        )
    }


def _assign_holdouts(project: Project, labeled: set[int]) -> dict:
    """Make sure every labeled photo is in a set before training. Splits are
    assigned when a photo is first labeled (Project.assign_splits, by the
    project's ratios and stable hash); this pass catches any that slipped
    through and reports the sizes the round trains and tests on."""
    moves = project.assign_splits(sorted(labeled))
    by_id = {r.id: r for r in project.list_images()}
    ratios = project.split_ratios
    return {
        "newly_held_out": sum(1 for v in moves.values() if v != "train"),
        "test_images": sum(1 for i in labeled if by_id[i].split == "test"),
        "validation_images": sum(1 for i in labeled if by_id[i].split == "valid"),
        "train_images": sum(1 for i in labeled if by_id[i].split == "train"),
        "ratios": ratios.model_dump(), "seed": project.manifest.split_seed,
    }


DETECTION_DEFAULT = "rfdetr-nano"
SEGMENTATION_DEFAULT = "rfdetr-seg-nano"


def default_model_for(project: Project, labeled: set[int]) -> tuple[str, str]:
    """(model key, reason): the loop never asks the user to pick a model.
    Polygons on most confirmed annotations mean the project wants masks →
    RF-DETR-Seg Nano; boxes → RF-DETR Nano. Both are the smallest, Jetson-
    friendly sizes; experts override through the Train page."""
    with_polygons = total = 0
    for image_id in labeled:
        for a in project.load_annotations(image_id).annotations:
            if a.status != "confirmed":
                continue
            total += 1
            with_polygons += bool(a.segmentation)
    if total and with_polygons * 2 >= total:
        return SEGMENTATION_DEFAULT, (
            f"{with_polygons} of {total} confirmed annotations are polygons → "
            f"instance segmentation model"
        )
    return DETECTION_DEFAULT, (
        f"{total - with_polygons} of {total} confirmed annotations are boxes → detection model"
    )


@capability(
    "loop.train",
    summary="Train this round's model on every labeled image (locks the validation split first)",
    web_route="/api/v1/loop/rounds/<int:number>/train",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos train' trains; the loop's automatic config is a UI default.",
)
def train_round(
    project: Project,
    number: int,
    *,
    model: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    resolution: int | None = None,
    device: str | None = None,
    seed: int = 42,
    extra: dict | None = None,
    entrypoint_override: str | None = None,
    warm_start: bool | None = None,
    ignore_short_classes: bool = False,
) -> LoopRound:
    """Start training for a round in state "labeling". Newly labeled photos
    are first bucketed into test / valid / train (see _assign_holdouts), only
    labeled images enter the snapshot (pending pre-labels are dropped by
    start_training), hyperparameters not given are derived by the E5 rules,
    and the run id is recorded on the round, which moves to "training".

    With the loop's training set to "continue" (or `warm_start=True`) the run
    starts from the newest completed run of the same model: weights kept,
    optimizer fresh, class head resized to today's classes. Without such a
    run — or with "fresh" — it starts from the published weights.

    `ignore_short_classes=True` trains without the classes under the
    per-class minimum: their annotations leave the snapshot and photos that
    carry nothing else leave with them (TrainRunConfig.categories,
    include_background=False), so the run learns the classes that have
    enough labels while the short ones keep collecting."""
    record = _reconcile_round(project, load_round(project, number))
    if record.state != "labeling":
        raise ProjectError(
            f"Round {number} is '{record.state}'; training starts from 'labeling'"
        )
    readiness = train_readiness(project)
    categories: list[str] | None = None
    ignored: list[str] = []
    if ignore_short_classes and readiness.short_classes:
        if not readiness.ready_without_short:
            others = [r for r in readiness.reasons if not r.startswith("class '")]
            raise ProjectError(
                "Not ready to train even without the short classes: "
                + ("; ".join(others) or "no class has enough labels")
            )
        ignored = readiness.short_classes
        categories = sorted(n for n in readiness.instances if n not in ignored)
    elif not readiness.ready:
        raise ProjectError("Not ready to train: " + "; ".join(readiness.reasons))
    labeled = _labeled_ids(project)
    settings = get_loop_settings(project)
    holdout = _assign_holdouts(project, labeled)
    if model is None:
        model = settings.model
        model_reason = "from the loop settings"
    else:
        model_reason = "chosen explicitly"
    if model is None:
        model, model_reason = default_model_for(project, labeled)
    continue_wanted = settings.training == "continue" if warm_start is None else warm_start
    init_from = init_reason = None
    if continue_wanted:
        same = [r for r in list_runs(project)
                if r.state == "completed" and r.checkpoint and r.model == model]
        source = max(same, key=lambda r: r.created_at) if same else None
        if source is not None and Path(source.checkpoint).is_file():
            init_from = source.run_id
            init_reason = f"continues from run {source.run_id} (newest completed {model})"
        else:
            init_reason = f"fresh start: no completed {model} run to continue from"
    else:
        init_reason = "fresh start by choice"
    config = TrainRunConfig(
        model=model, epochs=epochs, batch_size=batch_size, resolution=resolution,
        device=device, seed=seed, image_ids=sorted(labeled), extra=extra or {},
        init_from=source.checkpoint if init_from else None,
        entrypoint_override=entrypoint_override,
        categories=categories,
    )
    run = start_training(project, config)
    record = record.model_copy(update={
        "train_run_id": run.run_id,
        "training": {
            "model": model, "model_reason": model_reason,
            "init_from": init_from, "init_reason": init_reason,
            "labeled_images": len(labeled), "holdout": holdout,
            "labeled_in_round": sum(1 for i in record.image_ids if i in labeled),
            "ignored_classes": ignored,
            "ignored_reason": (
                f"trained without {', '.join(ignored)}: fewer than "
                f"{MIN_INSTANCES_PER_CLASS} instances each" if ignored else None
            ),
        },
    }).advance("training")
    return save_round(project, record)


def _reconcile_round(project: Project, record: LoopRound) -> LoopRound:
    """Move a training round on when its run has finished: completed →
    reviewing (with the run's best scores), failed/stopped → back to
    labeling with the error on record."""
    if record.state != "training" or not record.train_run_id:
        return record
    try:
        status = training_status(project, record.train_run_id)
    except HorosError as exc:
        training = {**record.training, "error": f"run record unreadable: {exc}"}
        return save_round(project, record.model_copy(update={"training": training})
                          .advance("labeling"))
    run = status.run
    if run.state == "completed":
        from horos.api.experiment import get_run_summary

        scores = get_run_summary(project, run.run_id, reference=None).scores
        update = {
            "metrics": dict(scores), "labeled_after": len(_labeled_ids(project)),
            "training": {**record.training, "evaluating": True},
        }
        record = save_round(project, record.model_copy(update=update).advance("reviewing"))
        _evaluate_in_background(project, record.number)
        return load_round(project, record.number)  # the evaluation may have run inline
    if run.state in ("failed", "stopped"):
        training = {**record.training, "error": run.error or f"run {run.state}"}
        return save_round(project, record.model_copy(update={"training": training})
                          .advance("labeling"))
    return record


class RoundTrainingStatus(BaseModel):
    round: LoopRound
    training: TrainStatus | None = None


@capability(
    "loop.training_status",
    summary="A round's training run: state and events after an index, plus the round",
    web_route="/api/v1/loop/rounds/<int:number>/training",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' shows the round table; live events are a UI need.",
)
def round_training_status(project: Project, number: int, *, after: int = 0) -> RoundTrainingStatus:
    record = _reconcile_round(project, load_round(project, number))
    if not record.train_run_id:
        return RoundTrainingStatus(round=record)
    return RoundTrainingStatus(
        round=record, training=training_status(project, record.train_run_id, after=after)
    )


def preannotate_events(
    project: Project,
    number: int,
    *,
    device: str | None = None,
    detector: ModelBackend | None = None,
    detector_label: str | None = None,
    cancel: CancelEvent | None = None,
) -> Iterator[Event]:
    """(Re)write pending pre-labels on a round's unlabeled images with the
    current scorer — a newer trained run, or OWLv2 before one exists. Human
    annotations are never touched; earlier pending pre-labels are replaced."""
    from horos.backends.base import RunCompleted, RunFailed, RunStarted

    record = load_round(project, number)
    if record.state == "closed":
        raise ProjectError(f"Round {number} is closed; pre-annotate the open round instead")
    ids = record.image_ids
    yield RunStarted(total=len(ids), config={"round": number})
    try:
        if detector is not None:
            backend, label, kind = detector, detector_label or detector.family, "run"
            name_of = dict(enumerate(c.name for c in project.categories))
        else:
            backend, label, name_of, kind = _scorer(project, device)
        settings = get_loop_settings(project)
        shapes = settings.shapes
        if shapes == "auto" and _wants_polygons(project, _labeled_ids(project)):
            shapes = "polygon"
        summary = yield from _preannotate_images(
            project, ids, backend, name_of, threshold=PRELABEL_THRESHOLD[kind], cancel=cancel,
            shapes=shapes, refiner_model=settings.refiner, device=device,
        )
        save_round(project, load_round(project, number).model_copy(
            update={"preannotation": {"scorer": label, **summary}}
        ))
        yield RunCompleted(result={"round": number, "cancelled": False, **summary})
    except _Cancelled:
        yield RunCompleted(result={"round": number, "cancelled": True})
    except Exception as exc:  # noqa: BLE001 — the stream must end with an event (R4)
        logger.exception("round pre-annotation failed")
        yield RunFailed(error_code=getattr(exc, "code", "backend_error"), message=str(exc))


@capability(
    "loop.preannotate",
    summary="Pre-label a round's unlabeled images with the current model (blocking)",
    not_web_because="The Web API runs it as a background job (loop.preannotate_job).",
    cli=None,
    not_cli_because="Selection pre-annotates automatically; redoing it is a UI action.",
)
def preannotate_round(project: Project, number: int, **kwargs) -> LoopRound:
    last = None
    for event in preannotate_events(project, number, **kwargs):
        last = event
    if last is None or last.type != "completed":
        raise ProjectError(
            f"Pre-annotation failed: {getattr(last, 'message', 'no result')}"
        )
    return load_round(project, number)


@capability(
    "loop.preannotate_job",
    summary="Pre-label a round's unlabeled images as a background job (poll via jobs.status)",
    web_route="/api/v1/loop/rounds/<int:number>/preannotate",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Selection pre-annotates automatically; redoing it is a UI action.",
)
def start_preannotate_job(project: Project, number: int, *, device: str | None = None) -> str:
    load_round(project, number)  # unknown rounds fail synchronously
    return start_job(
        project,
        "loop-preannotate",
        lambda cancel: preannotate_events(project, number, device=device, cancel=cancel),
    )


@capability(
    "loop.select",
    summary="Open the next round and pick its images (blocking; scripts and CLI)",
    not_web_because="The Web API starts the same work as a background job (loop.select_job).",
    cli="loop",
)
def select_round(project: Project, **kwargs) -> LoopRound:
    """Run `select_round_events` to completion and return the round."""
    last = None
    for event in select_round_events(project, **kwargs):
        last = event
    if last is None or last.type != "completed":
        message = getattr(last, "message", "selection produced no result")
        raise ProjectError(f"Round selection failed: {message}")
    if last.result.get("cancelled"):
        raise ProjectError("Round selection was cancelled")
    return load_round(project, int(last.result["round"]))


@capability(
    "loop.select_job",
    summary="Open the next round and pick its images as a background job (poll via jobs.status)",
    web_route="/api/v1/loop/rounds",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos loop select' runs the selection in the foreground.",
)
def start_round_job(
    project: Project,
    *,
    count: int | None = None,
    percent: float | None = None,
    strategy: Strategy = "auto",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
) -> str:
    if current_round(project) is not None:
        active = current_round(project)
        raise ProjectError(
            f"Round {active.number} is still '{active.state}'; close it before starting a new one."
        )
    return start_job(
        project,
        "loop-select",
        lambda cancel: select_round_events(
            project, count=count, percent=percent, strategy=strategy,
            embedding_model=embedding_model, device=device, cancel=cancel,
        ),
    )


# -------------------------------------------------------------- annotators


@capability(
    "loop.assign",
    summary="Hand a round's images to annotators, round robin (E10-S7)",
    web_route="/api/v1/loop/rounds/<int:number>/assign",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Assignment coordinates interactive annotators.",
)
def assign_round(
    project: Project, number: int, annotators: list[str], *, reassign: bool = False
) -> LoopRound:
    """Distribute the round's picks over `annotators` in pick order. Picks
    that already have an owner keep it unless `reassign` is set — a person
    joining late only receives what nobody holds yet."""
    names = [str(a).strip() for a in annotators]
    if not names:
        raise ProjectError("Give at least one annotator")
    if any(not n for n in names):
        raise ProjectError("Annotator names must not be empty")
    if len(set(names)) != len(names):
        raise ProjectError("Annotator names must be distinct")
    record = load_round(project, number)
    if record.state == "closed":
        raise ProjectError(f"Round {number} is closed; nothing left to assign")
    if record.selection is None:
        raise ProjectError(f"Round {number} has no picks yet")
    picks = []
    n = 0
    for pick in record.selection.picks:
        if pick.assigned_to is None or reassign:
            pick = pick.model_copy(update={"assigned_to": names[n % len(names)]})
            n += 1
        picks.append(pick)
    selection = record.selection.model_copy(update={"picks": picks})
    return save_round(project, record.model_copy(update={"selection": selection}))


class RoundQueueItem(BaseModel):
    image: ImageRecord
    assigned_to: str | None = None
    annotated: bool
    excluded: bool = False
    num_pending: int = 0
    #: another session's live claim on the image (E2-T8), None if free
    claimed_by: str | None = None
    score: float
    reason: str


@capability(
    "loop.queue",
    summary="A round's images for one annotator: their share plus unassigned, unlabeled first",
    web_route="/api/v1/loop/rounds/<int:number>/queue",
    web_methods=("GET",),
    cli=None,
    not_cli_because="The queue drives the interactive annotator.",
)
def round_queue(
    project: Project,
    number: int,
    *,
    annotator: str | None = None,
    session_id: str | None = None,
) -> list[RoundQueueItem]:
    record = load_round(project, number)
    by_id = {r.id: r for r in project.list_images()}
    claims = _load_claims(project)
    items: list[RoundQueueItem] = []
    for pick in record.selection.picks if record.selection else []:
        if annotator is not None and pick.assigned_to not in (None, annotator):
            continue
        image = by_id.get(pick.image_id)
        if image is None:
            continue  # deleted since the round was selected
        anns = project.load_annotations(pick.image_id).annotations
        holder = claims.get(pick.image_id)
        items.append(
            RoundQueueItem(
                image=image,
                assigned_to=pick.assigned_to,
                annotated=any(a.status == "confirmed" for a in anns),
                excluded=image.excluded,
                num_pending=sum(1 for a in anns if a.status == "pending"),
                claimed_by=(
                    holder["session"] if holder and holder["session"] != session_id else None
                ),
                score=pick.score,
                reason=pick.reason,
            )
        )
    # pick order, always: labeled photos keep their place and replacements
    # (appended to the picks) come last, so a reload after a skip never
    # reshuffles what the annotator is walking through; skipped picks sink
    items.sort(key=lambda i: i.excluded)  # stable
    return items


# ------------------------------------------------------------------ skipping


class SimilarImage(BaseModel):
    image: ImageRecord
    #: cosine similarity to the reference image (1 = identical)
    similarity: float
    annotated: bool = False


class SkipResult(BaseModel):
    skipped: list[int]
    #: ids that were already skipped / unknown to the round and left alone
    unchanged: int = 0
    skipped_images: int
    #: photos added to the open round to replace skipped picks (E10-T16)
    replacements: list[int] = Field(default_factory=list)


class ImageDetection(BaseModel):
    """One detection of the loop's current scorer on a project photo."""

    label: str
    bbox: tuple[float, float, float, float]  # COCO xywh, image pixels
    score: float


class ImagePredictions(BaseModel):
    image_id: int
    #: run id of the model used, or the zero-shot scorer's key
    model: str
    kind: Literal["run", "zero_shot"]
    threshold: float
    detections: list[ImageDetection] = Field(default_factory=list)


@capability(
    "images.predictions",
    summary="What the loop's current model sees on one photo (class suggestions)",
    web_route="/api/v1/images/<int:image_id>/predictions",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Feeds the annotator's class suggestion when a shape is accepted.",
)
def image_predictions(
    project: Project,
    image_id: int,
    *,
    threshold: float = 0.1,
    device: str | None = None,
) -> ImagePredictions:
    """Detections of the same scorer the loop pre-annotates with — the newest
    completed run, else OWLv2 prompted with the class names — on one photo,
    names mapped through the project's aliases. Nothing is written: the
    annotator uses them to suggest the class of a shape drawn where no
    pseudo label sits (SAM-T4)."""
    if not 0.0 <= threshold <= 1.0:
        raise ProjectError(f"threshold must be within [0, 1], got {threshold}")
    record = next((r for r in project.list_images() if r.id == image_id), None)
    if record is None:
        raise ProjectError(f"No image with id {image_id} in project {project.root}")
    backend, label, name_of, kind = _scorer(project, device)
    prediction = backend.infer_one(project.image_path(record), threshold=threshold)
    detections = []
    for inst in sorted(prediction.instances, key=lambda i: -i.score):
        if inst.score < threshold:
            continue
        name = inst.category_name or name_of.get(inst.category_id)
        if name is None:
            continue
        detections.append(
            ImageDetection(
                label=project.resolve_category_name(name), bbox=inst.bbox, score=inst.score
            )
        )
    return ImagePredictions(
        image_id=image_id, model=label, kind=kind, threshold=threshold, detections=detections
    )


@capability(
    "images.similar",
    summary="Unlabeled images that look like this one (embedding cosine similarity)",
    web_route="/api/v1/images/<int:image_id>/similar",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Drives the 'skip similar photos' sheet in the annotator.",
)
def similar_images(
    project: Project,
    image_id: int,
    *,
    threshold: float = 0.8,
    limit: int = 48,
    model: str = DEFAULT_EMBEDDING_MODEL,
    include_labeled: bool = False,
) -> list[SimilarImage]:
    """Other images whose embedding is within `threshold` cosine similarity
    of `image_id`, most similar first. Without a current embedding for the
    reference image the answer is an explicit error, never a silent empty
    list — the caller runs the embedding job first (E10-T3)."""
    if not 0.0 <= threshold <= 1.0:
        raise ProjectError(f"threshold must be within [0, 1], got {threshold}")
    records = project.list_images()
    reference = next((r for r in records if r.id == image_id), None)
    if reference is None:
        raise ProjectError(f"No image with id {image_id} in project {project.root}")
    ref_vec = load_embeddings(project, [image_id], model)
    if ref_vec is None:
        raise ProjectError(
            f"Image {image_id} has no current {model} embedding yet — run the embedding "
            f"job (POST /loop/embeddings) first."
        )
    labeled = _labeled_ids(project)
    candidates = [
        r for r in records
        if r.id != image_id and not r.excluded and (include_labeled or r.id not in labeled)
    ]
    vecs = load_embeddings(project, [r.id for r in candidates], model)
    if vecs is None:
        # some candidates lack a vector: score the ones that have one
        scored = []
        for r in candidates:
            v = load_embeddings(project, [r.id], model)
            if v is not None:
                scored.append((r, float(v[0] @ ref_vec[0])))
    else:
        sims = vecs @ ref_vec[0]
        scored = list(zip(candidates, (float(x) for x in sims), strict=True))
    scored = [(r, sim) for r, sim in scored if sim >= threshold]
    scored.sort(key=lambda t: -t[1])
    return [
        SimilarImage(image=r, similarity=round(sim, 4), annotated=r.id in labeled)
        for r, sim in scored[:limit]
    ]


@capability(
    "images.skip",
    summary="Skip images as unfit for training (they leave the pool, stats and snapshots)",
    web_route="/api/v1/images/skip",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Skipping is an annotator's judgement made in the editor.",
)
def skip_images(
    project: Project,
    image_ids: list[int],
    *,
    note: str = "",
    refill: bool = True,
    embedder: ImageEmbedder | None = None,
    detector: ModelBackend | None = None,
    device: str | None = None,
) -> SkipResult:
    """Skip photos. When some of them belong to the open round, the round is
    refilled with as many fresh photos from the pool (`refill`), so skipping
    never shrinks a round below the size that was asked for — otherwise a
    few unusable photos could leave the round short of the training
    threshold."""
    if not image_ids:
        raise ProjectError("Give at least one image id to skip")
    before = {r.id for r in project.list_images() if r.excluded}
    project.set_excluded(list(image_ids), True, note=note.strip())
    skipped = [i for i in image_ids if i not in before]
    replacements: list[int] = []
    active = current_round(project)
    if refill and skipped and active is not None and active.state == "labeling":
        if set(skipped) & set(active.image_ids):
            refilled = refill_round(
                project, active.number, embedder=embedder, detector=detector, device=device,
            )
            known = set(active.image_ids)
            replacements = [i for i in refilled.image_ids if i not in known]
    return SkipResult(
        skipped=skipped, unchanged=len(image_ids) - len(skipped),
        skipped_images=sum(1 for r in project.list_images() if r.excluded),
        replacements=replacements,
    )


@capability(
    "loop.refill",
    summary="Top an open round back up to its requested size after photos were skipped",
    web_route="/api/v1/loop/rounds/<int:number>/refill",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Skipping refills automatically; this is the manual retry.",
)
def refill_round(
    project: Project,
    number: int,
    *,
    embedder: ImageEmbedder | None = None,
    detector: ModelBackend | None = None,
    detector_label: str | None = None,
    device: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> LoopRound:
    """Add `requested − active picks` photos to a round in state "labeling".
    Replacements are chosen by diversity against everything already labeled
    or in the round (fast: the embeddings exist from the selection), inherit
    the skipped picks' annotators round robin, and get the same pre-labels
    the round got. With an empty pool the round notes say so."""
    record = _reconcile_round(project, load_round(project, number))
    if record.state != "labeling" or record.selection is None:
        raise ProjectError(
            f"Round {number} is '{record.state}'; only a round being labeled refills"
        )
    sel = record.selection
    by_id = {r.id: r for r in project.list_images()}
    active_picks = [p for p in sel.picks if p.image_id in by_id and not by_id[p.image_id].excluded]
    need = sel.requested - len(active_picks)
    if need <= 0:
        return record
    labeled = _labeled_ids(project)
    in_round = set(sel.image_ids)
    pool = [r for r in by_id.values() if _in_pool(r, labeled) and r.id not in in_round]
    notes = list(sel.notes)
    if not pool:
        notes.append(f"{need} photo(s) skipped could not be replaced — the pool is empty")
        return save_round(project, record.model_copy(
            update={"selection": sel.model_copy(update={"notes": notes})}
        ))
    need = min(need, len(pool))
    pool_ids = [r.id for r in pool]
    covered_ids = sorted(labeled | {p.image_id for p in active_picks})
    pool_vecs = load_embeddings(project, pool_ids, embedding_model)
    if pool_vecs is None:
        # new photos since the selection: embed just what is missing
        try:
            for _ in embedding_events(project, embedding_model, device=device, backend=embedder):
                pass
            pool_vecs = load_embeddings(project, pool_ids, embedding_model)
        except BackendError:
            pool_vecs = None
    covered = load_embeddings(project, covered_ids, embedding_model) if covered_ids else None
    if pool_vecs is not None:
        extra = kcenter_greedy(pool_vecs, need, covered=covered)
        how = "replacement for a skipped photo — "
    else:
        extra = random_picks(len(pool_ids), need)
        how = "replacement for a skipped photo (no embeddings) — "
        notes.append("replacements were picked at random — no embeddings available")
    # inherit the annotators of the skipped picks, round robin
    owners = [p.assigned_to for p in sel.picks
              if p.assigned_to and p.image_id in by_id and by_id[p.image_id].excluded]
    new_picks = []
    for n, pick in enumerate(extra):
        new_picks.append(PickedImage(
            image_id=pool_ids[pick.index], score=pick.score, reason=how + pick.reason,
            assigned_to=owners[n % len(owners)] if owners else None,
        ))
    notes.append(f"{len(new_picks)} photo(s) added to replace skipped ones")
    preannotation = dict(record.preannotation)
    settings = get_loop_settings(project)
    if project.categories and settings.preannotate:
        try:
            if detector is not None:
                backend, label, name_of, kind = detector, detector_label or detector.family, \
                    dict(enumerate(c.name for c in project.categories)), "run"
            else:
                backend, label, name_of, kind = _scorer(project, device)
            shapes = settings.shapes
            if shapes == "auto" and _wants_polygons(project, labeled):
                shapes = "polygon"
            gen = _preannotate_images(
                project, [p.image_id for p in new_picks], backend, name_of,
                threshold=PRELABEL_THRESHOLD[kind], shapes=shapes,
                refiner_model=settings.refiner, device=device,
            )
            summary = None
            while True:
                try:
                    next(gen)
                except StopIteration as stop:
                    summary = stop.value
                    break
            if summary:
                preannotation = {
                    "scorer": label, "threshold": summary["threshold"],
                    "images": preannotation.get("images", 0) + summary["images"],
                    "annotations": preannotation.get("annotations", 0) + summary["annotations"],
                }
        except BackendError as exc:
            notes.append(f"replacements not pre-labeled — scorer unavailable: {exc}")
    return save_round(project, record.model_copy(update={
        "selection": sel.model_copy(update={"picks": [*sel.picks, *new_picks], "notes": notes}),
        "preannotation": preannotation,
    }))


@capability(
    "images.restore",
    summary="Bring skipped images back into the pool",
    web_route="/api/v1/images/restore",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Undo of an editor action.",
)
def restore_images(project: Project, image_ids: list[int]) -> SkipResult:
    if not image_ids:
        raise ProjectError("Give at least one image id to restore")
    changed = project.set_excluded(list(image_ids), False)
    return SkipResult(
        skipped=[], unchanged=len(image_ids) - changed,
        skipped_images=sum(1 for r in project.list_images() if r.excluded),
    )


# ------------------------------------------------------------------- advice


Verdict = Literal["continue", "flattening", "target_reached", "check_labels", "nothing_left",
                  "first_round"]

#: below this absolute gain of a mAP-style metric (or relative loss drop) a
#: round counts as "flat"; two flat rounds in a row → flattening
FLAT_GAIN = 0.01
FLAT_LOSS_DROP = 0.02
#: train minus test mAP@50 above which the model is still learning the
#: training photos rather than the task — the goal does not count as reached
#: and more (diverse) labels are the advice
GAP_TOLERANCE = 0.05


class LoopAdvice(BaseModel):
    """The answer to "should I label another round?" (E10-S5), rule-based
    with the rule spelled out — the user decides, the loop explains."""

    verdict: Verdict
    title: str
    reason: str
    metric_key: str | None = None
    metric: float | None = None
    delta: float | None = None
    labels_spent: int = 0
    #: metric gain per 100 labeled photos over the last comparable rounds
    gain_per_100: float | None = None
    pool_size: int = 0
    rounds_with_metric: int = 0
    target: float | None = None
    #: train minus test mAP@50 of the last evaluated round, when both exist
    gap: float | None = None
    #: the trained run's own verdict findings (E7): validation too small, …
    findings: list[str] = Field(default_factory=list)


@capability(
    "loop.advice",
    summary="Should another round be labeled? Rule-based verdict with its reason",
    web_route="/api/v1/loop/advice",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos loop' prints the advice line with the status.",
)
def loop_advice(project: Project) -> LoopAdvice:
    history = [r for r in loop_history(project) if r.metric is not None]
    status = loop_status(project)
    settings = get_loop_settings(project)
    findings: list[str] = []
    if history and history[-1].train_run_id:
        try:
            from horos.api.verdict import run_verdict

            verdict = run_verdict(project, history[-1].train_run_id)
            findings = [
                f"{f.title} — {f.suggestion}" for f in verdict.findings if f.severity != "info"
            ][:3]
        except HorosError:
            findings = []
    if not history:
        return LoopAdvice(
            verdict="first_round", title="Label the first round",
            reason="There is no trained model yet; the first round gives the loop something to "
                   "measure and to pick from.",
            pool_size=status.pool_size, target=settings.target, findings=findings,
        )
    last = history[-1]
    key = last.metric_key or ""
    lower_is_better = key == "loss"
    base = dict(
        metric_key=last.metric_key, metric=last.metric, delta=last.delta,
        labels_spent=last.labels_spent, pool_size=status.pool_size,
        rounds_with_metric=len(history), target=settings.target, findings=findings,
    )
    # gain per 100 labels over the last two comparable rounds
    gain = None
    if len(history) >= 2 and history[-2].metric_key == key:
        spent = sum(r.labels_spent for r in history[-1:])
        change = last.metric - history[-2].metric
        if lower_is_better:
            change = -change
        if spent > 0:
            gain = round(100 * change / spent, 4)
    base["gain_per_100"] = gain
    metric_text = f"{key} {last.metric:.3f}" if last.metric is not None else ""
    gap = None
    if "train" in last.curve and "test" in last.curve:
        gap = round(last.curve["train"] - last.curve["test"], 4)
    base["gap"] = gap
    target_met = settings.target is not None and last.metric is not None and (
        (not lower_is_better and last.metric >= settings.target)
        or (lower_is_better and last.metric <= settings.target)
    )
    if gap is not None and gap > GAP_TOLERANCE and status.pool_size > 0:
        goal = (f" The goal of {settings.target:g} is met on the test set, but a gap of"
                if target_met else " A gap of")
        return LoopAdvice(
            verdict="continue", title="Keep going — train is still well above test",
            reason=f"mAP@50 is {last.curve['train']:.3f} on the training photos and "
                   f"{last.curve['test']:.3f} on the held-out test set.{goal} {gap:+.3f} means "
                   f"the model still learns these photos rather than the task; more labels, "
                   f"chosen for diversity, close it. Export when the two curves meet.",
            **base,
        )
    if target_met:
        return LoopAdvice(
            verdict="target_reached", title="Goal reached — export the model",
            reason=f"{metric_text} meets the goal of {settings.target:g}. Another round would "
                   f"spend labels on a model that already does what you asked.",
            **base,
        )
    if status.pool_size == 0:
        return LoopAdvice(
            verdict="nothing_left", title="Nothing left to label",
            reason="Every photo is labeled or skipped. Add photos on the Dataset page to keep "
                   "going, or export this model.",
            **base,
        )
    if len(history) == 1:
        return LoopAdvice(
            verdict="continue", title="Keep going — one round is not a trend",
            reason=f"{metric_text} after {last.labels_spent} labeled photos. A second round "
                   f"shows whether more labels still move the metric.",
            **base,
        )
    if last.improved is False and last.delta is not None:
        return LoopAdvice(
            verdict="check_labels", title="Worse than last round — check the new labels first",
            reason=f"{metric_text} moved {last.delta:+.3f} against the previous round. Before "
                   f"adding more, look at this round's labels and the model's worst cases: a "
                   f"few wrong or inconsistent labels usually explain a drop.",
            **base,
        )
    flat_now = (
        last.delta is not None and (
            (lower_is_better and abs(last.delta) < FLAT_LOSS_DROP * max(abs(last.metric), 1e-9))
            or (not lower_is_better and abs(last.delta) < FLAT_GAIN)
        )
    )
    prev = history[-2]
    flat_prev = (
        prev.delta is not None and (
            (lower_is_better and abs(prev.delta) < FLAT_LOSS_DROP * max(abs(prev.metric), 1e-9))
            or (not lower_is_better and abs(prev.delta) < FLAT_GAIN)
        )
    )
    if flat_now and flat_prev:
        return LoopAdvice(
            verdict="flattening", title="Gains are flattening",
            reason=f"The last two rounds moved {key} by less than "
                   f"{FLAT_GAIN if not lower_is_better else FLAT_LOSS_DROP:g} each "
                   f"({last.labels_spent} labels this round"
                   + (f", {gain:+.3f} per 100 labels" if gain is not None else "")
                   + "). Either stop and export, pick a bigger round, or try a larger model.",
            **base,
        )
    # no delta when the previous round was headlined by another metric (a
    # test set that only exists from this round on, say): not comparable
    change = (f", {last.delta:+.3f} against the previous round" if last.delta is not None
              else f" — not comparable to the previous round, which was measured on "
                   f"{history[-2].metric_key or 'another metric'}")
    return LoopAdvice(
        verdict="continue", title="Keep going — labels still pay off",
        reason=f"{metric_text}{change}"
               + (f" ({gain:+.3f} per 100 labels)" if gain is not None else "")
               + f". {status.pool_size} photos are still unlabeled.",
        **base,
    )


# --------------------------------------------------------- learning curve


_EVALUATING: set[tuple[str, int]] = set()
_EVALUATING_LOCK = threading.Lock()
#: serialises read-modify-write of a round record between the request thread
#: (close, reconcile) and the evaluation thread within this process
_ROUND_WRITE_LOCK = threading.RLock()


def _evaluate_in_background(project: Project, number: int) -> None:
    """Run the split evaluation on a daemon thread; the round record is the
    hand-off (training.evaluating flips back when done)."""
    key = (str(project.root), number)
    with _EVALUATING_LOCK:
        if key in _EVALUATING:
            return
        _EVALUATING.add(key)

    def run() -> None:
        try:
            evaluate_round_splits(project, number)
        except HorosError:
            logger.exception("round %s split evaluation failed", number)
        finally:
            with _EVALUATING_LOCK:
                _EVALUATING.discard(key)

    threading.Thread(target=run, name=f"horos-loop-eval-{number}", daemon=True).start()


@capability(
    "loop.evaluate",
    summary="mAP of a round's model on its train, valid and test splits (the learning curve)",
    web_route="/api/v1/loop/rounds/<int:number>/evaluate",
    web_methods=("POST",),
    cli=None,
    not_cli_because="Runs automatically when a round's training completes.",
)
def evaluate_round_splits(
    project: Project, number: int, *, device: str | None = None
) -> LoopRound:
    """COCO mAP of the round's trained model on every split its snapshot has
    (train, valid, test), stored as eval/<split>/map_50 and map_5095 on the
    round. Train mAP shows how well the model fits what it saw — scored on the
    run's own snapshot, since that is the set it saw — valid how it
    generalises to the fixed validation set, test — when the project has a
    labeled test split — how it does on data the loop never touched. A split
    that is missing or fails is recorded as a note, never as a zero."""
    from horos.api.evaluate import LabelSource, evaluate_run

    record = load_round(project, number)
    if not record.train_run_id:
        raise ProjectError(f"Round {number} has no training run to evaluate")
    metrics = dict(record.metrics)
    notes: list[str] = []
    for split in ("train", "valid", "test"):
        # the training line answers "how well does it fit what it saw", so it
        # is scored on the run's own snapshot; the current-labels default holds
        # back every photo the run trained on, which empties the train set by
        # construction. valid and test keep that holdback (E6-T13).
        labels: LabelSource = "snapshot" if split == "train" else "current"
        try:
            report = evaluate_run(
                project, record.train_run_id, split=split, labels=labels, device=device
            )
        except HorosError as exc:
            if "no '" in str(exc) and "split" in str(exc):
                continue  # the snapshot simply has no such split (usually test)
            notes.append(f"{split}: evaluation failed — {exc}")
            continue
        metrics[f"eval/{split}/map_50"] = float(report.map_50)
        metrics[f"eval/{split}/map_5095"] = float(report.map_5095)
    with _ROUND_WRITE_LOCK:
        latest = load_round(project, number)
        training = {**latest.training, "evaluating": False}
        if notes:
            training["evaluation_notes"] = notes
        else:
            training.pop("evaluation_notes", None)  # a re-run that worked clears the old reason
        merged = {**latest.metrics, **metrics}
        return save_round(
            project, latest.model_copy(update={"metrics": merged, "training": training})
        )

