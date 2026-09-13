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

import logging
import random
from collections.abc import Iterator
from threading import Event as CancelEvent
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from horos.api.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_events, load_embeddings
from horos.api.jobs import start_job
from horos.api.manifest import capability
from horos.api.train import TrainRunConfig, TrainStatus, list_runs, start_training, training_status
from horos.core import pal
from horos.core.project import Project
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
    from horos.backends.base import Event, ImageEmbedder, ImagePrediction, ModelBackend

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
]

Strategy = Literal["auto", "pal", "diversity", "random"]
#: a detection counts as "final" for PAL scoring at or above this confidence;
#: the backends report raw candidates below it (down to their own floor)
SCORE_THRESHOLD = 0.3
#: labeled images used to fit PAL's logistic classifiers per round — a cap
#: so a large labeled set does not cost a full inference pass every round
MAX_FIT_IMAGES = 300
ZERO_SHOT_SCORER = "owlv2-base"
#: pre-label confidence floor per scorer kind: a fine-tuned run is calibrated
#: on the project's classes, zero-shot OWLv2 scores run low (autolabel's
#: default is 0.1)
PRELABEL_THRESHOLD = {"run": SCORE_THRESHOLD, "zero_shot": 0.1}
#: training readiness (E10-T8): enough labeled images, and every class that
#: appears at all has enough instances for a validation split to mean anything
MIN_LABELED_IMAGES = 20
MIN_INSTANCES_PER_CLASS = 5
#: the validation split locked at the first training: a fixed share of the
#: labeled images, never grown afterwards (round metrics stay comparable)
VALID_FRACTION = 0.2
MIN_VALID_IMAGES = 2


class RoundSummary(BaseModel):
    number: int
    state: str
    strategy: str | None = None
    requested: int = 0
    picked: int = 0
    #: picks that now hold confirmed annotations
    labeled: int = 0
    train_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str


class LoopStatus(BaseModel):
    total_images: int
    labeled_images: int
    #: selectable images: unlabeled, in the train split, not in the open round
    pool_size: int
    validation_images: int
    categories: list[str]
    #: a completed training run exists — PAL will score with it
    has_model: bool
    #: what strategy "auto" would pick for the next round
    next_strategy: SelectionStrategy
    current: LoopRound | None = None
    rounds: list[RoundSummary] = Field(default_factory=list)


# --------------------------------------------------------------- helpers


def _labeled_ids(project: Project) -> set[int]:
    out = set()
    for record in project.list_images():
        if any(a.status == "confirmed" for a in project.load_annotations(record.id).annotations):
            out.add(record.id)
    return out


def _latest_completed_run(project: Project):
    runs = [r for r in list_runs(project) if r.state == "completed" and r.checkpoint]
    return max(runs, key=lambda r: r.created_at) if runs else None


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


def _summary(project: Project, record: LoopRound) -> RoundSummary:
    labeled = 0
    if record.selection:
        for image_id in record.selection.image_ids:
            try:
                anns = project.load_annotations(image_id).annotations
            except ProjectError:
                continue
            if any(a.status == "confirmed" for a in anns):
                labeled += 1
    return RoundSummary(
        number=record.number,
        state=record.state,
        strategy=record.selection.strategy if record.selection else None,
        requested=record.selection.requested if record.selection else 0,
        picked=len(record.image_ids),
        labeled=labeled,
        train_run_id=record.train_run_id,
        metrics=record.metrics,
        created_at=record.created_at,
    )


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
    pool = [
        r for r in images if r.id not in labeled and r.split == "train" and r.id not in reserved
    ]
    strategy, _ = _auto_strategy(project, labeled)
    return LoopStatus(
        total_images=len(images),
        labeled_images=len(labeled),
        pool_size=len(pool),
        validation_images=sum(1 for r in images if r.split == "valid"),
        categories=[c.name for c in project.categories],
        has_model=_latest_completed_run(project) is not None,
        next_strategy=strategy,
        current=active,
        rounds=[_summary(project, r) for r in list_rounds(project)],
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
    record = load_round(project, number)
    if record.state == "closed":
        return record
    return save_round(project, record.advance("closed"))


# --------------------------------------------------------------- selection


def _pal_detections(
    prediction: ImagePrediction,
    image_id: int,
    name_of: dict[int, str],
    *,
    threshold: float,
) -> list[pal.Detection]:
    """Backend prediction → PAL detections with support counts."""
    final = [i for i in prediction.instances if i.score >= threshold]
    cand_boxes = [c.bbox for c in prediction.candidates]
    support = pal.support_counts([f.bbox for f in final], cand_boxes) if cand_boxes else None
    out = []
    for n, inst in enumerate(final):
        name = inst.category_name or name_of.get(inst.category_id)
        if name is None:
            continue
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
    run = _latest_completed_run(project)
    if run is not None:
        from horos.api.evaluate import _load_run_backend

        backend, _ = _load_run_backend(project, run.run_id, device=device)
        return backend, run.run_id, {}, "run"
    if not project.categories:
        raise ProjectError(
            "No trained model and no classes to prompt a zero-shot scorer with — "
            "add the project's classes first."
        )
    from horos.backends import get_backend

    names = [c.name for c in project.categories]
    backend = get_backend(ZERO_SHOT_SCORER, device=device)
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
) -> Iterator[Event]:
    """Write the scorer's detections on `image_ids` as pending auto pre-labels
    (E10-T7, E10-T11). Predictions in `cached` (from the PAL scoring pass)
    are reused, so a PAL round costs no second inference. Images that
    already hold confirmed annotations are left alone. Returns the summary
    dict via StopIteration.value; use `result = yield from ...`."""
    from horos.api.autolabel import _ensure_categories, _write_pending
    from horos.backends.base import ProgressUpdated

    by_id = {r.id: r for r in project.list_images()}
    images = annotations = 0
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
        detections = []
        for inst in pred.instances:
            name = inst.category_name or name_of.get(inst.category_id)
            if name is None or inst.score < threshold:
                continue
            detections.append((name, inst.bbox, inst.score))
        cat_ids = _ensure_categories(project, {d[0] for d in detections}) if detections else {}
        annotations += _write_pending(project, image_id, detections, cat_ids)
        images += 1
        yield ProgressUpdated(
            current=n + 1, total=len(image_ids), phase=phase,
            message=f"{record.file_name}: {len(detections)} pre-label(s)",
        )
    return {"images": images, "annotations": annotations, "threshold": threshold}


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
    preannotate: bool = True,
) -> Iterator[Event]:
    """Open the next round and choose its images, as an R4 event stream.
    Ends with RunCompleted(result={"round", "picked", "strategy"}) and the
    round in state "labeling"; on failure the round is closed again so the
    loop is never stuck. `embedder`/`detector` are injectable for tests."""
    from horos.backends.base import ProgressUpdated, RunCompleted, RunFailed, RunStarted

    images = project.list_images()
    labeled = _labeled_ids(project)
    pool = [r for r in images if r.id not in labeled and r.split == "train"]
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
                "strategy": chosen, "pool": len(pool), "embedding_model": embedding_model},
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
            total_steps = len(fit_ids) + len(pool)
            step = 0
            cats = {c.id: c.name for c in project.categories}
            for image_id in fit_ids:
                if cancel is not None and cancel.is_set():
                    raise _Cancelled()
                rec = by_id[image_id]
                pred = backend.infer_one(project.image_path(rec), threshold=SCORE_THRESHOLD)
                dets = _pal_detections(pred, image_id, name_of, threshold=SCORE_THRESHOLD)
                final = [i for i in pred.instances if i.score >= SCORE_THRESHOLD]
                boxes = [i.bbox for i in final if (i.category_name or name_of.get(i.category_id))]
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
                                      message=f"{rec.file_name}: {len(dets)} detection(s)")
            unlabeled_dets: dict[int, list[pal.Detection]] = {}
            for rec in pool:
                if cancel is not None and cancel.is_set():
                    raise _Cancelled()
                pred = backend.infer_one(project.image_path(rec), threshold=SCORE_THRESHOLD)
                pool_preds[rec.id] = pred
                unlabeled_dets[rec.id] = _pal_detections(
                    pred, rec.id, name_of, threshold=SCORE_THRESHOLD
                )
                step += 1
                yield ProgressUpdated(current=step, total=total_steps, phase="scoring unlabeled",
                                      message=f"{rec.file_name}: "
                                              f"{len(unlabeled_dets[rec.id])} detection(s)")
            embeddings = (
                {image_id: pool_vecs[n] for n, image_id in enumerate(pool_ids)}
                if pool_vecs is not None else None
            )
            result = pal.select(unlabeled_dets, labeled_dets, wanted, embeddings=embeddings)
            notes.extend(result.notes)
            notes.append(
                "PAL class budgets: " + ", ".join(
                    f"{c}={b}" for c, b in sorted(result.budgets.items())
                )
            )
            picks = [
                PickedImage(image_id=p.image_id, score=p.score, reason=p.reason)
                for p in result.picks
            ]
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
                # a PAL pass already predicted at SCORE_THRESHOLD; only reuse
                # those predictions when the pre-label floor is not lower
                cached = pool_preds if threshold >= SCORE_THRESHOLD else None
                summary = yield from _preannotate_images(
                    project, [p.image_id for p in picks], backend, name_of,
                    threshold=threshold, cached=cached, cancel=cancel,
                )
                preannotation = {"scorer": scorer_label, **summary}
            except BackendError as exc:
                notes.append(f"pre-annotation skipped — scorer unavailable: {exc}")
        elif preannotate:
            notes.append("pre-annotation skipped — the project has no classes yet")
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
    #: labeled images already in the validation split (0 before the lock)
    validation_images: int = 0
    #: what still blocks training; empty when ready
    reasons: list[str] = Field(default_factory=list)


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
    labeled = _labeled_ids(project)
    instances = _confirmed_instances(project, labeled)
    reasons: list[str] = []
    if len(labeled) < min_images:
        reasons.append(
            f"{len(labeled)} labeled image(s); training needs at least {min_images}"
        )
    if not instances:
        reasons.append("no confirmed annotations yet")
    for name, count in sorted(instances.items()):
        if count < min_instances:
            reasons.append(
                f"class '{name}' has {count} instance(s); needs at least {min_instances}"
            )
    by_id = {r.id: r for r in project.list_images()}
    valid = sum(1 for i in labeled if by_id[i].split == "valid")
    return TrainReadiness(
        ready=not reasons, labeled_images=len(labeled), instances=instances,
        validation_images=valid, reasons=reasons,
    )


def _lock_validation(project: Project, labeled: set[int], *, seed: int) -> dict:
    """First training only: move a fixed share of the labeled train images to
    the valid split. Later rounds find the lock in place and change nothing,
    so every round's model is measured on the same images (E7-T2)."""
    by_id = {r.id: r for r in project.list_images()}
    already = sorted(i for i in labeled if by_id[i].split == "valid")
    if already:
        return {"locked_now": False, "validation_images": len(already)}
    candidates = sorted(i for i in labeled if by_id[i].split == "train")
    if len(candidates) < 2:
        raise ProjectError(
            "Cannot lock a validation split: fewer than two labeled train images"
        )
    share = max(MIN_VALID_IMAGES, round(len(candidates) * VALID_FRACTION))
    n_valid = min(len(candidates) - 1, share)
    chosen = sorted(random.Random(seed).sample(candidates, n_valid))
    project.update_image_splits({i: "valid" for i in chosen})
    return {"locked_now": True, "validation_images": n_valid, "seed": seed,
            "image_ids": chosen}


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
    model: str = "rfdetr-nano",
    epochs: int | None = None,
    batch_size: int | None = None,
    resolution: int | None = None,
    device: str | None = None,
    seed: int = 42,
    extra: dict | None = None,
    entrypoint_override: str | None = None,
) -> LoopRound:
    """Start training for a round in state "labeling". Only labeled images
    enter the snapshot (pending pre-labels are dropped by start_training),
    hyperparameters not given are derived by the E5 rules, and the run id
    is recorded on the round, which moves to "training"."""
    record = _reconcile_round(project, load_round(project, number))
    if record.state != "labeling":
        raise ProjectError(
            f"Round {number} is '{record.state}'; training starts from 'labeling'"
        )
    readiness = train_readiness(project)
    if not readiness.ready:
        raise ProjectError("Not ready to train: " + "; ".join(readiness.reasons))
    labeled = _labeled_ids(project)
    lock = _lock_validation(project, labeled, seed=seed)
    config = TrainRunConfig(
        model=model, epochs=epochs, batch_size=batch_size, resolution=resolution,
        device=device, seed=seed, image_ids=sorted(labeled), extra=extra or {},
        entrypoint_override=entrypoint_override,
    )
    run = start_training(project, config)
    record = record.model_copy(update={
        "train_run_id": run.run_id,
        "training": {
            "model": model, "labeled_images": len(labeled), "validation": lock,
            "labeled_in_round": sum(1 for i in record.image_ids if i in labeled),
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
        return save_round(project, record.model_copy(update={"metrics": dict(scores)})
                          .advance("reviewing"))
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
        summary = yield from _preannotate_images(
            project, ids, backend, name_of, threshold=PRELABEL_THRESHOLD[kind], cancel=cancel,
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
