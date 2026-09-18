"""Training run lifecycle (E5-T3): start, poll, stop.

Design decision (confirmed): training runs in a dedicated spawned subprocess
(`python -m horos.api.train_worker <run_dir>`), never inside the Flask/API
process — an OOM or crash cannot take the server down, stop is a real kill,
and the worker is spawn-safe by construction (E5-T6b). The worker appends R4
events to `<run_dir>/events.jsonl`; this module reads them back for polling.

Run artifacts live inside the project (confirmed): `<project>/runs/<run_id>/`
holds config.json, run.json, events.jsonl, the exported training dataset, and
checkpoints/ — a project directory carries its full experiment history (E7
scans this same layout later).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from horos.api.dataset import (
    dataset_stats,
    export_dataset,
    filter_dataset_categories,
    subset_dataset,
)
from horos.api.hparams import DerivedValue, HyperparameterPlan, derive_plan
from horos.api.manifest import capability
from horos.api.system import ensure_supported
from horos.core.fingerprint import DatasetFingerprint, fingerprint_dataset
from horos.core.fsutil import atomic_write_text, read_text_retry, rmtree_retry
from horos.core.project import Project
from horos.core.registry import get_model_info
from horos.core.streams import child_env
from horos.errors import LicenseError, ProjectError, UnknownModelError

logger = logging.getLogger(__name__)

__all__ = [
    "TrainRunConfig",
    "RunRecord",
    "TrainStatus",
    "delete_run",
    "derive_hyperparameters",
    "start_training",
    "training_status",
    "stop_training",
    "update_queued_run",
    "list_runs",
]

# queued: created and waiting for the active run to finish (no worker yet);
# pending: worker spawned but not yet reporting; the rest are terminal/live.
RunState = Literal["queued", "pending", "running", "completed", "failed", "stopped"]
ACTIVE_STATES = ("pending", "running")

_RUN_JSON = "run.json"
_CONFIG_JSON = "config.json"
_EVENTS_JSONL = "events.jsonl"
_STOP_FLAG = "stop.flag"
_SPAWN_CLAIM = "spawn.claim"
_WORKER_LOG = "worker.log"


class TrainRunConfig(BaseModel):
    """User-facing training request. Only `model` semantics are horos-level;
    everything else maps onto the backend-neutral TrainSpec.

    Hyperparameters left as None are derived from dataset statistics with a
    recorded reason (E5-T1); a set value is a user override and is marked as
    such in the plan without disturbing the other derivations (E5-T2).
    """

    model: str = "rfdetr-nano"
    epochs: int | None = None
    batch_size: int | None = None
    resolution: int | None = None
    lr: float | None = None
    #: mosaic composites per train image added to the snapshot (0 disables;
    #: None derives from dataset size). See horos/core/mosaic.py.
    mosaic_ratio: float | None = None
    device: str | None = None
    seed: int | None = None
    #: full-state resume (weights + optimizer + schedule); the class set must
    #: match the checkpoint's
    resume_from: str | None = None
    #: warm start from an earlier run's best checkpoint: the weights are kept,
    #: the optimizer starts fresh and the class head is resized, so classes
    #: may differ — the way a loop round continues from the previous one
    #: instead of training from the published weights again (E5-S6)
    init_from: str | None = None
    #: category names to train on; None = all. Unselected classes' objects
    #: become background in this run's dataset snapshot.
    categories: list[str] | None = None
    #: with a category subset: keep images that contain none of the selected
    #: classes as background negatives (True) or drop them (False, default —
    #: the dataset shrinks and the image-count rules follow)
    include_background: bool = False
    #: what "best checkpoint" means — "map" (detection quality, default),
    #: "smoothed_map" (mAP smoothed before comparison; robust when a tiny
    #: valid split makes per-epoch mAP noisy) or "loss" (lowest val loss)
    checkpoint_criterion: Literal["map", "smoothed_map", "loss"] = "map"
    acknowledge_non_apache: bool = False
    #: train on these images only (None = the whole project). The active-
    #: learning loop passes its labeled set so the thousands of still-unlabeled
    #: images are not exported as empty background (E10-T8)
    image_ids: list[int] | None = None
    #: expert passthrough to the backend's own knobs — applied last, wins (E5-S5)
    extra: dict[str, Any] = Field(default_factory=dict)
    #: testing hook: "module:ClassName" resolved instead of the model registry.
    #: Never set this in production code.
    entrypoint_override: str | None = None


class RunRecord(BaseModel):
    run_id: str
    model: str
    state: RunState
    created_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    #: derivation trail: every hyperparameter with its value, reason, and
    #: whether the user overrode it (E5-S2 — "why did the system pick this")
    hparams: list[DerivedValue] = Field(default_factory=list)
    hparam_notes: list[str] = Field(default_factory=list)
    device: str | None = None
    pid: int | None = None
    checkpoint: str | None = None
    error: str | None = None
    dataset_images: int = 0
    #: split sizes at training time — the verdict rules read these
    dataset_splits: dict[str, int] = Field(default_factory=dict)
    #: class names this run trained on (its snapshot's categories); resuming
    #: is only valid with exactly this set, so the UI locks the picker to it
    dataset_classes: list[str] = Field(default_factory=list)
    #: the full-state checkpoint (weights + optimizer + LR schedule; rfdetr's
    #: last.ckpt) — the right resume source. `checkpoint` stays the best
    #: weights for inference/export; resuming from it restarts the optimizer
    #: cold and the loss spikes for a few epochs.
    resume_checkpoint: str | None = None
    #: epochs this run actually finished — a resume's TOTAL epochs must exceed
    #: this or the trainer has nothing left to do
    epochs_completed: int | None = None
    #: content fingerprint of the data this run trained on (E7-T2), fixed at
    #: enqueue time; runs with different fingerprints are not comparable
    dataset_fingerprint: DatasetFingerprint | None = None


class TrainStatus(BaseModel):
    run: RunRecord
    events: list[dict[str, Any]] = Field(default_factory=list)  # events[after:]
    num_events: int = 0
    #: metrics events of the run(s) this one resumed from, oldest first, so a
    #: resumed run's curves start where training really started instead of
    #: at the checkpoint's epoch; only filled for a poll from the beginning
    #: (after == 0). Empty for a run that did not resume
    inherited_events: list[dict[str, Any]] = Field(default_factory=list)
    #: run ids of the resume chain, nearest parent first
    resumed_from: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ run storage


def runs_root(project: Project) -> Path:
    return project.root / "runs"


def _run_dir(project: Project, run_id: str) -> Path:
    run_dir = runs_root(project) / run_id
    if not (run_dir / _RUN_JSON).is_file():
        raise ProjectError(f"No such training run: {run_id}")
    return run_dir


def read_record(run_dir: Path) -> RunRecord:
    # the worker may be replacing run.json this very instant (Windows: a
    # transient PermissionError) — read_text_retry waits it out
    return RunRecord.model_validate_json(read_text_retry(run_dir / _RUN_JSON))


def write_record(run_dir: Path, record: RunRecord) -> None:
    """Atomic replace so a poll never reads a half-written run.json. The
    worker and the parent (reconciling) both write it, so the tmp name is
    per writer — a shared name made one replace steal the other's file
    (R7; see horos.core.fsutil)."""
    atomic_write_text(run_dir / _RUN_JSON, record.model_dump_json(indent=2))


def _read_events(run_dir: Path, after: int = 0) -> tuple[list[dict[str, Any]], int]:
    path = run_dir / _EVENTS_JSONL
    if not path.is_file():
        return [], 0
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # a killed worker can leave a torn final line — not an error
                continue
    return events[after:], len(events)


# Popen handles for workers this process spawned. Needed for liveness: an
# unreaped child that died is a zombie, and os.kill(pid, 0) still succeeds on
# zombies — poll() both answers correctly and reaps. Workers spawned by an
# earlier server process get reparented to init on its exit, so the plain pid
# check below is accurate for them.
_PROCESSES: dict[str, subprocess.Popen] = {}


def _worker_alive(record: RunRecord) -> bool:
    process = _PROCESSES.get(record.run_id)
    if process is not None:
        return process.poll() is None
    return _pid_alive(record.pid)


def _run_class_names(run_dir: Path) -> list[str] | None:
    """The class list a run trained on, read from its dataset snapshot."""
    gt_path = run_dir / "dataset" / "train" / "_annotations.coco.json"
    if not gt_path.is_file():
        return None
    try:
        gt = json.loads(gt_path.read_text("utf-8"))
        return sorted(c["name"] for c in gt.get("categories", []))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _run_completed_epochs(run_dir: Path) -> int | None:
    """Epochs a run actually finished, from its event log (the relay emits a
    progress event with phase="epoch completed" and current=epoch+1 at each
    train-epoch end). Other progress events must NOT count: weight-download
    progress carries BYTES in `current`, which would read as hundreds of
    millions of epochs."""
    events, _ = _read_events(run_dir)
    completed = [
        e["current"]
        for e in events
        if e.get("type") == "progress"
        and e.get("phase") == "epoch completed"
        and isinstance(e.get("current"), int)
    ]
    return max(completed) if completed else None


def _in_a_set(dataset):
    """Drop photos with no split: only labeled photos are set members, and a
    photo outside every set is neither a training example nor a negative."""
    return subset_dataset(dataset, image_ids=[i.id for i in dataset.images if i.split])


def _snapshot_class_names(checkpoint: Path) -> list[str] | None:
    """Class list for a checkpoint inside a horos run
    (runs/<id>/checkpoints/x.pth → runs/<id>/dataset/train/...);
    None when the checkpoint is not inside a run (nothing to verify)."""
    return _run_class_names(checkpoint.parent.parent)


def _split_counts(run_dir: Path, record: RunRecord) -> dict[str, int]:
    """Split sizes at training time; older runs predate the recorded field,
    so fall back to counting the images exported into the run directory."""
    if record.dataset_splits:
        return record.dataset_splits
    from horos.core.formats import IMAGE_SUFFIXES

    counts = {}
    for split in ("train", "valid", "test"):
        split_dir = run_dir / "dataset" / split
        counts[split] = (
            sum(1 for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if split_dir.is_dir()
            else 0
        )
    return counts


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reconcile(run_dir: Path, record: RunRecord) -> RunRecord:
    """A run that claims to be active but whose worker is gone died without a
    terminal event (killed, OOM-killed, machine crash). Settle its state."""
    if record.state == "queued":
        return record  # no worker yet by design — nothing to settle
    changed = False
    if not record.dataset_classes:  # backfill runs recorded before the field
        names = _run_class_names(run_dir)
        if names:
            record.dataset_classes = names
            changed = True
    if record.state not in ("pending", "running"):
        if record.resume_checkpoint is None:
            last = run_dir / "checkpoints" / "last.ckpt"
            if last.is_file():
                record.resume_checkpoint = str(last)
                changed = True
        if record.epochs_completed is None:
            completed = _run_completed_epochs(run_dir)
            if completed is not None:
                record.epochs_completed = completed
                changed = True
    if changed:
        write_record(run_dir, record)
    if record.state not in ("pending", "running") or _worker_alive(record):
        return record
    if (run_dir / _STOP_FLAG).is_file():
        record.state = "stopped"
    else:
        record.state = "failed"
        record.error = (
            "The training worker exited without reporting a result "
            f"(see {run_dir / _WORKER_LOG})."
        )
    write_record(run_dir, record)
    return record


#: knobs a user can set at start or edit on a queued run; None = re-derive
_EDITABLE_KNOBS = ("epochs", "batch_size", "resolution", "lr")
#: derived knobs that are TrainRunConfig fields; every other derived knob is
#: overridden through `extra`
_CONFIG_KNOBS = (*_EDITABLE_KNOBS, "mosaic_ratio")


def _criterion_entry(criterion: str) -> DerivedValue:
    """The checkpoint criterion as a hyperparameter-trail entry, so "why is
    THIS checkpoint the best one" survives with the run."""
    reason = {
        "map": "highest validation mAP wins — detection quality is what ships",
        "smoothed_map": (
            "mAP is smoothed (EMA) before comparison, so one noisy validation "
            "spike on a small valid split cannot lock in the best checkpoint"
        ),
        "loss": (
            "lowest validation loss wins — a smooth signal, but it can "
            "diverge from detection quality (mAP)"
        ),
    }[criterion]
    return DerivedValue(
        name="checkpoint_criterion",
        value=criterion,
        reason=reason,
        overridden=criterion != "map",
    )


def _check_resume_epochs(resume_from: str, total_epochs: int | None) -> None:
    """epochs is the TOTAL count and the trainer restores the checkpoint's
    epoch — a total at or below what is already done raises a raw
    MisconfigurationException deep in the backend; refuse it up front."""
    source_dir = Path(resume_from).parent.parent
    if not (source_dir / _RUN_JSON).is_file():
        return
    source_record = read_record(source_dir)
    completed = source_record.epochs_completed or _run_completed_epochs(source_dir)
    if completed and total_epochs is not None and total_epochs <= completed:
        raise ProjectError(
            f"Cannot resume: the checkpoint already completed {completed} "
            f"epochs and epochs is the TOTAL count — {total_epochs} "
            f"leaves nothing to train. Set epochs above {completed} "
            f"(e.g. {completed + max(10, completed // 2)})."
        )


# ----------------------------------------------------------------- the queue


def _spawn_worker(run_dir: Path, record: RunRecord) -> RunRecord:
    """Start the training worker for a prepared run directory."""
    with (run_dir / _WORKER_LOG).open("ab") as log:
        process = subprocess.Popen(  # noqa: S603 — our own module, our interpreter
            [sys.executable, "-m", "horos.api.train_worker", str(run_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            # the log is a file, so the child's stdout is the locale code page
            # unless we say otherwise — and a backend that prints a box-drawing
            # character would die on it (R7); the environment, not a
            # reconfigure, so spawned DataLoader workers inherit it too
            env=child_env(),
        )
    _PROCESSES[record.run_id] = process
    record.pid = process.pid
    record.state = "running"
    write_record(run_dir, record)
    logger.info("training run %s started (pid %d)", record.run_id, process.pid)
    return record


def advance_queue(root: Path) -> str | None:
    """Promote the oldest queued run when nothing is active; returns its id.

    Takes the runs ROOT (not a Project) so the exiting worker can chain into
    the next queued run with no project machinery. Callers race — the status
    poll from the UI, the worker's exit hook, a CLI — so the actual spawn is
    guarded by atomically creating `spawn.claim` in the run directory (O_EXCL:
    works on every platform, unlike fcntl — R7). The claim is one-shot: a run
    is only ever promoted once.
    """
    if not root.is_dir():
        return None
    queued: list[tuple[str, Path]] = []
    for run_dir in root.iterdir():
        if not (run_dir / _RUN_JSON).is_file():
            continue
        record = _reconcile(run_dir, read_record(run_dir))
        if record.state in ACTIVE_STATES:
            return None  # something is (still) training — nothing to promote
        if record.state == "queued":
            queued.append((record.created_at, run_dir))
    # FIFO by creation TIME: run ids only carry second precision plus a random
    # suffix, so two runs queued within the same second sort arbitrarily by id
    for _, run_dir in sorted(queued):
        try:
            fd = os.open(run_dir / _SPAWN_CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # a concurrent caller claimed this one — it is about to become
            # active, so promoting the NEXT queued run would double-train
            return None
        os.close(fd)
        try:
            record = read_record(run_dir)
        except OSError:  # deleted between the scan and the claim
            continue
        if record.state != "queued":  # e.g. stopped between scan and claim
            continue
        if (run_dir / _STOP_FLAG).is_file():
            record.state = "stopped"
            write_record(run_dir, record)
            continue
        try:
            record = _spawn_worker(run_dir, record)
        except OSError as exc:  # spawn failed: fail the run, try the next one
            record.state = "failed"
            record.error = f"Could not start the training worker: {exc}"
            write_record(run_dir, record)
            continue
        return record.run_id
    return None


# ---------------------------------------------------------------------- API


@capability(
    "train.derive",
    summary="Derive hyperparameters from dataset statistics, with reasons",
    web_route="/api/v1/train/derive",
    web_methods=("POST",),
    cli=None,
    not_cli_because=(
        "'horos train' derives automatically and records the plan in run.json; "
        "a standalone preview command ships with the training UI (E5-T8)."
    ),
)
def derive_hyperparameters(
    project: Project, config: TrainRunConfig | None = None
) -> HyperparameterPlan:
    """Rule-based plan (E5-T1): dataset statistics + model metadata + memory
    probe in, values with reasons out. Values set on `config` are honored as
    user overrides (E5-T2). The UI shows this plan before a run starts (E5-S2)."""
    from horos.backends.memory import probe_memory

    config = config or TrainRunConfig()
    try:
        model_info = get_model_info(config.model)
    except UnknownModelError:
        model_info = None  # testing backends: derive without model metadata
    memory = probe_memory(config.device.partition(":")[0] if config.device else None)
    notes: list[str] = []
    if config.categories is not None:
        # the rules must see the data this run will actually train on
        from horos.core.stats import compute_stats

        full = _in_a_set(subset_dataset(
            project.to_dataset(), image_ids=config.image_ids, confirmed_only=True
        ))
        filtered = filter_dataset_categories(
            full, config.categories, include_background=config.include_background
        )
        stats = compute_stats(filtered)
        background = len(full.images) - len({a.image_id for a in filtered.annotations})
        if background:
            notes.append(
                f"{background} of {len(full.images)} images contain none of the "
                f"selected classes {sorted(config.categories)}: "
                + (
                    f"kept as background negatives (include_background=True) — the "
                    f"image-count rules see all {stats.num_images} images"
                    if config.include_background
                    else f"excluded from this run (include_background=False) — the "
                    f"image-count rules see the remaining {stats.num_images} images"
                )
            )
    else:
        stats = dataset_stats(project, image_ids=config.image_ids, confirmed_only=True)
    plan = derive_plan(
        stats,
        model=config.model,
        model_info=model_info,
        memory=memory,
        warm_start=bool(config.init_from),
        overrides={
            # any derived knob the user set in `extra` (grad_accum_steps,
            # warmup_epochs, early_stopping_* ...) is an override too: the plan
            # shows it as such instead of a derived value the backend would
            # then silently ignore
            **{k: v for k, v in config.extra.items() if k not in _CONFIG_KNOBS},
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "resolution": config.resolution,
            "lr": config.lr,
            "mosaic_ratio": config.mosaic_ratio,
        },
    )
    plan.notes = [*notes, *plan.notes]
    return plan


@capability(
    "train.start",
    summary="Start a training run in a dedicated worker subprocess",
    web_route="/api/v1/train",
    web_methods=("POST",),
    cli="train",
)
def start_training(project: Project, config: TrainRunConfig | None = None) -> RunRecord:
    """Export the dataset into a new run directory and spawn the worker.

    Returns immediately with the run's record; poll with `training_status`.
    One run TRAINS at a time; starting while one is active queues the new run
    (state "queued") — it is promoted when the active run reaches a terminal
    state. The dataset snapshot and hyperparameter plan are fixed at enqueue
    time, so what a queued run will train on is already decided and visible.
    """
    config = config or TrainRunConfig()
    ensure_supported("training")

    if config.entrypoint_override is None:
        info = get_model_info(config.model)
        if info.requires_acknowledgement and not config.acknowledge_non_apache:
            raise LicenseError(
                f"Model '{config.model}' is licensed under {info.weights_license}, "
                f"not Apache 2.0 (see {info.license_url}). Pass "
                f"acknowledge_non_apache=True if you have reviewed and accept it."
            )

    busy = any(r.state in ACTIVE_STATES for r in list_runs(project))

    # pending pre-labels are unreviewed machine output — never ground truth;
    # a labeled photo that somehow has no set yet gets one first
    project.assign_splits()
    dataset = _in_a_set(subset_dataset(
        project.to_dataset(), image_ids=config.image_ids, confirmed_only=True
    ))
    if config.categories is not None:
        dataset = filter_dataset_categories(
            dataset, config.categories, include_background=config.include_background
        )
    # allocation follows the project's split assignment (the Dataset page):
    # train trains, valid validates, test stays untouched for later evaluation
    split_counts = {
        split: len(dataset.images_in_split(split))
        for split in ("train", "valid", "test")
    }
    train_count, valid_count = split_counts["train"], split_counts["valid"]
    if train_count == 0 or valid_count == 0:
        empty = [s for s in ("train", "valid") if split_counts[s] == 0]
        if config.categories is not None and not config.include_background:
            # the class selection emptied the split: say so, with the two
            # ways out (a wider selection or keeping background images)
            raise ProjectError(
                f"The selected categories {config.categories} have no annotated "
                f"images in the {' and '.join(empty)} split (found "
                f"train={train_count}, valid={valid_count}). Select more classes, "
                f"re-split on the Dataset page, or set include_background=True to "
                f"keep images without those classes as background."
            )
        raise ProjectError(
            f"Training needs a non-empty train and valid split (found "
            f"train={train_count}, valid={valid_count}). Re-split on the "
            f"Dataset page (or resplit() / 'horos split') first."
        )
    if config.categories is not None:
        train_ids = {i.id for i in dataset.images_in_split("train")}
        if not any(a.image_id in train_ids for a in dataset.annotations):
            raise ProjectError(
                f"The selected categories {config.categories} have no "
                f"annotations in the train split — nothing to learn from."
            )

    if config.init_from:
        if config.resume_from:
            raise ProjectError("Give either resume_from or init_from, not both")
        if not Path(config.init_from).is_file():
            raise ProjectError(f"init_from checkpoint not found: {config.init_from}")
    if config.resume_from:
        # the checkpoint's class head has a fixed shape: resuming with a
        # different class set fails deep inside the backend with a raw
        # state_dict size-mismatch — refuse it here with the actual fix
        source_names = _snapshot_class_names(Path(config.resume_from))
        new_names = sorted({c.name for c in dataset.categories})
        if source_names is not None and source_names != new_names:
            raise ProjectError(
                f"Cannot resume: the checkpoint was trained on classes "
                f"{source_names}, but this run would train on {new_names}. "
                f"The class head's shape is fixed by the checkpoint — select "
                f"exactly the source run's classes, or start a fresh run "
                f"(without resume) to train on the new class set."
            )

    # Fill every unset hyperparameter from the rule-based plan (E5-T1) and
    # hand the worker a fully resolved config; the plan (values + reasons)
    # goes into the run metadata so the "why" survives with the run (E5-S2).
    plan = derive_hyperparameters(project, config)
    spec_values = plan.spec_fields()
    resolved = config.model_copy(
        update={
            "epochs": spec_values["epochs"],
            "batch_size": spec_values["batch_size"],
            "resolution": spec_values.get("resolution", config.resolution),
            "mosaic_ratio": plan.values.get("mosaic_ratio", config.mosaic_ratio),
            # derived backend knobs ride in extra; the user's own extra wins
            "extra": {**plan.extra_fields(), **config.extra},
        }
    )

    if config.resume_from:
        _check_resume_epochs(config.resume_from, resolved.epochs)

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = runs_root(project) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # The run keeps the exact data it trained on — reproducible by design,
    # at the cost of copying images per run.
    export_dataset(
        project,
        run_dir / "dataset",
        format="coco",
        categories=config.categories,
        include_background=config.include_background,
        image_ids=config.image_ids,
        confirmed_only=True,
    )

    # mosaic composites are baked into the train snapshot here, before the
    # worker sees it — backend-neutral by construction (see core/mosaic.py)
    mosaic_ratio = float(plan.api_fields().get("mosaic_ratio") or 0.0)
    if mosaic_ratio > 0:
        from horos.core.mosaic import synthesize_mosaics

        synthesize_mosaics(
            run_dir / "dataset" / "train",
            count=max(1, round(train_count * mosaic_ratio)),
            seed=config.seed if config.seed is not None else 42,
        )

    (run_dir / _CONFIG_JSON).write_text(resolved.model_dump_json(indent=2), "utf-8")
    record = RunRecord(
        run_id=run_id,
        model=config.model,
        state="queued" if busy else "pending",
        created_at=datetime.now(timezone.utc).isoformat(),
        config=resolved.model_dump(exclude={"entrypoint_override"}),
        hparams=[*plan.derivations, _criterion_entry(config.checkpoint_criterion)],
        hparam_notes=plan.notes,
        device=config.device,
        dataset_images=len(dataset.images),
        dataset_splits=split_counts,
        dataset_classes=sorted({c.name for c in dataset.categories}),
        dataset_fingerprint=fingerprint_dataset(dataset),
    )
    write_record(run_dir, record)

    if busy:
        logger.info("training run %s queued behind the active run", run_id)
        return record
    # claim our own run so a concurrent advance_queue can never double-spawn it
    (run_dir / _SPAWN_CLAIM).touch()
    return _spawn_worker(run_dir, record)


@capability(
    "train.status",
    summary="Poll a training run: state plus events after a given index",
    web_route="/api/v1/train/runs/<run_id>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos train' runs in the foreground and prints events directly.",
)
def training_status(project: Project, run_id: str, *, after: int = 0) -> TrainStatus:
    # polling is also the queue's fallback heartbeat: if the finished worker's
    # own exit hook missed its chance (crash, kill -9), the next status call
    # promotes the oldest queued run
    advance_queue(runs_root(project))
    run_dir = _run_dir(project, run_id)
    record = _reconcile(run_dir, read_record(run_dir))
    events, num_events = _read_events(run_dir, after)
    inherited: list[dict[str, Any]] = []
    chain: list[str] = []
    if after == 0:
        inherited, chain = _inherited_metrics(project, record, events)
    return TrainStatus(
        run=record, events=events, num_events=num_events,
        inherited_events=inherited, resumed_from=chain,
    )


_MAX_RESUME_CHAIN = 12


def _inherited_metrics(
    project: Project, record: RunRecord, own_events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Metrics events of the resume chain (resume_from points into another
    run's checkpoints/), oldest run first, each cut at the epoch where the
    next run took over. A resumed run's own log starts at the checkpoint's
    epoch, so without this the training page showed a two-point curve."""
    root = runs_root(project).resolve()
    first_own = min(
        (e.get("step") for e in own_events if e.get("type") == "metrics"
         and isinstance(e.get("step"), int | float)),
        default=None,
    )
    chain: list[str] = []
    collected: list[list[dict[str, Any]]] = []
    resume_from = (record.config or {}).get("resume_from")
    cutoff = first_own
    while resume_from and len(chain) < _MAX_RESUME_CHAIN:
        source_dir = Path(resume_from).parent.parent
        try:
            resolved = source_dir.resolve()
        except OSError:
            break
        if resolved.parent != root or not (source_dir / _RUN_JSON).is_file():
            break
        source_id = resolved.name
        if source_id in chain or source_id == record.run_id:
            break
        chain.append(source_id)
        events, _ = _read_events(source_dir)
        metrics = [e for e in events if e.get("type") == "metrics"]
        if cutoff is not None:
            metrics = [e for e in metrics if (e.get("step") or 0) < cutoff]
        collected.append(metrics)
        cutoff = min(
            (e.get("step") for e in metrics if isinstance(e.get("step"), int | float)),
            default=cutoff,
        )
        try:
            resume_from = (read_record(source_dir).config or {}).get("resume_from")
        except (OSError, ValueError):
            break
    inherited = [e for group in reversed(collected) for e in group]
    return inherited, chain


@capability(
    "train.stop",
    summary="Stop a running training; checkpoints from finished epochs remain",
    web_route="/api/v1/train/runs/<run_id>/stop",
    web_methods=("POST",),
    cli=None,
    not_cli_because="'horos train' runs in the foreground; Ctrl+C stops it.",
)
def stop_training(project: Project, run_id: str) -> bool:
    """Best weights so far survive (E5-S4): rfdetr checkpoints every epoch, so
    terminating the worker only loses the epoch in flight."""
    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    if record.state == "queued":  # never spawned: settle directly
        (run_dir / _STOP_FLAG).touch()
        record.state = "stopped"
        write_record(run_dir, record)
        return True
    if record.state not in ACTIVE_STATES:
        return False
    (run_dir / _STOP_FLAG).touch()
    if record.pid is not None and _worker_alive(record):
        try:
            os.kill(record.pid, signal.SIGTERM)
        except OSError:  # already gone between the check and the kill
            pass
    return True


@capability(
    "train.delete",
    summary="Delete a training run and everything it stored",
    web_route="/api/v1/train/runs/<run_id>",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="Run housekeeping is a UI concern; 'rm -r runs/<id>' works too.",
)
def delete_run(project: Project, run_id: str) -> bool:
    """Remove runs/<id> — checkpoints, dataset snapshot, events, eval reports.
    An active run must be stopped first; deletion is permanent."""

    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    # queued runs have no worker and are always deletable (that's how a
    # queued run is cancelled or has its parameters redone)
    if record.state in ACTIVE_STATES and _worker_alive(record):
        raise ProjectError(
            f"Run {run_id} is still {record.state} — stop it before deleting."
        )
    _PROCESSES.pop(run_id, None)
    rmtree_retry(run_dir)
    logger.info("deleted training run %s", run_id)
    return True


@capability(
    "train.update",
    summary="Edit a queued run's hyperparameters in place before it starts",
    web_route="/api/v1/train/runs/<run_id>",
    web_methods=("PATCH",),
    cli=None,
    not_cli_because="Queue editing is a UI concern; delete and re-run 'horos train'.",
)
def update_queued_run(
    project: Project, run_id: str, updates: dict[str, Any]
) -> RunRecord:
    """Change a QUEUED run's knobs without losing its place in the queue.

    Editable: epochs, batch_size, resolution, lr (None = back to derived),
    seed, and checkpoint_criterion. The model, class selection, and dataset
    snapshot are fixed at enqueue time — change those by deleting the queued
    run and starting a new one. The plan is re-derived so dependent values
    (e.g. grad accumulation after a batch-size edit) stay consistent.
    """
    allowed = {*_EDITABLE_KNOBS, "seed", "checkpoint_criterion"}
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise ProjectError(
            f"Not editable on a queued run: {unknown}. Editable fields: "
            f"{sorted(allowed)} — for anything else, delete the queued run "
            f"and start a new one."
        )
    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    if record.state != "queued" or (run_dir / _SPAWN_CLAIM).is_file():
        raise ProjectError(
            f"Run {run_id} is {record.state} — only queued runs can be "
            f"edited. Its hyperparameters are already in use."
        )

    stored = TrainRunConfig.model_validate_json(
        (run_dir / _CONFIG_JSON).read_text("utf-8")
    )
    # config.json holds RESOLVED values; the hparams trail remembers which of
    # them were user overrides. Rebuild the request so untouched derived knobs
    # go back to None and re-derive cleanly against the new edits.
    overridden = {h.name for h in record.hparams if h.overridden}
    base = {
        knob: (getattr(stored, knob) if knob in overridden else None)
        for knob in _CONFIG_KNOBS
    }
    derived_names = {h.name for h in record.hparams}
    # keep the user's own extras AND the derived knobs they overrode through
    # extra (patience, scheduler...); drop only values the plan derived itself
    user_extra = {
        k: v for k, v in stored.extra.items()
        if k not in derived_names or k in overridden
    }
    new_config = stored.model_copy(update={**base, **updates, "extra": user_extra})
    # updates arrive as a raw dict and model_copy performs NO validation — a
    # fractional epochs value would land in config.json and only blow up when
    # the worker boots. Validate the merged config here, as a 400 with a reason.
    from pydantic import ValidationError

    try:
        new_config = TrainRunConfig.model_validate(new_config.model_dump(warnings=False))
    except ValidationError as exc:
        raise ProjectError(f"Invalid hyperparameters: {exc}") from exc

    plan = derive_hyperparameters(project, new_config)
    spec_values = plan.spec_fields()
    resolved = new_config.model_copy(
        update={
            "epochs": spec_values["epochs"],
            "batch_size": spec_values["batch_size"],
            "resolution": spec_values.get("resolution", new_config.resolution),
            "extra": {**plan.extra_fields(), **user_extra},
        }
    )
    if new_config.resume_from:
        _check_resume_epochs(new_config.resume_from, resolved.epochs)

    (run_dir / _CONFIG_JSON).write_text(resolved.model_dump_json(indent=2), "utf-8")
    record.config = resolved.model_dump(exclude={"entrypoint_override"})
    record.hparams = [
        *plan.derivations,
        _criterion_entry(new_config.checkpoint_criterion),
    ]
    record.hparam_notes = plan.notes
    write_record(run_dir, record)
    logger.info("queued run %s updated", run_id)
    return record


@capability(
    "train.runs",
    summary="List this project's training runs, newest first",
    web_route="/api/v1/train/runs",
    web_methods=("GET",),
    cli="models",
)
def list_runs(project: Project, *, advance: bool = False) -> list[RunRecord]:
    root = runs_root(project)
    if not root.is_dir():
        return []
    if advance:  # the UI's history refresh doubles as a queue heartbeat
        advance_queue(root)
    records = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if (run_dir / _RUN_JSON).is_file():
            records.append(_reconcile(run_dir, read_record(run_dir)))
    return records
