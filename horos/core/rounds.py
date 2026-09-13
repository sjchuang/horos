"""Active-learning rounds: data model, state machine and on-disk storage (E10-T1).

One round is one pass through the loop: select a batch → label it → train →
review. Rounds are first-class project objects stored under

    <root>/rounds/<number>/round.json

so "which round bought the biggest metric gain" is answerable later (E10-S5)
and every pick keeps the strategy, score and reason it was chosen with
(E10-S8). No backend or ML import belongs here (R1); the API layer fills the
records, this module only defines and persists them.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from horos.core.fsutil import atomic_write_text
from horos.core.project import Project
from horos.errors import ProjectError

ROUNDS_DIR = "rounds"
ROUND_JSON = "round.json"

RoundState = Literal["selecting", "labeling", "training", "reviewing", "closed"]

#: How a batch was chosen. "auto" is resolved to one of these before a round
#: is stored — the record always says what actually happened: "pal" is the
#: model-based acquisition (E10-T5), "diversity" the embedding cold start
#: (E10-T4), "random" the explicit fallback when no embedding model loads.
SelectionStrategy = Literal["pal", "diversity", "random"]

#: legal transitions; anything else is a programming error, not a user error
_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "selecting": ("labeling", "closed"),
    "labeling": ("training", "closed"),
    "training": ("reviewing", "labeling", "closed"),  # back to labeling when training fails
    "reviewing": ("closed",),
    "closed": (),
}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class PickedImage(BaseModel):
    """One image chosen for a round, with the evidence for choosing it."""

    image_id: int
    #: higher = more worth labeling; each strategy documents its scale in `reason`
    score: float
    reason: str
    #: annotator this image is handed to (E10-T10); None = anyone
    assigned_to: str | None = None


class SelectionRecord(BaseModel):
    strategy: SelectionStrategy
    #: the count the user asked for (already resolved from a percentage)
    requested: int
    requested_percent: float | None = None
    #: unlabeled images available when the round was selected
    pool_size: int
    embedding_model: str | None = None
    #: what produced the uncertainty scores: a run id, or a zero-shot model key
    scorer: str | None = None
    picks: list[PickedImage] = Field(default_factory=list)
    #: why this strategy (and not another) was used — surfaced in the UI
    notes: list[str] = Field(default_factory=list)

    @property
    def image_ids(self) -> list[int]:
        return [p.image_id for p in self.picks]


class LoopRound(BaseModel):
    number: int = Field(ge=1)
    state: RoundState = "selecting"
    created_at: str = Field(default_factory=_now)
    closed_at: str | None = None
    #: labeled images in the project when the round started (E10-S5: labels spent)
    labeled_before: int = 0
    #: labeled images when the round reached review / was closed; None while open
    labeled_after: int | None = None
    selection: SelectionRecord | None = None
    train_run_id: str | None = None
    #: headline validation metrics of this round's model, filled in at review
    metrics: dict[str, float] = Field(default_factory=dict)
    #: what pre-annotated the round's images (E10-T7): scorer, threshold,
    #: images and pending annotations written; empty when nothing could
    preannotation: dict[str, Any] = Field(default_factory=dict)
    #: how this round trained (E10-T8): model, labeled images in the
    #: snapshot, the validation lock, and an error when the run failed
    training: dict[str, Any] = Field(default_factory=dict)

    @property
    def image_ids(self) -> list[int]:
        return self.selection.image_ids if self.selection else []

    def advance(self, state: RoundState) -> LoopRound:
        """Return a copy in the new state; refuses illegal transitions."""
        allowed = _TRANSITIONS[self.state]
        if state not in allowed:
            raise ProjectError(
                f"Round {self.number} cannot go from '{self.state}' to '{state}' "
                f"(allowed: {', '.join(allowed) or 'none'})"
            )
        update: dict = {"state": state}
        if state == "closed":
            update["closed_at"] = _now()
        return self.model_copy(update=update)


# ----------------------------------------------------------------- storage


def rounds_dir(project: Project) -> Path:
    return project.root / ROUNDS_DIR


def round_dir(project: Project, number: int) -> Path:
    return rounds_dir(project) / str(number)


def _round_path(project: Project, number: int) -> Path:
    return round_dir(project, number) / ROUND_JSON


def list_rounds(project: Project) -> list[LoopRound]:
    """All rounds, oldest first. A project without a rounds/ dir has none."""
    base = rounds_dir(project)
    if not base.is_dir():
        return []
    rounds: list[LoopRound] = []
    for child in base.iterdir():
        if not child.name.isdigit() or not (child / ROUND_JSON).is_file():
            continue
        rounds.append(load_round(project, int(child.name)))
    rounds.sort(key=lambda r: r.number)
    return rounds


def load_round(project: Project, number: int) -> LoopRound:
    path = _round_path(project, number)
    if not path.is_file():
        raise ProjectError(f"No round {number} in project {project.root}")
    try:
        return LoopRound.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProjectError(f"Corrupt round record at {path}: {exc}") from exc


def save_round(project: Project, record: LoopRound) -> LoopRound:
    path = _round_path(project, record.number)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, record.model_dump_json(indent=2))
    return record


def current_round(project: Project) -> LoopRound | None:
    """The one round that is not closed, if any. Rounds are strictly
    sequential: a new one can only start once the previous is closed."""
    open_rounds = [r for r in list_rounds(project) if r.state != "closed"]
    return open_rounds[-1] if open_rounds else None


def create_round(project: Project, *, labeled_before: int) -> LoopRound:
    """Open the next round. Refuses while another round is still open —
    the loop has exactly one active step at a time (E10-T14)."""
    active = current_round(project)
    if active is not None:
        raise ProjectError(
            f"Round {active.number} is still '{active.state}'; close it before "
            f"starting a new round."
        )
    existing = list_rounds(project)
    number = existing[-1].number + 1 if existing else 1
    return save_round(project, LoopRound(number=number, labeled_before=labeled_before))
