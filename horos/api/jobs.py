"""Background jobs with polling (E3-T3 delivery mechanism).

Design decision (confirmed): long work runs on a background thread inside the
same process; progress is R4 events accumulated in memory and appended to a
JSONL file under <project>/jobs/, and the front-end polls. One running job at
a time — a second start is refused explicitly, never queued silently.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.project import Project
from horos.errors import ProjectError

if TYPE_CHECKING:
    from horos.backends.base import Event

logger = logging.getLogger(__name__)

__all__ = ["JobStatus", "RunningJob", "start_job", "job_status", "cancel_job", "running_job"]


class JobStatus(BaseModel):
    job_id: str
    kind: str
    state: str  # running | completed | failed | cancelled
    events: list[dict[str, Any]] = Field(default_factory=list)  # events[after:]
    num_events: int = 0


class RunningJob(BaseModel):
    """The one job in flight, so a reloaded page can pick its progress back up."""

    job_id: str
    kind: str
    num_events: int = 0


class _Job:
    def __init__(self, job_id: str, kind: str, project_root):
        self.job_id = job_id
        self.kind = kind
        self.state = "running"
        self.events: list[dict[str, Any]] = []
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.log_path = project_root / "jobs" / f"{job_id}.jsonl"


_REGISTRY: dict[str, _Job] = {}
_REGISTRY_LOCK = threading.Lock()


def _running_job() -> _Job | None:
    return next((j for j in _REGISTRY.values() if j.state == "running"), None)


def start_job(
    project: Project,
    kind: str,
    make_events: Callable[[threading.Event], Iterator[Event]],
) -> str:
    """Consume an R4 event stream on a background thread. Returns the job id.

    `make_events` receives the job's cancel flag so the stream can honor
    cancellation between work units. The stream owns its error handling (it
    must terminate with RunCompleted or RunFailed); the job state is derived
    from the terminal event.
    """
    from horos.backends.base import dump_event

    with _REGISTRY_LOCK:
        running = _running_job()
        if running is not None:
            raise ProjectError(
                f"A {running.kind} job ({running.job_id}) is already running. "
                f"Wait for it or cancel it first — horos runs one job at a time."
            )
        job = _Job(uuid.uuid4().hex[:12], kind, project.root)
        _REGISTRY[job.job_id] = job

    events = make_events(job.cancel)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        try:
            with job.log_path.open("a", encoding="utf-8") as log:
                for event in events:
                    line = dump_event(event)
                    log.write(line + "\n")
                    log.flush()
                    with job.lock:
                        job.events.append(event.model_dump(mode="json"))
                        if event.type == "completed":
                            job.state = (
                                "cancelled"
                                if event.result.get("cancelled")
                                else "completed"
                            )
                        elif event.type == "failed":
                            job.state = "failed"
        except Exception:  # noqa: BLE001 — a crashed stream must not hang "running"
            logger.exception("job %s crashed outside its event stream", job.job_id)
            with job.lock:
                job.state = "failed"
        finally:
            with job.lock:
                if job.state == "running":
                    job.state = "failed"

    threading.Thread(target=run, name=f"horos-job-{job.job_id}", daemon=True).start()
    return job.job_id


def _get(job_id: str) -> _Job:
    job = _REGISTRY.get(job_id)
    if job is None:
        raise ProjectError(f"No such job: {job_id}")
    return job


@capability(
    "jobs.status",
    summary="Poll a background job: state plus events after a given index",
    web_route="/api/v1/jobs/<job_id>",
    web_methods=("GET",),
    cli=None,
    not_cli_because="The CLI runs work in the foreground and prints events directly.",
)
def job_status(project: Project, job_id: str, *, after: int = 0) -> JobStatus:
    job = _get(job_id)
    with job.lock:
        return JobStatus(
            job_id=job.job_id,
            kind=job.kind,
            state=job.state,
            events=job.events[after:],
            num_events=len(job.events),
        )


@capability(
    "jobs.cancel",
    summary="Request cancellation of a running background job",
    web_route="/api/v1/jobs/<job_id>/cancel",
    web_methods=("POST",),
    cli=None,
    not_cli_because="The CLI runs work in the foreground; Ctrl+C cancels it.",
)
def cancel_job(project: Project, job_id: str) -> bool:
    """Signal the job's cancel event. The stream honors it between work units;
    already-finished jobs return False."""
    job = _get(job_id)
    if job.state != "running":
        return False
    job.cancel.set()
    return True


def running_job(kinds: tuple[str, ...] | None = None) -> RunningJob | None:
    """The running job (of one of `kinds`, when given) or None. Not a
    capability of its own: status endpoints embed it so a page that was
    reloaded mid-job finds the job again instead of offering to start another."""
    with _REGISTRY_LOCK:
        job = _running_job()
    if job is None or (kinds is not None and job.kind not in kinds):
        return None
    with job.lock:
        return RunningJob(job_id=job.job_id, kind=job.kind, num_events=len(job.events))


def cancel_event_for(job_id: str) -> threading.Event:
    return _get(job_id).cancel
