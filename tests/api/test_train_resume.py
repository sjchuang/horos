"""E5-T5: checkpoint management and resuming training (E5-S6)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status
from horos.errors import ProjectError

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _finish(project, run_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = training_status(project, run_id)
        if status.run.state not in ("pending", "running"):
            return status
        time.sleep(0.2)
    pytest.fail(f"run {run_id} still active after {timeout}s")


def test_resume_from_a_previous_runs_checkpoint(project):
    first = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=2))
    first_status = _finish(project, first.run_id)
    checkpoint = first_status.run.checkpoint
    assert checkpoint and Path(checkpoint).is_file()

    second = start_training(  # TOTAL epochs must exceed the source's 2
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=4, resume_from=checkpoint),
    )
    second_status = _finish(project, second.run_id)
    assert second_status.run.state == "completed"

    # the checkpoint path reached the backend's TrainSpec intact
    completed = [e for e in second_status.events if e["type"] == "completed"][-1]
    assert completed["result"]["resumed_from"] == checkpoint

    # and it is recorded in the run's own config for reproducibility
    config = json.loads(
        (project.root / "runs" / second.run_id / "config.json").read_text("utf-8")
    )
    assert config["resume_from"] == checkpoint
    assert second_status.run.config["resume_from"] == checkpoint

    # the training page draws the whole story: the first run's epochs come
    # back as inherited events, cut where the resumed run took over
    assert second_status.resumed_from == [first.run_id]
    # the fake trainer restarts its epoch count at 1, so nothing of the first
    # run lies before the resumed run's first epoch: no overlap is inherited
    assert second_status.inherited_events == []
    assert training_status(project, second.run_id, after=1).inherited_events == []
    assert first_status.inherited_events == [] and first_status.resumed_from == []

    # a real trainer continues the epoch count (rfdetr restores epoch 29 and
    # logs 29, 30, …): the parent's earlier epochs are handed over, cut where
    # the resumed run took over
    from horos.api.train import _inherited_metrics

    own = [{"type": "metrics", "step": 2, "metrics": {"loss": 0.1}}]
    inherited, chain = _inherited_metrics(project, second_status.run, own)
    assert chain == [first.run_id]
    assert [e["step"] for e in inherited] == [1]
    assert _inherited_metrics(project, second_status.run, []) == (
        [e for e in first_status.events if e["type"] == "metrics"], [first.run_id]
    )


def test_each_run_keeps_its_own_checkpoints(project):
    first = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    _finish(project, first.run_id)
    second = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    _finish(project, second.run_id)

    first_ckpt = project.root / "runs" / first.run_id / "checkpoints" / "best.fake"
    second_ckpt = project.root / "runs" / second.run_id / "checkpoints" / "best.fake"
    assert first_ckpt.is_file() and second_ckpt.is_file()
    assert first_ckpt != second_ckpt  # resuming can never overwrite the source run


def test_resume_with_a_different_class_set_is_refused(project):
    """The checkpoint's class head is shape-fixed: resuming with different
    classes fails deep in the backend with a raw state_dict size mismatch —
    horos must refuse it up front with the actual fix."""
    # include_background: the sample's valid split has no forklift image, and
    # the default (drop background images) would leave nothing to validate on
    source = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=1,
                       categories=["forklift"], include_background=True),
    )
    checkpoint = _finish(project, source.run_id).run.checkpoint

    with pytest.raises(ProjectError, match="Cannot resume.*forklift"):
        start_training(  # default = ALL classes ≠ the source run's one class
            project,
            TrainRunConfig(entrypoint_override=FAKE, epochs=2,
                           resume_from=checkpoint),
        )

    # matching selection resumes fine
    resumed = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=2, categories=["forklift"],
                       include_background=True, resume_from=checkpoint),
    )
    assert _finish(project, resumed.run_id).run.state == "completed"


def test_full_state_checkpoint_is_surfaced_for_resume(project):
    """rfdetr's last.ckpt carries optimizer + LR schedule; resuming from the
    best-weights .pth restarts them cold and the loss spikes. The record must
    point the UI at the full-state file when it exists."""
    record = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    status = _finish(project, record.run_id)
    assert status.run.resume_checkpoint is None  # the fake writes no last.ckpt

    ckpt_dir = project.root / "runs" / record.run_id / "checkpoints"
    (ckpt_dir / "last.ckpt").write_bytes(b"full-state")
    from horos.api.train import training_status

    refreshed = training_status(project, record.run_id).run
    assert refreshed.resume_checkpoint.endswith("last.ckpt")


def test_resume_total_epochs_must_exceed_completed(project):
    """epochs is the TOTAL count and the trainer restores the checkpoint's
    epoch: a total at or below what is done raises a raw
    MisconfigurationException in the backend — refuse it up front."""
    source = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=3))
    checkpoint = _finish(project, source.run_id).run.checkpoint

    for epochs in (2, 3):  # below and exactly-equal both leave nothing to train
        with pytest.raises(ProjectError, match="already completed 3 epochs"):
            start_training(
                project,
                TrainRunConfig(entrypoint_override=FAKE, epochs=epochs,
                               resume_from=checkpoint),
            )

    resumed = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=4, resume_from=checkpoint),
    )
    assert _finish(project, resumed.run_id).run.state == "completed"


def test_completed_epochs_are_recorded_and_backfilled(project):
    from horos.api.train import _run_dir, read_record, training_status, write_record

    record = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=3))
    status = _finish(project, record.run_id)
    assert status.run.epochs_completed == 3

    run_dir = _run_dir(project, record.run_id)
    stored = read_record(run_dir)
    stored.epochs_completed = None  # simulate a pre-field run.json
    write_record(run_dir, stored)
    assert training_status(project, record.run_id).run.epochs_completed == 3


def test_download_progress_events_do_not_count_as_epochs(project):
    """Weight-download progress carries BYTES in `current`; counting it as
    epochs would report hundreds of millions completed and wreck the
    resume-epochs guard."""
    from horos.api.train import _run_dir, read_record, training_status, write_record

    record = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=3))
    _finish(project, record.run_id)
    run_dir = _run_dir(project, record.run_id)
    download_event = {
        "ts": time.time(),
        "run_id": None,
        "type": "progress",
        "current": 386_045_550,
        "total": 386_045_550,
        "phase": "downloading rf-detr-small.pth: 368 / 368 MB",
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(download_event) + "\n")

    stored = read_record(run_dir)
    stored.epochs_completed = None  # force reconciliation from the event log
    write_record(run_dir, stored)
    assert training_status(project, record.run_id).run.epochs_completed == 3
