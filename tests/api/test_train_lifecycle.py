"""E5-T3: training run lifecycle — start, poll, stop — through the real worker
subprocess, with a fake backend so no ML dependency is needed."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import (
    TrainRunConfig,
    list_runs,
    start_training,
    stop_training,
    training_status,
)
from horos.errors import LicenseError, ProjectError, UnknownModelError

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    """The worker subprocess must resolve the fake backend entrypoint."""
    existing = os.environ.get("PYTHONPATH", "")
    joined = str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", joined)


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    # the sample dataset has train (2 images) and valid (1 image) splits
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _wait_terminal(project, run_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = training_status(project, run_id)
        if status.run.state not in ("pending", "running"):
            return status
        time.sleep(0.2)
    pytest.fail(f"run {run_id} did not reach a terminal state within {timeout}s")


def _config(**overrides):
    defaults = dict(entrypoint_override=FAKE, epochs=2)
    defaults.update(overrides)
    return TrainRunConfig(**defaults)


def test_init_from_is_validated_before_anything_is_written(project, tmp_path):
    """E5-S6 warm start: a missing checkpoint or init_from together with
    resume_from is refused up front, with no run directory left behind."""
    with pytest.raises(ProjectError, match="init_from checkpoint not found"):
        start_training(project, _config(init_from=str(tmp_path / "missing.pth")))
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x")
    with pytest.raises(ProjectError, match="either resume_from or init_from"):
        start_training(project, _config(init_from=str(weights), resume_from=str(weights)))
    assert not list((project.root / "runs").glob("*"))
    record = start_training(project, _config(init_from=str(weights)))
    assert record.config["init_from"] == str(weights)
    _wait_terminal(project, record.run_id)


def test_run_completes_and_keeps_artifacts(project):
    record = start_training(project, _config())
    assert record.state == "running" and record.pid

    status = _wait_terminal(project, record.run_id)
    assert status.run.state == "completed"
    assert status.run.checkpoint and Path(status.run.checkpoint).is_file()
    assert status.run.error is None

    types = [e["type"] for e in status.events]
    assert types[0] == "started" and types[-1] == "completed"
    assert "progress" in types and "metrics" in types

    run_dir = project.root / "runs" / record.run_id
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "dataset" / "train" / "_annotations.coco.json").is_file()


def test_status_pagination(project):
    record = start_training(project, _config())
    status = _wait_terminal(project, record.run_id)
    total = status.num_events
    assert total >= 3
    tail = training_status(project, record.run_id, after=total - 1)
    assert len(tail.events) == 1 and tail.num_events == total


def test_second_start_queues_instead_of_refusing(project):
    """One run TRAINS at a time, but a second start is queued, not refused —
    the full queue behavior lives in tests/api/test_train_queue.py."""
    record = start_training(
        project, _config(epochs=60, extra={"sleep_per_epoch": 0.5})
    )
    try:
        queued = start_training(project, _config())
        assert queued.state == "queued" and queued.pid is None
        # cancel the queued run so nothing chains after run1 stops below
        assert stop_training(project, queued.run_id) is True
        assert training_status(project, queued.run_id).run.state == "stopped"
    finally:
        stop_training(project, record.run_id)
        _wait_terminal(project, record.run_id)


def test_stop_settles_to_stopped(project):
    record = start_training(
        project, _config(epochs=60, extra={"sleep_per_epoch": 0.5})
    )
    time.sleep(1.0)  # let the worker boot and start its first epoch
    assert stop_training(project, record.run_id) is True
    status = _wait_terminal(project, record.run_id)
    assert status.run.state == "stopped"
    # stopping an already-terminal run reports False, never restarts anything
    assert stop_training(project, record.run_id) is False


def test_failed_backend_reports_failure(project):
    record = start_training(project, _config(extra={"fail": True}))
    status = _wait_terminal(project, record.run_id)
    assert status.run.state == "failed"
    assert "simulated failure" in (status.run.error or "")
    assert status.events[-1]["type"] == "failed"


def test_list_runs_newest_first(project):
    first = start_training(project, _config())
    _wait_terminal(project, first.run_id)
    time.sleep(1.1)  # run ids embed a second-resolution timestamp
    second = start_training(project, _config())
    _wait_terminal(project, second.run_id)
    runs = list_runs(project)
    assert [r.run_id for r in runs][:2] == [second.run_id, first.run_id]


def test_requires_train_and_valid_splits(tmp_path):
    proj = create_project(tmp_path / "empty")
    with pytest.raises(ProjectError, match="train and valid split"):
        start_training(proj, _config())


def test_unknown_model_is_rejected(project):
    with pytest.raises(UnknownModelError):
        start_training(project, TrainRunConfig(model="does-not-exist"))


def test_non_apache_model_needs_acknowledgement(project):
    with pytest.raises(LicenseError, match="PML"):
        start_training(project, TrainRunConfig(model="rfdetr-xl"))


def test_delete_run_removes_everything(project):
    from horos.api.train import delete_run

    record = start_training(project, _config())
    _wait_terminal(project, record.run_id)
    run_dir = project.root / "runs" / record.run_id
    assert run_dir.is_dir()

    assert delete_run(project, record.run_id) is True
    assert not run_dir.exists()
    with pytest.raises(ProjectError, match="No such training run"):
        training_status(project, record.run_id)
    assert record.run_id not in [r.run_id for r in list_runs(project)]


def test_delete_refuses_an_active_run(project):
    from horos.api.train import delete_run

    record = start_training(
        project, _config(epochs=60, extra={"sleep_per_epoch": 0.5})
    )
    try:
        with pytest.raises(ProjectError, match="stop it before deleting"):
            delete_run(project, record.run_id)
        assert (project.root / "runs" / record.run_id).is_dir()
    finally:
        stop_training(project, record.run_id)
        _wait_terminal(project, record.run_id)


def test_checkpoint_criterion_is_recorded_in_hparams(project):
    """The criterion is part of the run's hyperparameter trail (with a
    reason), so "why is THIS the best checkpoint" survives with the run."""
    record = start_training(project, _config(checkpoint_criterion="smoothed_map"))
    try:
        entry = next(h for h in record.hparams if h.name == "checkpoint_criterion")
        assert entry.value == "smoothed_map"
        assert entry.overridden is True and entry.reason
    finally:
        _wait_terminal(project, record.run_id)

    default = start_training(project, _config())
    entry = next(h for h in default.hparams if h.name == "checkpoint_criterion")
    assert entry.value == "map" and entry.overridden is False
    _wait_terminal(project, default.run_id)


def test_mosaic_snapshots_are_opt_in(project):
    """E5: mosaics are baked into the train snapshot only when the user opts
    in with mosaic_ratio > 0; the derived default is 0 (A/B-tested off)."""
    import json

    record = start_training(project, _config())
    try:
        ann = (
            project.root / "runs" / record.run_id
            / "dataset" / "train" / "_annotations.coco.json"
        )
        data = json.loads(ann.read_text(encoding="utf-8"))
        assert not any(
            i["file_name"].startswith("mosaic_") for i in data["images"]
        )
        entry = next(h for h in record.hparams if h.name == "mosaic_ratio")
        assert entry.value == 0.0 and entry.overridden is False and entry.reason
    finally:
        _wait_terminal(project, record.run_id)

    opted_in = start_training(project, _config(mosaic_ratio=0.5))
    try:
        ann = (
            project.root / "runs" / opted_in.run_id
            / "dataset" / "train" / "_annotations.coco.json"
        )
        data = json.loads(ann.read_text(encoding="utf-8"))
        mosaics = [
            i for i in data["images"] if i["file_name"].startswith("mosaic_")
        ]
        # 2 train images × ratio 0.5 → 1 composite
        assert len(mosaics) == 1
        assert (ann.parent / mosaics[0]["file_name"]).is_file()
        entry = next(h for h in opted_in.hparams if h.name == "mosaic_ratio")
        assert entry.overridden is True
    finally:
        _wait_terminal(project, opted_in.run_id)
