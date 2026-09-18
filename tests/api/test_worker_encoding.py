"""A worker's log never dies of the locale code page (R7).

The real failure this covers: on a Traditional Chinese Windows install the
training worker's stdout is worker.log, so its encoding is cp950, and rfdetr
prints its metrics table with box-drawing characters. The run died as the
first validation table was drawn, before any checkpoint existed, with
`'cp950' codec can't encode character '\u250f'`.

`PYTHONIOENCODING=cp950` is forced into the environment here so the test
fails the same way on any platform if the fix is removed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir
from helpers.fake_backend import TABLE_CORNER

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status
from horos.core.streams import child_env

TESTS_ROOT = Path(__file__).parent.parent
NOISY = "helpers.fake_backend:NoisyBackend"


@pytest.fixture(autouse=True)
def hostile_environment(monkeypatch):
    """A worker that inherits this environment unchanged cannot print a box."""
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else ""))
    monkeypatch.setenv("PYTHONIOENCODING", "cp950")


def test_child_env_overrides_a_hostile_ioencoding():
    env = child_env()
    assert env["PYTHONIOENCODING"] == "utf-8:backslashreplace"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONPATH"] == os.environ["PYTHONPATH"]  # rest is inherited
    assert child_env(PYTHONIOENCODING="cp437")["PYTHONIOENCODING"] == "cp437"


def test_training_survives_a_backend_printing_box_characters(tmp_path):
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(project, TrainRunConfig(entrypoint_override=NOISY, epochs=1))
    deadline = time.monotonic() + 60  # spawn interpreter startup is slow
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline, "noisy run never finished"
        time.sleep(0.2)

    status = training_status(project, record.run_id)
    assert status.run.state == "completed", status.run.error

    # and the characters are readable in the log, not mojibake or escapes
    log = (project.root / "runs" / record.run_id / "worker.log").read_text("utf-8")
    assert log.count(TABLE_CORNER) == 3  # stdout, stderr, spawned child
    assert "\U0001f4a1" in log
