"""Upload with progress (E1-T10, R4): import_dataset/import_zip progress
callbacks, staged uploads, and the polled import job."""

from __future__ import annotations

import time
import zipfile

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.api.dataset import import_dataset, import_zip
from horos.api.jobs import job_status
from horos.api.project import create_project
from horos.api.uploads import (
    STALE_UPLOAD_SECONDS,
    _purge_stale,
    discard_upload,
    stage_upload,
    start_upload_import,
    upload_import_events,
)
from horos.errors import DatasetFormatError, ProjectError


def _zip_dir(src, zip_path):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src))
    return zip_path


def _wait(project, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = job_status(project, job_id)
        if status.state != "running":
            return status
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


# ------------------------------------------------------------ progress callbacks


def test_import_dataset_reports_every_phase_in_order(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    events = []
    import_dataset(project, coco_dir, progress=events.append)
    progress = [e for e in events if e.type == "progress"]
    phases = []
    for e in progress:
        if not phases or phases[-1] != e.phase:
            phases.append(e.phase)
    assert phases == [
        "reading annotations",
        "checking for duplicates",
        "copying images",
        "saving annotations",
        "assigning splits",
    ]
    # counted phases end on current == total
    for phase in ("checking for duplicates", "copying images", "saving annotations"):
        last = [e for e in progress if e.phase == phase][-1]
        assert last.total == 3 and last.current == 3
    # every event is an R4 pydantic object, JSONL-serialisable
    from horos.backends.base import dump_event

    assert all(dump_event(e) for e in events)


def test_flat_source_reports_the_split_assignment_phase(tmp_path):
    # labeled photos whose source named no split join one by stable hash
    coco_dir = write_sample_coco_dir(tmp_path / "coco", split_layout=False)
    project = create_project(tmp_path / "proj")
    events = []
    summary = import_dataset(project, coco_dir, progress=events.append)
    assert "assigning splits" in {e.phase for e in events if e.type == "progress"}
    assert any("stable hash" in w for w in summary.warnings)


def test_reader_warnings_surface_as_warning_events(tmp_path):
    import json

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    for ann_file in coco_dir.rglob("_annotations.coco.json"):
        data = json.loads(ann_file.read_text("utf-8"))
        data["categories"].append({"id": 99, "name": "unused", "supercategory": "none"})
        ann_file.write_text(json.dumps(data), "utf-8")
    project = create_project(tmp_path / "proj")
    events = []
    summary = import_dataset(project, coco_dir, progress=events.append)
    # the reader itself has no warnings here; the empty-category drop is an
    # import warning and stays in the summary (nothing is lost either way)
    assert any("empty categor" in w for w in summary.warnings)
    assert all(e.type in ("progress", "warning") for e in events)


def test_progress_is_throttled_for_large_datasets(tmp_path):
    src = tmp_path / "big"
    lines = []
    for i in range(400):
        make_image(src / "train" / f"{i:04d}.png", 8, 8)
        lines.append(f"{i:04d}.png")
    # a Darknet dir: one .txt per image, plus the labels file
    for i in range(400):
        (src / "train" / f"{i:04d}.txt").write_text("0 0.5 0.5 0.5 0.5\n", "utf-8")
    (src / "train" / "_darknet.labels").write_text("thing\n", "utf-8")
    project = create_project(tmp_path / "proj")
    events = []
    import_dataset(project, src, progress=events.append)
    copying = [e for e in events if e.type == "progress" and e.phase == "copying images"]
    assert 40 <= len(copying) <= 60  # ~50 ticks, not 400
    assert copying[-1].current == copying[-1].total == 400


def test_import_zip_reports_extraction_first(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    zip_path = _zip_dir(coco_dir, tmp_path / "ds.zip")
    project = create_project(tmp_path / "proj")
    events = []
    import_zip(project, zip_path, progress=events.append)
    progress = [e for e in events if e.type == "progress"]
    assert progress[0].phase == "extracting" and progress[0].message == "ds.zip"
    extracting = [e for e in progress if e.phase == "extracting"]
    assert extracting[-1].current == extracting[-1].total > 0


# ------------------------------------------------------------ staged uploads


@pytest.fixture
def project(tmp_path):
    return create_project(tmp_path / "proj")


@pytest.fixture
def sample_zip(tmp_path):
    return _zip_dir(write_sample_coco_dir(tmp_path / "coco"), tmp_path / "sample.zip")


def test_stage_upload_from_path_and_from_stream(project, sample_zip):
    staged = stage_upload(project, sample_zip)
    assert staged.file_name == "sample.zip" and staged.size_bytes == sample_zip.stat().st_size
    assert (project.root / "uploads" / staged.upload_id / "sample.zip").is_file()

    with sample_zip.open("rb") as fh:
        streamed = stage_upload(project, fh, file_name="../evil/../data.zip")
    assert streamed.file_name == "data.zip"  # only the base name is used
    assert (project.root / "uploads" / streamed.upload_id / "data.zip").is_file()


def test_stage_upload_rejects_non_zip(project, tmp_path):
    bogus = tmp_path / "bogus.zip"
    bogus.write_text("not a zip", "utf-8")
    with pytest.raises(DatasetFormatError, match="not a valid zip"):
        stage_upload(project, bogus)
    assert not any((project.root / "uploads").iterdir())


def test_import_job_streams_progress_and_ends_with_the_summary(project, sample_zip):
    staged = stage_upload(project, sample_zip)
    job_id = start_upload_import(project, staged.upload_id)
    status = _wait(project, job_id)
    assert status.state == "completed" and status.kind == "import"
    types = [e["type"] for e in status.events]
    assert types[0] == "started" and types[-1] == "completed"
    phases = {e["phase"] for e in status.events if e["type"] == "progress"}
    assert {"extracting", "reading annotations", "copying images", "saving annotations"} <= phases
    result = status.events[-1]["result"]
    assert result["format"] == "coco" and result["num_images"] == 3
    # the staged zip is gone once imported
    assert not (project.root / "uploads" / staged.upload_id).exists()


def test_conflict_keeps_the_zip_and_retries_with_a_policy(project, sample_zip, tmp_path):
    import_zip(project, sample_zip)
    variant = write_sample_coco_dir(tmp_path / "variant")
    make_image(variant / "train" / "a.png", 64, 48, color=(1, 2, 3))
    staged = stage_upload(project, _zip_dir(variant, tmp_path / "variant.zip"))

    status = _wait(project, start_upload_import(project, staged.upload_id))
    assert status.state == "failed"
    failed = status.events[-1]
    assert failed["error_code"] == "import_conflict"
    assert failed["details"]["conflicts"] == ["a.png"]
    assert failed["details"]["retryable"] is True
    assert failed["details"]["upload_id"] == staged.upload_id
    assert (project.root / "uploads" / staged.upload_id).is_dir()  # kept for the retry

    status = _wait(project, start_upload_import(project, staged.upload_id, on_conflict="overwrite"))
    assert status.state == "completed"
    assert status.events[-1]["result"]["overwritten"] == 1
    assert not (project.root / "uploads" / staged.upload_id).exists()


def test_class_names_prompt_flows_through_the_job(project, tmp_path):
    src = tmp_path / "darknet"
    make_image(src / "img1.png")
    (src / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", "utf-8")
    staged = stage_upload(project, _zip_dir(src, tmp_path / "dn.zip"))
    status = _wait(project, start_upload_import(project, staged.upload_id))
    failed = status.events[-1]
    assert status.state == "failed" and failed["error_code"] == "class_names_required"
    assert failed["details"]["default_names"] == ["0"] and failed["details"]["retryable"]

    status = _wait(
        project, start_upload_import(project, staged.upload_id, class_names=["helmet"])
    )
    assert status.state == "completed"
    assert status.events[-1]["result"]["instances_per_category"] == {"helmet": 1}


def test_unretryable_failure_deletes_the_zip(project, tmp_path):
    src = tmp_path / "junk"
    (src / "readme.txt").parent.mkdir(parents=True)
    (src / "readme.txt").write_text("no dataset here", "utf-8")
    staged = stage_upload(project, _zip_dir(src, tmp_path / "junk.zip"))
    status = _wait(project, start_upload_import(project, staged.upload_id))
    failed = status.events[-1]
    assert status.state == "failed" and failed["error_code"] == "dataset_format_error"
    assert failed["details"]["retryable"] is False
    assert not (project.root / "uploads" / staged.upload_id).exists()


def test_start_rejects_bad_policy_and_unknown_upload_synchronously(project, sample_zip):
    staged = stage_upload(project, sample_zip)
    with pytest.raises(ProjectError, match="on_conflict"):
        start_upload_import(project, staged.upload_id, on_conflict="maybe")
    with pytest.raises(ProjectError, match="No staged upload"):
        start_upload_import(project, "deadbeef0000")
    with pytest.raises(ProjectError, match="Invalid upload id"):
        start_upload_import(project, "../escape")
    assert discard_upload(project, staged.upload_id) is True


def test_discard_and_stale_purge(project, sample_zip):
    staged = stage_upload(project, sample_zip)
    assert discard_upload(project, staged.upload_id) is True
    assert discard_upload(project, staged.upload_id) is False

    old = stage_upload(project, sample_zip)
    fresh = stage_upload(project, sample_zip)
    purged = _purge_stale(project, now=time.time() + STALE_UPLOAD_SECONDS + 1)
    assert purged == 2
    assert not (project.root / "uploads" / old.upload_id).exists()
    assert not (project.root / "uploads" / fresh.upload_id).exists()


def test_events_stream_is_framed_started_to_terminal(project, sample_zip):
    staged = stage_upload(project, sample_zip)
    events = list(upload_import_events(project, staged.upload_id))
    assert events[0].type == "started" and events[0].config["upload_id"] == staged.upload_id
    assert events[-1].type == "completed"
    assert all(e.type in ("progress", "warning") for e in events[1:-1])


# ------------------------------------------------------------ loose photos (E1-T11)


def test_stage_photos_imports_them_unlabeled(project, tmp_path):
    from horos.api.uploads import stage_photos

    a = make_image(tmp_path / "p1.jpg", 64, 48)
    b = make_image(tmp_path / "p2.png", 32, 32)
    with b.open("rb") as stream:
        staged = stage_photos(project, [("p1.jpg", a), ("p2.png", stream)])
    assert staged.file_name == "2 photos" and staged.size_bytes > 0
    job_id = start_upload_import(project, staged.upload_id)
    status = _wait(project, job_id)
    assert status.state == "completed"
    started = status.events[0]
    assert started["type"] == "started" and started["config"]["file_name"] == "2 photos"
    result = status.events[-1]["result"]
    assert result["format"] == "images" and result["num_images"] == 2
    assert [r.split for r in project.list_images()] == [None, None]
    # the staged photos are gone after a successful import
    assert discard_upload(project, staged.upload_id) is False


def test_stage_photos_refuses_non_photos_duplicates_and_nothing(project, tmp_path):
    from horos.api.uploads import stage_photos

    note = tmp_path / "notes.txt"
    note.write_text("x", encoding="utf-8")
    photo = make_image(tmp_path / "p.jpg")
    with pytest.raises(DatasetFormatError, match="Not photos: notes.txt"):
        stage_photos(project, [("p.jpg", photo), ("notes.txt", note)])
    with pytest.raises(DatasetFormatError, match="appears twice"):
        stage_photos(project, [("p.jpg", photo), ("sub/p.jpg", photo)])
    with pytest.raises(DatasetFormatError, match="No photos"):
        stage_photos(project, [])
    # a refused upload leaves nothing staged behind
    assert not list((project.root / "uploads").glob("*/photos")) if (
        project.root / "uploads"
    ).is_dir() else True


def test_label_conflict_keeps_the_upload_and_retries_with_a_policy(project, sample_zip):
    from horos.core.dataset import Annotation

    # the project has the sample photos with their labels; one photo is then
    # relabeled by hand, so the same zip brings different labels for it
    import_zip(project, sample_zip)
    record = next(r for r in project.list_images() if r.file_name == "a.png")
    current = project.load_annotations(record.id)
    project.save_annotations(
        record.id,
        [Annotation(id=1, image_id=record.id, category_id=1, bbox=(9.0, 9.0, 3.0, 3.0))],
        expected_version=current.version,
    )
    staged = stage_upload(project, sample_zip)
    status = _wait(project, start_upload_import(project, staged.upload_id))
    assert status.state == "failed"
    failed = status.events[-1]
    assert failed["error_code"] == "label_conflict"
    assert failed["details"]["conflicts"] == ["a.png"]
    assert failed["details"]["retryable"] is True
    # the zip is still there for the retry
    status = _wait(
        project, start_upload_import(project, staged.upload_id, on_annotations="replace")
    )
    assert status.state == "completed"
    result = status.events[-1]["result"]
    assert result["annotations_replaced"] == 1 and result["images_matched"] == 1
    assert result["duplicates_skipped"] == 2  # the other two photos: same labels again
    assert len(project.load_annotations(record.id).annotations) == 2
    with pytest.raises(ProjectError, match="on_annotations must be one of"):
        start_upload_import(project, staged.upload_id, on_annotations="maybe")
