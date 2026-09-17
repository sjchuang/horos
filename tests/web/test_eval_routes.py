"""E6-T9 (first slice): inference and evaluation Web API endpoints."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.web.app import create_app

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"

pytest.importorskip("pycocotools", reason="training stack not installed")


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    from horos.api.evaluate import _reset_backend_cache

    _reset_backend_cache()


@pytest.fixture
def trained_client(tmp_path):
    from horos.api import create_project, import_dataset

    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    app = create_app(proj.root)
    app.testing = True
    client = app.test_client()

    response = client.post(
        "/api/v1/train", json={"entrypoint_override": FAKE, "epochs": 1}
    )
    run_id = response.get_json()["run_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        state = client.get(f"/api/v1/train/runs/{run_id}").get_json()["run"]["state"]
        if state not in ("pending", "running"):
            break
        time.sleep(0.2)
    assert state == "completed"
    return client, run_id, tmp_path


def test_infer_upload_roundtrip(trained_client):
    client, run_id, tmp_path = trained_client
    image_path = make_image(tmp_path / "probe.png", 64, 48)
    response = client.post(
        f"/api/v1/train/runs/{run_id}/infer",
        data={"file": (io.BytesIO(image_path.read_bytes()), "probe.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["image"] == "probe.png"  # temp server path never leaks
    assert payload["instances"][0]["score"] == 0.9


def test_infer_without_file_is_a_client_error(trained_client):
    client, run_id, _ = trained_client
    response = client.post(f"/api/v1/train/runs/{run_id}/infer", data={})
    assert response.status_code == 400
    assert "multipart" in response.get_json()["error"]["message"]


def test_evaluate_job_and_persisted_report(trained_client):
    client, run_id, _ = trained_client
    response = client.post(
        f"/api/v1/train/runs/{run_id}/evaluate", json={"split": "valid"}
    )
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if job["state"] != "running":
            break
        time.sleep(0.2)
    assert job["state"] == "completed"
    assert job["events"][-1]["result"]["split"] == "valid"

    report = client.get(f"/api/v1/train/runs/{run_id}/eval/valid")
    assert report.status_code == 200
    assert report.get_json()["num_images"] == 1


def test_evaluate_missing_split_fails_synchronously(trained_client):
    client, run_id, _ = trained_client
    response = client.post(
        f"/api/v1/train/runs/{run_id}/evaluate", json={"split": "nope"}
    )
    assert response.status_code == 400
    assert "no 'nope' split" in response.get_json()["error"]["message"]


def _gif_bytes(frames: int = 3) -> bytes:
    from PIL import Image

    images = [Image.new("RGB", (48, 32), (40 * i, 80, 120)) for i in range(frames)]
    buffer = io.BytesIO()
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:],
        duration=80, loop=0,
    )
    return buffer.getvalue()


def _wait_job(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("media job never finished")


def test_media_gif_upload_gallery_and_frame_serving(trained_client):
    client, run_id, _ = trained_client
    response = client.post(
        f"/api/v1/train/runs/{run_id}/media",
        data={"file": (io.BytesIO(_gif_bytes(3)), "clip.gif")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert _wait_job(client, body["job_id"])["state"] == "completed"

    listing = client.get(f"/api/v1/train/runs/{run_id}/media").get_json()
    assert len(listing) == 1 and listing[0]["media_id"] == body["media_id"]
    assert listing[0]["kind"] == "video" and listing[0]["num_frames"] == 3

    detail = client.get(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}"
    ).get_json()
    assert len(detail["frames"]) == 3
    assert detail["frames"][0]["instances"][0]["score"] == 0.9

    frame = client.get(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}/"
        f"{detail['frames'][0]['file_name']}"
    )
    assert frame.status_code == 200
    assert frame.content_type.startswith("image/jpeg")
    # consume and close the download like a browser does: an unread test
    # response keeps the frame's file handle open, and Windows then refuses
    # the delete below (WinError 32) however long the API retries
    assert frame.data[:2] == b"\xff\xd8"
    frame.close()

    deleted = client.delete(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}"
    )
    assert deleted.get_json()["deleted"] is True
    assert client.get(f"/api/v1/train/runs/{run_id}/media").get_json() == []


def test_media_upload_without_file_is_a_client_error(trained_client):
    client, run_id, _ = trained_client
    response = client.post(f"/api/v1/train/runs/{run_id}/media", data={})
    assert response.status_code == 400


def test_relative_project_root_still_serves_frame_files(tmp_path, monkeypatch):
    """`horos ui my-project` hands create_app a RELATIVE path; Flask's file
    helpers resolve relative directories against the package dir, not the cwd
    — every frame 404'd until create_app resolved the root absolute."""
    from horos.api import create_project, import_dataset

    monkeypatch.chdir(tmp_path)
    proj = create_project(Path("proj"))
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    app = create_app("proj")  # relative on purpose
    app.testing = True
    client = app.test_client()

    run_id = client.post(
        "/api/v1/train", json={"entrypoint_override": FAKE, "epochs": 1}
    ).get_json()["run_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        state = client.get(f"/api/v1/train/runs/{run_id}").get_json()["run"]["state"]
        if state not in ("pending", "running"):
            break
        time.sleep(0.2)

    body = client.post(
        f"/api/v1/train/runs/{run_id}/media",
        data={"file": (io.BytesIO(_gif_bytes(2)), "clip.gif")},
        content_type="multipart/form-data",
    ).get_json()
    assert _wait_job(client, body["job_id"])["state"] == "completed"
    frame = client.get(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}/frames/00000.jpg"
    )
    assert frame.status_code == 200


# ------------------------------------------------------ error analysis (E6-T4/T5)


def _evaluate(client, run_id, split="valid"):
    response = client.post(f"/api/v1/train/runs/{run_id}/evaluate", json={"split": split})
    assert response.status_code == 202
    assert _wait_job(client, response.get_json()["job_id"])["state"] == "completed"


def test_error_analysis_follows_the_query_threshold(trained_client):
    client, run_id, _ = trained_client
    # the fake backend predicts one box of an unknown class at 0.9 on every image
    before = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/errors")
    assert before.status_code == 400
    assert "run an evaluation first" in before.get_json()["error"]["message"]

    _evaluate(client, run_id)
    response = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/errors")
    assert response.status_code == 200
    analysis = response.get_json()
    assert analysis["threshold"] == 0.5 and analysis["iou"] == 0.5
    assert analysis["classes"][-1] == "background"
    assert analysis["fn"] == 1 and analysis["fp"] == 1 and analysis["tp"] == 0
    assert len(analysis["matrix"]) == len(analysis["classes"])

    strict = client.get(
        f"/api/v1/train/runs/{run_id}/eval/valid/errors?threshold=0.95&iou=0.75"
    ).get_json()
    assert strict["threshold"] == 0.95 and strict["iou"] == 0.75
    assert strict["fp"] == 0  # the 0.9 prediction is gone above 0.95


def test_worst_cases_endpoint_lists_errors_and_honours_top(trained_client):
    client, run_id, _ = trained_client
    _evaluate(client, run_id)
    response = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/worst?top=1")
    assert response.status_code == 200
    report = response.get_json()
    assert report["top_k"] == 1 and report["total_images"] == 1
    assert report["images_with_errors"] == 1 and len(report["images"]) == 1
    worst = report["images"][0]
    assert worst["file_name"] == "c.png" and worst["errors"] == 2
    kinds = sorted(item["kind"] for item in worst["items"])
    assert kinds == ["fn", "fp"]

    bad = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/worst?threshold=7")
    assert bad.status_code == 400


def test_evaluation_scores_current_labels_unless_told_otherwise(trained_client):
    client, run_id, _ = trained_client
    _evaluate(client, run_id)
    report = client.get(f"/api/v1/train/runs/{run_id}/eval/valid").get_json()
    assert report["labels"] == "current"
    assert "as they are now" in report["notes"][0]

    response = client.post(
        f"/api/v1/train/runs/{run_id}/evaluate", json={"split": "valid", "labels": "snapshot"}
    )
    assert response.status_code == 202
    assert _wait_job(client, response.get_json()["job_id"])["state"] == "completed"
    frozen = client.get(f"/api/v1/train/runs/{run_id}/eval/valid").get_json()
    assert frozen["labels"] == "snapshot"


def test_threshold_endpoint_serves_the_sweep_and_honours_beta(trained_client):
    client, run_id, _ = trained_client
    before = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/threshold")
    assert before.status_code == 400
    assert "run an evaluation first" in before.get_json()["error"]["message"]

    _evaluate(client, run_id)
    response = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/threshold")
    assert response.status_code == 200
    advice = response.get_json()
    assert advice["run_id"] == run_id and advice["split"] == "valid"
    assert advice["iou"] == 0.5 and advice["beta"] == 1.0
    assert len(advice["points"]) == 91
    # the fake backend only ever predicts an unknown class: nothing is ever
    # correct, so there is no threshold to suggest and the default stands
    assert advice["confident"] is False
    assert advice["recommended"] == 0.5

    weighted = client.get(
        f"/api/v1/train/runs/{run_id}/eval/valid/threshold?beta=2&iou=0.75"
    ).get_json()
    assert weighted["beta"] == 2.0 and weighted["iou"] == 0.75

    bad = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/threshold?beta=0")
    assert bad.status_code == 400


def test_overlay_endpoint_streams_a_png_of_the_split_image(trained_client):
    client, run_id, _ = trained_client
    _evaluate(client, run_id)
    worst = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/worst").get_json()
    image_id = worst["images"][0]["image_id"]
    response = client.get(
        f"/api/v1/train/runs/{run_id}/eval/valid/images/{image_id}/overlay.png?threshold=0.3"
    )
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    from PIL import Image

    with Image.open(io.BytesIO(response.data)) as png:
        assert png.size == (32, 32)  # c.png of the sample dataset

    missing = client.get(f"/api/v1/train/runs/{run_id}/eval/valid/images/999/overlay.png")
    assert missing.status_code == 400
    assert "not part of" in missing.get_json()["error"]["message"]
