"""E8-T8: export routes are thin over horos.api and serve the files back."""

from __future__ import annotations

import time
import zipfile
from io import BytesIO

import pytest
from helpers.runs import completed_fake_run

from horos.web.app import create_app

pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return completed_fake_run(tmp_path_factory.mktemp("run"), epochs=2)


@pytest.fixture
def client(run):
    project, _ = run
    app = create_app(project.root)
    app.testing = True
    return app.test_client()


def _wait_job(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job still running")


def test_evaluation_chart_export_and_download(client, run):
    pytest.importorskip("pycocotools", reason="training stack not installed")
    from horos.api.evaluate import evaluate_run

    project, record = run
    evaluate_run(project, record.run_id, split="valid")
    body = client.post(
        f"/api/v1/train/runs/{record.run_id}/export/evaluation",
        json={"split": "valid", "threshold": 0.4, "iou": 0.5, "format": "png"},
    ).get_json()
    assert body["name"] == "evaluation_valid.png"
    response = client.get(body["download_url"])
    assert response.status_code == 200 and response.data[:8] == b"\x89PNG\r\n\x1a\n"

    bad = client.post(
        f"/api/v1/train/runs/{record.run_id}/export/evaluation", json={"format": "xlsx"}
    )
    assert bad.status_code == 400


def test_report_export_and_download(client, run):
    _, record = run
    body = client.post(
        f"/api/v1/train/runs/{record.run_id}/export/report", json={"format": "png"}
    ).get_json()
    assert body["name"] == "training_report.png"
    response = client.get(body["download_url"])
    assert response.status_code == 200
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert "attachment" in response.headers.get("Content-Disposition", "")
    # a viewable report also offers an inline URL the browser renders in a tab
    assert body["view_url"] == body["download_url"] + "?inline=1"
    inline = client.get(body["view_url"])
    assert inline.status_code == 200 and inline.mimetype == "image/png"
    assert "attachment" not in inline.headers.get("Content-Disposition", "")
    listed = client.get(f"/api/v1/train/runs/{record.run_id}/exports").get_json()
    png = next(a for a in listed if a["name"] == "training_report.png")
    assert png["view_url"].endswith("?inline=1")


def test_model_export_job_and_bundle_download(client, run):
    _, record = run
    response = client.post(
        f"/api/v1/train/runs/{record.run_id}/export/model", json={"format": "pytorch"}
    )
    assert response.status_code == 202
    job = _wait_job(client, response.get_json()["job_id"])
    assert job["state"] == "completed"
    bundle = job["events"][-1]["result"]["bundle"]

    listed = client.get(f"/api/v1/train/runs/{record.run_id}/exports").get_json()
    entry = next(a for a in listed if a["name"] == bundle)
    assert entry["kind"] == "model" and entry["download_url"].endswith(bundle)
    assert "view_url" not in entry  # a zip only downloads, even with ?inline=1
    forced = client.get(entry["download_url"] + "?inline=1")
    assert "attachment" in forced.headers.get("Content-Disposition", "")
    response = client.get(entry["download_url"])
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.data)) as zf:
        assert "model_card.json" in zf.namelist()


def test_bad_format_and_path_escape_are_client_errors(client, run):
    _, record = run
    response = client.post(
        f"/api/v1/train/runs/{record.run_id}/export/report", json={"format": "docx"}
    )
    assert response.status_code == 400
    response = client.get(f"/api/v1/train/runs/{record.run_id}/exports/..%2Frun.json")
    assert response.status_code in (400, 404)
