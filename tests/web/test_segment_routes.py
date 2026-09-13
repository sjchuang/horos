"""SAM-T3: interactive segmentation routes (thin over horos.api.segment)."""

from __future__ import annotations

import pytest
from helpers.data import write_sample_coco_dir
from helpers.fake_backend import FakePromptableSegmenter

import horos.backends
from horos.api.segment import _reset_segmenters
from horos.web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from horos.api import create_project, import_dataset

    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    fake = FakePromptableSegmenter()
    monkeypatch.setattr(horos.backends, "get_backend", lambda key, **kw: fake)
    _reset_segmenters()
    app = create_app(proj.root)
    app.testing = True
    yield app.test_client(), fake
    _reset_segmenters()


def test_click_then_box_then_accept_flow(client):
    client, fake = client
    response = client.post("/api/v1/images/1/segment", json={"points": [[30, 20]], "labels": [1]})
    assert response.status_code == 200
    body = response.get_json()
    assert body["shape_type"] == "polygon" and body["bbox"] == [20, 10, 20, 20]
    assert body["score"] == pytest.approx(0.9) and body["embedding_cached"] is False

    body = client.post(
        "/api/v1/images/1/segment", json={"box": [4, 4, 16, 12], "output": "bbox"}
    ).get_json()
    assert body["shape_type"] == "rectangle" and body["points"] == [[4, 4], [20, 16]]
    assert body["embedding_cached"] is True and len(fake.embed_calls) == 1

    # nothing was written by either call
    view = client.get("/api/v1/images/1/annotations").get_json()
    assert len(view["annotations"]) == 2  # the sample dataset's two boxes on image 1


def test_prefetch_route(client):
    client, fake = client
    first = client.post("/api/v1/images/2/segment/prefetch", json={}).get_json()
    assert first["embedding_cached"] is False and first["model"] == "sam2.1-tiny"
    again = client.post("/api/v1/images/2/segment/prefetch", json={"model": "sam2.1-tiny"})
    assert again.get_json()["embedding_cached"] is True
    assert client.post("/api/v1/images/2/segment/prefetch", json={"model": 5}).status_code == 400


def test_bad_requests_are_400_with_the_unified_shape(client):
    client, _ = client
    for body in (
        {},  # no prompt
        {"points": [[1, 1]], "labels": [1, 0]},  # mismatch
        {"points": [[999, 1]], "labels": [1]},  # outside
        {"points": "nope"},  # wrong type
        {"points": [[1, 1]], "labels": [1], "output": "mask"},  # unknown output
    ):
        response = client.post("/api/v1/images/1/segment", json=body)
        assert response.status_code == 400, body
        assert set(response.get_json()["error"]) >= {"code", "message"}
    assert client.post(
        "/api/v1/images/999/segment", json={"points": [[1, 1]], "labels": [1]}
    ).status_code == 400


def test_boxes_to_polygons_routes(client):
    """SAM-T6: one image synchronously, the project as a job."""
    client, fake = client
    before = client.get("/api/v1/images/1/annotations").get_json()
    boxes = [a for a in before["annotations"] if not a["segmentation"]]

    response = client.post("/api/v1/images/1/segment/boxes", json={"annotation_ids": "x"})
    assert response.status_code == 400
    response = client.post("/api/v1/images/1/segment/boxes", json={"categories": ["ghost"]})
    assert response.status_code == 400
    response = client.post("/api/v1/images/1/segment/boxes", json={"max_points": 2})
    assert response.status_code == 400
    response = client.post("/api/v1/images/1/segment/boxes", json={})
    assert response.status_code == 200
    body = response.get_json()
    assert body["converted"] == len(boxes) and body["skipped"] == 0
    assert body["version"] == before["version"] + 1
    after = client.get("/api/v1/images/1/annotations").get_json()
    assert all(a["segmentation"] for a in after["annotations"])
    assert len(fake.embed_calls) == 1

    assert client.post("/api/v1/segment/boxes", json={"categories": ["ghost"]}).status_code == 400
    response = client.post("/api/v1/segment/boxes", json={"include_pending": False})
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    import time

    deadline = time.time() + 10
    while True:
        status = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if status["state"] != "running":
            break
        assert time.time() < deadline
        time.sleep(0.05)
    assert status["state"] == "completed"
    result = status["events"][-1]["result"]
    assert result["skipped"] == 0 and result["images"] >= 0
