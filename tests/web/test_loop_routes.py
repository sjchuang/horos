"""E10-T12: loop Web API routes are thin wrappers over horos.api (R2)."""

from __future__ import annotations

import time

import pytest
from helpers.data import make_image
from helpers.fake_backend import fake_get_backend

from horos.core.dataset import Category
from horos.web.app import create_app


@pytest.fixture
def project_root(tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "proj")
    project.set_categories([Category(id=1, name="box")])
    for n in range(6):
        path = make_image(tmp_path / "src" / f"{n}.png", 40 + n, 30, (200, 30 * n, 30))
        project.add_image(path, width=40 + n, height=30)
    return project.root


@pytest.fixture
def client(project_root, monkeypatch):
    monkeypatch.setattr("horos.backends.get_backend", fake_get_backend)
    app = create_app(project_root)
    app.testing = True
    return app.test_client()


def _wait_job(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if body["state"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running")


def test_status_round_and_close_roundtrip(client):
    status = client.get("/api/v1/loop").get_json()
    assert status["pool_size"] == 6 and status["next_strategy"] == "diversity"
    assert status["current"] is None and status["rounds"] == [] and status["job"] is None

    emb = client.get("/api/v1/loop/embeddings?model=fake-embedder").get_json()
    assert emb["missing"] == 6

    resp = client.post("/api/v1/loop/rounds", json={"count": 2, "model": "fake-embedder"})
    assert resp.status_code == 202
    job = _wait_job(client, resp.get_json()["job_id"])
    assert job["state"] == "completed", job

    status = client.get("/api/v1/loop").get_json()
    assert status["current"]["number"] == 1 and status["pool_size"] == 4
    round_body = client.get("/api/v1/loop/rounds/1").get_json()
    assert round_body["state"] == "labeling"
    assert len(round_body["selection"]["picks"]) == 2
    assert all(p["reason"] for p in round_body["selection"]["picks"])

    # a second round while one is open is refused with the unified error format
    resp = client.post("/api/v1/loop/rounds", json={"count": 1})
    assert resp.status_code == 400
    assert "still 'labeling'" in resp.get_data(as_text=True)

    closed = client.post("/api/v1/loop/rounds/1/close").get_json()
    assert closed["state"] == "closed"
    assert client.get("/api/v1/loop").get_json()["current"] is None


def test_parameter_validation_is_explicit(client):
    assert client.post("/api/v1/loop/rounds", json={"count": "two"}).status_code == 400
    assert client.post("/api/v1/loop/rounds", json={"strategy": "magic"}).status_code == 400
    assert client.post("/api/v1/loop/rounds", json={"percent": True}).status_code == 400
    assert client.get("/api/v1/loop/rounds/7").status_code == 400


def test_embedding_job_route(client):
    resp = client.post("/api/v1/loop/embeddings", json={"model": "fake-embedder"})
    assert resp.status_code == 202
    assert _wait_job(client, resp.get_json()["job_id"])["state"] == "completed"
    emb = client.get("/api/v1/loop/embeddings?model=fake-embedder").get_json()
    assert emb["embedded"] == 6 and emb["missing"] == 0


def test_history_readiness_assign_and_queue_routes(client):
    assert client.get("/api/v1/loop/history").get_json() == []
    ready = client.get("/api/v1/loop/readiness").get_json()
    assert ready["ready"] is False and ready["reasons"]

    resp = client.post("/api/v1/loop/rounds", json={"count": 3, "model": "fake-embedder"})
    assert _wait_job(client, resp.get_json()["job_id"])["state"] == "completed"
    history = client.get("/api/v1/loop/history").get_json()
    assert len(history) == 1 and history[0]["labels_spent"] == 0

    assign = "/api/v1/loop/rounds/1/assign"
    assert client.post(assign, json={"annotators": "ann"}).status_code == 400
    body = client.post(assign, json={"annotators": ["ann", "bob"]}).get_json()
    owners = [p["assigned_to"] for p in body["selection"]["picks"]]
    assert owners == ["ann", "bob", "ann"]
    queue = client.get("/api/v1/loop/rounds/1/queue?annotator=bob").get_json()
    assert len(queue) == 1 and queue[0]["assigned_to"] == "bob" and queue[0]["reason"]
    assert len(client.get("/api/v1/loop/rounds/1/queue").get_json()) == 3

    # training is refused with the readiness reasons, not a stack trace
    resp = client.post("/api/v1/loop/rounds/1/train", json={})
    assert resp.status_code == 400 and "Not ready to train" in resp.get_data(as_text=True)
    assert "short_classes" in ready and "ready_without_short" in ready
    resp = client.post("/api/v1/loop/rounds/1/train", json={"ignore_short_classes": True})
    assert resp.status_code == 400  # six photos: dropping classes does not help either
    assert client.post("/api/v1/loop/rounds/1/train", json={"epochs": 0}).status_code == 400
    status = client.get("/api/v1/loop/rounds/1/training").get_json()
    assert status["round"]["number"] == 1 and status["training"] is None


def test_image_clusters_route(client):
    """E10-T20: the Dataset page's photo groups; needs embeddings first."""
    resp = client.get("/api/v1/images/clusters?model=fake-embedder")
    assert resp.status_code == 400 and "embedding job" in resp.get_data(as_text=True)
    job = client.post("/api/v1/loop/embeddings", json={"model": "fake-embedder"}).get_json()
    assert _wait_job(client, job["job_id"])["state"] == "completed"
    body = client.get("/api/v1/images/clusters?model=fake-embedder&k=2&samples=1").get_json()
    assert body["k"] == 2 and body["embedded"] == 6
    assert sum(c["size"] for c in body["clusters"]) == 6
    assert all(len(c["samples"]) == 1 and c["image_ids"] for c in body["clusters"])
    assert client.get("/api/v1/images/clusters?model=fake-embedder&k=abc").status_code == 400


def test_image_predictions_route(client):
    """SAM-T4: the annotator's class suggestion asks the loop's scorer."""
    body = client.get("/api/v1/images/1/predictions?threshold=0.2").get_json()
    assert body["image_id"] == 1 and body["kind"] == "zero_shot"
    assert all({"label", "bbox", "score"} <= set(d) for d in body["detections"])
    assert client.get("/api/v1/images/999/predictions").status_code == 400
