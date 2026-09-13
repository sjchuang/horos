"""E10-T12: loop Web API routes are thin wrappers over horos.api (R2)."""

from __future__ import annotations

import time

import pytest
from helpers.data import make_image
from helpers.fake_backend import FakeEmbedder

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
    monkeypatch.setattr("horos.backends.get_backend", lambda key, **kw: FakeEmbedder())
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
    assert status["current"] is None and status["rounds"] == []

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
