"""E2-T9: annotation Web API endpoints (thin routes over horos.api)."""

import pytest
from helpers.data import write_sample_coco_dir

from horos.web.app import create_app


@pytest.fixture
def project(tmp_path):
    from horos.api import create_project, import_dataset

    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


@pytest.fixture
def client(project):
    app = create_app(project.root)
    app.testing = True
    return app.test_client()


def _first_image_id(client) -> int:
    return client.get("/api/v1/queue").get_json()[0]["image"]["id"]


def test_queue_and_progress(client):
    queue = client.get("/api/v1/queue").get_json()
    assert len(queue) == 3
    names = [i["image"]["file_name"] for i in queue]
    assert names == sorted(names)  # all annotated -> file-name order
    progress = client.get("/api/v1/progress").get_json()
    assert progress["total_images"] == 3
    assert progress["annotated_images"] == 3


def test_queue_class_filter(client, project):
    pallet = project.categories[1].id
    names = [
        i["image"]["file_name"]
        for i in client.get(f"/api/v1/queue?category_id={pallet}").get_json()
    ]
    assert names == ["a.png", "c.png"]
    assert client.get("/api/v1/queue?category_id=99").status_code == 400


def test_get_put_annotations_roundtrip(client, project):
    image_id = _first_image_id(client)
    cat_id = project.categories[0].id
    view = client.get(f"/api/v1/images/{image_id}/annotations").get_json()
    response = client.put(
        f"/api/v1/images/{image_id}/annotations",
        json={
            "expected_version": view["version"],
            "annotations": [{"category_id": cat_id, "bbox": [10, 10, 20, 15]}],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["version"] == view["version"] + 1


def test_stale_version_is_409(client, project):
    image_id = _first_image_id(client)
    cat_id = project.categories[0].id
    version = client.get(f"/api/v1/images/{image_id}/annotations").get_json()["version"]
    body = {
        "expected_version": version,
        "annotations": [{"category_id": cat_id, "bbox": [1, 1, 2, 2]}],
    }
    assert client.put(f"/api/v1/images/{image_id}/annotations", json=body).status_code == 200
    stale = client.put(f"/api/v1/images/{image_id}/annotations", json=body)
    assert stale.status_code == 409
    assert stale.get_json()["error"]["code"] == "annotation_conflict"


def test_invalid_annotation_is_explicit(client):
    image_id = _first_image_id(client)
    response = client.put(
        f"/api/v1/images/{image_id}/annotations",
        json={
            "expected_version": 0,
            "annotations": [{"category_id": 999, "bbox": [1, 1, 2, 2]}],
        },
    )
    assert response.status_code == 400
    assert "unknown category" in response.get_json()["error"]["message"]


def test_image_file_served(client):
    image_id = _first_image_id(client)
    response = client.get(f"/api/v1/images/{image_id}/file")
    assert response.status_code == 200
    assert response.data[:8].startswith(b"\x89PNG") or response.data[:2] == b"\xff\xd8"


def test_claim_flow(client):
    image_id = _first_image_id(client)
    granted = client.post(f"/api/v1/images/{image_id}/claim", json={"session": "a"})
    assert granted.status_code == 200 and granted.get_json()["granted"]
    denied = client.post(f"/api/v1/images/{image_id}/claim", json={"session": "b"})
    assert denied.status_code == 409 and denied.get_json()["held_by"] == "a"
    released = client.delete(f"/api/v1/images/{image_id}/claim", json={"session": "a"})
    assert released.get_json()["released"]


def test_category_crud(client):
    created = client.post("/api/v1/categories", json={"name": "person"})
    assert created.status_code == 201
    cat_id = created.get_json()["id"]
    renamed = client.patch(f"/api/v1/categories/{cat_id}", json={"name": "worker"})
    assert renamed.get_json()["name"] == "worker"
    deleted = client.delete(f"/api/v1/categories/{cat_id}", json={})
    assert deleted.get_json()["deleted_annotations"] == 0


def test_delete_referenced_category_is_409_without_force(client, project):
    cat_id = project.categories[0].id
    response = client.delete(f"/api/v1/categories/{cat_id}", json={})
    assert response.status_code == 409
    forced = client.delete(f"/api/v1/categories/{cat_id}", json={"force": True})
    assert forced.status_code == 200
    assert forced.get_json()["deleted_annotations"] > 0


def test_merge_categories_route(client, project):
    forklift = next(c for c in project.categories if c.name == "forklift")
    pallet = next(c for c in project.categories if c.name == "pallet")
    response = client.post(
        "/api/v1/categories/merge", json={"sources": [pallet.id], "target": forklift.id}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["target"]["name"] == "forklift" and body["merged_annotations"] == 2
    assert body["removed_ids"] == [pallet.id]
    assert [c["name"] for c in client.get("/api/v1/project").get_json()["categories"]] == [
        "forklift"
    ]
    # validation: sources must be a list of ids, target an id
    assert client.post(
        "/api/v1/categories/merge", json={"sources": "1", "target": forklift.id}
    ).status_code == 400
    assert client.post("/api/v1/categories/merge", json={"sources": [1]}).status_code == 400


def test_class_deletion_runs_as_a_job_with_progress(client):
    """The annotate page deletes behind a blocking bar: POST starts a job,
    the job fails with category_in_use until force, then completes."""
    import time

    anns = client.get("/api/v1/images/1/annotations").get_json()["annotations"]
    used_id = anns[0]["category_id"]

    def wait(job_id):
        for _ in range(200):
            body = client.get(f"/api/v1/jobs/{job_id}").get_json()
            if body["state"] != "running":
                return body
            time.sleep(0.02)
        raise AssertionError("job still running")

    r = client.post(f"/api/v1/categories/{used_id}/delete", json={})
    assert r.status_code == 202
    body = wait(r.get_json()["job_id"])
    assert body["state"] == "failed"
    failed = next(e for e in body["events"] if e["type"] == "failed")
    assert failed["error_code"] == "category_in_use" and failed["details"]["annotations"] >= 1
    r = client.post(f"/api/v1/categories/{used_id}/delete", json={"force": True})
    body = wait(r.get_json()["job_id"])
    assert body["state"] == "completed"
    done = next(e for e in body["events"] if e["type"] == "completed")
    assert done["result"]["deleted_annotations"] >= 1
    assert any(e.get("phase") == "scanning" for e in body["events"])
    assert client.post("/api/v1/categories/999/delete", json={}).status_code == 400


def test_deleting_a_class_in_use_is_a_409_with_its_own_code(client):
    """The annotate page confirms only on category_in_use; a missing class is
    a plain error (the user saw both as bare 400s in the log)."""
    anns = client.get("/api/v1/images/1/annotations").get_json()["annotations"]
    used_id = anns[0]["category_id"]
    r = client.delete(f"/api/v1/categories/{used_id}", json={})
    assert r.status_code == 409
    body = r.get_json()["error"]
    assert body["code"] == "category_in_use"
    assert body["details"]["annotations"] >= 1 and " is used by " in body["message"]
    r = client.delete("/api/v1/categories/999", json={})
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "project_error"
