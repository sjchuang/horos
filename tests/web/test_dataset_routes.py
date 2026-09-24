"""E1-T9: dataset Web API endpoints (thin routes over horos.api)."""

import io
import zipfile

import pytest
from helpers.data import write_sample_coco_dir

from horos.web.app import create_app


@pytest.fixture
def project_root(tmp_path):
    from horos.api import create_project, import_dataset

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project.root


@pytest.fixture
def client(project_root):
    app = create_app(project_root)
    app.testing = True
    return app.test_client()


def _wait_job(client, job_id, timeout=15.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running")


def _zip_of(directory) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for f in directory.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(directory))
    buffer.seek(0)
    return buffer


def test_project_summary(client):
    body = client.get("/api/v1/project").get_json()
    assert body["name"] == "proj"
    assert body["num_images"] == 3
    assert {c["name"] for c in body["categories"]} == {"forklift", "pallet"}


def test_stats_route(client):
    body = client.get("/api/v1/dataset/stats").get_json()
    assert body["num_images"] == 3
    assert body["num_annotations"] == 4


def test_stats_route_for_a_class_selection(client):
    body = client.get("/api/v1/dataset/stats?categories=forklift").get_json()
    assert body["num_images"] == 2 and body["num_annotations"] == 2
    body = client.get(
        "/api/v1/dataset/stats?categories=forklift&include_background=1"
    ).get_json()
    assert body["num_images"] == 3 and body["unannotated_images"] == 1


def test_validation_issues_name_the_file(client, project_root):
    from horos.api import open_project
    from horos.core.dataset import Annotation

    # written through the core Project (the annotate API clamps boxes on save)
    project = open_project(project_root)
    image = project.list_images()[0]
    current = project.load_annotations(image.id)
    bad = Annotation(id=1, image_id=image.id, category_id=project.categories[0].id,
                     bbox=(-30.0, 0.0, 40.0, 10.0))
    project.save_annotations(image.id, [bad], expected_version=current.version)
    issue = next(i for i in client.get("/api/v1/dataset/validation").get_json()["issues"]
                 if i["kind"] == "bbox_out_of_bounds")
    # the UI builds /annotate#<image_id>:<annotation_id> from these
    assert issue["image_id"] == image.id and issue["annotation_id"] == 1
    assert issue["file_name"] == image.file_name and image.file_name in issue["message"]


def test_validation_route(client):
    body = client.get("/api/v1/dataset/validation").get_json()
    assert body["ok"] is True
    assert body["issues"] == []


def test_validation_fix_route(client, project_root):
    from horos.api import open_project

    project = open_project(project_root)
    record = project.list_images()[0]  # 64x48
    stored = project.load_annotations(record.id)
    jittered = [
        stored.annotations[0].model_copy(update={"bbox": (0.5, 4.0, 64.0, 12.0)}),
        *stored.annotations[1:],
    ]
    project.save_annotations(record.id, jittered, expected_version=stored.version)

    report = client.get("/api/v1/dataset/validation").get_json()
    assert any(i["fixable"] for i in report["issues"])

    body = client.post("/api/v1/dataset/validation/fix").get_json()
    assert body["num_fixed"] == 1
    assert body["ok"] is True
    assert body["report"]["issues"] == []


def test_images_route(client):
    body = client.get("/api/v1/images").get_json()
    assert len(body) == 3
    assert {"id", "file_name", "width", "height", "split"} <= set(body[0])


def test_images_delete_route(client):
    ids = [i["id"] for i in client.get("/api/v1/images").get_json()]
    body = client.post(
        "/api/v1/images/delete", json={"ids": ids[:2], "session": "s1"}
    ).get_json()
    assert body["deleted"] == ids[:2] and body["skipped_claimed"] == []
    assert len(client.get("/api/v1/images").get_json()) == 1


def test_images_delete_route_requires_ids(client):
    response = client.post("/api/v1/images/delete", json={})
    assert response.status_code == 400
    assert "ids" in response.get_json()["error"]["message"]


def test_split_route(client):
    body = client.post(
        "/api/v1/dataset/split",
        json={"train": 1.0, "valid": 0.0, "test": 0.0, "seed": 5, "reshuffle": True},
    ).get_json()
    assert body == {"train": 3, "valid": 0, "test": 0, "unassigned": 0}


def test_upload_route(tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "fresh")
    app = create_app(project.root)
    app.testing = True
    client = app.test_client()
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(coco_dir), "dataset.zip")},
        content_type="multipart/form-data",
    )
    # 202: the zip is staged and imported as a polled job (progress in the UI)
    assert response.status_code == 202
    body = response.get_json()
    assert set(body) >= {"upload_id", "job_id", "file_name", "size_bytes"}
    job = _wait_job(client, body["job_id"])
    assert job["state"] == "completed" and job["kind"] == "import"
    result = job["events"][-1]["result"]
    assert result["num_images"] == 3 and result["format"] == "coco"
    assert client.get("/api/v1/project").get_json()["num_images"] == 3


def test_export_route(client, tmp_path):
    body = client.post(
        "/api/v1/dataset/export",
        json={"out_dir": str(tmp_path / "out"), "format": "yolo"},
    ).get_json()
    assert body["path"].endswith("data.yaml")


def test_export_route_labelme(client, tmp_path):
    out = tmp_path / "out"
    body = client.post(
        "/api/v1/dataset/export",
        json={"out_dir": str(out), "format": "labelme"},
    ).get_json()
    assert body["path"] == str(out)
    assert any(out.rglob("*.json"))


def test_create_project_route(tmp_path):
    app = create_app()
    app.testing = True
    response = app.test_client().post(
        "/api/v1/projects", json={"path": str(tmp_path / "newproj"), "name": "n"}
    )
    assert response.status_code == 201
    assert response.get_json()["name"] == "n"


def test_models_route_carries_license(client):
    body = client.get("/api/v1/models").get_json()
    assert len(body) >= 4
    assert all(m["weights_license"] == "Apache-2.0" for m in body)


def test_capabilities_route(client):
    body = client.get("/api/v1/capabilities").get_json()
    features = {f["feature"] for f in body["features"]}
    assert "export_tensorrt" in features


def test_upload_conflict_flow(tmp_path, client, project_root):
    # phase 1: upload is staged and imported as a job; same names with
    # different content -> failed event carrying the conflict list
    from helpers.data import make_image
    from helpers.data import write_sample_coco_dir as sample

    variant = sample(tmp_path / "variant")
    make_image(variant / "train" / "a.png", 64, 48, color=(1, 2, 3))
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(variant), "dataset.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["file_name"] == "dataset.zip" and body["size_bytes"] > 0
    job = _wait_job(client, body["job_id"])
    assert job["state"] == "failed"
    failed = job["events"][-1]
    assert failed["error_code"] == "import_conflict"
    assert failed["details"]["conflicts"] == ["a.png"]
    assert failed["details"]["retryable"] is True
    # phase 2: retry with the chosen policy — no re-upload
    response = client.post(
        f"/api/v1/dataset/upload/{body['upload_id']}/import",
        json={"on_conflict": "overwrite"},
    )
    assert response.status_code == 202
    job = _wait_job(client, response.get_json()["job_id"])
    assert job["state"] == "completed"
    result = job["events"][-1]["result"]
    assert result["overwritten"] == 1
    assert result["duplicates_skipped"] == 2
    # progress events were streamed, not just the outcome
    phases = {e["phase"] for e in job["events"] if e["type"] == "progress"}
    assert {"extracting", "copying images"} <= phases
    # the staged zip is gone after a successful import
    response = client.delete(f"/api/v1/dataset/upload/{body['upload_id']}")
    assert response.get_json() == {"discarded": False}


def test_upload_darknet_class_names_flow(tmp_path):
    from helpers.data import make_image

    from horos.api import create_project
    from horos.web.app import create_app as make_app

    src = tmp_path / "darknet"
    make_image(src / "img1.png")
    (src / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    app = make_app(create_project(tmp_path / "fresh").root)
    app.testing = True
    web = app.test_client()
    response = web.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(src), "dataset.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    # phase 1: no _darknet.labels -> failed event with editable defaults
    job = _wait_job(web, body["job_id"])
    failed = job["events"][-1]
    assert job["state"] == "failed" and failed["error_code"] == "class_names_required"
    assert failed["details"]["default_names"] == ["0"]
    # phase 2: retry with the names the user typed
    response = web.post(
        f"/api/v1/dataset/upload/{body['upload_id']}/import",
        json={"class_names": ["helmet"]},
    )
    job = _wait_job(web, response.get_json()["job_id"])
    assert job["state"] == "completed"
    assert job["events"][-1]["result"]["instances_per_category"] == {"helmet": 1}


def test_upload_cancel_discards_the_staged_zip(tmp_path, client, project_root):
    from helpers.data import make_image
    from helpers.data import write_sample_coco_dir as sample

    variant = sample(tmp_path / "variant")
    make_image(variant / "train" / "a.png", 64, 48, color=(9, 9, 9))
    body = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(variant), "dataset.zip")},
        content_type="multipart/form-data",
    ).get_json()
    assert _wait_job(client, body["job_id"])["state"] == "failed"
    assert (project_root / "uploads" / body["upload_id"]).is_dir()
    response = client.delete(f"/api/v1/dataset/upload/{body['upload_id']}")
    assert response.get_json() == {"discarded": True}
    assert not (project_root / "uploads" / body["upload_id"]).exists()


def test_upload_non_zip_is_rejected_synchronously(client):
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (io.BytesIO(b"not a zip"), "dataset.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "dataset_format_error"


def test_upload_bad_class_names_is_400(tmp_path, client):
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(tmp_path), "dataset.zip"), "class_names": "not json"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "class_names" in response.get_json()["error"]["message"]


def test_clear_dataset_route_requires_the_project_name(client):
    refused = client.delete("/api/v1/dataset", json={"confirm": "wrong"})
    assert refused.status_code == 400
    assert "confirm must equal" in refused.get_json()["error"]["message"]
    assert client.delete("/api/v1/dataset", json={}).status_code == 400
    assert client.get("/api/v1/project").get_json()["num_images"] == 3

    name = client.get("/api/v1/project").get_json()["name"]
    body = client.delete("/api/v1/dataset", json={"confirm": name}).get_json()
    assert body["deleted_images"] == 3 and body["deleted_annotations"] == 4
    assert body["deleted_categories"] == 0 and body["skipped_claimed"] == []
    project = client.get("/api/v1/project").get_json()
    assert project["num_images"] == 0 and len(project["categories"]) == 2


def test_upload_photos_route(tmp_path):
    # E1-T11: several 'file' fields holding photos stage a loose-photo import
    from helpers.data import make_image

    from horos.api import create_project

    project = create_project(tmp_path / "fresh")
    app = create_app(project.root)
    app.testing = True
    client = app.test_client()
    a = make_image(tmp_path / "a.jpg", 64, 48)
    b = make_image(tmp_path / "b.png", 32, 32)
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": [(a.open("rb"), "a.jpg"), (b.open("rb"), "b.png")]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["file_name"] == "2 photos"
    job = _wait_job(client, body["job_id"])
    assert job["state"] == "completed"
    result = job["events"][-1]["result"]
    assert result["format"] == "images" and result["num_images"] == 2
    summary = client.get("/api/v1/project").get_json()
    assert summary["num_images"] == 2
    # a lone non-photo, non-zip file is refused synchronously
    note = tmp_path / "notes.txt"
    note.write_text("x", encoding="utf-8")
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (note.open("rb"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "dataset_format_error"


def test_upload_label_conflict_flow(tmp_path, client, project_root):
    # the project has the sample photos with labels; a zip carrying other
    # labels for one of them asks first, then retries with the decision
    from helpers.data import sample_dataset
    from helpers.data import write_sample_coco_dir as sample

    from horos.api import open_project
    from horos.core.dataset import Annotation

    project = open_project(project_root)
    project.set_categories(sample_dataset().categories)
    record = next(r for r in project.list_images() if r.file_name == "b.png")
    current = project.load_annotations(record.id)
    project.save_annotations(
        record.id,
        [Annotation(id=1, image_id=record.id, category_id=2, bbox=(1.0, 1.0, 2.0, 2.0))],
        expected_version=current.version,
    )
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(sample(tmp_path / "same")), "dataset.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    job = _wait_job(client, body["job_id"])
    assert job["state"] == "failed"
    failed = job["events"][-1]
    assert failed["error_code"] == "label_conflict"
    assert failed["details"]["conflicts"] == ["b.png"]
    assert failed["details"]["retryable"] is True
    response = client.post(
        f"/api/v1/dataset/upload/{body['upload_id']}/import",
        json={"on_annotations": "merge"},
    )
    assert response.status_code == 202
    job = _wait_job(client, response.get_json()["job_id"])
    assert job["state"] == "completed"
    result = job["events"][-1]["result"]
    assert result["annotations_merged"] == 1 and result["images_matched"] == 1
    assert result["duplicates_skipped"] == 2  # a.png and c.png: same labels again
    assert len(project.load_annotations(record.id).annotations) == 2
