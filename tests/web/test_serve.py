"""E8-T7: the local inference service — sources, the standalone serve app,
the framework-free ONNX executor, and start/stop through the project API."""

from __future__ import annotations

import json
import socket
import urllib.request
import zipfile
from pathlib import Path

import pytest
from helpers.data import make_image
from helpers.runs import completed_fake_run
from helpers.synthetic import synthetic_detector

from horos.api.serve import (
    _reset_servers,
    check_runtime,
    create_inference_server,
    load_source,
    resolve_source,
)
from horos.errors import BackendError, ProjectError, UnsupportedPlatformError
from horos.web.app import create_app
from horos.web.serve_app import create_serve_app


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return completed_fake_run(tmp_path_factory.mktemp("serve"), epochs=1)


def _client_for(source, **kwargs):
    app = create_serve_app(create_inference_server(source, **kwargs))
    app.testing = True
    return app.test_client()


def _post_image(client, path, url="/predict", **form):
    with path.open("rb") as handle:
        return client.post(
            url, data={"file": (handle, path.name), **form}, content_type="multipart/form-data"
        )


# ------------------------------------------------------------- checkpoint source


def test_checkpoint_source_serves_predictions(run, tmp_path):
    project, record = run
    source = resolve_source(project, run_id=record.run_id, format="checkpoint")
    assert source.kind == "checkpoint" and source.run_id == record.run_id
    assert source.classes == ["forklift", "pallet"]
    # R3: the licence rides along even for a bare checkpoint (from the registry)
    assert source.card["weights_license"] == "Apache-2.0"
    assert source.card["format"] == "checkpoint"

    client = _client_for(source, threshold=0.5)
    health = client.get("/health").get_json()
    assert health["status"] == "ok" and health["kind"] == "checkpoint" and health["classes"] == 2
    assert client.get("/model_card").get_json()["classes"] == ["forklift", "pallet"]
    assert "POST /predict" in client.get("/").get_json()["endpoints"]

    image = make_image(tmp_path / "a.png")
    response = _post_image(client, image)
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    body = response.get_json()
    assert body["image"] == "a.png" and body["threshold"] == 0.5
    assert body["instances"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert body["instances"][0]["score"] == pytest.approx(0.9)

    # per-request threshold above the fake's 0.9 filters everything out
    assert _post_image(client, image, url="/predict?threshold=0.95").get_json()["instances"] == []
    # a raw image body works too
    raw = client.post("/predict", data=image.read_bytes(), content_type="image/png")
    assert raw.status_code == 200 and raw.get_json()["instances"]
    # annotated=1 returns a JPEG overlay
    overlay = _post_image(client, image, url="/predict?annotated=1")
    assert overlay.status_code == 200 and overlay.mimetype == "image/jpeg"
    assert client.get("/health").get_json()["requests"] == 4


def test_serve_app_errors_use_the_unified_shape(run, tmp_path):
    project, record = run
    client = _client_for(resolve_source(project, run_id=record.run_id, format="checkpoint"))
    missing = client.post("/predict")
    assert missing.status_code == 400
    assert set(missing.get_json()["error"]) >= {"code", "message"}
    image = make_image(tmp_path / "b.png")
    assert _post_image(client, image, threshold="lots").status_code == 400
    assert _post_image(client, image, url="/predict?threshold=7").status_code == 400
    assert client.get("/nope").get_json()["error"]["code"] == "not_found"
    assert client.options("/predict").status_code == 204  # CORS preflight


def test_run_sources_are_validated(run):
    project, record = run
    with pytest.raises(ProjectError, match="no onnx export yet"):
        resolve_source(project, run_id=record.run_id, format="onnx")
    with pytest.raises(ProjectError, match="Unknown serve format"):
        resolve_source(project, run_id=record.run_id, format="bogus")
    with pytest.raises(ProjectError, match="No such training run"):
        resolve_source(project, run_id="ghost", format="checkpoint")
    with pytest.raises(ProjectError, match="needs a project and run_id"):
        resolve_source()


# ------------------------------------------------------------- path sources


def _write_bundle(root, *, fmt="onnx", artifact="model.onnx", classes=("a", "b")):
    root.mkdir(parents=True, exist_ok=True)
    (root / artifact).write_bytes(b"not really a model")
    (root / "model_card.json").write_text(json.dumps({
        "format": fmt, "artifact": artifact, "classes": list(classes),
        "model": "rfdetr-nano", "run_id": "r1", "weights_license": "Apache-2.0",
    }), "utf-8")
    return root


def test_path_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HOROS_WEIGHTS_DIR", str(tmp_path / "home" / "weights"))
    bundle = _write_bundle(tmp_path / "bundle")
    source = resolve_source(path=bundle)
    assert source.kind == "onnx" and source.classes == ["a", "b"] and source.run_id == "r1"
    assert resolve_source(path=bundle / "model_card.json").path == source.path

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in bundle.iterdir():
            zf.write(file, file.name)
    unpacked = resolve_source(path=zip_path)
    assert unpacked.kind == "onnx" and unpacked.classes == ["a", "b"]
    assert str(tmp_path / "home" / "serve") in unpacked.bundle_dir

    bare = tmp_path / "bare" / "model.onnx"
    bare.parent.mkdir()
    bare.write_bytes(b"x")
    (bare.parent / "class_names.txt").write_text("cat\ndog\n", "utf-8")
    assert resolve_source(path=bare).classes == ["cat", "dog"]

    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"x")
    with pytest.raises(ProjectError, match="needs the model key"):
        resolve_source(path=ckpt)
    assert resolve_source(path=ckpt, model="rfdetr-nano").kind == "checkpoint"
    with pytest.raises(ProjectError, match="No such model source"):
        resolve_source(path=tmp_path / "nowhere")
    with pytest.raises(ProjectError, match="Cannot serve"):
        resolve_source(path=_touch(tmp_path / "weird.bin"))
    with pytest.raises(ProjectError, match="not a horos export bundle"):
        resolve_source(path=tmp_path / "bare")


def _touch(path):
    path.write_bytes(b"")
    return path


def test_engine_and_tflite_sources_resolve_and_are_checked_first(tmp_path, monkeypatch):
    """An engine or TFLite bundle is a first-class source (Serve-T2); what
    this machine cannot execute is refused before a process is spawned."""
    from horos.api import system as system_mod
    from horos.backends.runtime import _graphs
    from horos.core.platform_info import PlatformInfo

    for fmt, artifact in (("tensorrt", "model.trt"), ("tflite", "model.tflite")):
        # Name the platform instead of inheriting the host's: check_runtime
        # refuses a TensorRT engine on macOS by capability (asserted on its
        # own below), which would pre-empt the missing-runtime hint this half
        # is about. Re-applied per iteration because monkeypatch.undo() at the
        # end of the loop body clears it.
        monkeypatch.setattr(
            system_mod, "detect_platform",
            lambda: PlatformInfo(os_family="linux", arch="x86_64",
                                 is_jetson=False, python_version="3.12.0"),
        )
        bundle = _write_bundle(tmp_path / fmt, fmt=fmt, artifact=artifact)
        source = resolve_source(path=bundle)
        assert source.kind == fmt and source.classes == ["a", "b"]
        # a bare artifact next to class_names.txt resolves by suffix, like .onnx
        (bundle / "class_names.txt").write_text("x\ny\nz\n", "utf-8")
        (bundle / "model_card.json").unlink()
        bare = resolve_source(path=bundle / artifact)
        assert bare.kind == fmt and bare.classes == ["x", "y", "z"]
        assert bare.card["format"] == fmt and bare.card["weights_license"] == "unknown"
        # the runtime is missing: the hint names the opt-in installer
        monkeypatch.setattr(_graphs, "_find_spec", lambda name: False)
        with pytest.raises(ProjectError, match=f"horos install --{fmt}"):
            check_runtime(source)
        with pytest.raises(ProjectError, match=f"horos install --{fmt}"):
            load_source(source)
        monkeypatch.undo()

    # an engine on macOS: refused by the capability list, whatever is installed
    engine = resolve_source(path=tmp_path / "tensorrt" / "model.trt")
    monkeypatch.setattr(
        system_mod, "detect_platform",
        lambda: PlatformInfo(os_family="macos", arch="arm64", is_jetson=False,
                             python_version="3.12.0"),
    )
    with pytest.raises(UnsupportedPlatformError, match="not supported on macOS"):
        check_runtime(engine)
    monkeypatch.undo()
    # .engine and .plan are engines too; anything else is not an artifact
    from horos.backends.runtime import kind_for

    assert kind_for(Path("m.engine")) == "tensorrt" and kind_for(Path("m.plan")) == "tensorrt"
    with pytest.raises(BackendError, match="Cannot execute"):
        kind_for(Path("m.bin"))


# ------------------------------------------------------------- ONNX executor


def _synthetic_detector(path, *, resolution=32, classes=3):
    pytest.importorskip("onnx")
    return synthetic_detector(path, resolution=resolution, classes=classes)


def test_onnx_executor_decodes_the_exported_contract(tmp_path):
    pytest.importorskip("onnxruntime")
    from horos.backends.runtime import ArtifactModel

    bundle = tmp_path / "onnx"
    bundle.mkdir()
    _synthetic_detector(bundle / "model.onnx")
    card = {
        "format": "onnx", "artifact": "model.onnx", "classes": ["a", "b", "c"],
        "input": {"shape": [1, 3, 32, 32]},
        "outputs": [{"name": "dets"}, {"name": "labels"}],
    }
    model = ArtifactModel(bundle / "model.onnx", card=card, device="cpu")
    image = make_image(tmp_path / "img.png", 64, 48)
    prediction = model.infer_one(image, threshold=0.4)
    assert model.device == "cpu" and (prediction.width, prediction.height) == (64, 48)
    assert [i.category_name for i in prediction.instances] == ["b", "a"]  # best first
    best, second = prediction.instances
    assert best.score == pytest.approx(0.9526, abs=1e-3)
    assert best.bbox == pytest.approx((16.0, 12.0, 32.0, 24.0))  # centre half-size box, pixels
    assert second.bbox == pytest.approx((12.8, 9.6, 6.4, 4.8))
    assert [i.category_name for i in model.infer_one(image, threshold=0.6).instances] == ["b"]

    # end to end through resolve_source → serve app, framework-free
    (bundle / "model_card.json").write_text(json.dumps(card), "utf-8")
    client = _client_for(resolve_source(path=bundle), threshold=0.4, device="cpu")
    health = client.get("/health").get_json()
    assert health["kind"] == "onnx" and health["device"] == "cpu"
    body = _post_image(client, image).get_json()
    assert [i["category_name"] for i in body["instances"]] == ["b", "a"]


def test_onnx_executor_never_falls_back_silently(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from horos.backends.runtime import ArtifactModel

    path = _synthetic_detector(tmp_path / "m.onnx")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        with pytest.raises(BackendError, match="no CUDAExecutionProvider"):
            ArtifactModel(path, device="cuda").load()
    with pytest.raises(BackendError, match="not supported by the ONNX runtime executor"):
        ArtifactModel(path, device="tpu").load()
    with pytest.raises(BackendError, match="not found"):
        ArtifactModel(tmp_path / "missing.onnx", device="cpu").load()


_CARD = {
    "classes": ["a", "b", "c"], "input": {"shape": [1, 3, 32, 32]},
    "outputs": [{"name": "dets"}, {"name": "labels"}],
}
_EXPECTED = {"names": ["b", "a"], "best": (16.0, 12.0, 32.0, 24.0), "second": (12.8, 9.6, 6.4, 4.8)}


def _assert_contract(model, image):
    """The same synthetic detector decodes identically whatever executes it."""
    prediction = model.infer_one(image, threshold=0.4)
    assert (prediction.width, prediction.height) == (64, 48)
    assert [i.category_name for i in prediction.instances] == _EXPECTED["names"]
    best, second = prediction.instances
    assert best.score == pytest.approx(0.9526, abs=1e-3)
    assert best.bbox == pytest.approx(_EXPECTED["best"], abs=1e-3)
    assert second.bbox == pytest.approx(_EXPECTED["second"], abs=1e-3)
    assert [i.category_name for i in model.infer_one(image, threshold=0.6).instances] == ["b"]


# ------------------------------------------------------------- TFLite executor


def _synthetic_tflite(path, *, classes=3):
    """The synthetic detector as a .tflite with a SignatureDef, the way
    onnx2tf writes it: the ONNX input/output names survive in the signature
    even though the tensors themselves are called PartitionedCall:N."""
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    dets = np.array([[[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.1, 0.1]]], dtype=np.float32)
    logits = np.full((1, 2, classes), -5.0, dtype=np.float32)
    logits[0, 0, 1] = 3.0
    logits[0, 1, 0] = 0.0

    class Detector(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 3, 32, 32], tf.float32, name="input")])
        def __call__(self, x):
            zero = tf.reduce_sum(x) * 0.0
            return {"dets": tf.constant(dets) + zero, "labels": tf.constant(logits) + zero}

    module = Detector()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.__call__.get_concrete_function()], module
    )
    path.write_bytes(converter.convert())
    return path


def test_tflite_executor_decodes_the_exported_contract(tmp_path):
    from horos.backends.runtime import ArtifactModel, runtime_available

    if not runtime_available("tflite"):
        pytest.skip("no TFLite interpreter installed")
    bundle = tmp_path / "tflite"
    bundle.mkdir()
    artifact = _synthetic_tflite(bundle / "model_float32.tflite")
    model = ArtifactModel(artifact, card=_CARD)
    image = make_image(tmp_path / "img.png", 64, 48)
    _assert_contract(model, image)
    assert model.kind == "tflite" and model.device == "cpu"
    assert model.runtime.startswith(("LiteRT", "tensorflow.lite"))
    assert model.load().output_names == ["dets", "labels"]  # from the SignatureDef
    # R7: TFLite runs on CPU here — asking for CUDA is an error, not a quiet CPU run
    with pytest.raises(BackendError, match="runs on CPU"):
        ArtifactModel(artifact, card=_CARD, device="cuda").load()

    # end to end: a tflite bundle through resolve_source → the serve app
    card = {**_CARD, "format": "tflite", "artifact": artifact.name}
    (bundle / "model_card.json").write_text(json.dumps(card), "utf-8")
    client = _client_for(resolve_source(path=bundle), threshold=0.4)
    health = client.get("/health").get_json()
    assert health["kind"] == "tflite" and health["device"] == "cpu"
    assert health["runtime"].startswith(("LiteRT", "tensorflow.lite"))
    body = _post_image(client, image).get_json()
    assert [i["category_name"] for i in body["instances"]] == ["b", "a"]


# ------------------------------------------------------------- TensorRT executor


def _build_engine(onnx_path, engine_path):
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(onnx_path.read_bytes()), [
        parser.get_error(i) for i in range(parser.num_errors)
    ]
    try:
        serialized = builder.build_serialized_network(network, builder.create_builder_config())
    except Exception as exc:  # noqa: BLE001 — no usable CUDA device on this runner
        pytest.skip(f"TensorRT cannot build here: {exc}")
    if serialized is None:
        pytest.skip("TensorRT cannot build an engine on this machine (no CUDA device?)")
    engine_path.write_bytes(bytes(serialized))
    return engine_path


def test_tensorrt_executor_runs_a_real_engine(tmp_path):
    pytest.importorskip("onnx")
    from horos.backends.runtime import ArtifactModel, runtime_available

    if not runtime_available("tensorrt"):
        pytest.skip("tensorrt not installed")
    bundle = tmp_path / "tensorrt"
    bundle.mkdir()
    engine = _build_engine(_synthetic_detector(bundle / "model.onnx"), bundle / "model.trt")
    image = make_image(tmp_path / "img.png", 64, 48)
    model = ArtifactModel(engine, card=_CARD)
    _assert_contract(model, image)
    assert model.kind == "tensorrt" and model.device == "cuda"
    assert model.runtime.startswith("TensorRT ") and "memory" in model.runtime
    # R7: an engine is CUDA-only — 'cpu' is refused, never emulated
    with pytest.raises(BackendError, match="CUDA only"):
        ArtifactModel(engine, card=_CARD, device="cpu").load()
    # a corrupt / foreign engine says why instead of a low-level crash
    (bundle / "broken.trt").write_bytes(b"not an engine")
    with pytest.raises(BackendError, match="could not deserialize"):
        ArtifactModel(bundle / "broken.trt", card=_CARD).load()

    card = {**_CARD, "format": "tensorrt", "artifact": "model.trt"}
    (bundle / "model_card.json").write_text(json.dumps(card), "utf-8")
    client = _client_for(resolve_source(path=bundle), threshold=0.4)
    health = client.get("/health").get_json()
    assert health["kind"] == "tensorrt" and health["device"] == "cuda"
    assert health["runtime"].startswith("TensorRT")
    body = _post_image(client, image).get_json()
    assert [i["category_name"] for i in body["instances"]] == ["b", "a"]


# ------------------------------------------------------------- process control


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_start_status_stop_over_the_project_api(run):
    project, record = run
    app = create_app(project.root)
    app.testing = True
    client = app.test_client()
    try:
        assert client.get("/api/v1/serve").get_json()["running"] is False
        assert client.post("/api/v1/serve", json={}).status_code == 400
        assert client.post(
            "/api/v1/serve", json={"run_id": record.run_id, "port": "x"}
        ).status_code == 400
        # no ONNX export on a fake run: refused before anything is spawned
        refused = client.post("/api/v1/serve", json={"run_id": record.run_id, "format": "onnx"})
        assert refused.status_code == 400
        assert "no onnx export" in refused.get_json()["error"]["message"]

        port = _free_port()
        started = client.post(
            "/api/v1/serve",
            json={"run_id": record.run_id, "format": "checkpoint", "port": port, "threshold": 0.3},
        )
        assert started.status_code == 201, started.get_json()
        status = started.get_json()
        assert status["running"] and status["port"] == port and status["pid"]
        assert status["source"]["kind"] == "checkpoint"

        with urllib.request.urlopen(status["url"] + "/health", timeout=5) as response:
            health = json.loads(response.read())
        assert health["status"] == "ok" and health["default_threshold"] == 0.3
        with urllib.request.urlopen(status["url"] + "/model_card", timeout=5) as response:
            assert json.loads(response.read())["classes"] == ["forklift", "pallet"]

        again = client.post(
            "/api/v1/serve", json={"run_id": record.run_id, "format": "checkpoint", "port": port}
        )
        assert again.status_code == 400
        assert "already running" in again.get_json()["error"]["message"]
        assert client.get("/api/v1/serve").get_json()["running"] is True

        stopped = client.delete("/api/v1/serve").get_json()
        assert stopped["running"] is False and stopped["pid"] is None
        assert client.get("/api/v1/serve").get_json()["running"] is False
        assert client.delete("/api/v1/serve").get_json()["running"] is False  # idempotent
    finally:
        _reset_servers()
