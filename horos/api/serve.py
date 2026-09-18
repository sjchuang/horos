"""Local inference service (E8-T7, E8-S7): serve a run's model over HTTP.

Design decisions (confirmed 2026-09-12):

- `horos serve` is its own lightweight Flask app (horos/web/serve_app.py)
  with /predict, /health and /model_card — no project, no UI. That is what
  gets deployed to a Jetson; the project Web API only starts and stops it.
- The preferred source is an export bundle: ONNX, TensorRT engines and
  TFLite models run through the framework-free executor in
  horos/backends/runtime/ (E8-S6: the artifact itself is what serves; a
  Jetson serves the engine it built, a CPU box the TFLite or ONNX graph);
  a PyTorch weights bundle or the run's checkpoint loads through the run's
  own backend. A runtime that is missing or cannot honour the requested
  device is an explicit error before anything is spawned, never a silent
  substitution (Serve-T1/T2, 2026-09-12).
- The model card travels with the service (/model_card): licence, classes
  and I/O contract are queryable wherever the model runs (R3).
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core.project import Project
from horos.core.registry import get_model_info
from horos.core.streams import child_env
from horos.errors import ProjectError, UnknownModelError

if TYPE_CHECKING:
    from horos.backends.base import ImagePrediction

logger = logging.getLogger(__name__)

__all__ = [
    "ServeSource",
    "ServeStatus",
    "InferenceServer",
    "resolve_source",
    "create_inference_server",
    "start_server",
    "server_status",
    "stop_server",
]

ServeKind = Literal["onnx", "pytorch", "checkpoint", "tensorrt", "tflite"]
SERVE_FORMATS: tuple[str, ...] = ("onnx", "tensorrt", "tflite", "pytorch", "checkpoint")
#: formats the framework-free executor runs (horos/backends/runtime/)
ARTIFACT_FORMATS: tuple[str, ...] = ("onnx", "tensorrt", "tflite")
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")
_STARTUP_TIMEOUT = 90.0


class ServeSource(BaseModel):
    kind: ServeKind
    #: the file the model is loaded from (model.onnx, weights.pt, best.pth ...)
    path: str
    bundle_dir: str | None = None
    run_id: str | None = None
    model: str | None = None
    classes: list[str] = Field(default_factory=list)
    #: model_card.json of the bundle, or a minimal card for a bare checkpoint
    card: dict[str, Any] = Field(default_factory=dict)
    #: testing hook carried over from the run's config (fake backends)
    entrypoint_override: str | None = None


class ServeStatus(BaseModel):
    running: bool = False
    pid: int | None = None
    host: str | None = None
    port: int | None = None
    url: str | None = None
    source: ServeSource | None = None
    started_at: str | None = None
    error: str | None = None
    log: str | None = None


# ------------------------------------------------------------------ sources


def _minimal_card(
    *, run_id: str | None, model: str | None, classes: list[str], fmt: str, artifact: Path
) -> dict[str, Any]:
    import horos

    display, code_lic, weights_lic, url = model or "unknown", "unknown", "unknown", ""
    if model:
        try:
            info = get_model_info(model)
            display, code_lic, weights_lic, url = (
                info.display_name, info.code_license, info.weights_license, info.license_url,
            )
        except UnknownModelError:
            pass
    return {
        "horos_version": horos.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id or "",
        "model": model or "unknown",
        "display_name": display,
        "code_license": code_lic,
        "weights_license": weights_lic,
        "license_url": url,
        "format": fmt,
        "artifact": artifact.name,
        "files": [artifact.name],
        "classes": list(classes),
        "input": {}, "outputs": [], "dataset": {}, "metrics": {}, "hyperparameters": {},
        "parity": {"status": "not_available"},
        "portability": "",
    }


def _source_from_bundle(bundle_dir: Path, *, run_id: str | None = None) -> ServeSource:
    card_path = bundle_dir / "model_card.json"
    if not card_path.is_file():
        raise ProjectError(
            f"{bundle_dir} is not a horos export bundle (no model_card.json). Export the "
            f"run first, or point at the run's checkpoint with format='checkpoint'."
        )
    try:
        card = json.loads(card_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Unreadable model card {card_path}: {exc}") from exc
    fmt = str(card.get("format") or "").lower()
    if fmt not in ("onnx", "pytorch", "tensorrt", "tflite"):
        raise ProjectError(f"Model card {card_path} declares unknown format '{fmt}'")
    artifact = bundle_dir / str(card.get("artifact") or "")
    if not artifact.is_file():
        raise ProjectError(f"Bundle {bundle_dir} is missing its artifact {artifact.name}")
    classes = list(card.get("classes") or [])
    names_path = bundle_dir / "class_names.txt"
    if not classes and names_path.is_file():
        classes = [n for n in names_path.read_text("utf-8").splitlines() if n.strip()]
    return ServeSource(
        kind=fmt,  # type: ignore[arg-type]
        path=str(artifact),
        bundle_dir=str(bundle_dir),
        run_id=run_id or (card.get("run_id") or None),
        model=card.get("model"),
        classes=classes,
        card=card,
    )


def _source_from_run(project: Project, run_id: str, fmt: str) -> ServeSource:
    from horos.api.export import exports_dir
    from horos.api.train import TrainRunConfig, _run_dir, read_record

    run_dir = _run_dir(project, run_id)
    record = read_record(run_dir)
    if fmt == "checkpoint":
        if record.state != "completed" or not record.checkpoint:
            raise ProjectError(
                f"Run {run_id} is '{record.state}' and has no usable checkpoint to serve."
            )
        checkpoint = Path(record.checkpoint)
        if not checkpoint.is_file():
            raise ProjectError(f"Checkpoint of run {run_id} is missing: {checkpoint}")
        config = TrainRunConfig.model_validate_json((run_dir / "config.json").read_text("utf-8"))
        return ServeSource(
            kind="checkpoint",
            path=str(checkpoint),
            run_id=run_id,
            model=record.model,
            classes=list(record.dataset_classes),
            card=_minimal_card(
                run_id=run_id, model=record.model, classes=record.dataset_classes,
                fmt="checkpoint", artifact=checkpoint,
            ),
            entrypoint_override=config.entrypoint_override,
        )
    if fmt not in ("onnx", "pytorch", "tensorrt", "tflite"):
        raise ProjectError(
            f"Unknown serve format '{fmt}' — one of {', '.join(SERVE_FORMATS)}"
        )
    bundle = exports_dir(project, run_id) / fmt
    if not (bundle / "model_card.json").is_file():
        raise ProjectError(
            f"Run {run_id} has no {fmt} export yet — export it on the Training page "
            f"(or 'horos export-model --format {fmt}') first, or serve format='checkpoint'."
        )
    return _source_from_bundle(bundle, run_id=run_id)


def _unpack_bundle(zip_path: Path) -> Path:
    """Extract an export zip once into ~/.horos/serve/<digest>/ and reuse it."""
    from horos.backends.weights import weights_root

    digest = hashlib.sha1(
        f"{zip_path.resolve()}:{zip_path.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    # next to the weight cache: ~/.horos/serve/<digest>/ (HOROS_WEIGHTS_DIR aware)
    target = weights_root().parent / "serve" / digest
    if not (target / "model_card.json").is_file():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                name = Path(member.filename).name  # flat bundles; never trust paths
                if not name or member.is_dir():
                    continue
                (target / name).write_bytes(zf.read(member))
    return target


@capability(
    "serve.source",
    summary="Resolve what to serve: a run's export bundle / checkpoint, or a bundle path",
    web_route=None,
    not_web_because="POST /api/v1/serve takes run_id + format and resolves the source itself.",
    cli="serve",
)
def resolve_source(
    project: Project | None = None,
    *,
    run_id: str | None = None,
    format: str = "onnx",
    path: Path | str | None = None,
    model: str | None = None,
) -> ServeSource:
    """What to serve: a run's export bundle or checkpoint (`project` + `run_id`
    + `format`), or a `path` — an export bundle directory or zip, a
    model_card.json, a bare .onnx file, or a checkpoint file (needs `model`)."""
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise ProjectError(f"No such model source: {path}")
        if path.is_dir():
            return _source_from_bundle(path)
        if path.name == "model_card.json":
            return _source_from_bundle(path.parent)
        if path.suffix.lower() == ".zip":
            return _source_from_bundle(_unpack_bundle(path))
        from horos.backends.runtime import SUFFIX_KINDS

        kind = SUFFIX_KINDS.get(path.suffix.lower())
        if kind is not None:  # a bare .onnx / .trt / .engine / .tflite file
            names = path.with_name("class_names.txt")
            classes = (
                [n for n in names.read_text("utf-8").splitlines() if n.strip()]
                if names.is_file() else []
            )
            card_path = path.with_name("model_card.json")
            card = json.loads(card_path.read_text("utf-8")) if card_path.is_file() else (
                _minimal_card(run_id=None, model=model, classes=classes, fmt=kind, artifact=path)
            )
            return ServeSource(
                kind=kind, path=str(path), bundle_dir=str(path.parent),  # type: ignore[arg-type]
                run_id=card.get("run_id") or None, model=card.get("model") or model,
                classes=classes or list(card.get("classes") or []), card=card,
            )
        if path.suffix.lower() in _CHECKPOINT_SUFFIXES:
            if not model:
                raise ProjectError(
                    f"Serving a bare checkpoint ({path.name}) needs the model key it was "
                    f"trained with (e.g. --model rfdetr-nano); a run id or an export "
                    f"bundle carries this itself."
                )
            return ServeSource(
                kind="checkpoint", path=str(path), model=model, classes=[],
                card=_minimal_card(run_id=None, model=model, classes=[], fmt="checkpoint",
                                   artifact=path),
            )
        raise ProjectError(
            f"Cannot serve {path.name}: expected an export bundle directory or zip, a "
            f"model_card.json, an .onnx / .trt / .engine / .tflite artifact or a "
            f"checkpoint (.pt/.pth/.ckpt)."
        )
    if project is None or not run_id:
        raise ProjectError("resolve_source needs a project and run_id, or a path")
    return _source_from_run(project, run_id, format)


# ------------------------------------------------------------------ server


def check_runtime(source: ServeSource) -> None:
    """Refuse, before spawning anything, what this machine cannot execute:
    a TensorRT engine on a platform without TensorRT (macOS) or without the
    tensorrt package, a TFLite model without an interpreter, ONNX without
    onnxruntime. Import-free, so it is cheap enough for the project API."""
    if source.kind not in ARTIFACT_FORMATS:
        return
    from horos.backends.runtime import INSTALL_HINTS, runtime_available

    if source.kind == "tensorrt":
        from horos.api.system import ensure_supported

        ensure_supported("export_tensorrt")  # engines run where they are built: never macOS
    if not runtime_available(source.kind):
        raise ProjectError(
            f"Cannot serve the {source.kind} artifact {Path(source.path).name}: "
            f"{INSTALL_HINTS[source.kind]}"
        )


def load_source(source: ServeSource, *, device: str | None = None):
    """The object whose `infer_one` serves predictions."""
    if source.kind in ARTIFACT_FORMATS:
        from horos.backends.runtime import ArtifactModel

        check_runtime(source)
        model = ArtifactModel(source.path, card=source.card, classes=source.classes,
                              device=device, kind=source.kind)
        model.load()  # fail at start-up, not on the first request
        return model
    checkpoint = Path(source.path)
    if source.entrypoint_override:
        import importlib

        module_name, _, class_name = source.entrypoint_override.partition(":")
        backend_cls = getattr(importlib.import_module(module_name), class_name)
        return backend_cls(None, device=device, checkpoint=checkpoint)
    if not source.model:
        raise ProjectError("A checkpoint source needs its model key to load the backend")
    from horos.backends import get_backend

    return get_backend(source.model, device=device, checkpoint=checkpoint)


class InferenceServer:
    """One loaded model plus the metadata /health and /model_card report."""

    def __init__(self, source: ServeSource, model, *, threshold: float = 0.5,
                 device: str | None = None):
        self.source = source
        self.model = model
        self.default_threshold = threshold
        # the executor records the provider it actually chose; framework
        # backends resolve their device lazily, so report the request ("auto")
        self.device = getattr(model, "device", None) or device or "auto"
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.requests = 0
        self._lock = threading.Lock()

    def predict(self, image: Path | str, *, threshold: float | None = None) -> ImagePrediction:
        thr = self.default_threshold if threshold is None else float(threshold)
        if not 0.0 <= thr <= 1.0:
            raise ProjectError(f"threshold must be within [0, 1], got {thr}")
        with self._lock:  # one request at a time per model — sessions are not reentrant
            self.requests += 1
            prediction = self.model.infer_one(Path(image), threshold=thr)
        # the threshold is this service's contract whatever the backend does with it
        kept = [i for i in prediction.instances if i.score >= thr]
        if len(kept) != len(prediction.instances):
            prediction = prediction.model_copy(update={"instances": kept})
        return prediction

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.source.model,
            "kind": self.source.kind,
            "artifact": Path(self.source.path).name,
            "run_id": self.source.run_id,
            "classes": len(self.source.classes),
            "device": self.device,
            "runtime": getattr(self.model, "runtime", None) or self.source.kind,
            "default_threshold": self.default_threshold,
            "started_at": self.started_at,
            "requests": self.requests,
        }


@capability(
    "serve.predict",
    summary="Run the served model on one image (the standalone 'horos serve' app)",
    web_route=None,
    not_web_because="Exposed by the standalone serve app (POST /predict), not the project API.",
    cli="serve",
)
def create_inference_server(
    source: ServeSource, *, device: str | None = None, threshold: float = 0.5
) -> InferenceServer:
    model = load_source(source, device=device)
    return InferenceServer(source, model, threshold=threshold, device=device)


# ------------------------------------------------------------ process control

_SERVERS: dict[str, dict[str, Any]] = {}  # project root -> {process, status}
_SERVERS_LOCK = threading.Lock()


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host if host not in ("0.0.0.0", "") else "127.0.0.1", port))
        except OSError:
            return False
    return True


def _probe(url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — not up yet
        return None


def _entry(project: Project) -> dict[str, Any] | None:
    with _SERVERS_LOCK:
        return _SERVERS.get(str(project.root))


def _status_of(entry: dict[str, Any] | None) -> ServeStatus:
    if entry is None:
        return ServeStatus()
    process: subprocess.Popen = entry["process"]
    status: ServeStatus = entry["status"]
    running = process.poll() is None
    if not running and status.running:
        status = status.model_copy(update={
            "running": False,
            "error": f"server exited with code {process.returncode}",
        })
        entry["status"] = status
    return status


@capability(
    "serve.start",
    summary="Start the local inference service for a run's export or checkpoint",
    web_route="/api/v1/serve",
    web_methods=("POST",),
    cli="serve",
)
def start_server(
    project: Project,
    *,
    run_id: str,
    format: str = "onnx",
    host: str = "127.0.0.1",
    port: int = 8080,
    threshold: float = 0.5,
    device: str | None = None,
    wait: float = _STARTUP_TIMEOUT,
) -> ServeStatus:
    """Spawn `horos serve` for this project's run in a subprocess (spawn-safe:
    a fresh interpreter, R7) and wait until /health answers. One service per
    project at a time; stop the running one first."""
    source = resolve_source(project, run_id=run_id, format=format)  # fail early
    check_runtime(source)  # ... and before spawning a process that cannot load it
    current = _status_of(_entry(project))
    if current.running:
        raise ProjectError(
            f"A service is already running for this project on {current.url} — stop it first."
        )
    if not 1 <= int(port) <= 65535:
        raise ProjectError(f"port must be within 1..65535, got {port}")
    if not _port_free(host, int(port)):
        raise ProjectError(f"Port {port} is already in use on {host}")
    from horos.api.train import _run_dir

    log_path = _run_dir(project, run_id) / "serve.log"
    cmd = [
        sys.executable, "-m", "horos.cli", "serve",
        "--run", run_id, "--format", format, "--project", str(project.root),
        "--host", host, "--port", str(int(port)), "--threshold", str(float(threshold)),
    ]
    if device:
        cmd += ["--device", device]
    log_handle = log_path.open("ab")
    process = subprocess.Popen(  # noqa: S603 — our own CLI, arguments built above
        cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        cwd=str(project.root), env=child_env(),
    )
    log_handle.close()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{int(port)}"
    status = ServeStatus(
        running=True, pid=process.pid, host=host, port=int(port), url=url, source=source,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        log=str(log_path),
    )
    with _SERVERS_LOCK:
        _SERVERS[str(project.root)] = {"process": process, "status": status}

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = _tail(log_path)
            status = status.model_copy(update={
                "running": False,
                "error": f"server exited with code {process.returncode}: {tail}",
            })
            with _SERVERS_LOCK:
                _SERVERS[str(project.root)]["status"] = status
            raise ProjectError(status.error or "serve process exited")
        if _probe(url) is not None:
            return status
        time.sleep(0.25)
    process.terminate()
    raise ProjectError(
        f"The service did not answer on {url}/health within {wait:.0f}s — see {log_path}"
    )


def _tail(path: Path, lines: int = 8) -> str:
    try:
        text = path.read_text("utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(text[-lines:])


@capability(
    "serve.status",
    summary="Whether the project's local inference service is running, and where",
    web_route="/api/v1/serve",
    web_methods=("GET",),
    cli=None,
    not_cli_because="'horos serve' runs in the foreground and prints its URL.",
)
def server_status(project: Project) -> ServeStatus:
    return _status_of(_entry(project))


@capability(
    "serve.stop",
    summary="Stop the project's local inference service",
    web_route="/api/v1/serve",
    web_methods=("DELETE",),
    cli=None,
    not_cli_because="Ctrl-C stops the foreground 'horos serve'.",
)
def stop_server(project: Project, *, timeout: float = 10.0) -> ServeStatus:
    entry = _entry(project)
    if entry is None:
        return ServeStatus()
    process: subprocess.Popen = entry["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
    status: ServeStatus = entry["status"]
    stopped = status.model_copy(update={"running": False, "pid": None, "error": None})
    with _SERVERS_LOCK:
        _SERVERS.pop(str(project.root), None)
    return stopped


def _reset_servers() -> None:  # tests only
    with _SERVERS_LOCK:
        for entry in _SERVERS.values():
            process = entry["process"]
            if process.poll() is None:
                process.kill()
        _SERVERS.clear()
