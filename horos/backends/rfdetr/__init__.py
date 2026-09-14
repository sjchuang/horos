"""RF-DETR backend (E4-T4) — the only place allowed to import `rfdetr` (R1).

Training goes through rfdetr's PyTorch Lightning stack (the `rfdetr[train]`
extra). rfdetr exposes no user callback hook in 1.9.4, so per-epoch progress
and metrics are captured by wrapping `rfdetr.training.build_trainer` and
appending one Lightning callback that relays into horos's R4 event types —
acceptable because R5 pins the version exactly.

All ML imports happen lazily on first use (R1b).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from horos.backends.base import (
    Event,
    ExportSpec,
    ImagePrediction,
    MetricsUpdated,
    ModelBackend,
    PredictedInstance,
    PredictionReady,
    ProgressUpdated,
    RunCompleted,
    RunFailed,
    RunStarted,
    TrainSpec,
    translate_backend_errors,
)
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

_MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
    # instance segmentation (RF-DETR-Seg): same trainer, masks on top
    "rfdetr-seg-nano": "RFDETRSegNano",
    "rfdetr-seg-small": "RFDETRSegSmall",
    "rfdetr-seg-medium": "RFDETRSegMedium",
    "rfdetr-seg-large": "RFDETRSegLarge",
    "rfdetr-seg-xlarge": "RFDETRSegXLarge",
    "rfdetr-seg-2xlarge": "RFDETRSeg2XLarge",
}

# Best-first order among the files rfdetr training writes to output_dir.
# best_total is rfdetr's own pick of the better of its regular/EMA tracks.
_CHECKPOINT_PREFERENCE = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "last.ckpt",
)

# EMA factor applied to the monitored mAP before best-checkpoint comparison
# under the "smoothed_map" criterion (rfdetr's own smooth_alpha knob). Chosen
# so one noisy validation spike on a tiny valid split cannot lock in a bad
# checkpoint, while a real improvement still wins within ~2 epochs.
_SMOOTH_ALPHA = 0.6


def _train_kwargs(spec: TrainSpec) -> dict[str, Any]:
    """Map the backend-neutral TrainSpec onto rfdetr.train() keyword arguments.

    horos owns progress reporting (R4), so rfdetr's own loggers and progress
    bar are off by default; `spec.extra` is applied last so an expert override
    wins over every derived value (E5-S5).
    """
    kwargs: dict[str, Any] = {
        "dataset_dir": str(spec.dataset_dir),
        "output_dir": str(spec.output_dir),
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
        "tensorboard": False,
        "progress_bar": None,
        "run_test": False,
        "early_stopping": False,
        "log_per_class_metrics": False,
    }
    if spec.resolution is not None:
        kwargs["resolution"] = spec.resolution
    if spec.device is not None:
        kwargs["device"] = spec.device
    if spec.seed is not None:
        kwargs["seed"] = spec.seed
    if spec.resume_from is not None:
        kwargs["resume"] = str(spec.resume_from)
    if spec.checkpoint_criterion == "smoothed_map":
        kwargs["smooth_alpha"] = _SMOOTH_ALPHA
    # "loss" has no train kwarg — the checkpoint callback is re-pointed after
    # build_trainer instead (see _repoint_checkpoint_monitor)
    kwargs.update(spec.extra)
    return kwargs


def _snapshot_num_classes(dataset_dir: Path) -> int:
    """Number of classes in a horos training snapshot: the categories of
    train/_annotations.coco.json (rfdetr assigns label indices by their
    position, so a class with labels only in valid/test still counts)."""
    import json

    path = Path(dataset_dir) / "train" / "_annotations.coco.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return len({int(c["id"]) for c in data["categories"]})


#: confidence floor for the raw candidate boxes reported next to the final
#: detections (see infer_one); below it a DETR query is noise, not a proposal
CANDIDATE_FLOOR = 0.05


def _detections_to_instances(
    detections: Any,
    class_names: list[str] | None = None,
    *,
    masks: bool = False,
    min_score: float = 0.0,
) -> list[PredictedInstance]:
    """supervision.Detections (xyxy) → PredictedInstance list (COCO xywh).

    Fine-tuned rfdetr emits 0-based indices into its class list — NOT the
    training dataset's COCO category ids. The name is the portable identity,
    so it is attached to every instance (see PredictedInstance).

    With `masks`, a segmentation model's boolean masks (N, H, W) become one
    polygon per instance (largest blob, simplified) for detections scoring
    at least `min_score` — polygonising every low-confidence candidate would
    cost more than the detection itself."""
    instances: list[PredictedInstance] = []
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    mask_stack = getattr(detections, "mask", None) if masks else None
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        label = int(class_id[i]) if class_id is not None else 0
        name = None
        if class_names is not None and 0 <= label < len(class_names):
            name = class_names[label]
        score = float(confidence[i]) if confidence is not None else 1.0
        segmentation = None
        if mask_stack is not None and score >= min_score:
            from horos.backends.sam.polygonize import mask_to_polygon

            polygon = mask_to_polygon(mask_stack[i])
            if polygon:
                segmentation = [polygon]
        instances.append(
            PredictedInstance(
                bbox=(x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)),
                score=score,
                category_id=label,
                category_name=name,
                segmentation=segmentation,
            )
        )
    return instances


def _epoch_metrics(callback_metrics: Any, *, train_side: bool) -> dict[str, float]:
    """One epoch's Lightning callback_metrics, split into the train-side or
    val-side slice.

    Lightning runs validation BEFORE `on_train_epoch_end`, and the epoch-
    aggregated `train/*` values are only published in that later hook — so a
    relay that reads everything at validation end reports train metrics one
    epoch late, misses epoch 0, and silently drops the final epoch. Each side
    must be captured in its own hook.
    """
    picked: dict[str, float] = {}
    for key, value in callback_metrics.items():
        if key.startswith("train/") != train_side:
            continue
        try:
            picked[key] = float(value)
        except (TypeError, ValueError):
            continue
    return picked


def _repoint_checkpoint_monitor(callback: Any, monitor: str, mode: str) -> None:
    """Re-target an already-constructed Lightning ModelCheckpoint.

    rfdetr 1.9.4 hardcodes the best-model monitor to val mAP inside
    build_trainer; the "loss" criterion needs val/loss with mode=min. Mode
    lives in the parent's name-mangled init helper (kth_value must be reset
    alongside), so it is re-run here — acceptable against a pinned version
    (R5)."""
    callback.monitor = monitor
    callback._ModelCheckpoint__init_monitor_mode(mode)  # noqa: SLF001


def _confident_detections(dets, logits, threshold: float) -> list[tuple[Any, int, float]]:
    """(box cxcywh, class, score) for queries whose best sigmoid score >= threshold."""
    import numpy as np

    scores = 1.0 / (1.0 + np.exp(-logits))
    classes = scores.argmax(axis=-1)
    best = scores.max(axis=-1)
    keep = np.nonzero(best >= threshold)[0]
    return [(dets[i], int(classes[i]), float(best[i])) for i in keep]


def _box_iou_cxcywh(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, float(min(ax1, bx1) - max(ax0, bx0)))
    ih = max(0.0, float(min(ay1, by1) - max(ay0, by0)))
    inter = iw * ih
    union = float(a[2] * a[3] + b[2] * b[3]) - inter
    return inter / union if union > 0 else 0.0


class _BestTracker:
    """Answers "which epoch do the saved best weights come from" by watching
    the actual BestModelCallback state each epoch — exact under every
    criterion (raw, smoothed, loss) and across the regular/EMA tracks,
    without re-implementing any comparison logic.
    """

    def __init__(self) -> None:
        self.callback: Any = None
        self._last_regular: float | None = None
        self._last_ema: float | None = None
        self._regular_epoch: int | None = None
        self._ema_epoch: int | None = None
        self._last_emitted: tuple[int, bool] | None = None

    def observe(self, epoch: int) -> dict[str, float] | None:
        """Metrics to publish when the best checkpoint changed, else None."""
        cb = self.callback
        if cb is None:
            return None
        score = getattr(cb, "best_model_score", None)
        regular = float(score) if score is not None else None
        ema = float(getattr(cb, "_best_ema", 0.0) or 0.0)
        if regular is not None and regular != self._last_regular:
            self._last_regular, self._regular_epoch = regular, epoch
        if ema and ema != self._last_ema:
            self._last_ema, self._ema_epoch = ema, epoch
        # mirror on_fit_end's winner rule: EMA wins on strict >, compared
        # against the raw (un-smoothed) regular value when smoothing is on
        raw_regular = regular if not getattr(cb, "_smooth_alpha", 0.0) else None
        if raw_regular is None:
            raw_regular = float(getattr(cb, "_best_raw_regular", 0.0) or 0.0)
        is_ema = (
            getattr(cb, "_monitor_ema", None) is not None
            and self._ema_epoch is not None
            and ema > raw_regular
        )
        best_epoch = self._ema_epoch if is_ema else self._regular_epoch
        if best_epoch is None or (best_epoch, is_ema) == self._last_emitted:
            return None
        self._last_emitted = (best_epoch, is_ema)
        return {"best/epoch": float(best_epoch), "best/is_ema": float(is_ema)}


def _default_weights_filename(model_class: Any) -> str | None:
    """The variant's published default `pretrain_weights` filename; None when
    no usable string default can be found (then rfdetr's own paths handle it).

    Most variants declare `_model_config_class`; RFDETRLarge instead overrides
    `get_model_config()` (its declared class attribute is the bare ModelConfig,
    default None), so the config default is checked first and the unbound
    method — which ignores `self` in 1.9.4 (R5) — is the fallback."""
    config_class = getattr(model_class, "_model_config_class", None)
    fields = getattr(config_class, "model_fields", None) or {}
    default = getattr(fields.get("pretrain_weights"), "default", None)
    if isinstance(default, str):
        return default
    try:
        config = model_class.get_model_config(model_class)
        default = getattr(config, "pretrain_weights", None)
    except Exception:  # noqa: BLE001 — best effort; downloading stays rfdetr's job
        return None
    return default if isinstance(default, str) else None


def _best_checkpoint(output_dir: Path) -> Path | None:
    for name in _CHECKPOINT_PREFERENCE:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    return None


class RFDETRBackend(ModelBackend):
    family = "rfdetr"

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        super().__init__(info, device=device, checkpoint=checkpoint)
        self._model = None  # loaded lazily on first real use

    # ------------------------------------------------------------------ model
    def _resolve_device(self) -> str:
        from horos.backends.device import select_device

        prefer = self.device.partition(":")[0] if self.device else None
        return select_device(prefer).torch_device  # type: ignore[arg-type]

    def _model_class(self):
        import rfdetr  # noqa: PLC0415 — the sanctioned import site (R1)

        class_name = _MODEL_CLASSES.get(self.info.key)
        if class_name is None:
            raise BackendError(
                f"No RF-DETR class mapping for model '{self.info.key}' "
                f"(rfdetr 1.9.4 ships {sorted(_MODEL_CLASSES)})",
                backend=self.family,
            )
        return getattr(rfdetr, class_name)

    def _load(self):
        if self._model is not None:
            return self._model
        with translate_backend_errors(self.family):
            from horos.backends import env

            env.check_environment()
            device = self._resolve_device()
            if self.checkpoint is not None:
                from rfdetr.detr import RFDETR

                # trust_checkpoint: these are horos's own training outputs
                self._model = RFDETR.from_checkpoint(
                    self.checkpoint, trust_checkpoint=True, device=device
                )
            else:
                self._model = self._model_class()(device=device)
        return self._model

    def _pretrained_weights_events(self) -> Iterator[Event]:
        """Pre-fetch the variant's pretrained weights with R4 progress events.

        rfdetr downloads its default weights silently inside model
        construction, so the first run of each size sat on a bare `started`
        event for minutes and looked hung. Fetching the same file into
        rfdetr's own cache first (same path, MD5-checked) makes rfdetr skip
        its download, and horos owns the progress stream."""
        from rfdetr.assets.model_weights import ModelWeights, get_model_cache_dir

        filename = _default_weights_filename(self._model_class())
        if filename is None or Path(filename).is_absolute():
            return
        target = Path(get_model_cache_dir()) / filename
        if target.is_file():
            return
        asset = ModelWeights.from_filename(filename)
        if asset is None:
            return  # unknown to the registry — leave rfdetr's fallbacks to it

        from rfdetr.utilities.files import _validate_file_md5

        from horos.backends.weights import download_events

        path = yield from download_events(
            asset.url,
            filename=filename,
            dest_dir=target.parent,
            label=f"downloading {filename}",
        )
        if asset.md5_hash and not _validate_file_md5(str(path), asset.md5_hash):
            path.unlink(missing_ok=True)
            raise BackendError(
                f"Downloaded weights {filename} failed MD5 validation; the "
                "corrupt file was removed — check the network and retry.",
                backend=self.family,
            )

    # --------------------------------------------------------------- training
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        import queue as queue_mod
        import threading

        kwargs = _train_kwargs(spec)
        if kwargs.get("device") is None:
            kwargs["device"] = self._resolve_device()
        # rfdetr 1.9.4: an explicit device index ("cuda:0") is mapped to PTL
        # devices=[0], a list its _requests_multiple_devices() cannot parse
        # (AttributeError: 'list' object has no attribute 'strip'). horos
        # always trains on the default device, so dropping the ":0" is the
        # same device with the bug sidestepped. Remove on the next rfdetr
        # upgrade (R5).
        if str(kwargs["device"]).endswith(":0"):
            kwargs["device"] = str(kwargs["device"]).partition(":")[0]
        yield RunStarted(
            config={k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
        )

        try:
            with translate_backend_errors(self.family):
                try:
                    import pytorch_lightning as pl
                    import rfdetr.training as rf_training
                except ImportError as exc:
                    raise BackendError(
                        "RF-DETR training needs the training extras: "
                        'pip install "rfdetr[train]==1.9.4" '
                        "(or run: horos doctor --fix)",
                        backend=self.family,
                    ) from exc

                if spec.init_from is not None:
                    # warm start from an earlier run. rfdetr sizes the class head
                    # from the checkpoint unless num_classes is given explicitly —
                    # a dataset with more classes then indexes past the head and
                    # training dies with a CUDA device-side assert. With the
                    # count from this run's snapshot, load_pretrain_weights
                    # expands or trims the head to it (classes may change)
                    model = self._model_class()(
                        device=kwargs["device"], pretrain_weights=str(spec.init_from),
                        num_classes=_snapshot_num_classes(spec.dataset_dir),
                    )
                else:
                    yield from self._pretrained_weights_events()
                    model = self._model_class()(device=kwargs["device"])
                events: queue_mod.Queue = queue_mod.Queue()
                tracker = _BestTracker()

                class _EventRelay(pl.Callback):
                    def on_train_epoch_start(self, trainer, module):  # noqa: ANN001
                        events.put(
                            ProgressUpdated(
                                current=trainer.current_epoch,
                                total=trainer.max_epochs,
                                phase=(
                                    f"epoch {trainer.current_epoch + 1}"
                                    f"/{trainer.max_epochs}"
                                ),
                            )
                        )

                    def on_validation_epoch_end(self, trainer, module):  # noqa: ANN001
                        if trainer.sanity_checking:
                            return
                        metrics = _epoch_metrics(
                            trainer.callback_metrics, train_side=False
                        )
                        if metrics:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=metrics
                                )
                            )

                    def on_train_epoch_end(self, trainer, module):  # noqa: ANN001
                        # train/* is only published in this hook (see
                        # _epoch_metrics) — capture it here, on its own epoch
                        metrics = _epoch_metrics(
                            trainer.callback_metrics, train_side=True
                        )
                        if metrics:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=metrics
                                )
                            )
                        # checkpointing ran during validation (before this
                        # hook) — the tracker now sees the settled best state
                        best = tracker.observe(trainer.current_epoch)
                        if best:
                            events.put(
                                MetricsUpdated(
                                    step=trainer.current_epoch, metrics=best
                                )
                            )
                        events.put(
                            ProgressUpdated(
                                current=trainer.current_epoch + 1,
                                total=trainer.max_epochs,
                                phase="epoch completed",
                            )
                        )

                original_build_trainer = rf_training.build_trainer

                def build_trainer_with_relay(*args, **kw):  # noqa: ANN002, ANN003
                    trainer = original_build_trainer(*args, **kw)
                    best_cb = next(
                        (
                            c
                            for c in trainer.callbacks
                            if isinstance(c, rf_training.BestModelCallback)
                        ),
                        None,
                    )
                    if best_cb is not None:
                        if spec.checkpoint_criterion == "loss":
                            _repoint_checkpoint_monitor(best_cb, "val/loss", "min")
                            # the EMA track still measures mAP; comparing a
                            # loss against an mAP for best_total is meaningless
                            best_cb._monitor_ema = None  # noqa: SLF001
                        tracker.callback = best_cb
                    trainer.callbacks.append(_EventRelay())
                    return trainer

                failure: list[BaseException] = []

                def run_training() -> None:
                    try:
                        model.train(**kwargs)
                    except BaseException as exc:  # noqa: BLE001 — relayed to the stream
                        failure.append(exc)

                rf_training.build_trainer = build_trainer_with_relay
                try:
                    worker = threading.Thread(
                        target=run_training, name="horos-rfdetr-train"
                    )
                    worker.start()
                    while worker.is_alive() or not events.empty():
                        try:
                            yield events.get(timeout=1.0)
                        except queue_mod.Empty:
                            continue
                    worker.join()
                finally:
                    rf_training.build_trainer = original_build_trainer

                if failure:
                    raise failure[0]

                checkpoint = _best_checkpoint(Path(kwargs["output_dir"]))
                if checkpoint is None:
                    raise BackendError(
                        "Training finished but no checkpoint was written to "
                        f"{kwargs['output_dir']}",
                        backend=self.family,
                    )
        except Exception as exc:  # noqa: BLE001 — R4: the stream terminates itself
            code = getattr(exc, "code", "backend_error")
            yield RunFailed(error_code=code, message=str(exc))
            return

        yield RunCompleted(result={"checkpoint": str(checkpoint)})

    # -------------------------------------------------------------- inference
    #: images per forward pass in infer_many; halved on CUDA OOM, down to 1
    INFER_BATCH = 16

    def _to_prediction(
        self, image: Path, size: tuple[int, int], detections: Any, class_names: list[str],
        *, threshold: float, masks: bool,
    ) -> ImagePrediction:
        # segmentation models carry masks: polygonised for the final
        # instances only; candidates stay boxes (the scorer needs counts)
        converted = _detections_to_instances(
            detections, class_names or None, masks=masks, min_score=threshold
        )
        return ImagePrediction(
            image=str(image),
            width=size[0],
            height=size[1],
            instances=[c for c in converted if c.score >= threshold],
            candidates=[c.model_copy(update={"segmentation": None}) for c in converted],
        )

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        """One forward pass at a low floor: the boxes at or above `threshold`
        are the instances, everything the query set proposed down to
        CANDIDATE_FLOOR is reported as `candidates` — the raw proposals the
        active-learning scorer counts as support (E10-T5). RF-DETR has no
        NMS, so its "pre-NMS boxes" are simply its low-confidence queries."""
        model = self._load()
        with translate_backend_errors(self.family):
            from PIL import Image

            with Image.open(image) as im:
                size = im.size
            floor = min(threshold, CANDIDATE_FLOOR)
            detections = model.predict(str(image), threshold=floor, include_source_image=False)
            class_names = list(getattr(model, "class_names", None) or [])
            return self._to_prediction(
                image, size, detections, class_names, threshold=threshold, masks=True
            )

    def infer_many(
        self, images: Iterable[Path], *, threshold: float = 0.5, masks: bool = True
    ) -> list[ImagePrediction]:
        """`infer_one` for many images, but batched: INFER_BATCH images per
        forward pass while a thread pool decodes the next batch, so the GPU
        is not idle between photos. Same floor, same candidates. The active-
        learning scorer runs thousands of photos through here (E10-T5) —
        one at a time it spent most of its time outside the model."""
        paths = [Path(p) for p in images]
        if not paths:
            return []
        model = self._load()
        out: list[ImagePrediction] = []
        with translate_backend_errors(self.family):
            from concurrent.futures import ThreadPoolExecutor

            from PIL import Image

            def load(path: Path):
                with Image.open(path) as im:
                    return im.convert("RGB")

            floor = min(threshold, CANDIDATE_FLOOR)
            class_names = list(getattr(model, "class_names", None) or [])
            batch = max(1, self.INFER_BATCH)
            chunks = [paths[i:i + batch] for i in range(0, len(paths), batch)]

            def predict(chunk, loaded):
                # a chunk that does not fit is split in two; never a silent CPU fallback
                try:
                    dets = model.predict(loaded, threshold=floor, include_source_image=False)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or len(chunk) == 1:
                        raise
                    half = len(chunk) // 2
                    return (predict(chunk[:half], loaded[:half])
                            + predict(chunk[half:], loaded[half:]))
                if not isinstance(dets, list):
                    dets = [dets]
                return [
                    self._to_prediction(path, im.size, det, class_names,
                                        threshold=threshold, masks=masks)
                    for path, im, det in zip(chunk, loaded, dets, strict=True)
                ]

            with ThreadPoolExecutor(max_workers=4) as pool:
                ahead = [pool.submit(load, p) for p in chunks[0]]
                for k, chunk in enumerate(chunks):
                    loaded = [f.result() for f in ahead]
                    if k + 1 < len(chunks):
                        ahead = [pool.submit(load, p) for p in chunks[k + 1]]
                    out.extend(predict(chunk, loaded))
        return out

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        paths = [Path(p) for p in images]
        yield RunStarted(total=len(paths), config={"model": self.info.key})
        for index, path in enumerate(paths):
            prediction = self.infer_one(path, threshold=threshold)
            yield PredictionReady(index=index, prediction=prediction)
            yield ProgressUpdated(current=index + 1, total=len(paths), phase="inference")
        yield RunCompleted(result={"images": len(paths)})

    # ----------------------------------------------------------------- export
    def _export_io_spec(self, model, resolution: int) -> dict[str, Any]:
        """Input/output description for the model card (E8-S5)."""
        return {
            "input": {
                "name": "input",
                "dtype": "float32",
                "shape": [1, int(getattr(model.model_config, "num_channels", 3)),
                          resolution, resolution],
                "layout": "NCHW, RGB scaled to [0,1] then normalised with mean/std",
                "mean": [float(v) for v in model.means],
                "std": [float(v) for v in model.stds],
            },
            "outputs": [
                {"name": "dets", "shape": ["batch", "queries", 4],
                 "description": "boxes as (cx, cy, w, h) normalised to [0,1] of the input"},
                {"name": "labels", "shape": ["batch", "queries", "num_classes"],
                 "description": "class logits; apply sigmoid, take the max per query"},
                *([{"name": "masks", "shape": ["batch", "queries", "mask_h", "mask_w"],
                    "description": "per-query mask logits at reduced resolution; apply "
                                   "sigmoid and threshold at 0.5, then resize to the input"}]
                  if getattr(model.model_config, "segmentation_head", False) else []),
            ],
        }

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        """pytorch → weights.pt + class_names.txt (rfdetr's own loadable bundle);
        onnx / tensorrt → rfdetr's exporter (the [onnx] extra, and NVIDIA's
        tensorrt package for engines); tflite → the ONNX graph converted with
        onnx2tf (horos/backends/convert/tflite.py, opt-in toolchain, E8-T3)."""
        try:
            yield RunStarted(config={"format": spec.format, "model": self.info.key,
                                     "checkpoint": str(checkpoint)})
            with translate_backend_errors(self.family):
                yield ProgressUpdated(current=0, total=None, phase="loading checkpoint")
                model = self._load()
                out = Path(spec.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                resolution = int(model.model.resolution)
                variants: dict[str, dict[str, str]] = {}  # precision/quantisation variants
                if spec.format == "pytorch":
                    yield ProgressUpdated(current=0, total=None, phase="writing weights bundle")
                    model.export_for_roboflow(str(out))
                    artifact = out / "weights.pt"
                elif spec.format in ("onnx", "tensorrt", "tflite"):
                    import importlib.util

                    if importlib.util.find_spec("onnx") is None:
                        raise BackendError(
                            "ONNX export needs rfdetr's [onnx] extra (onnx, onnxsim, "
                            "onnxruntime) — run 'horos install' to add it.",
                            backend=self.family,
                        )
                    if spec.format == "tensorrt" and importlib.util.find_spec("tensorrt") is None:
                        raise BackendError(
                            "TensorRT export needs NVIDIA's 'tensorrt' Python package on "
                            "this machine; horos does not install it.",
                            backend=self.family,
                        )
                    if spec.format == "tflite":
                        from horos.backends.convert.tflite import (
                            INSTALL_HINT,
                            toolchain_available,
                        )

                        if not toolchain_available():
                            raise BackendError(INSTALL_HINT, backend=self.family)
                    graph_format = "onnx" if spec.format == "tflite" else spec.format
                    yield ProgressUpdated(
                        current=0, total=None,
                        phase=f"tracing and exporting {graph_format} (this takes a while)",
                    )
                    path = model.export(
                        output_dir=str(out),
                        format=graph_format,
                        opset_version=int(spec.options.get("opset", 17)),
                        batch_size=int(spec.options.get("batch_size", 1)),
                        dynamic_batch=bool(spec.options.get("dynamic_batch", False)),
                        fp16=bool(spec.options.get("fp16", True)),
                        verbose=False,
                    )
                    artifact = Path(path)
                    if spec.format == "tflite":
                        from horos.backends.convert.tflite import convert_onnx_to_tflite

                        yield ProgressUpdated(
                            current=0, total=None,
                            phase="converting ONNX to TFLite with onnx2tf (this takes a while)",
                        )
                        produced = convert_onnx_to_tflite(
                            artifact, out, input_names=["input"], stem=self.info.key
                        )
                        if spec.options.get("int8"):
                            # E8-T3b: int8 WEIGHTS (dynamic-range). Static int8 is
                            # not viable for this architecture (docs/BACKLOG.md):
                            # TFLite's quantised LayerNorm divides by zero at run
                            # time. The legacy converter is the one whose dynamic
                            # range covers the transformer's weights; it needs Erf
                            # approximated (no builtin) and hands out NHWC input.
                            yield ProgressUpdated(
                                current=0, total=None,
                                phase="quantising weights to int8 through the TensorFlow "
                                      "converter (several minutes)",
                            )
                            work = out / "_int8"
                            quant = convert_onnx_to_tflite(
                                artifact, work, stem=self.info.key, precisions=("int8_dynamic",),
                                backend="tf_converter", pseudo_operators=["Erf"],
                            )
                            int8_path = out / f"{self.info.key}_int8.tflite"
                            shutil.move(str(quant["int8_dynamic"]), int8_path)
                            shutil.rmtree(work, ignore_errors=True)
                            variants["int8"] = {
                                "artifact": str(int8_path),
                                "method": "dynamic_range",
                                "weights": "int8",
                                "activations": "float32",
                                "input_layout": "NHWC",
                                "erf": "tanh approximation (TFLite has no builtin Erf)",
                            }
                        artifact.unlink()  # the bundle ships TFLite only
                        artifact = produced["float32"]
                        variants["float16"] = {"artifact": str(produced["float16"]),
                                               "method": "float16 cast"}
                    if spec.format == "tensorrt":
                        # rfdetr builds the engine from an intermediate ONNX graph;
                        # the bundle ships the engine only (export onnx separately)
                        for stray in out.glob("*.onnx"):
                            stray.unlink()
                else:
                    raise BackendError(
                        f"RF-DETR cannot export '{spec.format}' through horos "
                        f"(supported: pytorch, onnx, tensorrt)",
                        backend=self.family,
                    )
                names_path = out / "class_names.txt"
                if not names_path.is_file():
                    names_path.write_text(
                        "\n".join(list(getattr(model, "class_names", None) or [])) + "\n",
                        "utf-8",
                    )
                files = sorted(p.name for p in out.iterdir() if p.is_file())
                result = {"artifact": str(artifact), "files": files, "variants": variants,
                          **self._export_io_spec(model, resolution)}
        except Exception as exc:  # noqa: BLE001 — R4: the stream terminates itself
            code = getattr(exc, "code", "backend_error")
            yield RunFailed(error_code=code, message=str(exc))
            return
        yield RunCompleted(result=result)

    def export_parity(
        self,
        artifact: Path,
        spec: ExportSpec,
        images: list[Path],
        *,
        tolerance: float = 0.02,
    ) -> dict[str, Any] | None:
        """ONNX only: feed the same preprocessed tensors to the original weights
        (rfdetr's export-mode forward, on CPU) and to onnxruntime, then compare
        the DETECTIONS, not the raw query order (E8-T5).

        RF-DETR's two-stage transformer picks its decoder queries by encoder
        score; on near-ties a 1e-5 numeric difference flips the selection and
        whole rows swap places, so an element-wise diff of the raw tensors is
        meaningless for this architecture. What must match is what a user
        sees: every detection scoring >= 0.25 on one side must have a same-class
        partner on the other side with IoU >= 0.9 and a score within
        `tolerance`. The raw max abs diff is still recorded for the record."""
        if spec.format not in ("onnx", "tflite"):
            return None
        if not images:
            return {"status": "skipped", "message": "no images in the run's snapshot",
                    "passed": True, "images": 0}
        with translate_backend_errors(self.family):
            from copy import deepcopy

            import numpy as np
            import torch
            from PIL import Image

            model = self._load()
            resolution = int(model.model.resolution)
            means = np.asarray(model.means, dtype=np.float32)
            stds = np.asarray(model.stds, dtype=np.float32)
            reference = deepcopy(model.model.model).to("cpu").eval()
            reference.export()  # the same graph rfdetr traced for ONNX
            if spec.format == "onnx":
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(artifact), providers=["CPUExecutionProvider"]
                )
                input_name = session.get_inputs()[0].name

                def run_artifact(array):
                    return session.run(None, {input_name: array})

                side = "onnxruntime (CPU)"
            else:
                from horos.backends.convert.tflite import TFLiteRunner

                runner = TFLiteRunner(artifact)
                run_artifact = runner.run
                side = "TFLite interpreter (CPU)"

            score_threshold, iou_min = 0.25, 0.9
            raw_max_diff = 0.0
            max_score_diff = 0.0
            min_iou = 1.0
            compared = 0
            unmatched = 0
            for image in images:
                with Image.open(image) as im:
                    arr = np.asarray(
                        im.convert("RGB").resize((resolution, resolution), Image.BILINEAR),
                        dtype=np.float32,
                    ) / 255.0
                arr = ((arr - means) / stds).transpose(2, 0, 1)[None]
                tensor = torch.from_numpy(np.ascontiguousarray(arr))
                with torch.no_grad():
                    outs = reference(tensor)
                if not isinstance(outs, tuple | list):
                    outs = [outs]
                torch_outs = [o.detach().cpu().numpy().astype(np.float32) for o in outs]
                ort_outs = [np.asarray(o, dtype=np.float32) for o in run_artifact(tensor.numpy())]
                for a, b in zip(torch_outs, ort_outs, strict=False):
                    if a.shape != b.shape:
                        return {"status": "failed", "passed": False, "images": len(images),
                                "message": f"output shape mismatch {a.shape} vs {b.shape}"}
                    raw_max_diff = max(raw_max_diff, float(np.max(np.abs(a - b))))
                ref_dets = _confident_detections(
                    torch_outs[0][0], torch_outs[1][0], score_threshold
                )
                onnx_dets = _confident_detections(
                    ort_outs[0][0], ort_outs[1][0], score_threshold
                )
                for side_a, side_b in ((ref_dets, onnx_dets), (onnx_dets, ref_dets)):
                    for box, cls, score in side_a:
                        best_iou, best_score = 0.0, None
                        for box_b, cls_b, score_b in side_b:
                            if cls_b != cls:
                                continue
                            iou = _box_iou_cxcywh(box, box_b)
                            if iou > best_iou:
                                best_iou, best_score = iou, score_b
                        if best_score is None or best_iou < iou_min:
                            unmatched += 1
                            continue
                        compared += 1
                        min_iou = min(min_iou, best_iou)
                        max_score_diff = max(max_score_diff, abs(float(score) - float(best_score)))
            passed = unmatched == 0 and max_score_diff <= tolerance
            return {
                "images": len(images),
                "detections_compared": compared,
                "unmatched_detections": unmatched,
                "score_threshold": score_threshold,
                "max_score_diff": max_score_diff,
                "min_iou": min_iou if compared else None,
                "iou_min": iou_min,
                "tolerance": tolerance,
                "raw_max_abs_diff": raw_max_diff,
                "passed": passed,
                "method": f"detections >= 0.25 from the original weights (export-mode forward, "
                          f"CPU) vs {side} on identical inputs: same class, IoU >= 0.9, "
                          f"score within tolerance; raw tensor diff recorded for reference",
            }
