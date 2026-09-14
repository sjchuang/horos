"""Abstract backend interface (E4-T2) and the shared event types (E4-T3, R4).

Every model dependency lives behind subclasses of `ModelBackend` in
`horos/backends/<family>/`. This module itself must import no ML library —
it is imported by the lazy loader before any backend is resolved.

R4: long-running work reports progress through these event types and nothing
else. Backends must not invent their own reporting format.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, TypeAdapter

from horos.errors import BackendError, BackendOutOfMemoryError, HorosError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

# ---------------------------------------------------------------- inference payloads


class PredictedInstance(BaseModel):
    """One predicted object. bbox is absolute-pixel COCO xywh.

    `category_id` is in the backend's own label space (e.g. rfdetr emits
    0-based indices into its class list); `category_name` is the portable
    identity — callers matching predictions against a dataset must map by
    name, never by raw id."""

    bbox: tuple[float, float, float, float]
    score: float
    category_id: int
    category_name: str | None = None
    segmentation: list[list[float]] | None = None
    #: the detector's full per-class probability vector for this box (same
    #: order as the backend's class list), when the architecture exposes one.
    #: Feeds the class-weighted image entropy of the active-learning scorer
    #: (E10-T5); None means "only `score` is known".
    class_probs: list[float] | None = None


class ImagePrediction(BaseModel):
    image: str  # path or identifier the caller passed in
    width: int | None = None
    height: int | None = None
    instances: list[PredictedInstance] = Field(default_factory=list)
    #: every raw box the detector considered before its confidence threshold
    #: (and NMS, where the architecture has one), down to a low floor. The
    #: active-learning scorer counts how many of these support each final
    #: detection — the "pre-NMS box count" feature of PAL (E10-T5). Backends
    #: that cannot expose raw candidates leave this empty; the scorer then
    #: falls back to confidence alone and says so.
    candidates: list[PredictedInstance] = Field(default_factory=list)


# --------------------------------------------------------------------------- events


class _EventBase(BaseModel):
    ts: float = Field(default_factory=time.time)
    run_id: str | None = None


class RunStarted(_EventBase):
    type: Literal["started"] = "started"
    total: int | None = None  # total units of work, if known up front
    config: dict[str, Any] = Field(default_factory=dict)


class ProgressUpdated(_EventBase):
    type: Literal["progress"] = "progress"
    current: int
    total: int | None = None
    phase: str = ""  # e.g. "epoch 2/10", "downloading weights"
    message: str = ""


class MetricsUpdated(_EventBase):
    type: Literal["metrics"] = "metrics"
    step: int
    metrics: dict[str, float]


class WarningRaised(_EventBase):
    type: Literal["warning"] = "warning"
    message: str


class PredictionReady(_EventBase):
    """One item of a batch inference finished; payload is an ImagePrediction dump."""

    type: Literal["prediction"] = "prediction"
    index: int
    prediction: ImagePrediction


class RunCompleted(_EventBase):
    type: Literal["completed"] = "completed"
    result: dict[str, Any] = Field(default_factory=dict)


class RunFailed(_EventBase):
    type: Literal["failed"] = "failed"
    error_code: str = "backend_error"
    message: str = ""
    #: structured payload mirroring HorosError.details (e.g. an import's
    #: conflicting file names), so a job failure carries what the synchronous
    #: Web error format would have carried
    details: dict[str, Any] = Field(default_factory=dict)


Event = Annotated[
    RunStarted
    | ProgressUpdated
    | MetricsUpdated
    | WarningRaised
    | PredictionReady
    | RunCompleted
    | RunFailed,
    Field(discriminator="type"),
]

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(data: dict[str, Any] | str | bytes) -> Event:
    """Parse a serialized event back into its typed form (for SSE / JSONL)."""
    if isinstance(data, (str, bytes)):
        return _event_adapter.validate_json(data)
    return _event_adapter.validate_python(data)


def dump_event(event: Event) -> str:
    """One-line JSON, suitable for JSONL streams and SSE data fields."""
    return event.model_dump_json()


#: Serialises model construction across the process. transformers loads
#: weights through process-global state (its lazy import machinery and the
#: meta-device init context), so two request threads calling from_pretrained
#: at once leave one model with meta tensors ("Cannot copy out of meta
#: tensor") or a half-imported module. Every backend's _ensure_model takes it.
MODEL_LOAD_LOCK = threading.RLock()


# ------------------------------------------------------------------------- specs


#: what "best checkpoint" means: mAP (detection quality, the default),
#: smoothed mAP (an EMA over the metric before comparison — robust to the
#: per-epoch noise of tiny validation splits), or validation loss.
CheckpointCriterion = Literal["map", "smoothed_map", "loss"]


class TrainSpec(BaseModel):
    """Backend-neutral training request. Backends map these onto their own knobs
    and must reject (not ignore) anything they cannot honor."""

    dataset_dir: Path
    output_dir: Path
    epochs: int
    batch_size: int
    resolution: int | None = None
    device: str | None = None  # resolved via backends/device.py when None
    seed: int | None = None
    #: full-state resume: weights + optimizer + schedule, same class set
    resume_from: Path | None = None
    #: warm start: weights of an earlier run's best checkpoint, fresh optimizer;
    #: the class head is resized to this run's class set, so classes may be
    #: added or dropped between runs (E5-S6, E10-T8 continuous training)
    init_from: Path | None = None
    checkpoint_criterion: CheckpointCriterion = "map"
    extra: dict[str, Any] = Field(default_factory=dict)


#: "pytorch" is the weights bundle (weights.pt + class_names.txt) the source
#: framework loads directly; the others are deployment graphs/engines
ExportFormat = Literal["pytorch", "onnx", "tensorrt", "tflite"]


class ExportSpec(BaseModel):
    format: ExportFormat
    output_dir: Path
    options: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- interface


class ModelBackend(ABC):
    """The only surface upper layers may program against (R1).

    Construction must be cheap and must not load weights or import the heavy
    dependency yet — first use does (R1b is about `import horos`, but keeping
    construction light keeps registry-driven UIs snappy too).
    """

    family: ClassVar[str]

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        self.info = info
        self.device = device
        # When set, inference loads these trained weights instead of the
        # model's pretrained defaults. Ignored by backends that never train.
        self.checkpoint = checkpoint

    # -- training ------------------------------------------------------------
    @abstractmethod
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        """Run training, yielding events (R4). Final event is RunCompleted with
        result["checkpoint"] pointing at the best weights, or RunFailed."""

    # -- inference -----------------------------------------------------------
    @abstractmethod
    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        """Single-image inference, synchronous."""

    @abstractmethod
    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        """Batch inference as an event stream: ProgressUpdated + PredictionReady
        per image, terminated by RunCompleted/RunFailed."""

    def infer_many(
        self, images: Iterable[Path], *, threshold: float = 0.5, masks: bool = True
    ) -> list[ImagePrediction]:
        """Predictions for `images`, in order, synchronously — the same result
        as `infer_one` per image, but a backend that can batch its forward
        passes (and decode images ahead of the GPU) overrides this. With
        `masks=False` a segmentation backend may skip the mask → polygon step:
        scorers that only count boxes (E10-T5) do not pay for polygons.
        The default loops over `infer_one`."""
        return [self.infer_one(Path(p), threshold=threshold) for p in images]

    # -- export --------------------------------------------------------------
    def export_parity(
        self,
        artifact: Path,
        spec: ExportSpec,
        images: list[Path],
        *,
        tolerance: float = 0.02,
    ) -> dict[str, Any] | None:
        """Compare the exported artifact's outputs with the original weights on
        the given images (E8-T5). Return {"max_abs_diff", "passed", ...}, or
        None when the backend cannot verify this format. Never raises for an
        unsupported format — the caller records "not available"."""
        return None

    @abstractmethod
    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        """Export a trained checkpoint. RunCompleted carries result["artifact"]."""


class OpenVocabularyBackend(ModelBackend):
    """Zero-shot detectors driven by text prompts (OWLv2 and successors).

    `category_id` in predictions indexes into the prompt list passed to
    `configure_prompts` — the caller owns the prompt→class mapping (E3-T2)."""

    @abstractmethod
    def configure_prompts(self, prompts: list[str]) -> None:
        """Set the text prompts used by subsequent infer_one/infer_batch calls."""


class BoxToMaskBackend(ModelBackend):
    """Promptable segmenters that turn detection boxes into masks (SAM and
    successors) — the polygon output path of autolabel."""

    @abstractmethod
    def polygons_for_boxes(
        self, image: Path, boxes: list[tuple[float, float, float, float]]
    ) -> list[list[float] | None]:
        """One flat [x1, y1, x2, y2, ...] polygon per COCO-xywh box, in the
        same order; None where no usable mask came back."""


class ImageEmbedder(ModelBackend):
    """Image-level feature extractors (DINOv2 and successors) for the
    active-learning loop (E10-T2): cold-start diversity selection and the
    similarity penalty of PAL work on cosine distances between these
    vectors. Encoders only — training, detection and export are refused
    with an explicit error rather than left abstract, so a subclass
    implements just `embed_batch` and `embedding_dim`."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Length of one vector."""

    @abstractmethod
    def embed_batch(self, images: Sequence[Path]) -> list[list[float]]:
        """One L2-normalised vector per image, in the order given."""

    def _refuse(self, op: str) -> BackendError:
        return BackendError(
            f"{self.info.display_name} is an image-embedding model; it does not {op}.",
            backend=self.family,
        )

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        raise self._refuse("train")

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        raise self._refuse("detect objects")

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        raise self._refuse("detect objects")

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise self._refuse("export")


class SegmentPrompt(BaseModel):
    """One interactive prompt on an image (pixel coordinates): positive /
    negative clicks and/or a rough COCO-xywh box (SAM-T1)."""

    points: list[tuple[float, float]] = Field(default_factory=list)
    #: 1 = this is the object, 0 = this is not; one per point
    labels: list[int] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None

    def validated(self) -> SegmentPrompt:
        if len(self.points) != len(self.labels):
            raise ValueError(
                f"points ({len(self.points)}) and labels ({len(self.labels)}) differ in length"
            )
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("labels must be 1 (positive) or 0 (negative)")
        if not self.points and self.box is None:
            raise ValueError("a prompt needs at least one point or a box")
        if self.box is not None and (self.box[2] <= 0 or self.box[3] <= 0):
            raise ValueError("box width and height must be positive")
        return self


class SegmentResult(BaseModel):
    """What one prompt produced: the mask as a polygon, its box and the
    model's own confidence (predicted IoU)."""

    polygon: list[float] | None = None  # flat [x1, y1, x2, y2, ...], image pixels
    bbox: tuple[float, float, float, float] | None = None  # COCO xywh of the mask
    score: float = 0.0
    area: int = 0


class ImageEmbedding:
    """Opaque handle for one image's encoder output (SAM-T2 caches these).
    `data` is whatever the backend needs to run its prompt decoder again."""

    __slots__ = ("width", "height", "data", "model_key")

    def __init__(self, width: int, height: int, data: Any, model_key: str):
        self.width = width
        self.height = height
        self.data = data
        self.model_key = model_key


class PromptableSegmenter(BoxToMaskBackend):
    """Interactive segmenters (SAM 2.1, SAM): the image encoder runs ONCE per
    image (`embed`), every click then runs only the light prompt decoder
    (`segment`). Box-batch refinement comes for free through the same path."""

    @abstractmethod
    def embed(self, image: Path) -> ImageEmbedding:
        """Run the image encoder; the handle is reusable across prompts."""

    @abstractmethod
    def segment(self, embedding: ImageEmbedding, prompt: SegmentPrompt) -> SegmentResult:
        """Decode one prompt against a cached embedding."""

    def polygons_for_boxes(
        self, image: Path, boxes: list[tuple[float, float, float, float]]
    ) -> list[list[float] | None]:
        if not boxes:
            return []
        embedding = self.embed(image)
        return [
            self.segment(embedding, SegmentPrompt(box=box)).polygon for box in boxes
        ]


# ------------------------------------------------------------------ error bridge


@contextmanager
def translate_backend_errors(backend: str):
    """Wrap backend-library calls so upper layers only ever see HorosError (E4-T5).

    OOM is recognized structurally where possible and by message otherwise —
    torch.cuda.OutOfMemoryError cannot be imported here without violating R1b.
    """
    try:
        yield
    except HorosError:
        raise
    except MemoryError as exc:
        raise BackendOutOfMemoryError(
            f"[{backend}] out of memory: {exc}", backend=backend
        ) from exc
    except Exception as exc:  # noqa: BLE001 — the whole point is to catch everything
        message = str(exc)
        if "out of memory" in message.lower():
            raise BackendOutOfMemoryError(
                f"[{backend}] out of memory: {message}", backend=backend
            ) from exc
        raise BackendError(
            f"[{backend}] {type(exc).__name__}: {message}", backend=backend
        ) from exc
