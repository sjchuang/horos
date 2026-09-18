"""A fully functional fake backend used to test the interface contract (E4-T2)
and lazy loading (E4-T11) without any ML dependency."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from horos.backends.base import (
    Event,
    ExportSpec,
    ImageEmbedder,
    ImagePrediction,
    MetricsUpdated,
    ModelBackend,
    PredictedInstance,
    PredictionReady,
    ProgressUpdated,
    RunCompleted,
    RunStarted,
    TrainSpec,
    translate_backend_errors,
)
from horos.errors import BackendError

IMPORTED_MARKER = {"count": 0}
IMPORTED_MARKER["count"] += 1  # increments once per real import of this module


class FakeBackend(ModelBackend):
    family = "fake"

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        import time

        from horos.backends.base import RunFailed

        yield RunStarted(total=spec.epochs, config={"epochs": spec.epochs})
        # spec.extra is the expert passthrough; the fakes read pacing and
        # failure switches from it so subprocess tests can steer behavior.
        oom_above = spec.extra.get("oom_above_batch")
        if oom_above is not None and spec.batch_size > int(oom_above):
            yield RunFailed(
                error_code="backend_out_of_memory",
                message=f"simulated OOM at batch {spec.batch_size}",
            )
            return
        for epoch in range(spec.epochs):
            if spec.extra.get("sleep_per_epoch"):
                time.sleep(float(spec.extra["sleep_per_epoch"]))
            # same phase the real relay uses at train-epoch end — epoch
            # reconciliation counts only "epoch completed" progress events
            yield ProgressUpdated(
                current=epoch + 1, total=spec.epochs, phase="epoch completed"
            )
            yield MetricsUpdated(step=epoch + 1, metrics={"loss": 1.0 / (epoch + 1)})
        if spec.extra.get("fail"):
            yield RunFailed(error_code="backend_error", message="simulated failure")
            return
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = spec.output_dir / "best.fake"
        checkpoint.write_bytes(b"fake-weights")
        result = {"checkpoint": str(checkpoint), "batch_size": spec.batch_size}
        if spec.resume_from is not None:
            result["resumed_from"] = str(spec.resume_from)
        yield RunCompleted(result=result)

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        return ImagePrediction(
            image=str(image),
            instances=[
                PredictedInstance(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, category_id=0)
            ],
        )

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        images = list(images)
        yield RunStarted(total=len(images))
        for i, image in enumerate(images):
            yield PredictionReady(index=i, prediction=self.infer_one(image))
            yield ProgressUpdated(current=i + 1, total=len(images), phase="infer")
        yield RunCompleted(result={"count": len(images)})

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        yield RunStarted(config={"format": spec.format, "options": dict(spec.options)})
        yield ProgressUpdated(current=1, total=2, phase="exporting")
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"pytorch": "pt", "onnx": "onnx", "tensorrt": "trt"}.get(spec.format, spec.format)
        artifact = spec.output_dir / f"model.{suffix}"
        artifact.write_bytes(b"fake-export:" + checkpoint.read_bytes()[:16])
        result = {"artifact": str(artifact), "files": [artifact.name]}
        if spec.format == "tflite" and spec.options.get("int8"):
            # E8-T3b: a quantised variant next to the primary artifact, with
            # the metadata the real backend records
            int8 = spec.output_dir / "model_int8.tflite"
            int8.write_bytes(b"fake-int8")
            result["variants"] = {
                "int8": {"artifact": str(int8), "method": "dynamic_range", "weights": "int8",
                         "activations": "float32", "input_layout": "NHWC"},
            }
            result["files"].append(int8.name)
        yield ProgressUpdated(current=2, total=2, phase="exporting")
        yield RunCompleted(result=result)


class ExplodingBackend(FakeBackend):
    """Raises a foreign exception inside the translation wrapper (E4-T5)."""

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        with translate_backend_errors(self.family):
            raise RuntimeError("simulated library failure")

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        with translate_backend_errors(self.family):
            raise RuntimeError("CUDA error: out of memory (simulated)")


class FakeOpenVocabBackend(FakeBackend):
    """Deterministic open-vocabulary backend for the autolabel tests (E3).

    Default behavior: one detection per configured prompt, boxes spread along
    x, scores 0.9, 0.8, ... per prompt index. `score_by_name` overrides the
    base score per image file name so ranking tests can control uncertainty.
    """

    family = "fake-openvocab"

    def __init__(self, info=None, *, device=None, score_by_name=None):
        if info is not None:
            super().__init__(info, device=device)
        else:
            self.info = None
            self.device = device
        self.prompts: list[str] = []
        self.score_by_name = score_by_name or {}
        self.calls: list[str] = []

    def configure_prompts(self, prompts):
        self.prompts = list(prompts)

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        self.calls.append(Path(image).name)
        base = self.score_by_name.get(Path(image).name, 0.9)
        instances = [
            PredictedInstance(
                bbox=(10.0 + 30.0 * i, 10.0, 20.0, 20.0),
                score=max(base - 0.1 * i, 0.01),
                category_id=i,
            )
            for i in range(len(self.prompts))
        ]
        return ImagePrediction(image=str(image), instances=instances)


class FakeRefinerBackend:
    """Deterministic box->polygon refiner for the autolabel polygon tests:
    each box becomes a triangle inside it; boxes named in `fail_indices`
    return None (mask failure -> the box must survive as a box)."""

    family = "fake-refiner"

    def __init__(self, *, fail_indices=()):
        self.fail_indices = set(fail_indices)
        self.calls: list[tuple[str, int]] = []

    def polygons_for_boxes(self, image, boxes):
        self.calls.append((Path(image).name, len(boxes)))
        out = []
        for i, (x, y, w, h) in enumerate(boxes):
            if i in self.fail_indices:
                out.append(None)
            else:
                out.append([x, y, x + w, y, x + w / 2, y + h])
        return out


class FakePromptableSegmenter:
    """Deterministic interactive segmenter (SAM-T1/T2 tests): the mask is the
    prompt's box, or the bounding box of the positive points padded by 10 px;
    a negative point inside that box shaves its bottom half off. Counts
    encoder runs so the embedding cache can be asserted."""

    family = "fake-segmenter"

    def __init__(self):
        self.embed_calls: list[str] = []
        self.segment_calls = 0

    def embed(self, image):
        from PIL import Image

        from horos.backends.base import ImageEmbedding

        self.embed_calls.append(str(image))
        with Image.open(image) as im:
            width, height = im.size
        return ImageEmbedding(width, height, {"image": str(image)}, "fake-segmenter")

    def segment(self, embedding, prompt):
        from horos.backends.base import SegmentResult

        prompt = prompt.validated()
        self.segment_calls += 1
        if prompt.box is not None:
            x, y, w, h = prompt.box
        else:
            pos = [p for p, label in zip(prompt.points, prompt.labels, strict=True) if label == 1]
            if not pos:
                return SegmentResult(polygon=None, bbox=None, score=0.0, area=0)
            xs, ys = [p[0] for p in pos], [p[1] for p in pos]
            x, y = max(0.0, min(xs) - 10), max(0.0, min(ys) - 10)
            w = min(embedding.width, max(xs) + 10) - x
            h = min(embedding.height, max(ys) + 10) - y
        neg_inside = any(
            label == 0 and x <= px <= x + w and y <= py <= y + h
            for (px, py), label in zip(prompt.points, prompt.labels, strict=True)
        )
        if neg_inside:
            h = h / 2
        polygon = [x, y, x + w, y, x + w, y + h, x, y + h]
        return SegmentResult(
            polygon=polygon, bbox=(x, y, w, h), score=0.9 if not neg_inside else 0.8,
            area=int(w * h),
        )

    def polygons_for_boxes(self, image, boxes):
        embedding = self.embed(image)
        from horos.backends.base import SegmentPrompt

        return [self.segment(embedding, SegmentPrompt(box=b)).polygon for b in boxes]


def _spawn_probe_child(marker_path: str) -> None:
    """Runs in a spawn-context child process — must be module-level picklable."""
    Path(marker_path).write_text("spawned-child-ran", encoding="utf-8")


class SpawnProbeBackend(FakeBackend):
    """Mimics a DataLoader worker: starts a spawn-context child during
    training (E5-T6b). If the training worker were not `__main__`-guarded,
    the spawn re-import would re-execute it and the run would corrupt itself.
    """

    family = "fake-spawn"

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        import multiprocessing

        yield RunStarted(total=spec.epochs, config={"epochs": spec.epochs})
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        marker = spec.output_dir / "spawn_marker.txt"
        ctx = multiprocessing.get_context("spawn")  # the Windows/macOS default
        child = ctx.Process(target=_spawn_probe_child, args=(str(marker),))
        child.start()
        child.join(30)
        if not marker.is_file():
            from horos.backends.base import RunFailed

            yield RunFailed(error_code="backend_error",
                            message="spawned child never ran")
            return
        checkpoint = spec.output_dir / "best.fake"
        checkpoint.write_bytes(b"fake-weights")
        yield RunCompleted(result={"checkpoint": str(checkpoint)})


#: the character that killed a real training run: rfdetr draws its metrics
#: table with it and cp950 cannot encode it
TABLE_CORNER = "┏"


def _printing_child(text: str) -> None:
    """Runs in a spawn-context child — must be module-level picklable."""
    print(text, flush=True)  # noqa: T201


class NoisyBackend(FakeBackend):
    """Prints non-ASCII to stdout and stderr the way a real backend does (R7).

    The worker's stdout is a file, so without the UTF-8 environment the print
    raises UnicodeEncodeError and the run fails. A spawn-context child prints
    too: a reconfigure in the worker would not reach it, only the environment
    does.
    """

    family = "fake-noisy"

    def train(self, spec: TrainSpec) -> Iterator[Event]:
        import multiprocessing
        import sys

        yield RunStarted(total=spec.epochs, config={"epochs": spec.epochs})
        table = f"{TABLE_CORNER}━┓ Val ━ mAP 0.42 💡"
        print(table, flush=True)  # noqa: T201
        print(f"{table} (stderr)", file=sys.stderr, flush=True)  # noqa: T201
        ctx = multiprocessing.get_context("spawn")  # the Windows/macOS default
        child = ctx.Process(target=_printing_child, args=(f"{table} (child)",))
        child.start()
        child.join(30)
        if child.exitcode != 0:
            from horos.backends.base import RunFailed

            yield RunFailed(
                error_code="backend_error",
                message=f"spawned child died with exit code {child.exitcode}",
            )
            return
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = spec.output_dir / "best.fake"
        checkpoint.write_bytes(b"fake-weights")
        yield RunCompleted(result={"checkpoint": str(checkpoint)})


class FakeEmbedder(ImageEmbedder):
    """A deterministic image embedder for the active-learning loop tests
    (E10): the vector is built from the image's mean colour and size, so
    two files with the same content embed identically and different colours
    land apart. No ML dependency."""

    family = "fake-embedder"

    def __init__(self, info=None, *, device=None, checkpoint=None):
        super().__init__(info, device=device, checkpoint=checkpoint)
        self.calls: list[list[str]] = []

    @property
    def embedding_dim(self) -> int:
        return 6

    def embed_batch(self, images):
        import numpy as np
        from PIL import Image

        paths = [str(p) for p in images]
        self.calls.append(paths)
        out = []
        for path in paths:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                width, height = rgb.size
                mean = np.asarray(rgb, dtype=np.float64).reshape(-1, 3).mean(axis=0) / 255.0
            vec = np.asarray([*mean, width / 1000.0, height / 1000.0, 1.0])
            out.append((vec / (np.linalg.norm(vec) or 1.0)).tolist())
        return out


# colour → what the fake detector "sees": red = confident box, green = unsure
# box, blue = confident pallet, grey = nothing
COLOURS = {
    "red": (220, 20, 20),
    "green": (20, 220, 20),
    "blue": (20, 20, 220),
    "grey": (128, 128, 128),
}


def dominant_colour(path) -> str:
    from PIL import Image

    with Image.open(path) as im:
        r, g, b = im.convert("RGB").resize((1, 1)).getpixel((0, 0))
    if max(r, g, b) - min(r, g, b) < 30:
        return "grey"
    return {r: "red", g: "green", b: "blue"}[max(r, g, b)]


class FakeDetector(ModelBackend):
    """A detector that 'sees' by image colour (E10 loop tests): red = a
    confident 'box', green = a fence-sitting 'box', blue = a confident
    'pallet', grey = nothing. Candidates repeat the instance `support`
    times so PAL's support counts are meaningful."""

    family = "fake-detector"

    def __init__(self):
        super().__init__(None)
        self.seen: list[str] = []

    def infer_one(self, image, *, threshold: float = 0.5):
        self.seen.append(str(image))
        kind = dominant_colour(image)
        box = (10.0, 10.0, 30.0, 20.0)
        instances, candidates = [], []

        def add(name, score, support):
            inst = PredictedInstance(bbox=box, score=score, category_id=0, category_name=name)
            instances.append(inst)
            candidates.extend([inst] * support)

        if kind == "red":
            add("box", 0.95, 8)
        elif kind == "green":
            add("box", 0.5, 2)
        elif kind == "blue":
            add("pallet", 0.9, 7)
        return ImagePrediction(image=str(image), width=64, height=48,
                               instances=[i for i in instances if i.score >= threshold],
                               candidates=candidates)

    def configure_prompts(self, prompts):
        """Lets the fake stand in for the OWLv2 zero-shot scorer."""
        self.prompts = list(prompts)

    def train(self, spec):
        raise BackendError("fake", backend=self.family)

    def infer_batch(self, images, *, threshold=0.5):
        raise BackendError("fake", backend=self.family)

    def export(self, checkpoint, spec):
        raise BackendError("fake", backend=self.family)


def fake_get_backend(key, **kwargs):
    """Drop-in for horos.backends.get_backend in loop tests: embedding keys
    resolve to FakeEmbedder, everything else to FakeDetector."""
    if key in ("fake-embedder", "dinov2-small"):
        return FakeEmbedder()
    if key.startswith("sam"):
        return FakePromptableSegmenter()  # the polygon refiner
    return FakeDetector()
