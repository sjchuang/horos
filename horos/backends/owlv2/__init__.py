"""OWLv2 backend (E3-T1) — the only place allowed to import `transformers` (R1).

Zero-shot open-vocabulary detection driven by text prompts. Inference-only:
train/export raise explicit errors. transformers/torch are imported lazily on
first inference (R1b); weights are fetched by transformers into horos's cache
(`weights.hf_cache_dir()`), never bundled.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from horos.backends import weights
from horos.backends.base import (
    MODEL_LOAD_LOCK,
    Event,
    ExportSpec,
    ImagePrediction,
    OpenVocabularyBackend,
    PredictedInstance,
    PredictionReady,
    ProgressUpdated,
    RunCompleted,
    RunStarted,
    TrainSpec,
    translate_backend_errors,
)
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo


#: confidence floor for the raw candidate boxes reported next to the final
#: detections; the active-learning scorer counts them as support (E10-T5)
CANDIDATE_FLOOR = 0.05


class OWLv2Backend(OpenVocabularyBackend):
    family = "owlv2"

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        super().__init__(info, device=device, checkpoint=checkpoint)
        self._model = None
        self._processor = None
        self._prompts: list[str] = []

    def configure_prompts(self, prompts: list[str]) -> None:
        if not prompts or not all(p.strip() for p in prompts):
            raise BackendError(
                "OWLv2 needs at least one non-empty text prompt", backend=self.family
            )
        self._prompts = [p.strip() for p in prompts]

    # ------------------------------------------------------------------ model
    def _ensure_model(self):
        if self._model is not None:
            return
        with MODEL_LOAD_LOCK, translate_backend_errors(self.family):
            if self._model is not None:  # another thread loaded it while we waited
                return
            import torch  # noqa: F401 — resolved lazily on first use (R1b)
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            from horos.backends.device import select_device

            self.device = select_device(self.device).torch_device
            cache = str(weights.hf_cache_dir())
            self._processor = Owlv2Processor.from_pretrained(
                self.info.hf_id, cache_dir=cache
            )
            self._model = Owlv2ForObjectDetection.from_pretrained(
                self.info.hf_id, cache_dir=cache
            ).to(self.device)
            self._model.eval()

    def _predict(self, image_path: Path, threshold: float) -> ImagePrediction:
        if not self._prompts:
            raise BackendError(
                "No prompts configured — call configure_prompts() first",
                backend=self.family,
            )
        self._ensure_model()
        with translate_backend_errors(self.family):
            import torch
            from PIL import Image

            with Image.open(image_path) as im:
                image = im.convert("RGB")
                width, height = image.size
                inputs = self._processor(
                    text=[self._prompts], images=image, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                # threshold 0 keeps every query, in query order, so the rows
                # line up with the raw per-prompt probabilities below
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=0.0,
                    target_sizes=torch.tensor([[height, width]]).to(self.device),
                )[0]
                probs = torch.sigmoid(outputs.logits[0]).detach().cpu().numpy()
            candidates = []
            floor = min(threshold, CANDIDATE_FLOOR)
            for q, (box, score, label) in enumerate(zip(
                results["boxes"], results["scores"], results["labels"], strict=True
            )):
                if float(score) < floor:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                candidates.append(
                    PredictedInstance(
                        bbox=(x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)),
                        score=float(score),
                        category_id=int(label),
                        # per-prompt probabilities feed the class-weighted image
                        # entropy of the active-learning scorer (E10-T5)
                        class_probs=[float(v) for v in probs[q]] if q < len(probs) else None,
                    )
                )
            return ImagePrediction(
                image=str(image_path),
                width=width,
                height=height,
                instances=[c for c in candidates if c.score >= threshold],
                candidates=candidates,
            )

    # -------------------------------------------------------------- interface
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        raise BackendError(
            "OWLv2 is a zero-shot autolabeling model; it is not trainable in horos.",
            backend=self.family,
        )

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        return self._predict(Path(image), threshold)

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        paths = [Path(p) for p in images]
        yield RunStarted(total=len(paths), config={"prompts": self._prompts})
        for index, path in enumerate(paths):
            prediction = self._predict(path, threshold)
            yield PredictionReady(index=index, prediction=prediction)
            yield ProgressUpdated(
                current=index + 1, total=len(paths), phase="autolabel"
            )
        yield RunCompleted(result={"images": len(paths)})

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise BackendError(
            "OWLv2 export is not supported; it is an autolabeling backend only.",
            backend=self.family,
        )
