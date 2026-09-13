"""DINOv2 image-embedding backend (E10-T2) — imports `transformers` (R1).

`facebook/dinov2-small` (code and weights Apache 2.0) gives one 384-d
vector per image; the active-learning loop uses cosine distances between
these vectors for cold-start diversity selection (E10-T4) and for the
similarity penalty of PAL's GUIDE term (E10-T5). It is an encoder only:
train / infer / export are refused explicitly. transformers/torch load on
first `embed_batch` (R1b); weights go to horos's own cache, never bundled.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from horos.backends import weights
from horos.backends.base import MODEL_LOAD_LOCK, ImageEmbedder, translate_backend_errors

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

#: images per forward pass; small enough for Jetson memory, large enough to
#: keep the GPU busy on a desktop
BATCH_SIZE = 16


class DINOv2Backend(ImageEmbedder):
    family = "dinov2"

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

    def _ensure_model(self):
        if self._model is not None:
            return
        with MODEL_LOAD_LOCK, translate_backend_errors(self.family):
            if self._model is not None:  # another thread loaded it while we waited
                return
            import torch  # noqa: F401 — resolved lazily on first use (R1b)
            from transformers import AutoImageProcessor, AutoModel

            from horos.backends.device import select_device

            self.device = select_device(self.device).torch_device
            cache = str(weights.hf_cache_dir())
            self._processor = AutoImageProcessor.from_pretrained(
                self.info.hf_id, cache_dir=cache
            )
            self._model = AutoModel.from_pretrained(self.info.hf_id, cache_dir=cache).to(
                self.device
            )
            self._model.eval()

    @property
    def embedding_dim(self) -> int:
        self._ensure_model()
        return int(self._model.config.hidden_size)

    def embed_batch(self, images: Sequence[Path]) -> list[list[float]]:
        paths = [Path(p) for p in images]
        if not paths:
            return []
        self._ensure_model()
        out: list[list[float]] = []
        with translate_backend_errors(self.family):
            import torch
            from PIL import Image

            for start in range(0, len(paths), BATCH_SIZE):
                chunk = paths[start : start + BATCH_SIZE]
                pil = []
                for path in chunk:
                    with Image.open(path) as im:
                        pil.append(im.convert("RGB"))
                inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                # pooler_output is the layer-normed CLS token — DINOv2's
                # intended global image descriptor
                feats = torch.nn.functional.normalize(outputs.pooler_output, dim=-1)
                out.extend(feats.detach().cpu().tolist())
        return out
