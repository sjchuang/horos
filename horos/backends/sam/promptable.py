"""Shared prompt-decoding flow for the transformers-hosted SAM family.

SAM (v1) and SAM 2.1 expose the same shape through transformers: a processor
that resizes the image and scales prompts, `get_image_embeddings` for the
encoder, and a forward pass that accepts `image_embeddings` plus points /
labels / boxes. The two differ only in what `post_process_masks` needs and in
the embedding's type (a tensor vs a list of feature maps), so one mixin
serves both backends. Imports of torch/transformers stay inside methods (R1b).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from horos.backends.base import ImageEmbedding, SegmentPrompt, SegmentResult
from horos.backends.sam.polygonize import mask_to_shape


class TransformersPromptableMixin:
    """Requires: self._ensure_model() setting self._model / self._processor /
    self.device, self.info.key, self.family, and `_needs_reshaped_sizes`."""

    _needs_reshaped_sizes: bool = False  # SAM v1's post_process_masks wants them

    @property
    def _infer_lock(self) -> threading.RLock:
        # one prompt at a time per model instance: the processor and the
        # CUDA graph are not reentrant, and request threads do overlap
        lock = getattr(self, "_infer_lock_obj", None)
        if lock is None:
            lock = self._infer_lock_obj = threading.RLock()
        return lock

    def embed(self, image: Path) -> ImageEmbedding:
        self._ensure_model()
        with self._infer_lock:
            return self._embed(image)

    def _embed(self, image: Path) -> ImageEmbedding:
        import torch
        from PIL import Image

        with Image.open(image) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self._model.get_image_embeddings(pixel_values)
        data: dict[str, Any] = {
            "features": features,
            "original_sizes": inputs["original_sizes"],
            "reshaped_input_sizes": inputs.get("reshaped_input_sizes"),
        }
        return ImageEmbedding(width, height, data, self.info.key)

    def segment(self, embedding: ImageEmbedding, prompt: SegmentPrompt) -> SegmentResult:
        prompt = prompt.validated()
        self._ensure_model()
        with self._infer_lock:
            return self._segment(embedding, prompt)

    def _segment(self, embedding: ImageEmbedding, prompt: SegmentPrompt) -> SegmentResult:
        import torch
        from PIL import Image

        # the processor scales prompts from original-image pixels into model
        # space; it needs an image of the right size to know the scale — a
        # blank stand-in of the original dimensions is enough (no encoder run)
        stand_in = Image.new("RGB", (embedding.width, embedding.height))
        kwargs: dict[str, Any] = {}
        if prompt.points:
            kwargs["input_points"] = [[[list(p) for p in prompt.points]]]
            kwargs["input_labels"] = [[list(prompt.labels)]]
        if prompt.box is not None:
            x, y, w, h = prompt.box
            kwargs["input_boxes"] = [[[x, y, x + w, y + h]]]
        encoded = self._processor(images=stand_in, return_tensors="pt", **kwargs)
        model_kwargs: dict[str, Any] = {"image_embeddings": embedding.data["features"]}
        for key in ("input_points", "input_labels", "input_boxes"):
            if key in encoded:
                tensor = encoded[key]
                if tensor.dtype == torch.float64:  # MPS has no float64
                    tensor = tensor.float()
                model_kwargs[key] = tensor.to(self.device)
        with torch.no_grad():
            outputs = self._model(**model_kwargs, multimask_output=False)
        post_args = [outputs.pred_masks.cpu(), embedding.data["original_sizes"]]
        if self._needs_reshaped_sizes:
            post_args.append(embedding.data["reshaped_input_sizes"])
        masks = self._processor.post_process_masks(*post_args)[0]
        mask = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[0].numpy().astype(bool)
        score = float(outputs.iou_scores.flatten()[0])
        # the piece under the click (or the box centre) is the answer — the
        # model may paint the whole object, and a part kept earlier must not
        # come back as this prompt's polygon
        anchor = None
        positives = [p for p, label in zip(prompt.points, prompt.labels, strict=True) if label == 1]
        if positives:
            anchor = (int(positives[0][0]), int(positives[0][1]))
        elif prompt.box is not None:
            x, y, w, h = prompt.box
            anchor = (int(x + w / 2), int(y + h / 2))
        return self._result_from_mask(mask, score, anchor)

    @staticmethod
    def _result_from_mask(
        mask, score: float, anchor: tuple[int, int] | None = None
    ) -> SegmentResult:
        """Polygon, box and area all from one blob — the one under `anchor`
        when it is foreground, else the largest — so the three agree (stray
        specks neither widen the box nor become the polygon)."""
        shape = mask_to_shape(mask, anchor=anchor)
        if shape is None:
            return SegmentResult(polygon=None, bbox=None, score=score, area=0)
        return SegmentResult(polygon=shape.polygon, bbox=shape.bbox, score=score, area=shape.area)
