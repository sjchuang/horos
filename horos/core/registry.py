"""Model registry: static metadata only (E4-T1).

R3: license metadata is a first-class field, recorded separately for code and
weights — a permissive codebase does not imply permissive weights.

R1b: this module must stay importable without any backend dependency. Backends
are referenced by entrypoint strings ("module:ClassName") and resolved lazily
by `horos.backends.get_backend` (E4-T11).

§9: only Apache-licensed RF-DETR sizes (Nano/Small/Medium/Large) are listed in
the public registry. XL / 2XL (PML 1.0) live in a separate gated table and can
only be loaded with `acknowledge_non_apache=True` (E4-T9).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from horos.errors import UnknownModelError

Task = Literal["detection", "instance_segmentation", "embedding"]

APACHE_2_0 = "Apache-2.0"


class ModelInfo(BaseModel):
    """Static metadata for one loadable model. No backend imports allowed here."""

    key: str
    family: str
    display_name: str
    task: Task
    code_license: str
    weights_license: str
    license_url: str
    input_resolution: int
    params_millions: float
    latency_hint: str
    entrypoint: str  # "horos.backends.<family>:<ClassName>", resolved lazily
    hf_id: str | None = None  # HuggingFace hub id, for transformers-hosted weights
    #: can this model be fine-tuned through horos? (drives the training UI list)
    trainable: bool = False
    #: answers point/box prompts with a mask (SAM and successors): the only
    #: models the annotator's Draw tool and boxes → polygons may pick
    promptable: bool = False
    #: a training resolution must be a multiple of this (backbone patch size ×
    #: window count); the hyperparameter rules snap derived values to it
    resolution_step: int = 64
    notes: str = ""

    @property
    def license(self) -> str:
        """Display license: weights license governs what users can ship."""
        return self.weights_license

    @property
    def requires_acknowledgement(self) -> bool:
        return self.weights_license != APACHE_2_0 or self.code_license != APACHE_2_0


_RFDETR_ENTRYPOINT = "horos.backends.rfdetr:RFDETRBackend"
_RFDETR_LICENSE_URL = "https://github.com/roboflow/rf-detr/blob/main/LICENSE"
_OWLV2_ENTRYPOINT = "horos.backends.owlv2:OWLv2Backend"

# rfdetr==1.9.4 (pinned, R5). Sizes/params from the upstream release notes.
_MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            key="rfdetr-nano",
            family="rfdetr",
            display_name="RF-DETR Nano",
            task="detection",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=384,
            params_millions=30.5,
            latency_hint="fastest — Jetson-friendly real-time",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
        ),
        ModelInfo(
            key="rfdetr-small",
            family="rfdetr",
            display_name="RF-DETR Small",
            task="detection",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=512,
            params_millions=32.1,
            latency_hint="fast — good default for Jetson",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
        ),
        ModelInfo(
            key="rfdetr-medium",
            family="rfdetr",
            display_name="RF-DETR Medium",
            task="detection",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=576,
            params_millions=33.7,
            latency_hint="balanced accuracy/latency",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
        ),
        ModelInfo(
            key="rfdetr-large",
            family="rfdetr",
            display_name="RF-DETR Large",
            task="detection",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=704,
            params_millions=129.0,
            latency_hint="highest accuracy — desktop GPU recommended",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
        ),
        # RF-DETR-Seg (instance segmentation). Code and weights Apache 2.0 for
        # every size — unlike detection, the seg XL/2XL weights ship in the
        # open `rfdetr` package, not in rfdetr_plus (verified 2026-09-13 against
        # rfdetr/assets/model_weights.py and the upstream README).
        ModelInfo(
            key="rfdetr-seg-nano",
            family="rfdetr",
            display_name="RF-DETR-Seg Nano",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=312,
            params_millions=33.6,
            latency_hint="fastest masks — Jetson-friendly",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            resolution_step=12,
        ),
        ModelInfo(
            key="rfdetr-seg-small",
            family="rfdetr",
            display_name="RF-DETR-Seg Small",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=384,
            params_millions=33.7,
            latency_hint="fast masks — good default for Jetson",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            resolution_step=24,
        ),
        ModelInfo(
            key="rfdetr-seg-medium",
            family="rfdetr",
            display_name="RF-DETR-Seg Medium",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=432,
            params_millions=35.7,
            latency_hint="balanced mask quality/latency",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            resolution_step=24,
        ),
        ModelInfo(
            key="rfdetr-seg-large",
            family="rfdetr",
            display_name="RF-DETR-Seg Large",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=504,
            params_millions=36.2,
            latency_hint="high mask quality — desktop GPU recommended",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            resolution_step=24,
        ),
        ModelInfo(
            key="rfdetr-seg-xlarge",
            family="rfdetr",
            display_name="RF-DETR-Seg XLarge",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=624,
            params_millions=38.1,
            latency_hint="highest mask quality — desktop GPU only",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            resolution_step=24,
        ),
        ModelInfo(
            key="rfdetr-seg-2xlarge",
            family="rfdetr",
            display_name="RF-DETR-Seg 2XLarge",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url=_RFDETR_LICENSE_URL,
            input_resolution=768,
            params_millions=38.6,
            latency_hint="best masks, slowest — desktop GPU only",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            resolution_step=24,
        ),
        ModelInfo(
            key="owlv2-base",
            family="owlv2",
            display_name="OWLv2 Base (open-vocabulary autolabel)",
            task="detection",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url="https://huggingface.co/google/owlv2-base-patch16-ensemble",
            input_resolution=960,
            params_millions=155.0,
            latency_hint="zero-shot autolabeling, not for deployment",
            entrypoint=_OWLV2_ENTRYPOINT,
            hf_id="google/owlv2-base-patch16-ensemble",
        ),
        ModelInfo(
            key="owlv2-large",
            family="owlv2",
            display_name="OWLv2 Large (open-vocabulary autolabel)",
            task="detection",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url="https://huggingface.co/google/owlv2-large-patch14-ensemble",
            input_resolution=1008,
            params_millions=437.0,
            latency_hint="best zero-shot quality, slow — batch use only",
            entrypoint=_OWLV2_ENTRYPOINT,
            hf_id="google/owlv2-large-patch14-ensemble",
        ),
        # DINOv2 (code + weights Apache 2.0): image embeddings for the
        # active-learning loop's similarity-based selection (E10). Not a
        # detector. DINOv3 is deliberately absent (non-Apache license, §9).
        ModelInfo(
            key="dinov2-small",
            family="dinov2",
            display_name="DINOv2 Small (image embeddings)",
            task="embedding",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url="https://huggingface.co/facebook/dinov2-small",
            input_resolution=224,
            params_millions=22.1,
            latency_hint="one vector per image; drives batch selection, never deployed",
            entrypoint="horos.backends.dinov2:DINOv2Backend",
            hf_id="facebook/dinov2-small",
        ),
        ModelInfo(
            key="sam-base",
            family="sam",
            display_name="SAM ViT-B (box-to-mask refiner)",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url="https://huggingface.co/facebook/sam-vit-base",
            input_resolution=1024,
            params_millions=94.0,
            latency_hint="turns autolabel boxes into polygons; not a detector",
            entrypoint="horos.backends.sam:SAMBackend",
            hf_id="facebook/sam-vit-base",
        ),
        # SAM 2.1 (code + weights Apache 2.0): the interactive click/box
        # assist in the annotator. tiny is the Jetson-friendly default.
        # Deliberately absent: SAM 3 (custom SAM license), FastSAM (AGPL),
        # EdgeSAM (research only), ultralytics wrappers (AGPL) — §9.
        ModelInfo(
            key="sam2.1-tiny",
            family="sam2",
            display_name="SAM 2.1 Hiera-Tiny (interactive segmenter)",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url="https://huggingface.co/facebook/sam2.1-hiera-tiny",
            input_resolution=1024,
            params_millions=38.9,
            latency_hint="click/box to mask; encoder once per image, ms per click",
            entrypoint="horos.backends.sam2:SAM2Backend",
            promptable=True,
            hf_id="facebook/sam2.1-hiera-tiny",
        ),
        ModelInfo(
            key="sam2.1-small",
            family="sam2",
            display_name="SAM 2.1 Hiera-Small (interactive segmenter)",
            task="instance_segmentation",
            code_license=APACHE_2_0,
            weights_license=APACHE_2_0,
            license_url="https://huggingface.co/facebook/sam2.1-hiera-small",
            input_resolution=1024,
            params_millions=46.0,
            latency_hint="slightly better masks than tiny at a modest cost",
            entrypoint="horos.backends.sam2:SAM2Backend",
            promptable=True,
            hf_id="facebook/sam2.1-hiera-small",
        ),
    ]
}

# Known but license-gated (E4-T9). Never shown by list_models(); loading requires
# acknowledge_non_apache=True AND the rfdetr[plus] extra installed.
_GATED_MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            key="rfdetr-xl",
            family="rfdetr",
            display_name="RF-DETR XL",
            task="detection",
            code_license="PML-1.0",
            weights_license="PML-1.0",
            license_url="https://github.com/roboflow/rf-detr/blob/main/LICENSE_PLUS",
            input_resolution=700,
            params_millions=0.0,  # not published for the gated sizes
            latency_hint="requires rfdetr[plus]",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            notes="Roboflow PML 1.0 — not Apache. Commercial use restrictions apply.",
        ),
        ModelInfo(
            key="rfdetr-2xl",
            family="rfdetr",
            display_name="RF-DETR 2XL",
            task="detection",
            code_license="PML-1.0",
            weights_license="PML-1.0",
            license_url="https://github.com/roboflow/rf-detr/blob/main/LICENSE_PLUS",
            input_resolution=880,
            params_millions=0.0,
            latency_hint="requires rfdetr[plus]",
            entrypoint=_RFDETR_ENTRYPOINT,
            trainable=True,
            notes="Roboflow PML 1.0 — not Apache. Commercial use restrictions apply.",
        ),
    ]
}


def list_models(task: Task | None = None) -> list[ModelInfo]:
    """All openly registered models (Apache-licensed only), optionally by task."""
    models = list(_MODELS.values())
    if task is not None:
        models = [m for m in models if m.task == task]
    return models


def get_model_info(key: str) -> ModelInfo:
    """Look up one registered model. Raises UnknownModelError for unknown keys.

    Gated (non-Apache) models resolve here too so the license guard can explain
    *why* they are blocked instead of pretending they do not exist.
    """
    info = _MODELS.get(key) or _GATED_MODELS.get(key)
    if info is None:
        known = ", ".join(sorted(_MODELS))
        raise UnknownModelError(f"Unknown model '{key}'. Registered models: {known}")
    return info
