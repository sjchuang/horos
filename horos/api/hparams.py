"""Rule-based hyperparameter derivation (E5-T1) and overrides (E5-T2).

Pure logic: dataset statistics (E1-T7) + model metadata + a memory probe go
in, a plan with per-value reasons comes out. Every derived value records WHY
it was chosen — the plan is stored in the run metadata so users can see
"why did the system pick this resolution" (E5-S2). Search-based HPO is out of
scope for v1 by design (§6 E5).

Overrides: a user-supplied value replaces the derived one and is marked
`overridden`; the other derivations are computed exactly as before — partial
overrides never shift the rest of the plan. `extra` passthrough to the backend
still applies last on top of everything (E5-S5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from horos.backends.memory import MemoryInfo
    from horos.core.registry import ModelInfo
    from horos.core.stats import DatasetStats

__all__ = ["DerivedValue", "HyperparameterPlan", "derive_plan"]

#: effective batch size (batch × grad accumulation) the schedule is tuned for
_TARGET_EFFECTIVE_BATCH = 16
#: keep this share of measured available memory; the rest absorbs spikes
_MEMORY_HEADROOM = 0.7
#: assumed usable memory when availability is unknown (CPU / probe failure)
_CONSERVATIVE_BUDGET_GB = 3.0
#: reference cost: ~GB per sample for a ~30M-param model at 384px
_BASE_SAMPLE_GB = 1.5
_BASE_RESOLUTION = 384
_BASE_PARAMS_M = 30.5

#: TrainSpec-level knobs; everything else derived goes through spec.extra
_SPEC_FIELDS = ("epochs", "batch_size", "resolution")
#: knobs consumed by the training API itself (snapshot preparation) — they
#: are shown in the plan like any derived value but never reach the backend
_API_FIELDS = ("mosaic_ratio",)

#: fraction of the schedule's peak LR that cosine decay ends at
_LR_MIN_FACTOR = 0.01
#: below this image count the fine-tuning LR is halved to curb divergence
_SMALL_DATASET_IMAGES = 100

# Augmentation presets, keyed by dataset size. Plain data in Albumentations'
# transform-name convention: the backend maps them onto its own augmentation
# pipeline, so no model dependency is needed here (R1). Values follow the
# backend's own preset guidance (conservative under ~2000 images) with a mild
# affine added — small datasets are the ones that overfit without geometric
# variety.
_AUG_STANDARD = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
    "Affine": {
        "scale": (0.85, 1.15),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-10, 10),
        "p": 0.5,
    },
}
_AUG_HEAVY = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
}
#: image count at which the heavy augmentation preset takes over
_HEAVY_AUG_IMAGES = 2000


class DerivedValue(BaseModel):
    name: str
    value: Any
    reason: str
    overridden: bool = False


class HyperparameterPlan(BaseModel):
    model: str
    #: final effective values, overrides already applied
    values: dict[str, Any] = Field(default_factory=dict)
    #: one entry per value, in derivation order, each with its reason
    derivations: list[DerivedValue] = Field(default_factory=list)
    #: honest non-actionables (e.g. imbalance the backend has no knob for)
    notes: list[str] = Field(default_factory=list)

    def spec_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k in _SPEC_FIELDS}

    def extra_fields(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.values.items()
            if k not in _SPEC_FIELDS and k not in _API_FIELDS
        }

    def api_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k in _API_FIELDS}


#: a run that starts from an earlier run's weights needs fewer passes: the
#: features are learned, only the new photos and classes move the model
WARM_START_EPOCH_FACTOR = 0.5
WARM_START_MIN_EPOCHS = 5


def _derive_epochs(stats: DatasetStats) -> tuple[int, str]:
    n = stats.num_images
    if n < 500:
        # the epoch count is also the cosine schedule's horizon: it must be
        # short enough that the LR anneal actually completes on runs early
        # stopping is likely to end — a 100-epoch horizon stopped at ~25
        # leaves the LR near its peak and the consolidation phase never runs
        return 60, (
            f"{n} images: 60 epochs — enough passes for a small dataset to "
            f"converge, and a cosine horizon short enough that the LR anneal "
            f"completes before early stopping ends the run"
        )
    for limit, epochs in ((2000, 40), (10000, 25)):
        if n < limit:
            return epochs, (
                f"{n} images: small datasets need more passes to converge "
                f"(<{limit} images → {epochs} epochs)"
            )
    return 15, f"{n} images: large dataset, 15 epochs suffice per pass volume"


def _derive_resolution(
    stats: DatasetStats, model_info: ModelInfo | None
) -> tuple[int, str] | None:
    if model_info is None:
        return None  # unknown model: leave the backend default alone
    base = model_info.input_resolution
    area = stats.relative_area
    if area is not None and area.median < 0.01:
        # a third more pixels per side, snapped UP to the model's resolution
        # step (patch size × windows: 64 for RF-DETR detection, 12/24 for
        # RF-DETR-Seg) — an unsnapped value is rejected by the backend
        step = max(1, model_info.resolution_step)
        raised = -(-(base + 128) // step) * step
        return raised, (
            f"median object covers {area.median:.2%} of its image (<1%): "
            f"small objects need more pixels — raised {base} → {raised} "
            f"(multiple of {step})"
        )
    return base, f"model's native input resolution ({base}px), objects are not tiny"


def _derive_batch(
    memory: MemoryInfo, model_info: ModelInfo | None, resolution: int | None
) -> tuple[int, str]:
    params = model_info.params_millions if model_info else _BASE_PARAMS_M
    res = resolution or _BASE_RESOLUTION
    per_sample = (
        _BASE_SAMPLE_GB
        * (res / _BASE_RESOLUTION) ** 2
        * math.sqrt(params / _BASE_PARAMS_M)
    )
    if memory.available_gb is not None:
        budget = memory.available_gb * _MEMORY_HEADROOM
        budget_reason = (
            f"{memory.available_gb:.1f} GB available on {memory.kind} "
            f"({memory.source}), {_MEMORY_HEADROOM:.0%} budgeted"
        )
    else:
        budget = _CONSERVATIVE_BUDGET_GB
        budget_reason = (
            f"memory availability unknown on {memory.kind} ({memory.source}): "
            f"conservative {_CONSERVATIVE_BUDGET_GB:g} GB budget"
        )
    raw = budget / per_sample
    batch = 2 ** int(math.log2(raw)) if raw >= 1 else 1
    batch = max(1, min(batch, 16))
    return batch, (
        f"{budget_reason}; ~{per_sample:.1f} GB/sample at {res}px "
        f"→ batch {batch} (power of two, capped at 16)"
    )


def derive_plan(
    stats: DatasetStats,
    *,
    model: str,
    model_info: ModelInfo | None,
    memory: MemoryInfo,
    overrides: dict[str, Any] | None = None,
    warm_start: bool = False,
) -> HyperparameterPlan:
    """Apply the derivation rules; `overrides` values (non-None) win per-key
    without disturbing how the other keys are derived (E5-T2). `warm_start`
    says the run continues from an earlier run's weights, which needs fewer
    passes than learning the task from the published weights."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    plan = HyperparameterPlan(model=model)

    def put(name: str, value: Any, reason: str) -> Any:
        if name in overrides:
            value = overrides[name]
            entry = DerivedValue(
                name=name, value=value, reason="user override", overridden=True
            )
        else:
            entry = DerivedValue(name=name, value=value, reason=reason)
        plan.values[name] = value
        plan.derivations.append(entry)
        return value

    epochs, why = _derive_epochs(stats)
    if warm_start:
        epochs = max(WARM_START_MIN_EPOCHS, round(epochs * WARM_START_EPOCH_FACTOR))
        why = (f"continuing from an earlier run's weights: {WARM_START_EPOCH_FACTOR:g}× the "
               f"fresh-start count (the model already knows the task) — {why}")
    put("epochs", epochs, why)

    resolution: int | None = None
    derived_res = _derive_resolution(stats, model_info)
    if derived_res is not None:
        resolution = put("resolution", *derived_res)
    elif "resolution" in overrides:
        resolution = put("resolution", None, "")

    batch, why = _derive_batch(memory, model_info, resolution)
    batch = put("batch_size", batch, why)

    put(
        "grad_accum_steps",
        max(1, round(_TARGET_EFFECTIVE_BATCH / batch)),
        f"accumulate gradients to an effective batch of {_TARGET_EFFECTIVE_BATCH} "
        f"(batch {batch} × accumulation)",
    )

    if stats.num_images < _SMALL_DATASET_IMAGES:
        put(
            "lr",
            5e-5,
            f"{stats.num_images} images (<{_SMALL_DATASET_IMAGES}): the "
            f"backend's tuned default (1e-4) diverges on very small datasets "
            f"once early convergence ends — halved to 5e-5",
        )
    else:
        put(
            "lr",
            1e-4,
            f"backend's tuned fine-tuning default; the schedule is calibrated "
            f"for an effective batch of {_TARGET_EFFECTIVE_BATCH}, which the "
            f"gradient accumulation above maintains — no scaling needed",
        )

    put(
        "lr_scheduler",
        "cosine",
        "backend default 'step' only drops the LR at its lr_drop epoch (100), "
        "i.e. never within a typical run — full LR to the last epoch destroys "
        "converged fine-tunes; cosine decays smoothly to the floor below",
    )
    put(
        "lr_scheduler_kwargs",
        {"min_factor": _LR_MIN_FACTOR},
        f"cosine decays to {_LR_MIN_FACTOR:.0%} of the peak LR by the final "
        f"epoch (the convention YOLO-family trainers use for fine-tuning)",
    )

    populated = [c for c in stats.per_class if c.instances > 0]
    weakest = min(populated, key=lambda c: c.instances, default=None)
    if stats.num_images < 500:
        put(
            "warmup_epochs",
            3.0,
            f"{stats.num_images} images: few optimizer steps per epoch, so a "
            f"3-epoch linear warmup replaces the usual step-count warmup and "
            f"stabilizes the re-initialized detection head",
        )
    elif weakest is not None and weakest.instances < 100:
        put(
            "warmup_epochs",
            1.0,
            f"weakest class '{weakest.name}' has only {weakest.instances} "
            f"instances (<100): one warmup epoch stabilizes early training",
        )
    else:
        put("warmup_epochs", 0.0, "every class has ≥100 instances, no warmup needed")

    patience = 15 if stats.num_images < 500 else 10
    put(
        "early_stopping",
        True,
        "stop when validation mAP plateaus instead of training to the last "
        "epoch — small datasets degrade catastrophically past their peak, and "
        "the best checkpoint is already kept either way",
    )
    put(
        "early_stopping_patience",
        patience,
        (
            f"{stats.num_images} images: a small valid split makes per-epoch "
            f"mAP noisy — wait {patience} epochs without improvement"
            if stats.num_images < 500
            else f"{patience} epochs without validation improvement"
        ),
    )
    put(
        "early_stopping_use_ema",
        True,
        "monitor the EMA weights' mAP: smoother than per-epoch raw mAP, so "
        "one noisy validation dip cannot stop training early",
    )

    heavy = stats.num_images >= _HEAVY_AUG_IMAGES
    put(
        "aug_config",
        _AUG_HEAVY if heavy else _AUG_STANDARD,
        (
            f"{stats.num_images} images (≥{_HEAVY_AUG_IMAGES}): heavy "
            f"augmentation (flip, brightness/contrast, affine with shear, "
            f"color jitter) — large datasets tolerate and benefit from it"
            if heavy
            else f"{stats.num_images} images: standard augmentation (flip, "
            f"brightness/contrast, mild affine) — the backend's default "
            f"pipeline applies only a horizontal flip, which is not enough "
            f"variety to prevent overfitting"
        ),
    )
    put(
        "augmentation_backend",
        "cpu",
        "CPU augmentation gives identical pixels on CUDA, MPS and CPU "
        "platforms (R7); the GPU path would make runs device-dependent",
    )

    # Off by default after a controlled A/B (balloon, 74 images, seed 42,
    # 60 epochs): at ratio 0.5 the composites carried ~70% of the training
    # annotation mass at quarter scale, and val mAP decayed monotonically
    # after its early peak (0.155 → 0.013) while the mosaic-free twin peaked
    # higher and stayed healthy (0.186) with 3× better confidence
    # calibration. Opt in per run via mosaic_ratio when the deployment
    # distribution really is many-small-objects.
    put(
        "mosaic_ratio",
        0.0,
        "mosaic snapshots are opt-in: at the ratios that add meaningful "
        "variety they dominate the annotation mass with quarter-scale "
        "objects, which measurably degraded validation mAP and confidence "
        "calibration on small datasets — set mosaic_ratio explicitly for "
        "many-small-object domains",
    )

    put(
        "num_workers",
        0 if stats.num_images < 200 else 2,
        (
            f"{stats.num_images} images (<200): single-process data loading — "
            f"worker spawn overhead outweighs the gain"
            if stats.num_images < 200
            else f"{stats.num_images} images: 2 loader workers"
        ),
    )

    if stats.imbalance_ratio is not None and stats.imbalance_ratio > 3.0:
        plan.notes.append(
            f"Class imbalance is {stats.imbalance_ratio:.1f}:1. The current "
            f"backend exposes no sampling-strategy knob; consider adding data "
            f"for the underrepresented classes."
        )
    return plan
