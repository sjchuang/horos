"""E5-T1: rule-based hyperparameter derivation — every value carries a reason."""

from __future__ import annotations

import pytest

from horos.api.hparams import derive_plan
from horos.backends.memory import MemoryInfo
from horos.core.registry import get_model_info
from horos.core.stats import ClassStats, DatasetStats, RelativeAreaStats

NANO = get_model_info("rfdetr-nano")


def test_warm_start_halves_the_epochs_and_says_why():
    from horos.api.hparams import WARM_START_MIN_EPOCHS

    def _mem():
        return MemoryInfo(kind="cuda", total_gb=24.0, available_gb=22.0, source="test")

    fresh = derive_plan(_stats(num_images=50), model="rfdetr-nano",
                        model_info=get_model_info("rfdetr-nano"), memory=_mem())
    warm = derive_plan(_stats(num_images=50), model="rfdetr-nano",
                       model_info=get_model_info("rfdetr-nano"), memory=_mem(), warm_start=True)
    assert warm.values["epochs"] == max(WARM_START_MIN_EPOCHS, round(fresh.values["epochs"] / 2))
    reason = next(d.reason for d in warm.derivations if d.name == "epochs")
    assert "continuing from an earlier run" in reason
    # an explicit override still wins
    forced = derive_plan(_stats(num_images=50), model="rfdetr-nano",
                         model_info=get_model_info("rfdetr-nano"), memory=_mem(),
                         warm_start=True, overrides={"epochs": 7})
    assert forced.values["epochs"] == 7


def _stats(
    *,
    num_images=50,
    per_class=None,
    median_area=0.05,
    imbalance=None,
) -> DatasetStats:
    per_class = per_class or [
        ClassStats(category_id=1, name="block", instances=200, images=40)
    ]
    return DatasetStats(
        num_images=num_images,
        num_annotations=sum(c.instances for c in per_class),
        num_categories=len(per_class),
        per_class=per_class,
        split_counts={"train": num_images},
        image_sizes=[],
        relative_area=RelativeAreaStats(
            minimum=median_area / 2,
            maximum=median_area * 2,
            mean=median_area,
            median=median_area,
            histogram=[0] * 10,
        ),
        imbalance_ratio=imbalance,
    )


def _memory(kind="cuda", available=16.0, total=16.0) -> MemoryInfo:
    return MemoryInfo(kind=kind, total_gb=total, available_gb=available, source="test")


def _plan(stats=None, memory=None, **kw):
    return derive_plan(
        stats or _stats(),
        model="rfdetr-nano",
        model_info=NANO,
        memory=memory or _memory(),
        **kw,
    )


def test_every_derivation_has_a_nonempty_reason():
    plan = _plan()
    assert plan.derivations, "the plan must not be empty"
    for entry in plan.derivations:
        assert entry.reason.strip(), f"{entry.name} has no reason"
        assert entry.name in plan.values


@pytest.mark.parametrize(
    ("num_images", "expected_epochs"),
    # <500 images share one 60-epoch tier: the epoch count doubles as the
    # cosine anneal horizon, which must stay reachable under early stopping
    [(30, 60), (300, 60), (1500, 40), (5000, 25), (50000, 15)],
)
def test_epochs_scale_down_with_dataset_size(num_images, expected_epochs):
    plan = _plan(_stats(num_images=num_images))
    assert plan.values["epochs"] == expected_epochs
    reason = next(d.reason for d in plan.derivations if d.name == "epochs")
    assert str(num_images) in reason


def test_batch_size_follows_available_memory():
    # nano at 384px ≈ 1.5 GB/sample; 16 GB * 0.7 = 11.2 GB → 7.4 samples → 4
    plan = _plan(memory=_memory(available=16.0))
    assert plan.values["batch_size"] == 4
    # more memory, bigger batch — still a power of two
    plan = _plan(memory=_memory(available=48.0, total=48.0))
    assert plan.values["batch_size"] == 16


def test_unknown_memory_is_conservative():
    plan = _plan(memory=MemoryInfo(kind="cpu", total_gb=32.0, available_gb=None,
                                   source="test"))
    assert plan.values["batch_size"] == 2  # 3 GB budget / 1.5 GB per sample
    reason = next(d.reason for d in plan.derivations if d.name == "batch_size")
    assert "conservative" in reason


def test_grad_accum_targets_effective_batch_16():
    plan = _plan(memory=_memory(available=16.0))  # batch 4
    assert plan.values["grad_accum_steps"] == 4
    plan = _plan(memory=_memory(available=48.0, total=48.0))  # batch 16
    assert plan.values["grad_accum_steps"] == 1


def test_tiny_objects_raise_resolution():
    plan = _plan(_stats(median_area=0.005))
    assert plan.values["resolution"] == NANO.input_resolution + 128
    assert plan.values["resolution"] % 64 == 0  # rfdetr divisibility constraint
    plan = _plan(_stats(median_area=0.2))
    assert plan.values["resolution"] == NANO.input_resolution


def test_resolution_bump_raises_per_sample_memory_cost():
    # 12 GB: 8.4 budget / 1.5 per sample at 384px → 4; at 512px (~2.7/sample) → 2
    at_base = _plan(_stats(median_area=0.2), memory=_memory(available=12.0))
    bumped = _plan(_stats(median_area=0.005), memory=_memory(available=12.0))
    assert bumped.values["batch_size"] < at_base.values["batch_size"]


def test_scarce_class_adds_warmup():
    # ≥500 images so the small-dataset 3-epoch warmup rule does not apply and
    # the instance-count rule is what decides
    scarce = [
        ClassStats(category_id=1, name="common", instances=500, images=100),
        ClassStats(category_id=2, name="rare", instances=12, images=8),
    ]
    plan = _plan(_stats(num_images=800, per_class=scarce))
    assert plan.values["warmup_epochs"] == 1.0
    reason = next(d.reason for d in plan.derivations if d.name == "warmup_epochs")
    assert "rare" in reason

    plentiful = [ClassStats(category_id=1, name="common", instances=500, images=100)]
    plan = _plan(_stats(num_images=800, per_class=plentiful))
    assert plan.values["warmup_epochs"] == 0.0


def test_small_dataset_disables_loader_workers():
    assert _plan(_stats(num_images=50)).values["num_workers"] == 0
    assert _plan(_stats(num_images=5000)).values["num_workers"] == 2


def test_imbalance_is_noted_not_silently_ignored():
    plan = _plan(_stats(imbalance=8.0))
    assert any("8.0:1" in note for note in plan.notes)
    assert _plan(_stats(imbalance=1.2)).notes == []


def test_unknown_model_skips_resolution_but_derives_the_rest():
    plan = derive_plan(
        _stats(), model="fake", model_info=None, memory=_memory()
    )
    assert "resolution" not in plan.values
    assert {"epochs", "batch_size", "grad_accum_steps"} <= set(plan.values)


def test_lr_is_derived_with_reason_and_overridable():
    # ≥100 images: the backend default applies (below 100 it is halved, see
    # test_hparam_schedule.py)
    plan = _plan(_stats(num_images=300))
    entry = next(d for d in plan.derivations if d.name == "lr")
    assert entry.value == pytest.approx(1e-4)
    assert "effective batch" in entry.reason
    assert "lr" in plan.extra_fields()  # rides to the backend via spec.extra

    overridden = _plan(_stats(num_images=300), overrides={"lr": 5e-5})
    entry = next(d for d in overridden.derivations if d.name == "lr")
    assert entry.value == pytest.approx(5e-5) and entry.overridden is True
    # an lr override does not disturb the other derivations (E5-T2)
    assert overridden.values["epochs"] == plan.values["epochs"]
    assert overridden.values["batch_size"] == plan.values["batch_size"]


def test_tiny_objects_snap_the_raised_resolution_to_the_model_step():
    """E4-T15: RF-DETR-Seg needs multiples of 12/24, not 64 — 312+128=440 is
    rejected by the backend, 444 (nano) / 528 (small) are what get derived."""
    def plan_for(key):
        info = get_model_info(key)
        return derive_plan(_stats(median_area=0.005), model=key, model_info=info, memory=_memory())

    plan = plan_for("rfdetr-seg-nano")
    assert plan.values["resolution"] == 444 and 444 % 12 == 0
    reasons = " ".join(getattr(d, "reason", "") for d in plan.derivations)
    assert "multiple of 12" in reasons
    plan = plan_for("rfdetr-seg-small")
    assert plan.values["resolution"] == 528 and 528 % 24 == 0
    # detection keeps its 64-step behaviour
    plan = _plan(_stats(median_area=0.005))
    assert plan.values["resolution"] == NANO.input_resolution + 128
