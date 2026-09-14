"""E10-T5: PAL acquisition — LIUS instance uncertainty, GUIDE terms, class
budgets and the combined per-class selection (arXiv 2605.10349)."""

import numpy as np
import pytest

from horos.core import pal
from horos.core.pal import (
    Detection,
    InstanceScorer,
    binary_entropy,
    class_budgets,
    cwie,
    match_true_positives,
    rarity_weights,
    rcdi,
    rcsp,
    select,
    support_counts,
)
from horos.errors import ProjectError


def _labeled(rng, n=60, category="box", image_offset=0):
    """Synthetic labeled detections: true positives are confident and well
    supported, false positives are shaky and lonely."""
    dets = []
    for i in range(n):
        tp = i % 2 == 0
        conf = rng.uniform(0.7, 0.99) if tp else rng.uniform(0.05, 0.4)
        support = int(rng.integers(6, 15)) if tp else int(rng.integers(1, 3))
        dets.append(
            Detection(
                image_id=image_offset + i,
                category=category,
                confidence=conf,
                support=support,
                true_positive=tp,
            )
        )
    return dets


# ------------------------------------------------------------ features


def test_support_counts_overlapping_candidates():
    final = [(0, 0, 10, 10), (50, 50, 10, 10)]
    cands = [(0, 0, 10, 10), (1, 1, 10, 10), (9, 9, 10, 10), (50, 50, 10, 10)]
    assert support_counts(final, cands) == [2, 1]


def test_true_positive_matching_is_one_to_one_by_class():
    preds = [(0, 0, 10, 10), (0, 0, 10, 10), (50, 50, 10, 10)]
    classes = ["box", "box", "pallet"]
    scores = [0.9, 0.8, 0.7]
    gts = [(0, 0, 10, 10), (50, 50, 10, 10)]
    gt_classes = ["box", "box"]
    # second duplicate box is a false positive; pallet vs box is a class mismatch
    assert match_true_positives(preds, classes, scores, gts, gt_classes) == [True, False, False]


# --------------------------------------------------------------- LIUS


def test_logistic_scorer_learns_true_positive_probability():
    rng = np.random.default_rng(0)
    scorer = InstanceScorer(_labeled(rng))
    assert scorer.notes == []
    sure = Detection(image_id=999, category="box", confidence=0.95, support=12)
    shaky = Detection(image_id=998, category="box", confidence=0.15, support=1)
    # squarely between the two labeled populations (0.7-0.99 / 6-14 vs 0.05-0.4 / 1-2)
    fence = Detection(image_id=997, category="box", confidence=0.55, support=4)
    p_sure, mode = scorer.probability(sure)
    p_shaky, _ = scorer.probability(shaky)
    p_fence, _ = scorer.probability(fence)
    assert mode == "class"
    assert p_sure > 0.9 and p_shaky < 0.1
    assert p_shaky < p_fence < p_sure
    h_sure, _, _ = scorer.uncertainty(sure)
    h_shaky, _, _ = scorer.uncertainty(shaky)
    h_fence, _, _ = scorer.uncertainty(fence)
    assert h_fence > max(h_sure, h_shaky)


def test_binary_entropy_peaks_at_half():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-4)
    assert binary_entropy(0.9) < binary_entropy(0.6)


def test_class_without_enough_labels_falls_back_and_says_so():
    rng = np.random.default_rng(1)
    labeled = _labeled(rng) + [
        Detection(image_id=500, category="pallet", confidence=0.8, support=5, true_positive=True)
    ]
    scorer = InstanceScorer(labeled)
    assert scorer.mode["pallet"] == "pooled"
    assert any("pallet" in n and "pooled" in n for n in scorer.notes)
    pallet = Detection(image_id=1, category="pallet", confidence=0.9, support=9)
    p, mode = scorer.probability(pallet)
    assert mode == "pooled" and p > 0.5
    # a class never seen in labels also goes to the pooled classifier
    _, mode = scorer.probability(Detection(image_id=1, category="cone", confidence=0.9, support=9))
    assert mode == "pooled"


def test_no_labels_at_all_uses_confidence():
    scorer = InstanceScorer([])
    p, mode = scorer.probability(Detection(image_id=1, category="box", confidence=0.7))
    assert (p, mode) == (0.7, "confidence")
    assert "no labeled detections" in scorer.notes[0]


# -------------------------------------------------------------- GUIDE


def test_rarity_weights_follow_eq5():
    r = rarity_weights({"box": 90, "pallet": 10}, {"box": 80, "pallet": 20})
    assert r["box"] == pytest.approx(1 - 0.5 * (0.9 + 0.8))
    assert r["pallet"] == pytest.approx(1 - 0.5 * (0.1 + 0.2))
    assert r["pallet"] > r["box"]
    # a class absent from one side still gets a weight
    assert rarity_weights({}, {"box": 5})["box"] == pytest.approx(0.5)


def test_class_budgets_favour_rare_classes_and_respect_availability():
    rarity = {"box": 0.15, "pallet": 0.85}
    budgets = class_budgets(10, rarity, {"box": 100, "pallet": 100})
    assert sum(budgets.values()) == 10
    assert budgets["pallet"] > budgets["box"]
    # availability caps a class; the remainder flows to the other one
    capped = class_budgets(10, rarity, {"box": 100, "pallet": 2})
    assert capped == {"pallet": 2, "box": 8}
    # nothing available at all
    assert class_budgets(10, rarity, {"box": 0, "pallet": 0}) == {"box": 0, "pallet": 0}
    assert sum(class_budgets(3, rarity, {"box": 1, "pallet": 1}).values()) == 2


def test_cwie_and_rcdi():
    rarity = {"box": 0.2, "pallet": 0.8}
    fence = [Detection(image_id=1, category="pallet", confidence=0.5)]
    sure = [Detection(image_id=2, category="pallet", confidence=0.99)]
    assert cwie(fence, rarity) > cwie(sure, rarity)
    # a full probability vector is used when the detector provides one
    spread = [Detection(image_id=3, category="box", confidence=0.5, class_probs=[0.5, 0.5])]
    peaked = [Detection(image_id=4, category="box", confidence=0.98, class_probs=[0.98, 0.02])]
    assert cwie(spread, rarity) > cwie(peaked, rarity)
    both = [
        Detection(image_id=5, category="box", confidence=0.9),
        Detection(image_id=5, category="pallet", confidence=0.9),
        Detection(image_id=5, category="pallet", confidence=0.8),
    ]
    assert rcdi(both, rarity) == pytest.approx(1.0)
    assert rcdi(both[:1], rarity) == pytest.approx(0.2)


def test_rcsp_penalises_lookalikes_of_higher_ranked_images():
    e = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    out = rcsp(e)
    assert out[0] == pytest.approx(1.0)  # the top-ranked image is never penalised
    assert out[1] == pytest.approx(0.0)  # identical to the image above it
    assert out[2] == pytest.approx(1.0)  # orthogonal to everything above
    assert len(rcsp(None)) == 0


# ---------------------------------------------------------- selection


def _pool(rng):
    """40 unlabeled images: ids 0-19 hold confident 'box' detections, 20-29
    fence-sitting 'box' detections, 30-34 rare 'pallet' detections, 35-39 empty."""
    unlabeled = {}
    for i in range(20):
        unlabeled[i] = [Detection(i, "box", rng.uniform(0.9, 0.99), int(rng.integers(8, 14)))]
    for i in range(20, 30):
        unlabeled[i] = [Detection(i, "box", rng.uniform(0.45, 0.55), int(rng.integers(3, 6)))]
    for i in range(30, 35):
        unlabeled[i] = [Detection(i, "pallet", rng.uniform(0.5, 0.7), int(rng.integers(3, 6)))]
    for i in range(35, 40):
        unlabeled[i] = []
    return unlabeled


def test_select_prefers_uncertain_images_and_spends_rare_class_budget():
    rng = np.random.default_rng(2)
    labeled = _labeled(rng, image_offset=1000) + _labeled(
        rng, n=12, category="pallet", image_offset=2000
    )
    result = select(_pool(rng), labeled, 8)
    ids = [p.image_id for p in result.picks]
    assert len(ids) == 8 and len(set(ids)) == 8
    assert result.shortfall == 0
    assert all(20 <= i < 35 for i in ids), ids  # never the confident ones
    assert any(30 <= i < 35 for i in ids)  # the rare class got part of the budget
    assert result.budgets["pallet"] >= 1
    assert all(p.reason.startswith("PAL for class") for p in result.picks)
    assert all("LIUS" in p.reason and "RCSP" in p.reason for p in result.picks)


def test_select_uses_embeddings_for_rcsp_and_reports_shortfall():
    rng = np.random.default_rng(3)
    labeled = _labeled(rng, image_offset=1000)
    pool = _pool(rng)
    embeddings = {i: rng.normal(size=8) for i in pool}
    result = select(pool, labeled, 40, embeddings=embeddings)
    # 35 images have detections; the 5 empty ones are unreachable by PAL
    assert len(result.picks) == 35
    assert result.shortfall == 5
    assert any("could not be chosen by PAL" in n for n in result.notes)
    # pallet had no labeled detections in this run → pooled classifier, noted
    assert any(p.category == "pallet" for p in result.picks)


def test_select_validates_inputs():
    with pytest.raises(ProjectError, match="budget"):
        select({1: []}, [], 0)
    with pytest.raises(ProjectError, match="non-negative"):
        select({1: []}, [], 1, alpha=-1)


def test_paper_weights_are_the_defaults():
    assert (pal.ALPHA, pal.BETA, pal.GAMMA) == (0.9, 0.04, 0.02)
    assert pal.CANDIDATE_FACTOR == 2


def test_balance_weights_tilt_budgets_towards_under_labeled_classes():
    """horos choice on top of Eq. 5/6: inverse label frequency (clipped) —
    a class with 5 % of the labels gets a far larger share than the paper's
    gentle rarity alone would give; balance=0 restores the paper."""
    from horos.core.pal import BALANCE_CAP, balance_weights

    w = balance_weights({"box": 95, "pallet": 5}, ["box", "pallet"])
    assert w["box"] == pytest.approx(0.5 / 0.95) and w["pallet"] == pytest.approx(BALANCE_CAP)
    even = balance_weights({"box": 50, "pallet": 50}, ["box", "pallet"])
    assert even == {"box": 1.0, "pallet": 1.0}
    assert balance_weights({}, ["box"]) == {"box": 1.0}

    def det(i, c, conf, support, tp=None):
        return Detection(image_id=i, category=c, confidence=conf, support=support,
                         true_positive=tp)

    labeled = [det(1, "box", 0.9, 8, True) for _ in range(19)] + [det(2, "pallet", 0.9, 8, True)]
    unlabeled = {}
    for i in range(100, 120):
        unlabeled[i] = [det(i, "box", 0.5, 3)]
    for i in range(200, 220):
        unlabeled[i] = [det(i, "pallet", 0.5, 3)]
    paper = select(unlabeled, labeled, 10, balance=0.0)
    ours = select(unlabeled, labeled, 10)
    assert paper.budgets["pallet"] <= 7  # Eq. 5 alone: a mild tilt (r ≈ 0.73 vs 0.28)
    assert ours.budgets["pallet"] >= 9 > paper.budgets["pallet"]
    assert ours.balance["pallet"] > 1 > ours.balance["box"]
    assert sum(ours.budgets.values()) == 10
