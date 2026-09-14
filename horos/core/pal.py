"""Portable Active Learning acquisition (E10-T5).

Implements the image-selection method of Sharma, Bersamin & Subramanian,
"Portable Active Learning for Object Detection" (CVPR 2026, arXiv
2605.10349) as the loop's default uncertainty metric. It works only on
detector outputs — no model internals — which is exactly what the horos
backend boundary (R1) exposes, so any backend can drive it.

The method
----------
LIUS — logistic-based instance uncertainty scorer. For each class a binary
logistic classifier is fitted on the LABELED images' detections, learning
P(true positive | support, confidence) where `support` is the number of
raw candidate boxes that overlap the detection ("pre-NMS box count" in the
paper). On unlabeled images the classifier's probability is turned into a
Shannon entropy: a detection the classifier cannot call is worth labeling.

GUIDE — global uncertainty and image diversity estimator, three terms
computed over the candidate images of a class and min-max scaled to [0, 1]:

    CWIE(I) = −Σ_i r_{c_i} Σ_j p_ij log p_ij   class-weighted image entropy
    RCDI(I) = Σ_{k∈K(I)} r_k                   rare-class diversity index
    RCSP(I) = 1 − max_{m<i} cos(e_i, e_m)      rank-conditioned similarity penalty

with class-rarity weights r_c = 1 − ½(n_{c,l}/N_l + n_{c,u}/N_u) (Eq. 5) and
per-class budgets b_c = min(n_{c,u}, b · r_c / Σ r) (Eq. 6).

Score(I) = α·LIUS(I) + γ·RCSP(I) + β·(CWIE(I) + RCDI(I)), α = 0.9, β = 0.04,
γ = 0.02. Per class the top 2·b_c images by LIUS are candidates, the top
b_c by Score are selected.

Where the paper leaves a detail open, horos chooses and says so here:

- image LIUS = the maximum instance entropy of that class in the image
  (the paper selects "images containing the highest LIUS instances")
- candidate support is counted at IoU ≥ 0.5; a detection is a true positive
  when it matches an unmatched ground-truth box of its class at IoU ≥ 0.5
  (COCO's loose threshold)
- b_c is capped by the number of unlabeled IMAGES containing class c —
  budgets are spent in images, so an instance-count cap could overshoot
- an image selected under one class is skipped when it comes up again under
  another; the next candidate takes its place
- for detectors that expose one confidence per box, CWIE uses the Bernoulli
  distribution (p, 1 − p); when `class_probs` is available the full vector
  is used as the paper writes it
- a class whose labeled detections cannot train a classifier (too few, or
  all one label) falls back to a classifier pooled over all classes, and if
  that is degenerate too, to P(TP) = confidence. The fallback is recorded in
  the result notes and in every affected reason
- a round with no labeled images cannot train LIUS at all; the API layer
  uses diversity selection instead (cold start, E10-T4) — the paper seeds
  with a random 2 % which the confirmed E10 design replaces

Pure numpy, no I/O: the API layer runs the detector and brings the outputs
here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from horos.core.selection import Pick, cosine_distance, normalize
from horos.errors import ProjectError

ALPHA = 0.9  # LIUS weight
BETA = 0.04  # CWIE + RCDI weight
GAMMA = 0.02  # RCSP weight

# horos choice (2026-09-14, on the user's request): Eq. 5's rarity is gentle —
# a class holding 60 % of the labels still gets weight 0.7 against 0.99 for
# one at 2 %, so budgets barely move. Budgets are therefore additionally
# weighted by inverse label frequency, mean_frac / frac_c, clipped to
# [1/BALANCE_CAP, BALANCE_CAP] and raised to BALANCE (0 restores the paper).
# Rarity itself still drives CWIE, RCDI and the class order.
BALANCE = 1.0
BALANCE_CAP = 8.0
CANDIDATE_FACTOR = 2  # candidates per class = CANDIDATE_FACTOR · b_c
SUPPORT_IOU = 0.5
TRUE_POSITIVE_IOU = 0.5
#: fewer labeled detections than this, or a single label, cannot fit a classifier
MIN_FIT_SAMPLES = 6
_L2 = 1e-2  # ridge on the logistic weights: keeps separable classes finite
_EPS = 1e-7


# --------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Detection:
    """One final detection on one image, as the detector reported it."""

    image_id: int
    category: str
    confidence: float
    #: raw candidate boxes overlapping this detection (see support_counts);
    #: None when the backend exposed no candidates
    support: int | None = None
    #: full per-class probability vector when the detector has one
    class_probs: Sequence[float] | None = None
    #: only meaningful on labeled images: matched a ground-truth box?
    true_positive: bool | None = None


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two COCO xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def support_counts(
    final_boxes: Sequence[Sequence[float]],
    candidate_boxes: Sequence[Sequence[float]],
    *,
    iou: float = SUPPORT_IOU,
) -> list[int]:
    """How many raw candidate boxes overlap each final detection at IoU ≥
    `iou` — the pre-NMS box count of the paper. A detection the detector
    reached through many overlapping proposals is well supported; a lone
    box is suspicious."""
    return [
        sum(1 for cand in candidate_boxes if box_iou(box, cand) >= iou) for box in final_boxes
    ]


def match_true_positives(
    pred_boxes: Sequence[Sequence[float]],
    pred_classes: Sequence[str],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    gt_classes: Sequence[str],
    *,
    iou: float = TRUE_POSITIVE_IOU,
) -> list[bool]:
    """Greedy one-to-one matching in descending confidence: a prediction is
    a true positive when it overlaps a still-unmatched ground-truth box of
    the same class at IoU ≥ `iou`."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    taken = [False] * len(gt_boxes)
    result = [False] * len(pred_boxes)
    for i in order:
        best, best_j = 0.0, -1
        for j, (gbox, gcls) in enumerate(zip(gt_boxes, gt_classes, strict=True)):
            if taken[j] or gcls != pred_classes[i]:
                continue
            overlap = box_iou(pred_boxes[i], gbox)
            if overlap >= iou and overlap > best:
                best, best_j = overlap, j
        if best_j >= 0:
            taken[best_j] = True
            result[i] = True
    return result


# ------------------------------------------------------------------ LIUS


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def binary_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p) in bits: 1.0 at p = 0.5, 0 when sure."""
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


class _Logistic:
    """P(Y = 1 | x) = σ(β₀ + β₁x₁ + β₂x₂) fitted by Newton's method on
    standardised features with a small ridge term."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        z = self._design(x)
        w = np.zeros(z.shape[1])
        reg = np.full(z.shape[1], _L2)
        reg[0] = 0.0  # no penalty on the bias
        for _ in range(50):
            p = _sigmoid(z @ w)
            grad = z.T @ (p - y) + reg * w
            hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(reg)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w

    def _design(self, x: np.ndarray) -> np.ndarray:
        scaled = (x - self.mean) / self.std
        return np.hstack([np.ones((len(scaled), 1)), scaled])

    def proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self._design(x) @ self.w)


def _features(dets: Iterable[Detection]) -> np.ndarray:
    return np.asarray(
        [[float(d.support if d.support is not None else 0), float(d.confidence)] for d in dets],
        dtype=np.float64,
    )


def _fittable(y: np.ndarray) -> bool:
    return len(y) >= MIN_FIT_SAMPLES and 0 < int(y.sum()) < len(y)


class InstanceScorer:
    """LIUS: per-class logistic classifiers over (support, confidence),
    trained on labeled detections with true/false-positive flags.

    `notes` records every class that fell back to the pooled classifier or
    to raw confidence, so a round can say which classes were scored with
    less than the full method.
    """

    def __init__(self, labeled: Iterable[Detection]):
        dets = [d for d in labeled if d.true_positive is not None]
        self.notes: list[str] = []
        self._per_class: dict[str, _Logistic] = {}
        self._pooled: _Logistic | None = None
        self.mode: dict[str, str] = {}
        by_class: dict[str, list[Detection]] = {}
        for d in dets:
            by_class.setdefault(d.category, []).append(d)
        if dets:
            y_all = np.asarray([1.0 if d.true_positive else 0.0 for d in dets])
            if _fittable(y_all):
                self._pooled = _Logistic(_features(dets), y_all)
        for name, group in sorted(by_class.items()):
            y = np.asarray([1.0 if d.true_positive else 0.0 for d in group])
            if _fittable(y):
                self._per_class[name] = _Logistic(_features(group), y)
                self.mode[name] = "class"
            elif self._pooled is not None:
                self.mode[name] = "pooled"
                self.notes.append(
                    f"class '{name}': {len(group)} labeled detections cannot train a "
                    f"classifier ({int(y.sum())} true positives); scored with the "
                    f"classifier pooled over all classes"
                )
            else:
                self.mode[name] = "confidence"
                self.notes.append(
                    f"class '{name}': no usable labeled detections; P(true positive) "
                    f"taken as the detection confidence"
                )
        if not dets:
            self.notes.append(
                "no labeled detections at all; P(true positive) taken as the detection "
                "confidence for every class"
            )

    def probability(self, det: Detection) -> tuple[float, str]:
        """P(true positive) for one detection and how it was obtained."""
        model = self._per_class.get(det.category)
        mode = self.mode.get(det.category)
        if model is None and mode is None:
            # a class never seen on labeled images: pooled if we have it
            mode = "pooled" if self._pooled is not None else "confidence"
        if model is not None:
            return float(model.proba(_features([det]))[0]), "class"
        if mode == "pooled" and self._pooled is not None:
            return float(self._pooled.proba(_features([det]))[0]), "pooled"
        return float(min(max(det.confidence, 0.0), 1.0)), "confidence"

    def uncertainty(self, det: Detection) -> tuple[float, float, str]:
        """(entropy, P(true positive), mode) for one detection."""
        p, mode = self.probability(det)
        return binary_entropy(p), p, mode


# ----------------------------------------------------------------- GUIDE


def rarity_weights(
    labeled_counts: Mapping[str, int], unlabeled_counts: Mapping[str, int]
) -> dict[str, float]:
    """Eq. 5: r_c = 1 − ½(n_{c,l}/N_l + n_{c,u}/N_u). Rare classes approach 1,
    a class that is everything approaches 0. Counts are detections."""
    classes = set(labeled_counts) | set(unlabeled_counts)
    n_l = sum(labeled_counts.values())
    n_u = sum(unlabeled_counts.values())
    weights = {}
    for c in classes:
        frac_l = labeled_counts.get(c, 0) / n_l if n_l else 0.0
        frac_u = unlabeled_counts.get(c, 0) / n_u if n_u else 0.0
        weights[c] = 1.0 - 0.5 * (frac_l + frac_u)
    return weights


def balance_weights(
    labeled_counts: Mapping[str, int], classes: Iterable[str], *, cap: float = BALANCE_CAP
) -> dict[str, float]:
    """Inverse label frequency per class, mean_frac / frac_c, clipped to
    [1/cap, cap]; a class without a single label gets `cap`. 1.0 everywhere
    when the labels are balanced (or absent)."""
    classes = list(classes)
    total = sum(labeled_counts.get(c, 0) for c in classes)
    if not classes or total <= 0:
        return {c: 1.0 for c in classes}
    mean_frac = 1.0 / len(classes)
    out = {}
    for c in classes:
        frac = labeled_counts.get(c, 0) / total
        out[c] = cap if frac <= 0 else min(cap, max(1.0 / cap, mean_frac / frac))
    return out


def class_budgets(
    total: int, rarity: Mapping[str, float], available: Mapping[str, int]
) -> dict[str, int]:
    """Eq. 6: b_c = min(available_c, b · r_c / Σ r). `available` is the number
    of unlabeled images containing the class. Rounded to whole images so the
    budgets sum to `total` (or to everything available when that is less),
    with the remainder going to the classes with the largest fractional
    share and room left."""
    if total <= 0:
        return {c: 0 for c in rarity}
    classes = [c for c in rarity if available.get(c, 0) > 0]
    if not classes:
        return {c: 0 for c in rarity}
    weight_sum = sum(rarity[c] for c in classes) or 1.0
    ideal = {c: total * rarity[c] / weight_sum for c in classes}
    budgets = {c: min(available[c], int(math.floor(ideal[c]))) for c in classes}
    remaining = total - sum(budgets.values())
    # hand out the remainder, largest fractional share first, never past availability
    while remaining > 0:
        room = [c for c in classes if budgets[c] < available[c]]
        if not room:
            break
        room.sort(key=lambda c: -(ideal[c] - budgets[c]))
        for c in room:
            if remaining == 0:
                break
            budgets[c] += 1
            remaining -= 1
    return {c: budgets.get(c, 0) for c in rarity}


def cwie(dets: Iterable[Detection], rarity: Mapping[str, float]) -> float:
    """Class-weighted image entropy (Eq. 7), natural log."""
    total = 0.0
    for d in dets:
        if d.class_probs:
            probs = np.clip(np.asarray(d.class_probs, dtype=np.float64), _EPS, 1.0)
        else:
            p = min(max(d.confidence, _EPS), 1.0 - _EPS)
            probs = np.asarray([p, 1.0 - p])
        total += rarity.get(d.category, 0.5) * float(-(probs * np.log(probs)).sum())
    return total


def rcdi(dets: Iterable[Detection], rarity: Mapping[str, float]) -> float:
    """Rare-class diversity index (Eq. 8): rarity summed over the classes present."""
    return sum(rarity.get(c, 0.5) for c in {d.category for d in dets})


def rcsp(embeddings: np.ndarray | None) -> np.ndarray:
    """Rank-conditioned similarity penalty (Eq. 9) for candidates given in
    rank order: the top image scores 1, every other 1 − max cosine
    similarity to the images ranked above it. Without embeddings every
    image scores 1 (no penalty, no bonus)."""
    if embeddings is None:
        return np.ones(0)
    e = normalize(embeddings)
    n = len(e)
    out = np.ones(n)
    if n <= 1:
        return out
    sim = 1.0 - cosine_distance(e, e)
    for i in range(1, n):
        out[i] = 1.0 - float(sim[i, :i].max())
    return np.clip(out, 0.0, 1.0)


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 0:
        return np.full(len(values), 0.5)
    return (values - lo) / (hi - lo)


# ------------------------------------------------------------- selection


@dataclass
class PalPick:
    image_id: int
    score: float
    reason: str
    category: str


@dataclass
class PalResult:
    picks: list[PalPick]
    rarity: dict[str, float]
    budgets: dict[str, int]
    #: images the method could not fill (classes with too few candidates,
    #: or images with no detections at all); the caller tops up otherwise
    shortfall: int
    notes: list[str] = field(default_factory=list)
    #: inverse-label-frequency weights that shaped the budgets (1.0 = neutral)
    balance: dict[str, float] = field(default_factory=dict)

    def as_picks(self, index_of: Mapping[int, int]) -> list[Pick]:
        """Translate to selection.Pick rows via an image_id → index map."""
        return [
            Pick(index=index_of[p.image_id], score=p.score, reason=p.reason) for p in self.picks
        ]


def select(
    unlabeled: Mapping[int, Sequence[Detection]],
    labeled: Iterable[Detection],
    budget: int,
    *,
    embeddings: Mapping[int, np.ndarray] | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    balance: float = BALANCE,
) -> PalResult:
    """Run one PAL acquisition round. `balance` is the exponent on the
    inverse-label-frequency weight that tilts the class budgets towards
    under-labeled classes (0 = the paper's Eq. 6 alone).

    `unlabeled` maps image id → its detections (images with none are simply
    unreachable by the method and count towards `shortfall`); `labeled` are
    the detections on labeled images with `true_positive` set; `embeddings`
    (image id → vector) enable RCSP. Returns at most `budget` distinct images.
    """
    if budget <= 0:
        raise ProjectError(f"budget must be positive, got {budget}")
    if min(alpha, beta, gamma) < 0:
        raise ProjectError("alpha, beta and gamma must be non-negative")
    scorer = InstanceScorer(labeled)
    notes = list(scorer.notes)

    labeled_counts: dict[str, int] = {}
    for d in labeled:
        labeled_counts[d.category] = labeled_counts.get(d.category, 0) + 1
    unlabeled_counts: dict[str, int] = {}
    images_with: dict[str, set[int]] = {}
    for image_id, dets in unlabeled.items():
        for d in dets:
            unlabeled_counts[d.category] = unlabeled_counts.get(d.category, 0) + 1
            images_with.setdefault(d.category, set()).add(image_id)
    rarity = rarity_weights(labeled_counts, unlabeled_counts)
    reachable = {i for i, dets in unlabeled.items() if dets}
    balance_w = balance_weights(labeled_counts, rarity) if balance > 0 else {c: 1.0 for c in rarity}
    budget_weights = {c: rarity[c] * balance_w[c] ** balance for c in rarity}
    budgets = class_budgets(
        min(budget, len(reachable)), budget_weights, {c: len(s) for c, s in images_with.items()}
    )

    # LIUS per image per class: the most uncertain detection of that class
    lius: dict[tuple[int, str], tuple[float, Detection, float, str]] = {}
    for image_id, dets in unlabeled.items():
        for d in dets:
            h, p, mode = scorer.uncertainty(d)
            key = (image_id, d.category)
            if key not in lius or h > lius[key][0]:
                lius[key] = (h, d, p, mode)

    selected: dict[int, PalPick] = {}
    # rarest classes first so their (small) budgets are not eaten by duplicates
    for category in sorted(budgets, key=lambda c: -rarity[c]):
        b_c = budgets[category]
        if b_c <= 0:
            continue
        ranked = sorted(
            ((lius[(i, category)][0], i) for i in images_with.get(category, ())),
            key=lambda t: (-t[0], t[1]),
        )
        candidates = [i for _, i in ranked[: CANDIDATE_FACTOR * b_c]]
        if not candidates:
            continue
        l_scores = np.asarray([lius[(i, category)][0] for i in candidates])
        cw = _minmax(np.asarray([cwie(unlabeled[i], rarity) for i in candidates]))
        rd = _minmax(np.asarray([rcdi(unlabeled[i], rarity) for i in candidates]))
        if embeddings is not None and all(i in embeddings for i in candidates):
            rs = rcsp(np.asarray([np.asarray(embeddings[i]) for i in candidates]))
        else:
            rs = np.ones(len(candidates))
        final = alpha * l_scores + gamma * rs + beta * (cw + rd)
        order = sorted(range(len(candidates)), key=lambda k: (-final[k], candidates[k]))
        taken = 0
        for k in order:
            if taken >= b_c:
                break
            image_id = candidates[k]
            if image_id in selected:
                continue  # already chosen under another class; take the next one
            h, det, p, mode = lius[(image_id, category)]
            how = {
                "class": "class classifier",
                "pooled": "pooled classifier",
                "confidence": "raw confidence",
            }[mode]
            support = "unknown" if det.support is None else str(det.support)
            reason = (
                f"PAL for class '{category}' (round budget {b_c}): LIUS {h:.2f} — "
                f"P(true positive) {p:.2f} by {how} for its most uncertain "
                f"'{category}' detection (confidence {det.confidence:.2f}, support "
                f"{support}); CWIE {cw[k]:.2f}, RCDI {rd[k]:.2f}, RCSP {rs[k]:.2f}; "
                f"score {final[k]:.3f}"
            )
            selected[image_id] = PalPick(
                image_id=image_id, score=float(final[k]), reason=reason, category=category
            )
            taken += 1

    picks = sorted(selected.values(), key=lambda p: (-p.score, p.image_id))[:budget]
    shortfall = budget - len(picks)
    if shortfall:
        notes.append(
            f"{shortfall} of {budget} images could not be chosen by PAL: only "
            f"{len(reachable)} unlabeled images have detections and class budgets "
            f"ran out of candidates"
        )
    return PalResult(balance=balance_w,
        picks=picks, rarity=rarity, budgets=budgets, shortfall=shortfall, notes=notes
    )
