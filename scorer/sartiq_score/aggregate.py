"""Policy, review gates and :class:`~sartiq_score.types.ScoreReport` assembly.

This module turns a pair of canonicalised images into the final
"reference similarity" score.  Policy is FROZEN before submissions (SPEC.md
"Metrics" / "Weights" / "Status rules"); weights are never fitted to synthetic
data.  Augmentations only ever test invariance and monotonicity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import align, metrics
from .metrics import MetricContext
from .types import (
    ScoreReport,
    flags_from_canon,
)

# ---------------------------------------------------------------------------
# Frozen policy: family weights and gates (SPEC.md)
# ---------------------------------------------------------------------------

#: Family weights for the final score.  Must sum to 1.0.
WEIGHTS: dict[str, float] = {
    "colour": 0.40,
    "lpips": 0.25,
    "clip": 0.15,
    "geometry": 0.15,
    "ssim": 0.05,
}

#: CLIP similarity below this triggers a manual review (SPEC "Status rules").
CLIP_REVIEW_THRESHOLD: float = 0.30

#: v2.1 reference-free blank gate: a submission whose whole-image luma standard
#: deviation is below this (0..255 scale) is a flat field, not a garment.
#: u2net happily returns a ~40-50 % blob on a blank canvas, which scored up to
#: 36/100 before this rule existed.
UNIFORM_LUMA_STD: float = 4.0
#: Machine-readable reason for the blank gate.
BLANK_REASON: str = "blank_or_uniform"

#: Sub-score keys consumed by the aggregation, in report order.
SUBSCORE_KEYS: tuple[str, ...] = ("colour", "lpips", "clip", "geometry", "ssim")

#: Raw metric keys reported (SPEC "Raw values and all sub-scores always reported").
RAW_KEYS: tuple[str, ...] = (
    "colour_global_D",
    "colour_spatial_dE",
    "lpips_d",
    "clip_d",
    "geometry_iou",
    "ssim",
)


def _clip01(x: float) -> float:
    if not (x == x):  # NaN
        return 0.0
    return float(min(1.0, max(0.0, x)))


# ---------------------------------------------------------------------------
# v2.1 reference-free blank / flat-field gate
# ---------------------------------------------------------------------------


def luma_std(rgb_u8: Any) -> float:
    """Standard deviation (0..255) of the Rec.601 luma of an RGB uint8 image."""
    a = np.asarray(rgb_u8)
    if a.ndim != 3 or a.shape[2] < 3:
        return 0.0
    # v2.2: bounded memory -- subsample very large images (std of a 4 MPix grid is plenty)
    step = max(1, int(np.ceil(np.sqrt(a.shape[0] * a.shape[1] / 4_000_000))))
    a = a[::step, ::step, :3].astype(np.float32)
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    val = float(luma.std())
    return val if np.isfinite(val) else 0.0


def is_uniform_submission(rgb_u8: Any, threshold: float = UNIFORM_LUMA_STD) -> bool:
    """True when the submission is a flat field (blank / solid blob).

    Reference-free by construction: it only looks at the submission.  A real
    garment photograph -- even a very light garment on a white sweep -- always
    has texture and an edge, so its luma standard deviation is far above 4/255.
    """
    return luma_std(rgb_u8) < float(threshold)


# ---------------------------------------------------------------------------
# Sub-scores + raw values
# ---------------------------------------------------------------------------


def subscores_from_metrics(m: dict[str, float]) -> dict[str, float]:
    """Map the flat metric dict to the five family sub-scores in [0, 1]."""
    return {
        "colour": _clip01(m.get("s_colour", 0.0)),
        "lpips": _clip01(m.get("s_lpips", 0.0)),
        "clip": _clip01(m.get("s_clip", 0.0)),
        "geometry": _clip01(m.get("geometry_iou", 0.0)),
        "ssim": _clip01(m.get("ssim", 0.0)),
    }


def raw_from_metrics(m: dict[str, float]) -> dict[str, float]:
    """Extract the published raw metric values (finite floats)."""
    return {k: float(m.get(k, 0.0)) for k in RAW_KEYS}


def final_score(subscores: dict[str, float]) -> float:
    """``100 * sum(w * s)`` rounded to 2 decimals."""
    total = 0.0
    for key, w in WEIGHTS.items():
        total += w * _clip01(subscores.get(key, 0.0))
    return round(100.0 * total, 2)


# ---------------------------------------------------------------------------
# Status / review flags
# ---------------------------------------------------------------------------


def review_flags(
    ref: Any, sub: Any, subscores: dict[str, float]
) -> list[str]:
    """Review flags triggered by the submission (SPEC "Status rules").

    - ``mask_empty`` / ``mask_suspect`` on either image,
    - ``truncated`` on the submission,
    - ``multi_component`` on the submission while the reference is single,
    - ``clip_low`` when the CLIP sub-score < 0.30.
    """
    ref_flags = set(flags_from_canon(ref))
    sub_flags = set(flags_from_canon(sub))
    flags: list[str] = []

    if "mask_empty" in ref_flags or "mask_empty" in sub_flags:
        flags.append("mask_empty")
    if "mask_suspect" in ref_flags or "mask_suspect" in sub_flags:
        flags.append("mask_suspect")
    if "truncated" in sub_flags:
        flags.append("truncated")
    if "multi_component" in sub_flags and "multi_component" not in ref_flags:
        flags.append("multi_component")
    if subscores.get("clip", 0.0) < CLIP_REVIEW_THRESHOLD:
        flags.append("clip_low")
    return flags


def status_from_flags(flags: list[str]) -> str:
    """``review`` when any review flag fired, else ``ok``."""
    return "review" if flags else "ok"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    ref: Any,
    sub: Any,
    ctx: MetricContext,
    *,
    versions: dict[str, Any] | None = None,
    timing_s: float = 0.0,
    view: str | None = None,
) -> ScoreReport:
    """Compute metrics + policy and return a populated :class:`ScoreReport`.

    v2.1: before scoring, the submission canon is translated by at most +-1 %
    of each side onto the reference using masks only
    (:func:`sartiq_score.align.micro_align_shift`, identity preferred).  The
    applied shift is reported in ``raw['align_dy'/'align_dx']``.
    """
    dy, dx = align.micro_align_shift(ref, sub)
    sub_aligned = align.shift_canon(sub, dy, dx)
    m = metrics.all_metrics(ref, sub_aligned, ctx)
    subscores = subscores_from_metrics(m)
    raw = raw_from_metrics(m)
    raw["align_dy"] = int(dy)
    raw["align_dx"] = int(dx)
    raw["antialias_sigma"] = float(metrics.CANON_ANTIALIAS_SIGMA)
    flags = review_flags(ref, sub, subscores)
    status = status_from_flags(flags)
    report = ScoreReport(
        score=final_score(subscores),
        status=status,
        subscores=subscores,
        raw=raw,
        flags=flags,
        versions=dict(versions or {}),
        timing_s=float(timing_s),
    )
    if view is not None:
        # additive, defaulted report extras (never replaces frozen fields)
        report.raw = {**report.raw, "view": view} if isinstance(report.raw, dict) else report.raw
    return report


def invalid_report(
    reason: str,
    *,
    versions: dict[str, Any] | None = None,
    view: str | None = None,
    detail: str = "",
) -> ScoreReport:
    """Report for an unreadable / limit-violating input: score 0, status invalid.

    Only unreadable files and explicit submission-rule violations reach here;
    model uncertainty is expressed as ``review`` instead (never a hard cap).
    """
    raw: dict[str, Any] = {"invalid_reason": reason}
    if detail:
        raw["invalid_detail"] = detail
    if view is not None:
        raw["view"] = view
    return ScoreReport(
        score=0.0,
        status="invalid",
        subscores={k: 0.0 for k in SUBSCORE_KEYS},
        raw=raw,
        flags=[f"invalid:{reason}"],
        versions=dict(versions or {}),
        timing_s=0.0,
    )
