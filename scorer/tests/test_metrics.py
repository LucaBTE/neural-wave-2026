"""Tests for metrics, aggregate policy, and the public api glue.

Synthetic :class:`Canon` objects are built directly from arrays so the tests do
not depend on segmentation/alignment.  A couple of tests use real
packshots from ``../dataset/*/out.jpg`` resized to the canonical 512x384 canvas
to exercise LPIPS/CLIP.

Run with::

    .venv/bin/python -m pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pytest

# Make ``sartiq_score`` importable regardless of pytest's rootdir handling.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sartiq_score import aggregate, metrics  # noqa: E402
from sartiq_score.types import BG_RGB, CANON_H, CANON_W, Canon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.normpath(os.path.join(HERE, "..", "..", "dataset"))


# ---------------------------------------------------------------------------
# Synthetic Canon helpers
# ---------------------------------------------------------------------------


def make_canon(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    n_components: int = 1,
    truncated: bool = False,
    area_frac: float | None = None,
) -> Canon:
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    assert rgb.shape == (CANON_H, CANON_W, 3), rgb.shape
    if mask is None:
        mask = np.ones((CANON_H, CANON_W), np.float32)
    mask = np.ascontiguousarray(np.asarray(mask, dtype=np.float32))
    af = float((mask > 0.5).sum()) / mask.size if area_frac is None else float(area_frac)
    return Canon(
        rgb=rgb,
        mask=mask,
        bbox_frac=(0.0, 0.0, 1.0, 1.0),
        area_frac=af,
        n_components=n_components,
        truncated=truncated,
        warnings=[],
    )


def rect_canon(
    color=(120, 80, 60),
    box=(96, 128, 288, 384),
    *,
    soft: float = 1.0,
    n_components: int = 1,
    truncated: bool = False,
    area_frac: float | None = None,
) -> Canon:
    x0, y0, x1, y1 = box
    rgb = np.empty((CANON_H, CANON_W, 3), np.uint8)
    rgb[...] = np.asarray(BG_RGB, np.uint8)
    mask = np.zeros((CANON_H, CANON_W), np.float32)
    rgb[y0:y1, x0:x1] = np.asarray(color, np.uint8)
    mask[y0:y1, x0:x1] = soft
    af = float((y1 - y0) * (x1 - x0)) / (CANON_H * CANON_W) if area_frac is None else float(area_frac)
    return Canon(
        rgb=rgb,
        mask=mask,
        bbox_frac=(x0 / CANON_W, y0 / CANON_H, (x1 - x0) / CANON_W, (y1 - y0) / CANON_H),
        area_frac=af,
        n_components=n_components,
        truncated=truncated,
        warnings=[],
    )


def lab_color_rgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    from skimage import color as skcolor

    lab = np.zeros((1, 1, 3), np.float64)
    lab[0, 0] = (L, a, b)
    rgb = np.clip(skcolor.lab2rgb(lab)[0, 0] * 255.0, 0, 255)
    return tuple(int(v) for v in np.round(rgb))


def load_real_canon(path: str) -> Canon:
    """Resize a real packshot to the canonical canvas (mask = whole image)."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((CANON_W, CANON_H), Image.BICUBIC)
        rgb = np.asarray(im, dtype=np.uint8)
    return make_canon(rgb, np.ones((CANON_H, CANON_W), np.float32))


def real_pairs(limit: int = 2) -> list[str]:
    files = sorted(glob.glob(os.path.join(DATASET, "*", "out.jpg")))
    return files[:limit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer():
    from sartiq_score.api import Scorer

    return Scorer()


@pytest.fixture(scope="module")
def ctx(scorer):
    return scorer.ctx


# ---------------------------------------------------------------------------
# 1. Pure-function metric invariants (no models)
# ---------------------------------------------------------------------------


def test_identical_pair_all_subscores_one():
    ref = rect_canon((120, 80, 60))
    sub = rect_canon((120, 80, 60))
    cg = metrics.colour_global(ref, sub)
    cs = metrics.colour_spatial(ref, sub)
    col = metrics.colour(ref, sub)
    geo = metrics.geometry(ref, sub)
    ssim = metrics.ssim_lstar(ref, sub)
    assert abs(cg["s_colour_global"] - 1.0) <= 1e-6
    assert abs(cs["s_colour_spatial"] - 1.0) <= 1e-6
    assert abs(col["s_colour"] - 1.0) <= 1e-6
    assert abs(geo["geometry_iou"] - 1.0) <= 1e-6
    assert abs(ssim["ssim"] - 1.0) <= 1e-6
    assert cg["colour_global_D"] <= 1e-6


def test_colour_global_monotone_with_shift():
    ref = rect_canon(lab_color_rgb(60.0, 10.0, 10.0))
    scores = []
    for delta in (0.0, 2.0, 5.0, 10.0, 20.0):
        sub = rect_canon(lab_color_rgb(60.0, 10.0 + delta, 10.0))
        scores.append(metrics.colour_global(ref, sub)["s_colour_global"])
    # strictly decreasing as the a* shift grows
    for a, b in zip(scores, scores[1:]):
        assert b < a, scores
    assert scores[0] > 0.999
    assert scores[-1] < scores[2]


def test_colour_spatial_missing_support_is_zero():
    ref = rect_canon((120, 80, 60))
    assert metrics.colour_spatial(ref, rect_canon((120, 80, 60)))["s_colour_spatial"] > 0.999

    # delete the left half of the garment from the submission
    sub = rect_canon((120, 80, 60))
    sub.mask[:, :192] = 0.0
    sub.rgb[:, :192] = np.asarray(BG_RGB, np.uint8)
    s = metrics.colour_spatial(ref, sub)["s_colour_spatial"]
    assert s < 0.999, s
    assert s >= 0.0


def test_colour_spatial_empty_submission_is_zero():
    ref = rect_canon((120, 80, 60))
    sub = rect_canon((120, 80, 60))
    sub.mask[:] = 0.0
    res = metrics.colour_spatial(ref, sub)
    assert res["s_colour_spatial"] == 0.0
    assert np.isfinite(res["s_colour_spatial"])


def test_geometry_iou_known_shapes():
    ref = rect_canon(box=(100, 100, 250, 300))  # 150x200
    sub = rect_canon(box=(150, 100, 300, 300))  # shifted 50 px right
    res = metrics.geometry(ref, sub)
    assert abs(res["geometry_iou"] - 0.5) < 1e-6, res

    # identical
    assert abs(metrics.geometry(ref, rect_canon(box=(100, 100, 250, 300)))["geometry_iou"] - 1.0) < 1e-6

    # disjoint
    ref2 = rect_canon(box=(10, 10, 100, 200))
    sub2 = rect_canon(box=(200, 300, 300, 450))
    assert metrics.geometry(ref2, sub2)["geometry_iou"] == 0.0


def test_empty_and_nan_masks_never_nan():
    ref = rect_canon((120, 80, 60))
    # empty mask
    empty = rect_canon((120, 80, 60))
    empty.mask[:] = 0.0
    # NaN mask (must be sanitised)
    nan_canon = rect_canon((120, 80, 60))
    nan_canon.mask = np.full((CANON_H, CANON_W), np.nan, np.float32)

    for sub in (empty, nan_canon):
        out = {}
        out.update(metrics.colour_global(ref, sub))
        out.update(metrics.colour_spatial(ref, sub))
        out.update(metrics.colour(ref, sub))
        out.update(metrics.geometry(ref, sub))
        out.update(metrics.ssim_lstar(ref, sub))
        for k, v in out.items():
            assert np.isfinite(v), (k, v)
            assert 0.0 <= v <= 1.0 or k.endswith("_D") or k.endswith("_dE"), (k, v)


def test_weighted_quantile_basic():
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = np.ones(5)
    q = metrics.weighted_quantile(vals, w, np.array([0.0, 0.5, 1.0]))
    assert q[0] == 0.0 and q[-1] == 4.0
    assert abs(q[1] - 2.0) < 1e-9
    # zero weights -> NaN
    assert np.isnan(metrics.weighted_quantile(vals, np.zeros(5), np.array([0.5]))[0])


# ---------------------------------------------------------------------------
# 2. Model-backed metrics (LPIPS + CLIP)
# ---------------------------------------------------------------------------


def test_lpips_clip_identical(ctx):
    ref = rect_canon((120, 80, 60))
    lp = metrics.lpips_metric(ref, ref, ctx)
    cl = metrics.clip_metric(ref, ref, ctx)
    assert abs(lp["s_lpips"] - 1.0) <= 1e-6, lp
    assert abs(cl["s_clip"] - 1.0) <= 1e-6, cl
    assert lp["lpips_d"] >= 0.0


def test_lpips_clip_real_images(ctx):
    files = real_pairs(2)
    if len(files) < 2:
        pytest.skip("need at least two ../dataset/*/out.jpg packshots")
    a = load_real_canon(files[0])
    b = load_real_canon(files[1])

    self_lp = metrics.lpips_metric(a, a, ctx)["s_lpips"]
    self_cl = metrics.clip_metric(a, a, ctx)["s_clip"]
    assert abs(self_lp - 1.0) <= 1e-6, self_lp
    assert abs(self_cl - 1.0) <= 1e-6, self_cl

    wrong_lp = metrics.lpips_metric(a, b, ctx)["s_lpips"]
    wrong_cl = metrics.clip_metric(a, b, ctx)["s_clip"]
    assert wrong_lp < self_lp, (wrong_lp, self_lp)
    assert wrong_cl < self_cl, (wrong_cl, self_cl)


def test_real_image_self_full_score(ctx):
    files = real_pairs(1)
    if not files:
        pytest.skip("no dataset packshots")
    a = load_real_canon(files[0])
    m = metrics.all_metrics(a, a, ctx)
    for key, val in m.items():
        assert np.isfinite(val), (key, val)
    subs = aggregate.subscores_from_metrics(m)
    for k, v in subs.items():
        assert abs(v - 1.0) <= 1e-6, (k, v)


# ---------------------------------------------------------------------------
# 3. Aggregate policy
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert abs(sum(aggregate.WEIGHTS.values()) - 1.0) < 1e-12
    assert set(aggregate.WEIGHTS) == set(aggregate.SUBSCORE_KEYS)


def test_final_score_bounds():
    assert aggregate.final_score({k: 1.0 for k in aggregate.SUBSCORE_KEYS}) == 100.0
    assert aggregate.final_score({k: 0.0 for k in aggregate.SUBSCORE_KEYS}) == 0.0
    # rounding to two decimals
    s = aggregate.final_score({"colour": 0.333, "lpips": 0.0, "clip": 0.0, "geometry": 0.0, "ssim": 0.0})
    assert s == round(0.40 * 0.333 * 100.0, 2)


def test_status_rules_flags():
    single_ok = rect_canon((120, 80, 60))
    subs_ok = {k: 0.8 for k in aggregate.SUBSCORE_KEYS}
    assert aggregate.review_flags(single_ok, single_ok, subs_ok) == []
    assert aggregate.status_from_flags([]) == "ok"

    # mask_empty on either side
    empty = rect_canon((120, 80, 60), area_frac=0.0)
    flags = aggregate.review_flags(empty, single_ok, subs_ok)
    assert "mask_empty" in flags

    # mask_suspect
    suspect = rect_canon((120, 80, 60), area_frac=0.95)
    assert "mask_suspect" in aggregate.review_flags(single_ok, suspect, subs_ok)

    # truncated on submission
    trunc = rect_canon((120, 80, 60), truncated=True)
    assert "truncated" in aggregate.review_flags(single_ok, trunc, subs_ok)

    # multi_component on submission while reference single
    multi = rect_canon((120, 80, 60), n_components=2)
    assert "multi_component" in aggregate.review_flags(single_ok, multi, subs_ok)
    # ... but not when the reference is multi too
    assert "multi_component" not in aggregate.review_flags(multi, multi, subs_ok)

    # low clip
    low = dict(subs_ok, clip=0.2)
    assert "clip_low" in aggregate.review_flags(single_ok, single_ok, low)
    assert aggregate.status_from_flags(aggregate.review_flags(single_ok, single_ok, low)) == "review"


def test_build_report_identical_is_100(ctx):
    ref = rect_canon((120, 80, 60))
    sub = rect_canon((120, 80, 60))
    rep = aggregate.build_report(ref, sub, ctx, versions={"package": "test"})
    assert rep.score == 100.0, rep.score
    assert rep.status == "ok"
    assert set(rep.subscores) == set(aggregate.SUBSCORE_KEYS)
    assert set(aggregate.RAW_KEYS) <= set(rep.raw)
    assert rep.versions == {"package": "test"}


def test_build_report_blank_submission_low(ctx):
    ref = rect_canon((120, 80, 60))
    blank = rect_canon((120, 80, 60))
    blank.rgb[...] = np.asarray(BG_RGB, np.uint8)
    blank.mask[:] = 0.0
    blank.area_frac = 0.0
    rep = aggregate.build_report(ref, blank, ctx)
    assert rep.score < 100.0
    assert rep.status == "review"  # mask_empty on submission


def test_invalid_report():
    rep = aggregate.invalid_report("unreadable", versions={"package": "test"})
    assert rep.score == 0.0
    assert rep.status == "invalid"
    assert any(f.startswith("invalid:") for f in rep.flags)


# ---------------------------------------------------------------------------
# 4. api glue (no segmentation: score pre-built canons)
# ---------------------------------------------------------------------------


def test_scorer_score_canons_and_versions(scorer):
    ref = rect_canon((120, 80, 60))
    sub = rect_canon((120, 80, 60))
    rep = scorer.score_canons(ref, sub)
    assert rep.score == 100.0
    assert rep.status == "ok"
    v = rep.versions
    assert v.get("lpips_model") == "alex@v0.1"
    assert v.get("clip_model") == "ViT-B-32/openai"
    # weight hashes recorded
    assert len(v.get("lpips_alex_sha256", "")) == 64
    assert len(v.get("clip_sha256", "")) == 64


def test_scorer_invalid_path_is_invalid(scorer, tmp_path):
    bad = tmp_path / "not_an_image.txt"
    bad.write_text("hello")
    rep = scorer.score_pair(str(bad), str(bad))
    assert rep.status == "invalid"
    assert rep.score == 0.0


# ---------------------------------------------------------------------------
# 5. Warm per-pair timing (CPU)
# ---------------------------------------------------------------------------


def test_warm_per_pair_timing(ctx):
    files = real_pairs(1)
    if not files:
        pytest.skip("no dataset packshots")
    a = load_real_canon(files[0])
    # warm-up
    metrics.all_metrics(a, a, ctx)
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        metrics.all_metrics(a, a, ctx)
    per_pair = (time.perf_counter() - t0) / n
    print(f"\n[warm] all_metrics per pair: {per_pair:.4f} s ({1/per_pair:.2f} pair/s)")
    assert per_pair < 30.0


# ---------------------------------------------------------------------------
# 6. v2.1 policy: blank gate, micro-alignment, acceptance monotonicity rules
# ---------------------------------------------------------------------------


def test_uniform_gate_detects_blank_and_blob():
    flat = np.full((64, 64, 3), 245, np.uint8)
    assert aggregate.luma_std(flat) == 0.0
    assert aggregate.is_uniform_submission(flat)
    blob = np.full((64, 64, 3), (110, 122, 138), np.uint8)
    assert aggregate.is_uniform_submission(blob)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert not aggregate.is_uniform_submission(noise)


def test_uniform_gate_keeps_real_garments():
    # a dark garment on a white sweep, deliberately low-contrast and small
    img = np.full((256, 256, 3), 250, np.uint8)
    img[100:150, 110:140] = np.array([240, 240, 235], np.uint8)  # faint, very light
    img[102:148, 112:138] = np.array([180, 180, 175], np.uint8)
    assert aggregate.luma_std(img) > aggregate.UNIFORM_LUMA_STD
    assert not aggregate.is_uniform_submission(img)


def test_uniform_gate_constants_documented():
    assert aggregate.UNIFORM_LUMA_STD == 4.0
    assert aggregate.BLANK_REASON == "blank_or_uniform"


def test_micro_align_recovers_translation_and_prefers_identity():
    from sartiq_score import align

    ref = rect_canon((120, 80, 60))
    assert align.micro_align_shift(ref, ref) == (0, 0)
    for dy, dx in ((2, -3), (-4, 4), (1, 1)):
        shifted = align.shift_canon(ref, dy, dx)
        assert align.micro_align_shift(ref, shifted) == (-dy, -dx)


def test_micro_align_is_mask_only_identity_for_colour_change():
    from sartiq_score import align

    a = rect_canon((120, 80, 60))
    b = rect_canon((20, 200, 210))  # same silhouette, wildly different colour
    assert align.micro_align_shift(a, b) == (0, 0)


def test_build_report_publishes_alignment(ctx):
    from sartiq_score import align

    ref = rect_canon((120, 80, 60))
    sub = align.shift_canon(rect_canon((120, 80, 60)), 2, -2)
    rep = aggregate.build_report(ref, sub, ctx)
    assert rep.raw["align_dy"] == -2 and rep.raw["align_dx"] == 2
    assert rep.score >= 99.0, rep.score


def _mono_record(garment, family, variant, score, **extra):
    return {
        "garment": garment,
        "ref": "ref.png",
        "family": family,
        "variant": variant,
        "score": float(score),
        "status": "ok",
        "flags": [],
        "wall_s": 0.1,
        **extra,
    }


def _check_map(checks):
    return {(c["family"], c["check"]): c for c in checks}


def test_monotone_policy_tolerates_small_step_but_requires_total_drop():
    from sartiq_score import sanity

    recs = [_mono_record("g", "self", "self", 100.0)]
    for v, s in (("jpeg_q80", 99.0), ("jpeg_q40", 98.6), ("jpeg_q10", 90.0)):
        recs.append(_mono_record("g", "monotone", v, s))
    by = _check_map(sanity.evaluate(recs))
    assert by[("monotone", "jpeg:non_increasing")]["ok"]
    assert by[("monotone", "jpeg:total_drop>=2")]["ok"]


def test_monotone_policy_flags_a_score_increase():
    from sartiq_score import sanity

    recs = [_mono_record("g", "self", "self", 100.0)]
    for v, s in (("jpeg_q80", 99.0), ("jpeg_q40", 99.6), ("jpeg_q10", 90.0)):
        recs.append(_mono_record("g", "monotone", v, s))
    by = _check_map(sanity.evaluate(recs))
    assert not by[("monotone", "jpeg:non_increasing")]["ok"]


def test_monotone_policy_flags_a_tiny_total_drop():
    from sartiq_score import sanity

    recs = [_mono_record("g", "self", "self", 100.0)]
    for v, s in (("jpeg_q80", 99.0), ("jpeg_q40", 98.5), ("jpeg_q10", 98.5)):
        recs.append(_mono_record("g", "monotone", v, s))
    by = _check_map(sanity.evaluate(recs))
    assert by[("monotone", "jpeg:non_increasing")]["ok"]
    assert not by[("monotone", "jpeg:total_drop>=2")]["ok"]


def test_hue_rotate_has_no_total_drop_floor():
    from sartiq_score import sanity

    assert "hue_rotate" not in sanity.TOTAL_DROP_MIN
    recs = [_mono_record("g", "self", "self", 100.0)]
    for v, s in (("hue_10deg", 98.4), ("hue_20deg", 98.0), ("hue_40deg", 95.0)):
        recs.append(_mono_record("g", "monotone", v, s))
    by = _check_map(sanity.evaluate(recs))
    assert by[("monotone", "hue_rotate:non_increasing")]["ok"]
    assert not any(k[1].startswith("hue_rotate:total_drop") for k in by)


def test_truncated_reference_excludes_removal_and_crop():
    from sartiq_score import sanity

    recs = [_mono_record("g", "self", "self", 100.0, ref_truncated=True)]
    for v, s in (("remove_5pct", 76.5), ("remove_15pct", 79.8), ("remove_30pct", 68.0)):
        recs.append(_mono_record("g", "monotone", v, s))
    for v, s in (("crop_5pct", 80.0), ("crop_15pct", 70.0), ("crop_30pct", 60.0)):
        recs.append(_mono_record("g", "monotone", v, s))
    by = _check_map(sanity.evaluate(recs))
    assert by[("monotone", "removal:excluded(ref_truncated)")]["ok"]
    assert by[("monotone", "crop_away:excluded(ref_truncated)")]["ok"]
    assert ("monotone", "removal:non_increasing") not in by


def test_non_truncated_reference_still_checks_removal():
    from sartiq_score import sanity

    recs = [_mono_record("g", "self", "self", 100.0, ref_truncated=False)]
    for v, s in (("remove_5pct", 92.0), ("remove_15pct", 82.0), ("remove_30pct", 70.0)):
        recs.append(_mono_record("g", "monotone", v, s))
    by = _check_map(sanity.evaluate(recs))
    assert by[("monotone", "removal:non_increasing")]["ok"]
    assert by[("monotone", "removal:total_drop>=8")]["ok"]
    assert ("monotone", "removal:excluded(ref_truncated)") not in by
