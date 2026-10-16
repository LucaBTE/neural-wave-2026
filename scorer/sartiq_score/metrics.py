"""Pure metric functions on canonical (:class:`~sartiq_score.types.Canon`) pairs.

Every function here is a *pure* function of its inputs plus an
optional pre-loaded :class:`MetricContext` (which only wraps already-initialised,
deterministic torch/OpenCLIP/LPIPS models).  Given the same inputs they must
return the same numbers, so no module-level mutable state lives here.

Design rules from SPEC.md ("Metrics"):
- colour_global  : Lab D65 quantile distance on each image's OWN soft mask.
- colour_spatial : 4x4 grid, ΔE00 on mutually supported cells, missing support = 0.
- lpips          : alex v0.1 on the flat-background canonical RGB.
- ssim           : skimage SSIM on the L* channel (data_range 100, sigma 1.5).
- geometry       : soft IoU of the two canonical masks.
- clip           : OpenCLIP ViT-B-32/openai on the whole-garment letterbox.

Every returned similarity is finite and in [0, 1]; NaNs, empty masks and
degenerate supports are handled explicitly (never propagate a NaN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import color as skcolor
from skimage.metrics import structural_similarity

from .types import BG_RGB

# ---------------------------------------------------------------------------
# Frozen policy constants (SPEC.md "Metrics" / "Weights")
# ---------------------------------------------------------------------------

#: colour_global half-score anchor: s_cg = 2 ** (-D / A_CG).
A_CG: float = 6.0
#: colour_spatial per-cell half-score anchor (ΔE00): sim = 2 ** (-dE / A_CS).
A_CS: float = 5.0
#: LPIPS half-score anchor: s = 2 ** (-d / A_LPIPS).
A_LPIPS: float = 0.20
#: CLIP half-score anchor: s = 2 ** (-(1 - cos) / A_CLIP).
A_CLIP: float = 0.10

#: Spatial colour grid (4x4) and its minimum per-cell support fraction.
SPATIAL_GRID: int = 4
SPATIAL_MIN_SUPPORT: float = 0.05

#: SSIM on the L* channel.
SSIM_DATA_RANGE: float = 100.0
SSIM_SIGMA: float = 1.5

#: v2.1: anti-alias applied to BOTH canon RGB images before LPIPS and SSIM.
#: The canonical crop is derived from the segmentation bbox, so two
#: presentations of the same garment (padded, translated, half-resolution)
#: resample it on a slightly different sub-pixel phase.  LPIPS's 1x1 features
#: read that phase as a photometric difference (measured: lpips_d 0.031 and
#: ssim 0.848 for a +20 % padding of an unchanged garment, worth ~3.5 points),
#: which is presentation, not fidelity.  A symmetric sigma=1 px Gaussian --
#: applied to *both* sides, and only for these two metrics -- removes the phase
#: noise while leaving content differences (colour, occlusion, blur, wrong
#: garment, text overlay) comfortably above their thresholds.  The LPIPS/SSIM
#: anchors are unchanged.
CANON_ANTIALIAS_SIGMA: float = 1.0

#: Quantiles for the global colour distance (0.05 .. 0.95 step 0.05).
COLOUR_QUANTILES: np.ndarray = np.linspace(0.05, 0.95, 19)

#: Background fill of the canonical canvas (single source: ``types.BG_RGB``).

#: Fraction of the reference colour score made of the global term (rest spatial).
COLOUR_W_GLOBAL: float = 0.60
COLOUR_W_SPATIAL: float = 0.40

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Metric context (loaded models, created once by api.Scorer)
# ---------------------------------------------------------------------------


@dataclass
class MetricContext:
    """Holds the deterministic, already-loaded models used by lpips/clip.

    ``colour``, ``geometry`` and ``ssim`` are model-free; they accept (and
    ignore) this context so callers can pass a single context everywhere.
    """

    lpips_fn: Any = None
    clip_model: Any = None
    clip_preprocess: Any = None
    clip_size: int = 224
    device: Any = "cpu"


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _mask_of(canon: Any) -> np.ndarray:
    """Soft mask as finite float64 in [0, 1]."""
    m = np.asarray(getattr(canon, "mask"), dtype=np.float64)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(m, 0.0, 1.0)


def _rgb_of(canon: Any) -> np.ndarray:
    """RGB as uint8 (H, W, 3)."""
    rgb = np.asarray(getattr(canon, "rgb"))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 (H, W, 3) -> CIELAB D65 float64 (H, W, 3).

    L* in [0, 100], a*/b* roughly in [-128, 127].
    """
    rgb = np.asarray(rgb_u8, dtype=np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb, illuminant="D65")
    return np.nan_to_num(lab, nan=0.0, posinf=0.0, neginf=0.0)


def _antialias_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """Symmetric anti-alias for the phase-sensitive LPIPS/SSIM comparison."""
    if CANON_ANTIALIAS_SIGMA <= 0.0:
        return rgb_u8
    from PIL import Image, ImageFilter  # noqa: PLC0415

    img = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8))
    out = img.filter(ImageFilter.GaussianBlur(float(CANON_ANTIALIAS_SIGMA)))
    return np.asarray(out, dtype=np.uint8)


def _support(mask: np.ndarray) -> np.ndarray:
    """Boolean support: soft mask > 0.5, eroded by at most 1 px.

    Falls back to the un-eroded thresholded mask when erosion would empty it
    (very thin garments), so we never lose the whole support.
    """
    binary = mask > 0.5
    if not binary.any():
        return binary
    eroded = ndimage.binary_erosion(binary, structure=np.ones((3, 3), bool))
    if not eroded.any():
        return binary
    return eroded


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Weighted quantiles via the cumulative-midpoint interpolation rule.

    Deterministic, order independent, and NaN-free for positive total weight.
    Returns NaN only when the total weight is not finite/positive.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    quantiles = np.asarray(quantiles, dtype=np.float64).ravel()
    out = np.full(quantiles.shape, np.nan, dtype=np.float64)
    if values.size == 0 or weights.size == 0:
        return out
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return out
    # Cumulative weight minus half the current weight -> interval midpoints.
    cum = np.cumsum(w) - 0.5 * w
    cum /= total
    out = np.interp(quantiles, cum, v)
    return out


def _sim_from_distance(d: float, anchor: float) -> float:
    """2 ** (-d / anchor), guarding against NaN/inf input."""
    if not np.isfinite(d):
        return 0.0
    s = 2.0 ** (-float(d) / anchor)
    if not np.isfinite(s):
        return 0.0
    return float(min(1.0, max(0.0, s)))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    m = float(np.nanmean(values))
    if not np.isfinite(m):
        return 0.0
    return m


# ---------------------------------------------------------------------------
# 1. colour_global
# ---------------------------------------------------------------------------


def colour_global(ref: Any, sub: Any) -> dict[str, float]:
    """Lab quantile distance on each image's own soft mask.

    ``D = mean over c in {L,a,b}, q in {0.05..0.95} of |Qref(c,q) - Qsub(c,q)|``
    where Q is a soft-mask-weighted quantile, then ``s_cg = 2 ** (-D / 6)``.

    Missing support on either side -> distance is undefined, similarity 0.
    """
    lab_r = rgb_to_lab(_rgb_of(ref))
    lab_s = rgb_to_lab(_rgb_of(sub))
    mask_r = _mask_of(ref)
    mask_s = _mask_of(sub)
    sup_r = _support(mask_r)
    sup_s = _support(mask_s)

    wr = mask_r[sup_r]
    ws = mask_s[sup_s]
    if wr.size == 0 or ws.size == 0 or float(wr.sum()) <= 0.0 or float(ws.sum()) <= 0.0:
        return {"colour_global_D": 0.0, "s_colour_global": 0.0}

    diffs: list[np.ndarray] = []
    for c in range(3):
        q_ref = weighted_quantile(lab_r[..., c][sup_r], wr, COLOUR_QUANTILES)
        q_sub = weighted_quantile(lab_s[..., c][sup_s], ws, COLOUR_QUANTILES)
        diffs.append(np.abs(q_ref - q_sub))
    D = _safe_mean(np.concatenate(diffs))
    return {"colour_global_D": float(D), "s_colour_global": _sim_from_distance(D, A_CG)}


# ---------------------------------------------------------------------------
# 2. colour_spatial
# ---------------------------------------------------------------------------


def _cell_slices(shape: tuple[int, int], grid: int) -> list[tuple[slice, slice]]:
    """Row-major list of (row_slice, col_slice) for a grid x grid split."""
    h, w = shape
    rows = np.linspace(0, h, grid + 1).astype(int)
    cols = np.linspace(0, w, grid + 1).astype(int)
    cells: list[tuple[slice, slice]] = []
    for i in range(grid):
        for j in range(grid):
            cells.append((slice(rows[i], rows[i + 1]), slice(cols[j], cols[j + 1])))
    return cells


def _cell_mean_lab(
    lab: np.ndarray, mask: np.ndarray, sl: tuple[slice, slice]
) -> np.ndarray | None:
    """Soft-mask-weighted mean Lab over mask > 0.5 pixels inside one cell."""
    m = mask[sl]
    sel = m > 0.5
    if not sel.any():
        return None
    w = m[sel]
    total = float(w.sum())
    if total <= 0.0:
        return None
    vals = lab[sl][sel]
    return (vals * w[:, None]).sum(axis=0) / total


def _spatial_direction(
    lab_ref: np.ndarray,
    lab_sub: np.ndarray,
    mask_ref: np.ndarray,
    mask_sub: np.ndarray,
    cells: list[tuple[slice, slice]],
) -> tuple[float, float]:
    """One direction of the spatial term.

    Iterate cells with support on the *leading* image (ref for the forward
    direction, sub for the reverse).  Similarity is ``2 ** (-ΔE00 / 5)`` when the
    other image also has support in the cell, else 0.  Cells are weighted by the
    leading image's foreground area.  Returns (weighted_mean, total_weight).
    """
    num = 0.0
    den = 0.0
    for sl in cells:
        m_lead = mask_ref[sl]
        lead_sel = m_lead > 0.5
        if lead_sel.size == 0:
            continue
        if float(lead_sel.mean()) < SPATIAL_MIN_SUPPORT:
            continue
        area = float(m_lead.sum())
        if area <= 0.0:
            continue
        mean_lead = _cell_mean_lab(lab_ref, mask_ref, sl)
        if mean_lead is None:
            continue

        m_other = mask_sub[sl]
        other_sel = m_other > 0.5
        if other_sel.size == 0 or float(other_sel.mean()) < SPATIAL_MIN_SUPPORT:
            sim = 0.0
        else:
            mean_other = _cell_mean_lab(lab_sub, mask_sub, sl)
            if mean_other is None:
                sim = 0.0
            else:
                dE = float(np.asarray(skcolor.deltaE_ciede2000(mean_lead, mean_other)))
                sim = _sim_from_distance(dE, A_CS)
        num += area * sim
        den += area
    if den <= 0.0:
        return 0.0, 0.0
    return num / den, den


def colour_spatial(ref: Any, sub: Any) -> dict[str, float]:
    """4x4 spatial colour agreement (ΔE00 on mutually supported cells)."""
    lab_r = rgb_to_lab(_rgb_of(ref))
    lab_s = rgb_to_lab(_rgb_of(sub))
    mask_r = _mask_of(ref)
    mask_s = _mask_of(sub)
    cells = _cell_slices(mask_r.shape, SPATIAL_GRID)

    fwd, w_fwd = _spatial_direction(lab_r, lab_s, mask_r, mask_s, cells)
    rev, w_rev = _spatial_direction(lab_s, lab_r, mask_s, mask_r, cells)

    parts = []
    if w_fwd > 0.0:
        parts.append(fwd)
    if w_rev > 0.0:
        parts.append(rev)
    s_cs = float(np.mean(parts)) if parts else 0.0
    return {
        "colour_spatial_dE": float(np.mean([fwd, rev])) if (w_fwd > 0 or w_rev > 0) else 0.0,
        "s_colour_spatial": float(min(1.0, max(0.0, s_cs))),
    }


def colour(ref: Any, sub: Any) -> dict[str, float]:
    """Combined colour similarity: ``0.6 * colour_global + 0.4 * colour_spatial``."""
    cg = colour_global(ref, sub)
    cs = colour_spatial(ref, sub)
    s = COLOUR_W_GLOBAL * cg["s_colour_global"] + COLOUR_W_SPATIAL * cs["s_colour_spatial"]
    return {
        **cg,
        **cs,
        "s_colour": float(min(1.0, max(0.0, s))),
    }


# ---------------------------------------------------------------------------
# 5. geometry
# ---------------------------------------------------------------------------


def geometry(ref: Any, sub: Any) -> dict[str, float]:
    """Soft mask IoU: ``sum(min) / sum(max)``.  Empty union -> 0."""
    mask_r = _mask_of(ref)
    mask_s = _mask_of(sub)
    union = float(np.maximum(mask_r, mask_s).sum())
    if union <= _EPS:
        return {"geometry_iou": 0.0}
    inter = float(np.minimum(mask_r, mask_s).sum())
    iou = inter / union
    if not np.isfinite(iou):
        iou = 0.0
    return {"geometry_iou": float(min(1.0, max(0.0, iou)))}


# ---------------------------------------------------------------------------
# 4. ssim
# ---------------------------------------------------------------------------


def ssim_lstar(ref: Any, sub: Any) -> dict[str, float]:
    """skimage SSIM on the L* channel (data_range 100, gaussian sigma 1.5).

    v2.1: both sides are anti-aliased first (``CANON_ANTIALIAS_SIGMA``).
    """
    l_r = rgb_to_lab(_antialias_rgb(_rgb_of(ref)))[..., 0]
    l_s = rgb_to_lab(_antialias_rgb(_rgb_of(sub)))[..., 0]
    try:
        val = structural_similarity(
            l_r,
            l_s,
            data_range=SSIM_DATA_RANGE,
            gaussian_weights=True,
            sigma=SSIM_SIGMA,
        )
    except ValueError:
        return {"ssim": 0.0}
    val = float(val)
    if not np.isfinite(val):
        val = 0.0
    return {"ssim": float(min(1.0, max(0.0, val)))}


# ---------------------------------------------------------------------------
# 3. lpips
# ---------------------------------------------------------------------------


def _to_lpips_tensor(rgb_u8: np.ndarray, ctx: MetricContext):
    import torch

    arr = np.asarray(rgb_u8, dtype=np.float32) / 255.0 * 2.0 - 1.0  # [-1, 1]
    ten = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    return ten.unsqueeze(0).to(ctx.device, dtype=torch.float32)


def lpips_metric(ref: Any, sub: Any, ctx: MetricContext) -> dict[str, float]:
    """LPIPS alex v0.1 on the canonical RGB; ``s = 2 ** (-d / 0.20)``.

    v2.1: both sides are anti-aliased first (``CANON_ANTIALIAS_SIGMA``), so the
    sub-pixel resampling phase of the canonical crop does not dominate the
    distance.
    """
    if ctx.lpips_fn is None:
        raise RuntimeError("MetricContext.lpips_fn is not loaded")
    import torch

    a = _to_lpips_tensor(_antialias_rgb(_rgb_of(ref)), ctx)
    b = _to_lpips_tensor(_antialias_rgb(_rgb_of(sub)), ctx)
    with torch.inference_mode():
        out = ctx.lpips_fn(a, b)
    d = float(np.asarray(out.detach().cpu()).reshape(-1)[0])
    if not np.isfinite(d):
        d = 0.0
    return {"lpips_d": float(max(0.0, d)), "s_lpips": _sim_from_distance(d, A_LPIPS)}


# ---------------------------------------------------------------------------
# 6. clip
# ---------------------------------------------------------------------------


def letterbox_square(
    rgb_u8: np.ndarray, size: int, bg: tuple[int, int, int] = BG_RGB
) -> np.ndarray:
    """Fit the whole image into a square canvas of side ``size`` on a flat bg.

    Aspect ratio preserved (no centre-crop of the garment), padded with ``bg``.
    """
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8))
    w, h = img.size
    scale = float(size) / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), tuple(int(v) for v in bg))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _clip_embedding(rgb_u8: np.ndarray, ctx: MetricContext):
    """Raw (unnormalised) CLIP image features, float32, shape [1, D]."""
    import torch
    from PIL import Image

    if ctx.clip_model is None or ctx.clip_preprocess is None:
        raise RuntimeError("MetricContext.clip_model is not loaded")
    square = letterbox_square(rgb_u8, ctx.clip_size, BG_RGB)
    tensor = ctx.clip_preprocess(Image.fromarray(square)).unsqueeze(0).to(ctx.device)
    with torch.inference_mode():
        feats = ctx.clip_model.encode_image(tensor)
    return feats


def clip_metric(ref: Any, sub: Any, ctx: MetricContext) -> dict[str, float]:
    """OpenCLIP ViT-B-32/openai cosine distance; ``s = 2 ** (-(1-cos) / 0.10)``.

    The cosine is computed in float64 with explicit normalisation so that two
    identical inputs give exactly ``cos == 1`` up to float64 rounding (a plain
    float32 dot of two unit vectors is only accurate to ~1e-6).
    """
    fa = _clip_embedding(_rgb_of(ref), ctx).double()
    fb = _clip_embedding(_rgb_of(sub), ctx).double()
    num = (fa * fb).sum(dim=-1)
    den = (fa.norm(dim=-1) * fb.norm(dim=-1)).clamp_min(_EPS)
    cos = float((num / den).reshape(-1)[0].clamp(-1.0, 1.0).detach().cpu())
    d = 1.0 - cos
    if not np.isfinite(d):
        d = 1.0
    return {"clip_d": float(max(0.0, d)), "s_clip": _sim_from_distance(d, A_CLIP)}


# ---------------------------------------------------------------------------
# Convenience: all metric families at once
# ---------------------------------------------------------------------------


def all_metrics(ref: Any, sub: Any, ctx: MetricContext) -> dict[str, float]:
    """Run every metric family and return a flat raw + sub-score dict."""
    out: dict[str, float] = {}
    out.update(colour(ref, sub))
    out.update(geometry(ref, sub))
    out.update(ssim_lstar(ref, sub))
    out.update(lpips_metric(ref, sub, ctx))
    out.update(clip_metric(ref, sub, ctx))
    return out
