"""Canonical alignment.

Contract: ``canonicalize(loaded, mask) -> Canon`` (see :mod:`sartiq_score.types`).

Frozen v2.0 policy (SPEC.md "Preprocessing" / "Alignment"):
  * crop the mask bbox expanded by 2 %,
  * uniform (isotropic) scale so the binding dimension of the mask bbox
    occupies 90 % of the 384x512 canvas,
  * centre it, background :data:`BG_RGB`, soft mask transformed identically,
  * **no** rotation, flip, anisotropic scaling or refinement,
  * empty mask -> whole image used as bbox, flag ``mask_empty``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .segment import MaskStats, binary, mask_stats
from .types import (
    BG_RGB,
    CANON_FILL,
    CANON_H,
    CANON_MARGIN,
    CANON_W,
    Canon,
    Loaded,
)


def _crop_with_pad(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill) -> np.ndarray:
    """Crop ``arr[y0:y1, x0:x1]`` allowing coordinates to fall outside."""
    h, w = arr.shape[:2]
    ch, cw = y1 - y0, x1 - x0
    if arr.ndim == 3:
        out = np.empty((ch, cw, arr.shape[2]), dtype=arr.dtype)
        out[...] = np.asarray(fill, dtype=arr.dtype)
    else:
        out = np.full((ch, cw), fill, dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def _resize(arr: np.ndarray, size: tuple[int, int], is_mask: bool) -> np.ndarray:
    """Deterministic bilinear resize (PIL, float path for masks)."""
    if is_mask:
        img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32), mode="F")
    else:
        img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), mode="RGB")
    img = img.resize(size, Image.BILINEAR)
    out = np.asarray(img)
    return out.astype(np.float32 if is_mask else np.uint8, copy=False)


def _paste(canvas: np.ndarray, patch: np.ndarray, fill) -> None:
    ch, cw = patch.shape[:2]
    ox = (canvas.shape[1] - cw) // 2
    oy = (canvas.shape[0] - ch) // 2
    canvas[oy : oy + ch, ox : ox + cw] = patch


def canonicalize(loaded: Loaded, mask: np.ndarray) -> Canon:
    """Canonicalise one image + its soft mask onto the 512x384 canvas."""
    rgb = np.asarray(loaded.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Loaded.rgb must be HxWx3, got {rgb.shape}")
    h, w = rgb.shape[:2]

    soft = np.asarray(mask, dtype=np.float32)
    warnings: list[str] = list(loaded.warnings)
    if soft.ndim == 3:
        soft = soft[..., 0]
    if soft.shape != (h, w):
        soft = np.asarray(
            Image.fromarray(np.ascontiguousarray(soft, dtype=np.float32), mode="F").resize(
                (w, h), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        warnings.append("mask_resized_to_image")
    soft = np.clip(soft, 0.0, 1.0)

    stats: MaskStats = mask_stats(soft)
    used_whole_image = False
    if stats.empty:
        x0, y0, x1, y1 = 0, 0, w, h
        used_whole_image = True
    else:
        x0, y0, x1, y1 = stats.bbox

    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:  # pragma: no cover - guarded by empty check
        x0, y0, x1, y1 = 0, 0, w, h
        bw, bh = w, h
        used_whole_image = True

    # --- scale: bbox's binding dimension occupies CANON_FILL of the canvas
    scale = CANON_FILL * min(CANON_W / float(bw), CANON_H / float(bh))

    # --- crop window = bbox + 2 % margin (symmetric, so the centre is kept)
    mx = CANON_MARGIN * bw
    my = CANON_MARGIN * bh
    cx0 = int(np.floor(x0 - mx))
    cy0 = int(np.floor(y0 - my))
    cx1 = int(np.ceil(x1 + mx))
    cy1 = int(np.ceil(y1 + my))

    crop_rgb = _crop_with_pad(rgb, cx0, cy0, cx1, cy1, BG_RGB)
    crop_mask = _crop_with_pad(soft, cx0, cy0, cx1, cy1, 0.0)

    cw, ch = crop_mask.shape[1], crop_mask.shape[0]
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    new_w = min(new_w, CANON_W)
    new_h = min(new_h, CANON_H)

    res_rgb = _resize(crop_rgb, (new_w, new_h), is_mask=False)
    res_mask = _resize(crop_mask, (new_w, new_h), is_mask=True)
    res_mask = np.clip(res_mask, 0.0, 1.0)

    canvas_rgb = np.empty((CANON_H, CANON_W, 3), dtype=np.uint8)
    canvas_rgb[...] = np.asarray(BG_RGB, dtype=np.uint8)
    canvas_mask = np.zeros((CANON_H, CANON_W), dtype=np.float32)
    _paste(canvas_rgb, res_rgb, BG_RGB)
    _paste(canvas_mask, res_mask, 0.0)

    flags = list(stats.flags)
    if used_whole_image and "mask_empty" not in flags:
        flags.insert(0, "mask_empty")
    warnings.extend(f for f in flags if f not in warnings)

    # Mask content outside the pasted patch is zero by construction.
    return Canon(
        rgb=np.ascontiguousarray(canvas_rgb),
        mask=np.ascontiguousarray(canvas_mask, dtype=np.float32),
        bbox_frac=stats.bbox_frac,
        area_frac=stats.area_frac,
        n_components=stats.n_components,
        truncated=stats.truncated,
        warnings=warnings,
        used_whole_image=used_whole_image,
        soft_area_frac=stats.soft_area_frac,
    )


def canonicalize_pair(
    loaded: Loaded, mask: np.ndarray
) -> tuple[Canon, MaskStats]:
    """Same as :func:`canonicalize` but also returns the raw :class:`MaskStats`."""
    canon = canonicalize(loaded, mask)
    return canon, mask_stats(np.asarray(mask, dtype=np.float32))


# ---------------------------------------------------------------------------
# v2.1 mask-only micro-alignment (pair level)
# ---------------------------------------------------------------------------

#: Maximum micro-alignment translation, as a fraction of each canvas side.
MICRO_ALIGN_FRAC: float = 0.01
#: Minimum soft-IoU gain required to accept a non-identity shift.
MICRO_ALIGN_MIN_GAIN: float = 1e-4


def _shift_arr(arr: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Translate *arr* by (dy, dx) px, filling the vacated border with *fill*."""
    out = np.empty_like(arr)
    out[...] = fill
    h, w = arr.shape[:2]
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx] = arr[ys0:ys1, xs0:xs1]
    return out


def shift_canon(canon: Canon, dy: int, dx: int) -> Canon:
    """Return *canon* translated by (dy, dx) canonical pixels (bg/0 fill)."""
    if not dy and not dx:
        return canon
    rgb = _shift_arr(np.asarray(canon.rgb), dy, dx, np.asarray(BG_RGB, dtype=np.uint8))
    mask = _shift_arr(np.asarray(canon.mask), dy, dx, np.float32(0.0))
    return Canon(
        rgb=np.ascontiguousarray(rgb),
        mask=np.ascontiguousarray(mask, dtype=np.float32),
        bbox_frac=canon.bbox_frac,
        area_frac=canon.area_frac,
        n_components=canon.n_components,
        truncated=canon.truncated,
        warnings=list(canon.warnings),
        used_whole_image=canon.used_whole_image,
        soft_area_frac=canon.soft_area_frac,
    )


def micro_align_shift(
    ref: Canon,
    sub: Canon,
    max_frac: float = MICRO_ALIGN_FRAC,
    min_gain: float = MICRO_ALIGN_MIN_GAIN,
) -> tuple[int, int]:
    """Best mask-only translation of *sub* onto *ref*, inside +/-*max_frac*.

    The canonical crop is derived from the segmentation bbox, so any residual
    contour jitter becomes a sub-pixel framing shift that LPIPS/SSIM punish out
    of proportion to the actual difference (acceptance ``pad_20`` /
    ``resize_0.5x`` / ``translate_10``).  This searches a tiny window
    (default +/-1 % of each side, i.e. +/-4 px on the 384x512 canvas) for the
    translation that maximises soft-mask IoU, and returns the identity when the
    gain is below *min_gain*, so identical inputs stay byte-identical and the
    "identity preferred" tie-break holds.  Masks only -- colour, LPIPS and CLIP
    never influence the shift, so it cannot be gamed with colour or content.
    """
    mr = np.asarray(ref.mask)
    ms = np.asarray(sub.mask)
    if mr.shape != ms.shape or mr.size == 0:
        return 0, 0
    h, w = mr.shape
    my = max(1, int(round(max_frac * h)))
    mx = max(1, int(round(max_frac * w)))
    best = (0, 0)
    best_iou = soft_iou(mr, ms)
    for dy in range(-my, my + 1):
        for dx in range(-mx, mx + 1):
            if dy == 0 and dx == 0:
                continue
            iou = soft_iou(mr, _shift_arr(ms, dy, dx, np.float32(0.0)))
            if iou > best_iou + min_gain:
                best_iou = iou
                best = (dy, dx)
    return best


# ---------------------------------------------------------------------------
# reference-mask override (SPEC: ``--ref-mask-dir``)
# ---------------------------------------------------------------------------


def load_reference_mask(mask_dir: str, stem: str, size: tuple[int, int]) -> np.ndarray | None:
    """Load a hand-fixed reference mask PNG (same stem) resized to *size* (w, h).

    Returns None when no PNG/JPEG with that stem exists.  Masks are read as
    greyscale and normalised to [0, 1].
    """
    import os  # noqa: PLC0415

    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".webp"):
        path = os.path.join(mask_dir, stem + ext)
        if os.path.exists(path):
            with Image.open(path) as im:
                arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
            if arr.shape != (size[1], size[0]):
                arr = np.asarray(
                    Image.fromarray(arr, mode="F").resize(size, Image.BILINEAR), dtype=np.float32
                )
            return np.clip(arr, 0.0, 1.0)
    return None


def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Soft IoU = sum(min) / sum(max) -- used by metrics and by masks QA."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.maximum(a, b).sum())
    if denom <= 0:
        return 0.0
    return float(np.minimum(a, b).sum()) / denom


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = binary(a)
    b = binary(b)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union
