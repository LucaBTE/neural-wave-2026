"""Garment segmentation.

Frozen policy (SPEC.md "Preprocessing"): ONE model for everyone -- ``rembg``
``u2net`` (ONNX CPU), session created once.  Feed the composited RGB.  Output a
soft mask in [0, 1]; no per-image model routing.

Post-processing is split deliberately:
  * the *soft* mask is what metrics see (SPEC keeps soft masks);
  * threshold ``> 0.5`` is used only for bbox / components / area
    (:func:`mask_stats`), holes are kept, secondary components are kept.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import numpy as np

from .types import (
    BORDER_TOUCH_FRAC,
    MASK_EMPTY_AREA,
    MASK_SUSPECT_AREA,
    MASK_THRESHOLD,
    MIN_COMPONENT_AREA,
)

_MODEL_NAME = "u2net"
_session = None
_session_lock = threading.Lock()

#: v2.1 scale-invariant segmentation (see :func:`segment`).
REFINE_MARGIN: float = 0.08          # crop = pass-1 bbox + 8 % of its size
REFINE_MIN_SIDE: int = 48            # never re-segment a sliver
REFINE_MAX_AREA_FRAC: float = 0.98   # skip when the crop is ~the whole frame


def _ensure_model_home() -> None:
    """Point rembg at the pinned weights cache if the caller has not."""
    if not os.environ.get("U2NET_HOME"):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["U2NET_HOME"] = os.environ.get("SCORER_MODEL_DIR", os.path.join(here, ".models"))


def get_session():
    """Create (once) and return the u2net ONNX session."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _ensure_model_home()
                from rembg import new_session  # noqa: PLC0415

                _session = new_session(_MODEL_NAME)
    return _session


def reset_session() -> None:
    """Drop the cached session (tests / low-memory hosts)."""
    global _session
    with _session_lock:
        _session = None


def model_id() -> str:
    return f"rembg:{_MODEL_NAME}"


def segment(img_rgb_u8: np.ndarray, *, refine: bool = True) -> np.ndarray:
    """Soft garment mask for an RGB uint8 HxWx3 image.

    Returns ``float32`` HxW in [0, 1], same size as the input.

    v2.1: two-pass (scale-invariant) segmentation.  u2net always sees a
    320x320 letterbox, so on a 3400 px packshot the mask contour is quantised
    to ~10 original pixels; the same garment placed 20 % further from the frame
    edge (padding / translation / 0.5x resize) therefore produced a slightly
    different bbox, and the canonical crop inherited a sub-pixel framing shift
    that LPIPS/SSIM punished (acceptance ``pad_20``/``translate_10``/
    ``resize_0.5x`` failures).  Pass 1 segments the full image; pass 2
    re-segments a crop of the pass-1 bbox plus an 8 % margin, where the garment
    occupies far more of the network's 320 px input, and the sharper contour is
    pasted back.  Deterministic, still one model for everyone.
    """
    arr = _as_rgb(img_rgb_u8)
    mask = _segment_raw(arr)
    if not refine:
        return mask
    return _refine_scale_invariant(arr, mask)


def _as_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_rgb_u8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"segment() expects HxWx3 uint8 RGB, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _segment_raw(arr: np.ndarray) -> np.ndarray:
    """One u2net forward pass (soft mask, input resolution)."""
    from PIL import Image  # noqa: PLC0415

    from rembg import remove  # noqa: PLC0415

    out = remove(
        Image.fromarray(np.ascontiguousarray(arr), mode="RGB"),
        session=get_session(),
        only_mask=True,
        post_process_mask=False,
        alpha_matting=False,
    )
    mask = np.asarray(out if not hasattr(out, "convert") else out.convert("L"), dtype=np.float32)
    if mask.ndim == 3:  # defensive: some backends return RGBA
        mask = mask[..., 0]
    mask = mask / 255.0
    if mask.shape != arr.shape[:2]:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.float32), mode="F").resize(
                (arr.shape[1], arr.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        )
    return np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)


def _refine_scale_invariant(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Re-segment the pass-1 bbox (+margin) and keep the sharper of the two.

    v2.1 uses the pass-2 contour *conservatively*: the result is
    ``min(dilate(pass1, 1 px), pass2)``, so the refinement may tighten the
    contour (which is where the scale invariance comes from) but may not add
    foreground further than 1 px beyond the pass-1 support.  Without that clamp,
    a higher-resolution second pass can legitimately find *more* foreground on
    some garments (a penumbra, a shadow, legs under a dress) and the mask then
    changes with the framing again -- measured on one jacket reference, plain pass-2
    dragged the canon-mask IoU of a +20 % padded copy down to 0.958 (score
    96.41) while the clamped version reaches 0.9955 (99.78).
    """
    from PIL import Image  # noqa: PLC0415
    from scipy import ndimage  # noqa: PLC0415

    b = mask > MASK_THRESHOLD
    if not b.any():
        return mask
    rows = np.flatnonzero(b.any(axis=1))
    cols = np.flatnonzero(b.any(axis=0))
    h, w = arr.shape[:2]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    my = int(round((y1 - y0) * REFINE_MARGIN))
    mx = int(round((x1 - x0) * REFINE_MARGIN))
    cy0, cy1 = max(0, y0 - my), min(h, y1 + my)
    cx0, cx1 = max(0, x0 - mx), min(w, x1 + mx)
    ch, cw = cy1 - cy0, cx1 - cx0
    if ch < REFINE_MIN_SIDE or cw < REFINE_MIN_SIDE:
        return mask
    # No resolution gain when the crop is already (almost) the whole frame.
    if ch * cw >= REFINE_MAX_AREA_FRAC * h * w:
        return mask
    crop = np.ascontiguousarray(arr[cy0:cy1, cx0:cx1])
    m2 = _segment_raw(crop)
    if m2.shape[:2] != (ch, cw):
        m2 = np.asarray(
            Image.fromarray(np.ascontiguousarray(m2, dtype=np.float32), mode="F").resize(
                (cw, ch), Image.BILINEAR
            ),
            dtype=np.float32,
        )
    allow = ndimage.grey_dilation(mask, size=(3, 3))[cy0:cy1, cx0:cx1]
    m2 = np.minimum(np.clip(m2, 0.0, 1.0), allow)
    out = np.zeros((h, w), dtype=np.float32)
    out[cy0:cy1, cx0:cx1] = m2
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# binary statistics (threshold-driven: bbox / components / area only)
# ---------------------------------------------------------------------------


@dataclass
class MaskStats:
    """Thresholded-mask statistics used for bbox, components and flags."""

    bbox: tuple[int, int, int, int]        # x0, y0, x1, y1 (x1/y1 exclusive)
    bbox_frac: tuple[float, float, float, float]  # normalised xywh
    area_frac: float
    n_components: int
    truncated: bool
    empty: bool
    flags: list[str] = field(default_factory=list)
    soft_area_frac: float = 0.0


def binary(mask: np.ndarray, threshold: float = MASK_THRESHOLD) -> np.ndarray:
    """Threshold a soft mask (``>`` threshold, per SPEC)."""
    arr = np.asarray(mask, dtype=np.float32)
    return arr > float(threshold)


def _count_components(binary_mask: np.ndarray) -> int:
    """8-connected components whose area is >= 0.5 % of the image area."""
    total = binary_mask.size
    if total == 0 or not binary_mask.any():
        return 0
    try:
        from scipy import ndimage  # noqa: PLC0415

        labels, n = ndimage.label(binary_mask, structure=np.ones((3, 3), dtype=np.int32))
        if n == 0:
            return 0
        counts = np.bincount(labels.ravel())
        min_area = MIN_COMPONENT_AREA * total
        return int(np.count_nonzero(counts[1:] >= min_area))
    except Exception:  # pragma: no cover - scipy always present in the venv
        return 1


def _is_truncated(binary_mask: np.ndarray) -> bool:
    h, w = binary_mask.shape
    if h == 0 or w == 0:
        return False
    sides = (
        (binary_mask[0, :], w),
        (binary_mask[-1, :], w),
        (binary_mask[:, 0], h),
        (binary_mask[:, -1], h),
    )
    return any(np.count_nonzero(side) > BORDER_TOUCH_FRAC * length for side, length in sides)


def mask_stats(mask: np.ndarray, soft: bool = True) -> MaskStats:
    """Bbox / area / component / truncation stats of a mask.

    ``soft=True`` (default) recomputes ``soft_area_frac``; pass the already
    thresholded array with ``soft=False`` when the soft mask is not available.
    """
    soft_mask = np.asarray(mask, dtype=np.float32)
    if soft_mask.ndim == 3:
        soft_mask = soft_mask[..., 0]
    binmask = binary(soft_mask)

    h, w = binmask.shape
    empty = not bool(binmask.any())
    if empty:
        bbox = (0, 0, w, h)
    else:
        cols = np.flatnonzero(binmask.any(axis=0))
        rows = np.flatnonzero(binmask.any(axis=1))
        bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)

    x0, y0, x1, y1 = bbox
    area_frac = float(binmask.sum()) / float(h * w) if h and w else 0.0
    n_components = _count_components(binmask)
    truncated = False if empty else _is_truncated(binmask)

    flags: list[str] = []
    if area_frac < MASK_EMPTY_AREA:
        flags.append("mask_empty")
    if area_frac > MASK_SUSPECT_AREA:
        flags.append("mask_suspect")
    if n_components > 1:
        flags.append("multi_component")
    if truncated:
        flags.append("truncated")

    return MaskStats(
        bbox=bbox,
        bbox_frac=(
            x0 / w if w else 0.0,
            y0 / h if h else 0.0,
            (x1 - x0) / w if w else 0.0,
            (y1 - y0) / h if h else 0.0,
        ),
        area_frac=area_frac,
        n_components=n_components,
        truncated=truncated,
        empty=empty,
        flags=flags,
        soft_area_frac=float(soft_mask.mean()) if soft else 0.0,
    )


def segment_with_stats(img_rgb_u8: np.ndarray) -> tuple[np.ndarray, MaskStats]:
    mask = segment(img_rgb_u8)
    return mask, mask_stats(mask)
