"""Frozen data contract for the Sartiq reference-similarity scorer (v2).

Field names in ``Loaded``, ``Canon`` and ``ScoreReport`` are FROZEN by SPEC.md --
every other module imports them.  Do not rename or reorder.

Only additive, defaulted fields may be introduced after the freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants (frozen by SPEC.md "Data contract" / "Preprocessing")
# ---------------------------------------------------------------------------

#: Background used for alpha compositing *and* for the canonical canvas.
#: v2.1: pure white.  The reference packshots are photographed on a white
#: sweep (median background ~254/255), so compositing a transparent-bg
#: submission on a grey 245 canvas used to cost it 2-4.5 similarity points
#: purely for presentation (acceptance ``rgba_transparent`` /
#: ``bg_255``).  White removes that asymmetry and stays symmetric for the
#: canonical canvas fill (both sides of every pair use the same fill).
BG_RGB: tuple[int, int, int] = (255, 255, 255)

#: Canonical canvas is H=512, W=384 (3:4).
CANON_H: int = 512
CANON_W: int = 384
CANON_SHAPE: tuple[int, int] = (CANON_H, CANON_W)

#: Garment occupies 90 % of the binding dimension on the canonical canvas.
CANON_FILL: float = 0.90

#: Mask bbox is expanded by 2 % before cropping.
CANON_MARGIN: float = 0.02

#: Only the thresholded mask drives bbox / components / area.
MASK_THRESHOLD: float = 0.5

# Input limits (violations -> status "invalid").
MIN_SIDE: int = 256
MAX_SIDE: int = 12000
MAX_ASPECT: float = 3.0
MAX_IMAGE_PIXELS: float = 150e6
MAX_FILE_BYTES: int = 100 * 1024 * 1024   # v2.2: refuse to read bigger files
WORK_MAX_PIXELS: int = 50_000_000         # v2.2: decoded images above this are downscaled first

# Flag thresholds (SPEC "Preprocessing").
MASK_EMPTY_AREA: float = 0.01       # area < 1 %  -> mask_empty
MASK_SUSPECT_AREA: float = 0.92     # area > 92 % -> mask_suspect
MIN_COMPONENT_AREA: float = 0.005   # components >= 0.5 % of image area
BORDER_TOUCH_FRAC: float = 0.10     # > 10 % of a side -> truncated


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidImage(Exception):
    """Raised by :func:`sartiq_score.io.load_image` for rejected inputs.

    ``reason`` is a short machine-readable token (see ``InvalidReason``).
    Callers (``api.score_pair``) must turn this into
    ``ScoreReport(score=0.0, status="invalid")``.
    """

    def __init__(self, reason: str, path: str | None = None, detail: str = "") -> None:
        self.reason = reason
        self.path = path
        self.detail = detail
        msg = reason if not detail else f"{reason}: {detail}"
        if path:
            msg = f"{msg} [{path}]"
        super().__init__(msg)


class InvalidReason:
    """Machine-readable rejection reasons."""

    UNREADABLE = "unreadable"                  # decode / open failed
    BOMB = "decompression_bomb"                # header bigger than MAX_IMAGE_PIXELS
    TOO_SMALL = "min_side_too_small"           # min side < 256
    TOO_LARGE = "max_side_too_large"           # max side > 12000
    ASPECT = "aspect_too_extreme"              # max/min > 3.0
    ANIMATED = "animated_or_multiframe"        # GIF/WebP/APNG with > 1 frame
    FILE_TOO_LARGE = "file_too_large"          # encoded bytes > MAX_FILE_BYTES
    SYMLINK = "symlink_not_allowed"            # submission path is a symlink


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass
class Loaded:
    """A decoded, orientation- and colour-corrected image.

    Attributes
    ----------
    rgb:
        ``uint8`` HxWx3, sRGB, EXIF orientation applied, alpha already
        composited on :data:`BG_RGB`.
    alpha_present:
        True when the source file carried an alpha channel or a P-mode
        transparency table (regardless of whether any pixel was transparent).
    orig_size:
        ``(width, height)`` of the RGB array *after* EXIF transpose.
    warnings:
        Non-fatal preprocessing notes (ICC fallback, odd modes, ...).
    sha256:
        Hex digest of the source file bytes.
    """

    rgb: np.ndarray
    alpha_present: bool
    orig_size: tuple[int, int]
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""

    # --- additive, defaulted, optional -------------------------------------
    path: str = ""

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass
class Canon:
    """Canonicalised pair member: mask bbox cropped, uniformly scaled, centred.

    Attributes
    ----------
    rgb:
        ``uint8`` 512x384x3 (H=512, W=384), background :data:`BG_RGB`.
    mask:
        ``float32`` 512x384 soft mask, transformed identically, 0 outside.
    bbox_frac:
        Mask bbox in the *original* image as normalised ``xywh`` (x, y, w, h).
    area_frac:
        Thresholded mask area / image area.
    n_components:
        Number of 8-connected components each >= 0.5 % of the image area.
    truncated:
        Mask touches an image border on >= 1 side over > 10 % of that side.
    warnings:
        Flags / notes (``mask_empty``, ``mask_suspect``, ``multi_component``).
    """

    rgb: np.ndarray
    mask: np.ndarray
    bbox_frac: tuple[float, float, float, float]
    area_frac: float
    n_components: int
    truncated: bool
    warnings: list[str] = field(default_factory=list)

    # --- additive, defaulted, optional -------------------------------------
    #: True when the canonical canvas used the whole image as bbox
    #: (empty mask fallback -> submission status "review").
    used_whole_image: bool = False
    #: Soft-mask statistics straight from segmentation (mean of soft mask).
    soft_area_frac: float = 0.0


@dataclass
class ScoreReport:
    """Final report for one (reference, submission) pair.

    ``score`` is 0-100 rounded to 2 decimals; ``status`` is one of
    ``"ok" | "review" | "invalid"``.  ``subscores`` holds the per-family
    similarities, ``raw`` the un-mapped metric values, ``flags`` the triggered
    review flags and ``versions`` the pinned package/model identities.
    """

    score: float
    status: str
    subscores: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)
    timing_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers shared by owners
# ---------------------------------------------------------------------------

VALID_STATUSES = ("ok", "review", "invalid")

#: Review flags derivable from a Canon (SPEC "Metrics" status rules).
DERIVED_FLAGS = ("mask_empty", "mask_suspect", "multi_component", "truncated")


def flags_from_canon(canon: Canon) -> list[str]:
    """Review flags implied by one canonicalised image."""
    flags: list[str] = []
    if canon.area_frac < MASK_EMPTY_AREA:
        flags.append("mask_empty")
    if canon.area_frac > MASK_SUSPECT_AREA:
        flags.append("mask_suspect")
    if canon.n_components > 1:
        flags.append("multi_component")
    if canon.truncated:
        flags.append("truncated")
    return flags
