"""Image decoding / normalisation.

Contract: ``load_image(path) -> Loaded`` (see :mod:`sartiq_score.types`).

Rules frozen by SPEC.md "Preprocessing":
  * Pillow decode, ``ImageOps.exif_transpose``, ICC -> sRGB via ``ImageCms``
    (on failure: warning + treat as sRGB).
  * Modes P / LA / RGBA / CMYK / I / 16-bit handled; alpha composited on
    :data:`BG_RGB` and RGB under ``alpha == 0`` ignored (never used).
  * Reject with :class:`InvalidImage` when decode fails, min side < 256,
    max side > 12000, aspect > 3.0, or the header is a decompression bomb
    (``Image.MAX_IMAGE_PIXELS = 150e6``).
"""

from __future__ import annotations

import hashlib
import io as _io
import os
import warnings as _warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .types import (
    BG_RGB,
    MAX_ASPECT,
    MAX_IMAGE_PIXELS,
    MAX_FILE_BYTES,
    MAX_SIDE,
    MIN_SIDE,
    WORK_MAX_PIXELS,
    InvalidImage,
    InvalidReason,
    Loaded,
)

# Decompression-bomb ceiling: 150 MPix (SPEC).
Image.MAX_IMAGE_PIXELS = int(MAX_IMAGE_PIXELS)

try:  # optional: HEIC / HEIF support (never fatal)
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

_BG = np.asarray(BG_RGB, dtype=np.float32)

_UINT16_MODES = {"I;16", "I;16B", "I;16L", "I;16N"}
_PLAIN_MODES = {"RGB", "L", "1", "LAB", "YCbCr", "HSV"}
_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_bytes(path: str | os.PathLike) -> bytes:
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            raise InvalidImage(InvalidReason.FILE_TOO_LARGE, str(path), f"{size} bytes > {MAX_FILE_BYTES}")
        with open(path, "rb") as fh:
            return fh.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise InvalidImage(InvalidReason.UNREADABLE, str(path), str(exc)) from exc


def _open(data: bytes, path: str):
    try:
        with _warnings.catch_warnings():
            # A header larger than MAX_IMAGE_PIXELS is a rejection, not a warning.
            _warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(_io.BytesIO(data))
            # v2.2: enforce the limits on the HEADER, before paying for a full decode
            _check_limits(im.size, path)
            if getattr(im, "is_animated", False) and im.format in ("GIF", "WEBP", "PNG"):
                raise InvalidImage(InvalidReason.ANIMATED, path,
                                   f"{im.format} with {getattr(im, 'n_frames', '?')} frames")
            im.load()
        return im
    except InvalidImage:
        raise
    except Image.DecompressionBombError as exc:
        raise InvalidImage(InvalidReason.BOMB, path, str(exc)) from exc
    except Image.DecompressionBombWarning as exc:
        raise InvalidImage(InvalidReason.BOMB, path, str(exc)) from exc
    except Exception as exc:  # UnidentifiedImageError, OSError, ...
        raise InvalidImage(InvalidReason.UNREADABLE, path, f"{type(exc).__name__}: {exc}") from exc


def _check_limits(size: tuple[int, int], path: str) -> None:
    width, height = size
    if min(width, height) < MIN_SIDE:
        raise InvalidImage(InvalidReason.TOO_SMALL, path, f"{width}x{height} < {MIN_SIDE} px")
    if max(width, height) > MAX_SIDE:
        raise InvalidImage(InvalidReason.TOO_LARGE, path, f"{width}x{height} > {MAX_SIDE} px")
    if max(width, height) / float(min(width, height)) > MAX_ASPECT:
        raise InvalidImage(
            InvalidReason.ASPECT, path, f"aspect {max(width, height) / min(width, height):.2f}"
        )


def _scale_wide_mode(im: Image.Image) -> np.ndarray:
    """Monochrome-ify exotic modes (I, I;16, F) to float64 0..255."""
    arr = np.asarray(im).astype(np.float64)
    if im.mode == "F":
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
    elif im.mode in _UINT16_MODES:
        arr = arr / 257.0  # v2.2: 16-bit -> 8-bit by bit depth, never by image content
    elif arr.size and float(arr.max()) > 255.0:
        arr = arr / (65535.0 / 255.0)  # 32-bit "I": best effort
    return np.clip(arr, 0.0, 255.0)


def _icc_to_srgb(
    im: Image.Image, icc_bytes: bytes | None, warnings: list[str]
) -> Image.Image:
    """Apply the embedded ICC profile, converting to sRGB. Never raises."""
    if not icc_bytes:
        return im
    try:
        src = ImageCms.ImageCmsProfile(_io.BytesIO(icc_bytes))
        dst = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(im, src, dst, renderingIntent=0, outputMode="RGB")
    except Exception as exc:  # pragma: no cover - profile dependent
        warnings.append(f"icc_conversion_failed:{type(exc).__name__}")
        return im


def _composite(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.float32)[..., None] / 255.0
    out = rgb.astype(np.float32) * a + _BG * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def load_image(path: str | os.PathLike, *, raise_on_error: bool = True) -> Loaded:
    """Decode *path* into a :class:`Loaded` (sRGB uint8, EXIF applied, alpha composited).

    Raises :class:`InvalidImage` on limit violations or unreadable files.
    """
    path = str(path)
    data = _read_bytes(path)
    sha = hashlib.sha256(data).hexdigest()
    warnings: list[str] = []
    im = _open(data, path)

    icc_bytes = im.info.get("icc_profile")

    # --- EXIF orientation -------------------------------------------------
    try:
        im = ImageOps.exif_transpose(im)
    except Exception as exc:  # pragma: no cover - malformed EXIF
        warnings.append(f"exif_transpose_failed:{type(exc).__name__}")

    # --- hard limits (also checked on the header in _open) ----------------
    _check_limits(im.size, path)

    # --- v2.2 memory bound: downscale very large decodes before any float work
    if im.size[0] * im.size[1] > WORK_MAX_PIXELS:
        f = (WORK_MAX_PIXELS / float(im.size[0] * im.size[1])) ** 0.5
        new = (max(1, int(im.size[0] * f)), max(1, int(im.size[1] * f)))
        if im.mode not in ("RGB", "RGBA", "L", "LA", "CMYK"):
            im = im.convert("RGBA" if ("transparency" in im.info or "A" in im.getbands()) else "RGB")
        im = im.resize(new, Image.LANCZOS)
        warnings.append(f"downscaled_for_memory:{new[0]}x{new[1]}")

    mode = im.mode
    # v2.2: colour-key transparency (tRNS on RGB / L PNGs) -> real alpha
    if mode in ("RGB", "L") and "transparency" in im.info:
        im = im.convert("RGBA")
        mode = "RGBA"
    alpha_present = False
    alpha: np.ndarray | None = None

    if mode == "P":
        if "transparency" in im.info:
            im = im.convert("RGBA")
            mode = "RGBA"
            alpha_present = True
        else:
            im = im.convert("RGB")
            mode = "RGB"
    if mode in ("LA", "PA", "RGBa"):
        im = im.convert("RGBA")
        mode = "RGBA"
        alpha_present = True
    if mode == "RGBA":
        alpha_present = True
        alpha = np.asarray(im.getchannel("A"))
        im = im.convert("RGB")
        mode = "RGB"

    # --- per-mode pixel extraction ---------------------------------------
    if mode == "CMYK":
        icc_src = im
        icc_applied = _icc_to_srgb(icc_src, icc_bytes, warnings)
        if icc_applied is im:
            # no profile (or failed) -> plain CMYK->RGB conversion
            rgb = np.asarray(im.convert("RGB"))
        else:
            rgb = np.asarray(icc_applied.convert("RGB"))
    elif mode == "RGB":
        icc_src = im
        icc_applied = _icc_to_srgb(icc_src, icc_bytes, warnings)
        rgb = np.asarray(icc_applied.convert("RGB"))
    elif mode in _UINT16_MODES or mode == "I" or mode == "F":
        gray = _scale_wide_mode(im)
        if icc_bytes:
            # Re-run ICC through an 8-bit L image so ICC conversions stay uniform.
            l_img = Image.fromarray(gray.astype(np.uint8), mode="L")
            l_img = _icc_to_srgb(l_img, icc_bytes, warnings)
            gray = np.asarray(l_img.convert("L")).astype(np.float64)
        rgb = np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)
    else:
        # L, 1, LAB, YCbCr, HSV, ... : let Pillow do the colour conversion.
        icc_src = im.convert("RGB")
        icc_src = _icc_to_srgb(icc_src, icc_bytes, warnings)
        rgb = np.asarray(icc_src.convert("RGB"))

    rgb = np.ascontiguousarray(rgb[..., :3])

    if alpha is not None:
        if alpha.shape[:2] != rgb.shape[:2]:  # defensive
            alpha = np.asarray(
                Image.fromarray(alpha).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            )
        rgb = _composite(rgb, alpha)
    elif alpha_present:
        warnings.append("alpha_present_but_not_decoded")

    loaded = Loaded(
        rgb=rgb,
        alpha_present=alpha_present,
        orig_size=(int(rgb.shape[1]), int(rgb.shape[0])),
        warnings=warnings,
        sha256=sha,
        path=path,
    )
    return loaded


def load_image_or_none(path: str | os.PathLike) -> tuple[Loaded | None, str | None]:
    """Convenience wrapper: ``(Loaded, None)`` or ``(None, reason)``."""
    try:
        return load_image(path), None
    except InvalidImage as exc:
        return None, exc.reason


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
