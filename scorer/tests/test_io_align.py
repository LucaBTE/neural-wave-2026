"""Tests for io (decode / colour / limits) and align (canonicalisation).

Run from ``scorer/``::

    U2NET_HOME=$PWD/.models .venv/bin/python -m pytest tests/test_io_align.py -q

Optional real-data tests: point SCORER_TEST_REFS at a folder of <name>/out.jpg packshots and
SCORER_TEST_P3_PNG at a PNG with a Display P3 profile and EXIF orientation 6; otherwise they skip.
"""

from __future__ import annotations

import hashlib
import io as _io
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sartiq_score.align import (  # noqa: E402
    binary_iou,
    canonicalize,
    load_reference_mask,
    soft_iou,
)
from sartiq_score.io import load_image  # noqa: E402
from sartiq_score.segment import (  # noqa: E402
    binary,
    mask_stats,
    model_id,
    segment,
    segment_with_stats,
)
from sartiq_score.types import (  # noqa: E402
    BG_RGB,
    CANON_H,
    CANON_W,
    InvalidImage,
    InvalidReason,
    Loaded,
)

SCORER = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("SCORER_TEST_REFS", SCORER.parent / "_no_test_refs"))
P3_PNG = Path(os.environ.get("SCORER_TEST_P3_PNG", SCORER.parent / "_no_p3_sample.png"))

# Targets known to be white / cream garments on a white background.
WHITE_GARMENTS = [n for n in os.environ.get("SCORER_TEST_WHITE", "").split(",") if n] or ["white_1"]

HAVE_DATASET = DATASET.is_dir() and any(DATASET.iterdir())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _png_with_size(width: int, height: int, bit_depth: int = 8, color_type: int = 0) -> bytes:
    """Minimal PNG container carrying *width*/{height} in IHDR (header-only)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _synthetic(h: int, w: int, bbox_frac=(0.10, 0.08, 0.62, 0.80)):
    """An image with a soft-edged rounded rectangle 'garment' + its soft mask."""
    ss = 4
    mask_hi = Image.new("L", (w * ss, h * ss), 0)
    from PIL import ImageDraw

    x0, y0, bw, bh = bbox_frac
    box = [
        int(x0 * w * ss),
        int(y0 * h * ss),
        int((x0 + bw) * w * ss),
        int((y0 + bh) * h * ss),
    ]
    ImageDraw.Draw(mask_hi).rounded_rectangle(box, radius=int(0.06 * w * ss), fill=255)
    mask = np.asarray(mask_hi.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    rgb = np.full((h, w, 3), BG_RGB, dtype=np.uint8)
    rgb[..., 0] = 40
    rgb[..., 1] = 70
    rgb[..., 2] = 120
    body = mask > 0.5
    rgb[body] = np.array([180, 40, 60], dtype=np.uint8)
    return rgb, mask


def _translate_pad(rgb, mask, dx, dy, pad=None):
    """Place *rgb*/*mask* at ``(pad+dx, pad+dy)`` on a larger BG canvas."""
    pad = max(abs(dx), abs(dy)) + 1 if pad is None else pad
    h, w = rgb.shape[:2]
    H, W = h + 2 * pad, w + 2 * pad
    out_rgb = np.full((H, W, 3), BG_RGB, dtype=np.uint8)
    out_mask = np.zeros((H, W), dtype=np.float32)
    oy, ox = pad + dy, pad + dx
    sy0, sy1 = max(0, -oy), min(h, H - oy)
    sx0, sx1 = max(0, -ox), min(w, W - ox)
    assert sy1 > sy0 and sx1 > sx0, "content clipped - increase pad"
    out_rgb[oy + sy0 : oy + sy1, ox + sx0 : ox + sx1] = rgb[sy0:sy1, sx0:sx1]
    out_mask[oy + sy0 : oy + sy1, ox + sx0 : ox + sx1] = mask[sy0:sy1, sx0:sx1]
    return out_rgb, out_mask


def _scale(rgb, mask, factor):
    h, w = rgb.shape[:2]
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    s_rgb = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    s_mask = np.asarray(
        Image.fromarray(mask.astype(np.float32), mode="F").resize((nw, nh), Image.BILINEAR),
        dtype=np.float32,
    )
    return s_rgb, np.clip(s_mask, 0, 1)


def _loaded(rgb, path="synthetic") -> Loaded:
    return Loaded(
        rgb=rgb,
        alpha_present=False,
        orig_size=(rgb.shape[1], rgb.shape[0]),
        warnings=[],
        sha256="",
        path=path,
    )


# ---------------------------------------------------------------------------
# io: EXIF / colour / modes
# ---------------------------------------------------------------------------


def test_exif_rotation_real_png():
    """On-site reference PNGs carry orientation 6 -> displayed size is transposed."""
    p = P3_PNG
    if not p.exists():
        pytest.skip("P3 sample not present")
    with Image.open(p) as im:
        raw_size = im.size  # (w, h) as stored
    loaded = load_image(p)
    assert loaded.orig_size == (raw_size[1], raw_size[0]), "EXIF orientation 6 must be applied"
    assert loaded.rgb.shape[:2] == (loaded.orig_size[1], loaded.orig_size[0])
    assert loaded.rgb.dtype == np.uint8


def test_exif_rotation_synthetic(tmp_path):
    arr = np.zeros((600, 400, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]  # marker top-left
    path = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(arr).save(path, exif=exif)
    loaded = load_image(path)
    assert loaded.orig_size == (600, 400)
    assert loaded.rgb.shape[:2] == (400, 600)
    assert not np.array_equal(loaded.rgb[:50, :50], arr[:50, :50])


def test_p3_profile_is_converted_to_srgb(tmp_path):
    """Display P3 data must be converted, not passed through."""
    src_png = P3_PNG
    if src_png.exists():
        with Image.open(src_png) as im:
            icc = im.info.get("icc_profile")
    else:  # pragma: no cover
        pytest.skip("no P3 sample available")
    assert icc, "expected the reference PNG to embed a Display P3 profile"
    desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(_io.BytesIO(icc)))
    assert "P3" in desc

    arr = np.zeros((320, 300, 3), dtype=np.uint8)
    arr[:, :150] = [255, 128, 0]
    arr[:, 150:] = [10, 200, 90]
    path = tmp_path / "p3.png"
    Image.fromarray(arr).save(path, icc_profile=icc)

    loaded = load_image(path)
    assert loaded.warnings == []
    with Image.open(path) as im:
        raw = np.asarray(im.convert("RGB"))
    assert raw.dtype == np.uint8
    assert np.abs(loaded.rgb.astype(int) - raw.astype(int)).mean() > 0.5, "P3 -> sRGB not applied"
    # fully saturated primary stays saturated-ish, mid-grey stays neutral
    grey = np.full((256, 256, 3), 128, dtype=np.uint8)
    gpath = tmp_path / "p3grey.png"
    Image.fromarray(grey).save(gpath, icc_profile=icc)
    g = load_image(gpath).rgb.astype(int)
    assert abs(g[..., 0].mean() - g[..., 2].mean()) < 6


def test_broken_icc_warns_but_does_not_crash(tmp_path):
    arr = np.full((256, 300, 3), 77, dtype=np.uint8)
    path = tmp_path / "bad_icc.png"
    Image.fromarray(arr).save(path, icc_profile=b"not-a-real-profile")
    loaded = load_image(path)
    assert any(w.startswith("icc_conversion_failed") for w in loaded.warnings)
    assert np.all(loaded.rgb == 77)


def test_rgba_composite_and_hidden_rgb_ignored(tmp_path):
    h, w = 300, 300
    visible = np.zeros((h, w, 4), dtype=np.uint8)
    visible[:, :, :3] = [15, 160, 60]
    visible[:, :, 3] = 255
    visible[:, :150, 3] = 0  # left half transparent

    hidden_a = visible.copy()
    hidden_a[:, :150, :3] = 255  # bright hidden RGB
    hidden_b = visible.copy()
    hidden_b[:, :150, :3] = 0  # dark hidden RGB

    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(hidden_a, "RGBA").save(pa)
    Image.fromarray(hidden_b, "RGBA").save(pb)

    la, lb = load_image(pa), load_image(pb)
    assert la.alpha_present and lb.alpha_present
    assert np.array_equal(la.rgb, lb.rgb), "RGB under alpha=0 must not influence the result"
    assert np.all(la.rgb[:, :20] == np.asarray(BG_RGB, dtype=np.uint8))
    assert np.all(la.rgb[150:, 200:] == np.array([15, 160, 60], dtype=np.uint8))


def test_semi_transparent_alpha_is_composited(tmp_path):
    arr = np.zeros((256, 256, 4), dtype=np.uint8)
    arr[..., :3] = 0
    arr[..., 3] = 128  # 50 % black over the compositing background
    p = tmp_path / "half.png"
    Image.fromarray(arr, "RGBA").save(p)
    got = load_image(p).rgb.astype(int)
    expected = int(round(0 * 0.5 + BG_RGB[0] * 0.5))
    assert np.abs(got - expected).max() <= 1


def test_palette_mode_with_transparency(tmp_path):
    pal = Image.new("P", (300, 320))
    pal.putpalette([10, 20, 30, 200, 30, 40] + [0, 0, 0] * 254)
    pal.paste(1, (0, 0, 320, 160))
    pal.info["transparency"] = 1
    p = tmp_path / "pal.png"
    pal.save(p, transparency=1)
    loaded = load_image(p)
    assert loaded.alpha_present
    top = loaded.rgb[40, 40].astype(int)
    assert np.abs(top - np.asarray(BG_RGB)).max() <= 2, "transparent palette index -> BG"
    assert np.all(loaded.rgb[250, 40] == np.array([10, 20, 30], dtype=np.uint8))


def test_la_mode(tmp_path):
    arr = np.zeros((256, 300, 2), dtype=np.uint8)
    arr[..., 0] = 90
    arr[..., 1] = 255
    p = tmp_path / "la.png"
    Image.fromarray(arr, "LA").save(p)
    loaded = load_image(p)
    assert loaded.alpha_present and loaded.rgb.shape == (256, 300, 3)
    assert np.all(loaded.rgb == 90)


def test_cmyk_jpeg(tmp_path):
    arr = np.zeros((300, 300, 4), dtype=np.uint8)
    arr[..., 0] = 255  # cyan ink
    p = tmp_path / "cmyk.jpg"
    Image.fromarray(arr, "CMYK").save(p)
    loaded = load_image(p)
    assert loaded.rgb.shape == (300, 300, 3)
    px = loaded.rgb[150, 150].astype(int)
    assert px[1] > 200 and px[2] > 200 and px[0] < 60, f"cyan ink -> G/B high, R low, got {px}"


def test_16bit_png(tmp_path):
    ramp = np.zeros((256, 600), dtype=np.uint16)
    ramp[:] = np.linspace(0, 65535, 600, dtype=np.uint16)[None, :]
    p = tmp_path / "u16.png"
    Image.fromarray(ramp, mode="I;16").save(p)
    loaded = load_image(p)
    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (256, 600, 3)
    row = loaded.rgb[128, :, 0].astype(int)
    assert row[0] == 0 and row[-1] == 255
    assert np.all(np.diff(row) >= 0)


def test_float_tiff_like_mode(tmp_path):
    arr = (np.linspace(0, 1, 256 * 300, dtype=np.float32) * 1.0).reshape(256, 300)
    p = tmp_path / "f.tif"
    Image.fromarray(arr, mode="F").save(p)
    loaded = load_image(p)
    assert loaded.rgb.shape == (256, 300, 3) and loaded.rgb.dtype == np.uint8


# ---------------------------------------------------------------------------
# io: limits and robustness
# ---------------------------------------------------------------------------


def test_1x1_is_invalid(tmp_path):
    p = tmp_path / "one.png"
    Image.new("RGB", (1, 1), (0, 0, 0)).save(p)
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.TOO_SMALL


def test_min_side_limit(tmp_path):
    p = tmp_path / "small.png"
    Image.new("RGB", (255, 400), (0, 0, 0)).save(p)
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.TOO_SMALL


def test_max_side_limit(tmp_path):
    p = tmp_path / "wide.png"
    Image.new("L", (13000, 256), 0).save(p)
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.TOO_LARGE


def test_aspect_limit(tmp_path):
    p = tmp_path / "long.png"
    Image.new("RGB", (1000, 256), (0, 0, 0)).save(p)
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.ASPECT


def test_decompression_bomb_header_is_invalid(tmp_path):
    p = tmp_path / "bomb.png"
    p.write_bytes(_png_with_size(20000, 20000))
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.BOMB


def test_decompression_bomb_real_file_is_invalid(tmp_path):
    p = tmp_path / "bomb2.png"
    Image.new("1", (20000, 20000), 0).save(p)  # 1 bpp -> cheap to build, huge when decoded
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.BOMB


def test_corrupt_file_is_invalid(tmp_path):
    p = tmp_path / "nope.png"
    p.write_bytes(b"this is definitely not an image")
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.UNREADABLE


def test_missing_file_is_invalid(tmp_path):
    with pytest.raises(InvalidImage) as e:
        load_image(tmp_path / "absent.jpg")
    assert e.value.reason == InvalidReason.UNREADABLE


def test_empty_file_is_invalid(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.UNREADABLE


def test_sha256_matches_file_bytes(tmp_path):
    arr = np.random.default_rng(0).integers(0, 255, (300, 320, 3), dtype=np.uint8)
    p = tmp_path / "s.png"
    Image.fromarray(arr).save(p)
    loaded = load_image(p)
    assert loaded.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_webp_input(tmp_path):
    arr = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    p = tmp_path / "x.webp"
    Image.fromarray(arr).save(p, lossless=True)
    loaded = load_image(p)
    assert loaded.orig_size == (300, 300) and np.array_equal(loaded.rgb, arr)


# ---------------------------------------------------------------------------
# align: canonical geometry
# ---------------------------------------------------------------------------


def test_canon_shape_and_dtype():
    rgb, mask = _synthetic(1000, 800)
    canon = canonicalize(_loaded(rgb), mask)
    assert canon.rgb.shape == (CANON_H, CANON_W, 3) and canon.rgb.dtype == np.uint8
    assert canon.mask.shape == (CANON_H, CANON_W) and canon.mask.dtype == np.float32
    assert 0.0 <= float(canon.mask.min()) and float(canon.mask.max()) <= 1.0


def test_canon_occupies_90_percent_of_binding_dimension():
    for bbox in [(0.10, 0.08, 0.62, 0.80), (0.05, 0.30, 0.80, 0.30), (0.35, 0.05, 0.20, 0.90)]:
        rgb, mask = _synthetic(1000, 800, bbox)
        canon = canonicalize(_loaded(rgb), mask)
        b = binary(canon.mask)
        rows = np.flatnonzero(b.any(axis=1))
        cols = np.flatnonzero(b.any(axis=0))
        frac_h = (rows[-1] - rows[0] + 1) / CANON_H
        frac_w = (cols[-1] - cols[0] + 1) / CANON_W
        assert max(frac_h, frac_w) == pytest.approx(0.90, abs=0.02), (bbox, frac_h, frac_w)


def test_canon_is_centred():
    rgb, mask = _synthetic(1000, 800, (0.02, 0.02, 0.55, 0.90))
    canon = canonicalize(_loaded(rgb), mask)
    b = binary(canon.mask)
    rows = np.flatnonzero(b.any(axis=1))
    cols = np.flatnonzero(b.any(axis=0))
    assert abs((rows[0] + rows[-1]) / 2 - (CANON_H - 1) / 2) <= 2
    assert abs((cols[0] + cols[-1]) / 2 - (CANON_W - 1) / 2) <= 2


def test_canon_background_is_bg_rgb():
    rgb, mask = _synthetic(700, 700, (0.3, 0.3, 0.3, 0.3))
    canon = canonicalize(_loaded(rgb, path="solid"), mask)
    assert np.all(canon.rgb[0, 0] == np.asarray(BG_RGB, dtype=np.uint8))
    assert canon.mask[0, 0] == 0.0


def test_empty_mask_uses_whole_image():
    rgb = np.full((500, 600, 3), 30, dtype=np.uint8)
    canon = canonicalize(_loaded(rgb), np.zeros((500, 600), dtype=np.float32))
    assert canon.used_whole_image
    assert "mask_empty" in canon.warnings
    assert canon.area_frac == 0.0
    assert canon.bbox_frac == (0.0, 0.0, 1.0, 1.0)
    assert canon.truncated is False
    assert canon.rgb.shape == (CANON_H, CANON_W, 3)


def test_blank_white_image_is_not_empty_mask():
    """A featureless white image must not silently become a 'no garment' case."""
    if not HAVE_DATASET:
        pytest.skip("dataset not present")
    rgb = np.full((600, 600, 3), 255, dtype=np.uint8)
    m = segment(rgb)
    st = mask_stats(m)
    assert 0.0 <= st.area_frac <= 1.0
    assert isinstance(st.flags, list)


def test_border_touching_mask_is_truncated():
    mask = np.zeros((400, 400), dtype=np.float32)
    mask[:380, 100:300] = 1.0  # touches the top border over 200/400 = 50 % of that side
    st = mask_stats(mask)
    assert st.truncated and "truncated" in st.flags


def test_side_touch_below_threshold_is_not_truncated():
    mask = np.zeros((400, 400), dtype=np.float32)
    mask[100:300, 100:300] = 1.0
    mask[0, 40:70] = 1.0  # 30/400 = 7.5 % of the top side -> below 10 %
    st = mask_stats(mask)
    assert not st.truncated


def test_components_are_counted_and_kept():
    mask = np.zeros((500, 500), dtype=np.float32)
    mask[100:200, 100:200] = 1.0  # 10000 px = 4 % of image
    mask[300:330, 300:330] = 1.0  # 900 px = 0.36 % -> below the 0.5 % bar
    mask[350:390, 350:390] = 1.0  # 1600 px = 0.64 % -> counted
    st = mask_stats(mask)
    assert st.n_components == 2
    assert "multi_component" in st.flags
    canon = canonicalize(_loaded(np.full((500, 500, 3), 200, np.uint8)), mask)
    assert canon.n_components == 2
    assert float(canon.mask.sum()) > 0


def test_mask_suspect_flag():
    mask = np.ones((300, 300), dtype=np.float32)
    st = mask_stats(mask)
    assert "mask_suspect" in st.flags and st.area_frac == 1.0


# ---------------------------------------------------------------------------
# align: invariance of canonicalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dx,dy", [(137, 91), (-64, 210), (7, -5)])
def test_canonicalize_invariant_to_translate_and_pad(dx, dy):
    rgb, mask = _synthetic(900, 700)
    c0 = canonicalize(_loaded(rgb), mask)
    rgb2, mask2 = _translate_pad(rgb, mask, dx, dy)
    c1 = canonicalize(_loaded(rgb2), mask2)
    assert binary_iou(c0.mask, c1.mask) >= 0.97
    assert soft_iou(c0.mask, c1.mask) >= 0.97


def test_canonicalize_invariant_to_margins():
    rgb, mask = _synthetic(900, 700)
    c0 = canonicalize(_loaded(rgb), mask)
    rgb2, mask2 = _translate_pad(rgb, mask, 0, 0, pad=170)  # +~20 % margins
    c1 = canonicalize(_loaded(rgb2), mask2)
    assert binary_iou(c0.mask, c1.mask) >= 0.97


@pytest.mark.parametrize("factor", [0.5, 0.75, 1.6])
def test_canonicalize_invariant_to_aspect_preserving_resize(factor):
    rgb, mask = _synthetic(900, 700)
    c0 = canonicalize(_loaded(rgb), mask)
    rgb2, mask2 = _scale(rgb, mask, factor)
    c1 = canonicalize(_loaded(rgb2), mask2)
    assert binary_iou(c0.mask, c1.mask) >= 0.97
    assert soft_iou(c0.mask, c1.mask) >= 0.97


def test_canonicalize_fill_and_no_anisotropic_scaling():
    """A wide garment keeps its aspect ratio on the canvas."""
    rgb, mask = _synthetic(1000, 1000, (0.05, 0.35, 0.90, 0.25))
    canon = canonicalize(_loaded(rgb), mask)
    b = binary(canon.mask)
    rows = np.flatnonzero(b.any(axis=1))
    cols = np.flatnonzero(b.any(axis=0))
    got = (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)
    want = 0.90 * 1000 / (0.25 * 1000)
    assert got == pytest.approx(want, rel=0.03)


# ---------------------------------------------------------------------------
# segment: real images
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_DATASET, reason="dataset not present")
@pytest.mark.parametrize("name", WHITE_GARMENTS)
def test_white_garment_on_white_background_has_sane_mask(name):
    p = DATASET / name / "out.jpg"
    if not p.exists():
        pytest.skip(f"{name} missing")
    import time

    loaded = load_image(p)
    t0 = time.perf_counter()
    mask, st = segment_with_stats(loaded.rgb)
    elapsed = time.perf_counter() - t0
    assert mask.shape == loaded.rgb.shape[:2]
    assert mask.dtype == np.float32 and 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    assert 0.02 < st.area_frac < 0.95, f"{name}: area_frac={st.area_frac}"
    assert st.n_components >= 1
    assert "mask_empty" not in st.flags
    canon = canonicalize(loaded, mask)
    assert 0.02 < float(binary(canon.mask).mean()) < 0.95, f"{name}: canonical coverage off"
    assert elapsed < 20.0


@pytest.mark.skipif(not HAVE_DATASET, reason="dataset not present")
def test_segment_is_deterministic():
    p = DATASET / WHITE_GARMENTS[0] / "out.jpg"
    if not p.exists():
        pytest.skip("target missing")
    small = np.asarray(Image.fromarray(load_image(p).rgb).resize((640, 640), Image.BILINEAR))
    a = segment(small)
    b = segment(small)
    assert np.array_equal(a, b)
    assert model_id() == "rembg:u2net"


@pytest.mark.skipif(not HAVE_DATASET, reason="dataset not present")
def test_reference_mask_override(tmp_path):
    p = DATASET / WHITE_GARMENTS[0] / "out.jpg"
    if not p.exists():
        pytest.skip("target missing")
    loaded = load_image(p)
    assert load_reference_mask(str(tmp_path), p.stem, loaded.orig_size) is None
    hand = np.zeros((200, 150), dtype=np.uint8)
    hand[50:150, 40:110] = 255
    hand_path = tmp_path / (p.stem + ".png")
    Image.fromarray(hand).save(hand_path)
    got = load_reference_mask(str(tmp_path), p.stem, loaded.orig_size)
    assert got is not None and got.shape == (loaded.orig_size[1], loaded.orig_size[0])
    assert got.dtype == np.float32 and float(got.max()) == 1.0


def test_segment_shape_contract():
    rgb, _ = _synthetic(320, 240)
    m = segment(rgb)
    assert m.shape == (320, 240) and m.dtype == np.float32


def test_segment_rejects_bad_shape():
    with pytest.raises(ValueError):
        segment(np.zeros((10, 10), dtype=np.uint8))
