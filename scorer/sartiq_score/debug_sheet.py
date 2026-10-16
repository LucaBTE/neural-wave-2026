#!/usr/bin/env python3
"""Per-pair debug contact sheet.

Layout, left to right::

    ref | sub | canon ref | canon sub | masks | per-cell ΔE00 heat map

The score breakdown (score, status, sub-scores, raw metrics, flags) is printed
to stdout and returned alongside the written image path.

The ΔE00 panel recomputes the 4x4 cell distances from the two :class:`Canon`
objects so the sheet stays meaningful even if the aggregate's internal grid
changes; it mirrors the SPEC's ``colour_spatial`` definition (cell supported by
the reference's *own* mask, ΔE00 between masked mean L*a*b*).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cli import format_report_text, report_to_dict
from .types import BG_RGB, CANON_H, CANON_W, Canon

PANEL_W, PANEL_H = CANON_W, CANON_H
LABEL_H = 26
GAP = 6

#: ΔE00 half-way anchors for the heat map (value -> RGB).
_HEAT_ANCHORS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (46, 160, 67)),
    (2.0, (140, 200, 60)),
    (5.0, (255, 214, 0)),
    (10.0, (255, 140, 0)),
    (20.0, (205, 40, 40)),
    (40.0, (110, 0, 0)),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _font(size: int = 16):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # pragma: no cover - old Pillow
        return ImageFont.load_default()


def _letterbox(rgb: np.ndarray, w: int = PANEL_W, h: int = PANEL_H) -> Image.Image:
    """Aspect-preserving fit into w x h on the canonical background."""
    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), BG_RGB)
    canvas.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _lab(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab  # noqa: PLC0415

    return rgb2lab(rgb.astype(np.float64) / 255.0)


def _heat_colour(de: float) -> tuple[int, int, int]:
    anchors = _HEAT_ANCHORS
    if de <= anchors[0][0]:
        return anchors[0][1]
    for (v0, c0), (v1, c1) in zip(anchors, anchors[1:]):
        if de <= v1:
            t = 0.0 if v1 == v0 else (de - v0) / (v1 - v0)
            return tuple(int(round(c0[i] + t * (c1[i] - c0[i]))) for i in range(3))
    return anchors[-1][1]


# ---------------------------------------------------------------------------
# canon utilities
# ---------------------------------------------------------------------------


def _mean_lab_in_region(lab: np.ndarray, mask: np.ndarray, sl: tuple[slice, slice]):
    m = mask[sl]
    if m.size == 0:
        return None
    solid = m > 0.5
    if not solid.any():
        return None
    region = lab[sl]
    weights = m[solid]
    return np.average(region[solid], axis=0, weights=weights)


def per_cell_delta_e(
    canon_ref: Canon, canon_sub: Canon, n: int = 4, support: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(delta_e, support_code)`` grids of shape ``(n, n)``.

    ``support_code``: 0 = both supported, 1 = reference only, 2 = submission only,
    3 = no reference support (cell ignored).
    """
    from skimage.color import deltaE_ciede2000  # noqa: PLC0415

    lab_r, lab_s = _lab(canon_ref.rgb), _lab(canon_sub.rgb)
    H, W = canon_ref.rgb.shape[:2]
    ch, cw = H // n, W // n
    de = np.zeros((n, n), dtype=np.float64)
    code = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            sl = (slice(i * ch, (i + 1) * ch), slice(j * cw, (j + 1) * cw))
            cell = canon_ref.mask[sl]
            ref_fill = float((cell > 0.5).mean()) if cell.size else 0.0
            sub_cell = canon_sub.mask[sl]
            sub_fill = float((sub_cell > 0.5).mean()) if sub_cell.size else 0.0
            if ref_fill < support:
                code[i, j] = 3
                de[i, j] = np.nan
                continue
            if sub_fill < support:
                code[i, j] = 1
                de[i, j] = 40.0  # missing support: maximum documented penalty
                continue
            mr = _mean_lab_in_region(lab_r, canon_ref.mask, sl)
            ms = _mean_lab_in_region(lab_s, canon_sub.mask, sl)
            if mr is None or ms is None:
                code[i, j] = 3
                de[i, j] = np.nan
                continue
            de[i, j] = float(
                deltaE_ciede2000(
                    mr.reshape(1, 1, 3), ms.reshape(1, 1, 3), channel_axis=2
                )[0, 0]
            )
            # reverse direction: submission support with no reference support is
            # a *missing in reference* case and is reported separately (code 2).
            if cell.size and float((cell > 0.5).mean()) < support and sub_fill >= support:
                code[i, j] = 2
    return de, code


def _mask_panel(canon_ref: Canon, canon_sub: Canon) -> Image.Image:
    img = np.asarray(canon_ref.rgb, dtype=np.float64) * 0.35
    img[..., 0] += 210.0 * canon_ref.mask
    img[..., 1] += 210.0 * canon_sub.mask
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _heat_panel(de: np.ndarray, code: np.ndarray) -> Image.Image:
    n = de.shape[0]
    H, W = PANEL_H, PANEL_W
    ch, cw = H // n, W // n
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if code[i, j] == 3:
                colour = (30, 30, 30)  # no reference support
            elif code[i, j] == 2:
                colour = (0, 80, 160)  # submission-only cell
            elif code[i, j] == 1 or not np.isfinite(de[i, j]):
                colour = (120, 0, 0) if code[i, j] == 1 else (30, 30, 30)
            else:
                colour = _heat_colour(float(de[i, j]))
            y0, y1 = i * ch, (i + 1) * ch if i < n - 1 else H
            x0, x1 = j * cw, (j + 1) * cw if j < n - 1 else W
            canvas[y0:y1, x0:x1] = colour
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for i in range(1, n):
        draw.line([(i * cw, 0), (i * cw, H)], fill=(255, 255, 255), width=1)
        draw.line([(0, i * ch), (W, i * ch)], fill=(255, 255, 255), width=1)
    font = _font(15)
    for i in range(n):
        for j in range(n):
            label = "n/a" if code[i, j] == 3 else ("-" if code[i, j] == 1 else f"{de[i, j]:.0f}")
            draw.text((j * cw + 6, i * ch + 6), label, fill=(255, 255, 255), font=font)
    return img


def _titled(panel: Image.Image, title: str) -> Image.Image:
    out = Image.new("RGB", (panel.width, panel.height + LABEL_H), (24, 24, 24))
    out.paste(panel, (0, LABEL_H))
    ImageDraw.Draw(out).text((6, 5), title, fill=(240, 240, 240), font=_font(17))
    return out


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def canonical_pair(ref_path, sub_path):
    """Load, segment and canonicalise both members of a pair."""
    from .align import canonicalize  # noqa: PLC0415
    from .io import load_image  # noqa: PLC0415
    from .segment import segment  # noqa: PLC0415

    ref_loaded = load_image(ref_path)
    sub_loaded = load_image(sub_path)
    ref_mask = segment(ref_loaded.rgb)
    sub_mask = segment(sub_loaded.rgb)
    return (
        ref_loaded,
        sub_loaded,
        canonicalize(ref_loaded, ref_mask),
        canonicalize(sub_loaded, sub_mask),
    )


def render_debug_sheet(
    ref,
    sub,
    out,
    view: str | None = None,
    report: Any = None,
    canon_ref: Canon | None = None,
    canon_sub: Canon | None = None,
    panels: int = 6,
) -> Path:
    """Write the contact sheet for one pair and return its path.

    ``report`` (a :class:`ScoreReport`) is computed when not supplied.  When
    ``canon_ref``/``canon_sub`` are supplied the pair is not canonicalised again.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if canon_ref is None or canon_sub is None:
        ref_loaded, sub_loaded, canon_ref, canon_sub = canonical_pair(ref, sub)
    else:
        from .io import load_image  # noqa: PLC0415

        ref_loaded = load_image(ref)
        sub_loaded = load_image(sub)

    if report is None:
        from .cli import build_scorer, score_with  # noqa: PLC0415

        report = score_with(build_scorer(), ref, sub, view)

    de, code = per_cell_delta_e(canon_ref, canon_sub)

    title = out_path.stem
    panel_list = [_titled(_letterbox(ref_loaded.rgb), f"ref  {ref_loaded.orig_size[0]}x{ref_loaded.orig_size[1]}")]
    if panels >= 6:
        panel_list.append(_titled(_letterbox(sub_loaded.rgb), f"sub  {sub_loaded.orig_size[0]}x{sub_loaded.orig_size[1]}"))
        panel_list.append(_titled(Image.fromarray(canon_ref.rgb), "canon ref  512x384"))
        panel_list.append(_titled(Image.fromarray(canon_sub.rgb), "canon sub  512x384"))
        panel_list.append(_titled(_mask_panel(canon_ref, canon_sub), "masks R=ref G=sub"))
        panel_list.append(_titled(_heat_panel(de, code), "cell dE00 (4x4)"))
    else:
        panel_list.append(_titled(Image.fromarray(canon_ref.rgb), "canon ref"))
        panel_list.append(_titled(Image.fromarray(canon_sub.rgb), "canon sub"))

    total_w = sum(p.width for p in panel_list) + GAP * (len(panel_list) - 1)
    total_h = max(p.height for p in panel_list) + LABEL_H
    sheet = Image.new("RGB", (total_w, total_h), (16, 16, 16))
    x = 0
    for panel in panel_list:
        sheet.paste(panel, (x, LABEL_H))
        x += panel.width + GAP
    ImageDraw.Draw(sheet).text(
        (6, 5), f"{title}  [{report_to_dict(report).get('status', '?')}]", fill=(255, 235, 120), font=_font(18)
    )
    sheet.save(out_path, quality=92)

    print(format_report_text(report, str(ref), str(sub)))
    print(f"  sheet   : {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - convenience
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Render a debug contact sheet for one pair.")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default=None)
    args = ap.parse_args(argv)
    render_debug_sheet(args.ref, args.sub, args.out, args.view)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
