"""Smoke test for the acceptance (sanity) suite entry points.

Keeps the plumbing honest — the transforms, the worker and the report writer —
without needing the full 97-reference run.  Real weights are required, so the
test is skipped when the model bundle cannot be built.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

SCORER = Path(__file__).resolve().parents[1]
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))


def _synthetic_packshot(path: Path, size=(768, 1024)) -> Path:
    """A flat-lay-ish rectangle garment on a light background."""
    w, h = size
    bg = (248, 248, 246)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [w * 0.22, h * 0.18, w * 0.78, h * 0.82], radius=int(w * 0.05),
        fill=(74, 96, 138),
    )
    draw.rectangle([w * 0.44, h * 0.06, w * 0.56, h * 0.24], fill=(74, 96, 138))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
    return path


@pytest.fixture()
def reference(tmp_path: Path) -> Path:
    return _synthetic_packshot(tmp_path / "ref" / "out.jpg")


def _models_available() -> bool:
    try:
        from sartiq_score.cli import build_scorer

        return build_scorer() is not None
    except Exception:
        return False


needs_models = pytest.mark.skipif(
    os.environ.get("SARTIQ_SKIP_MODEL_TESTS") == "1",
    reason="model-dependent smoke test disabled via SARTIQ_SKIP_MODEL_TESTS=1",
)


def test_transforms_are_deterministic_and_shaped(reference: Path):
    from sartiq_score import sanity

    base = np.asarray(Image.open(reference).convert("RGB"))
    mask = np.zeros(base.shape[:2], dtype=np.float32)
    mask[base.shape[0] // 4 : base.shape[0] * 3 // 4, base.shape[1] // 4 : base.shape[1] * 3 // 4] = 1.0
    rng = np.random.default_rng(7)

    replaced = sanity.t_remove_area(base, mask, 30)
    assert replaced.shape == base.shape
    removed = int((base.astype(int) - replaced.astype(int)).any(axis=2).sum())
    garment = int((mask > 0.5).sum())
    assert 0 < removed <= garment

    cropped = sanity.t_crop_away(base, mask, 30)
    assert cropped.shape[0] < base.shape[0]

    shifted = sanity.t_colour_shift(base, mask, 10)
    assert shifted.shape == base.shape and shifted.dtype == np.uint8

    rgba = sanity.t_rgba_transparent(base, mask, rng)
    assert rgba.shape[:2] == base.shape[:2] and rgba.shape[2] == 4

    bomb = sanity.t_bomb()
    assert bomb.startswith(b"\x89PNG")

    swap = sanity.t_collage_swap(base)
    assert swap.shape == base.shape


def test_select_refs_is_deterministic(tmp_path: Path):
    from sartiq_score import sanity

    for i in range(10):
        _synthetic_packshot(tmp_path / f"g{i}" / "out.jpg", size=(320, 420))
    first = sanity.select_refs(str(tmp_path / "*" / "out.jpg"), limit=4)
    second = sanity.select_refs(str(tmp_path / "*" / "out.jpg"), limit=4)
    assert first == second
    assert len(first) == 4


@needs_models
def test_sanity_smoke_end_to_end(reference: Path, tmp_path: Path):
    """One garment, invariance + ordering only: the suite must produce a report."""
    from sartiq_score import sanity

    out = tmp_path / "sanity"
    summary = sanity.run_suite(
        refs_glob=str(reference),
        out_dir=out,
        limit=1,
        seed=1,
        jobs=1,
        families=["invariance", "ordering"],
        wrong_per_garment=1,
        keep_variants=True,
    )
    written = sanity.write_report(summary)
    assert Path(written["report_path"]).is_file()
    assert Path(written["results_path"]).is_file()

    records = summary["records"]
    assert records, "no variants were scored"
    scores = {r["variant"]: r["score"] for r in records}
    assert scores["self"] >= 0.0
    # a submission identical to the reference must be the best score available
    assert scores["self"] >= max(scores.values()) - 1e-6
    # the JSONL must be parseable line by line
    for line in Path(written["results_path"]).read_text().splitlines():
        json.loads(line)
    assert summary["n_checks"] > 0


def test_cli_pair_and_wrapper_parse():
    from sartiq_score import cli

    parser = cli.build_parser()
    args = parser.parse_args(["pair", "--ref", "r.jpg", "--sub", "s.jpg", "--json"])
    assert args.ref == "r.jpg" and args.sub == "s.jpg" and args.json is True

    sys.path.insert(0, str(SCORER))
    import sartiq_scorer  # noqa: PLC0415

    wrapper = sartiq_scorer.build_parser()
    ns = wrapper.parse_args(["--reference", "r.jpg", "--generated", "s.jpg"])
    assert ns.reference == "r.jpg" and ns.generated == "s.jpg"


def test_report_to_dict_handles_missing_fields():
    from sartiq_score.cli import format_report_text, report_to_dict

    class Partial:
        score = 77.5
        status = "review"
        subscores = {"colour": 0.8}
        raw = {"lpips_d": 0.1}
        flags = ["mask_suspect"]

    d = report_to_dict(Partial())
    assert d["score"] == 77.5 and d["status"] == "review"
    text = format_report_text(Partial(), "r.jpg", "s.jpg")
    assert "77.5" in text and "mask_suspect" in text


# ---------------------------------------------------------------------------
# v2.1: the text-overlay transform must actually draw big text on the garment
# ---------------------------------------------------------------------------


def test_text_overlay_is_large_and_on_the_garment():
    from sartiq_score import sanity

    h, w = 1200, 900
    base = np.full((h, w, 3), 250, np.uint8)
    base[180:1020, 120:780] = np.array([205, 200, 190], np.uint8)  # light garment
    mask = np.zeros((h, w), np.float32)
    mask[180:1020, 120:780] = 1.0

    out = sanity.t_text_overlay(base, mask)
    assert out.shape == base.shape and out.dtype == np.uint8
    changed = (out.astype(int) - base.astype(int)).any(axis=2)
    frac = float(changed.mean())
    # a real, large overlay (the pre-v2.1 bug produced < 0.5 % of the frame)
    assert frac > 0.03, frac
    # and it lands on the garment, not in a background corner
    inside = changed & (mask > 0.5)
    assert inside.sum() > 0.6 * changed.sum()
    rows = np.flatnonzero(changed.any(axis=1))
    assert (rows[-1] - rows[0]) > 0.08 * h  # tall enough to be readable


def test_text_overlay_uses_a_real_typeface_when_available():
    from sartiq_score import sanity

    candidates = [p for p in sanity._FONT_CANDIDATES if os.path.exists(p)]
    if not candidates:
        pytest.skip("no TrueType font on this host")
    font = sanity._pick_font(64)
    assert font.getlength("SARTIQ") > 100.0
