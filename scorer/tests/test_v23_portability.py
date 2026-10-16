"""v2.3 cross-platform regressions (clean-machine container / macOS / Windows runs).

Model-free: the font fallback, the RSS helper, the UTF-8 writers and the
"no checks ran is not a pass" CLI guard are all exercised without weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCORER = Path(__file__).resolve().parents[1]
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))


# --------------------------------------------------------------------------- #
# font fallback on a host with no TrueType font at all
# --------------------------------------------------------------------------- #

def _garment_frame(h: int = 1200, w: int = 900):
    base = np.full((h, w, 3), 250, np.uint8)
    base[180:1020, 120:780] = np.array([205, 200, 190], np.uint8)  # light garment
    mask = np.zeros((h, w), np.float32)
    mask[180:1020, 120:780] = 1.0
    return base, mask


def _no_fonts(monkeypatch):
    from sartiq_score import sanity

    monkeypatch.setattr(sanity, "_FONT_CANDIDATES", ())
    monkeypatch.setattr(sanity, "_system_font_paths", lambda: [])


def test_builtin_fallback_font_is_scalable(monkeypatch):
    """``_pick_font`` must honour the requested size even with no font installed."""
    _no_fonts(monkeypatch)
    from sartiq_score import sanity

    small = sanity._pick_font(16).getlength("SARTIQ")
    big = sanity._pick_font(256).getlength("SARTIQ")
    assert big > 4 * small


def test_text_overlay_is_large_with_no_font_installed(monkeypatch):
    """The grow loop must converge (not run away) and still draw a big overlay."""
    _no_fonts(monkeypatch)
    from sartiq_score import sanity

    base, mask = _garment_frame()
    out = sanity.t_text_overlay(base, mask)
    assert out.shape == base.shape and out.dtype == np.uint8
    changed = (out.astype(int) - base.astype(int)).any(axis=2)
    frac = float(changed.mean())
    assert frac > 0.03, frac
    inside = changed & (mask > 0.5)
    assert inside.sum() > 0.6 * changed.sum()


def test_font_size_and_stroke_are_capped(monkeypatch):
    """A non-scaling font cannot push the size (or the stroke) without bound."""
    _no_fonts(monkeypatch)
    from sartiq_score import sanity

    assert sanity._MAX_FONT_PX <= 8192
    assert sanity._pick_font(10 ** 9).size <= sanity._MAX_FONT_PX


# --------------------------------------------------------------------------- #
# resource / RSS helper (Windows has no ``resource`` module)
# --------------------------------------------------------------------------- #

def test_max_rss_kb_is_int_or_none():
    from sartiq_score import sanity

    value = sanity._max_rss_kb()
    assert value is None or isinstance(value, int)
    child = sanity._max_rss_kb(children=True)
    assert child is None or isinstance(child, int)


def test_module_imports_without_resource(monkeypatch):
    """Simulate Windows: importing the module must not need ``resource``."""
    import importlib

    import sartiq_score.sanity as sanity

    monkeypatch.setattr(sanity, "_resource", None)
    assert sanity._max_rss_kb() is None
    monkeypatch.undo()
    reloaded = importlib.reload(sanity)
    if reloaded._resource is None:  # real Windows: no resource module, helper reports None
        assert reloaded._max_rss_kb() is None
    else:
        assert reloaded._max_rss_kb() is not None


# --------------------------------------------------------------------------- #
# UTF-8 CSV / text output
# --------------------------------------------------------------------------- #

def test_leaderboard_and_jsonl_are_utf8(tmp_path):
    from sartiq_score import batch

    rows = [{"rank": 1, "team": "Équipe Zürich,東京", "score": 12.34, "colour": 0.1,
             "lpips": 0.2, "clip": 0.3, "geometry": 0.4, "ssim": 0.5,
             "pairs_scored": 1, "pairs_missing": 0, "garments_scored": 1,
             "garments_missing": 0, "n_review": 0, "n_invalid": 0}]
    lb = tmp_path / "leaderboard.csv"
    batch.write_leaderboard(rows, lb)
    assert "Équipe Zürich,東京" in lb.read_text(encoding="utf-8")

    scores = tmp_path / "scores.jsonl"
    batch.write_scores_jsonl([{"team": "Équipe Zürich", "score": 1.0}], scores)
    assert json.loads(scores.read_text(encoding="utf-8"))["team"] == "Équipe Zürich"


def test_sanity_report_is_utf8(tmp_path):
    from sartiq_score import sanity

    summary = {
        "out_dir": str(tmp_path), "refs_glob": "*", "limit": None, "seed": 1, "jobs": 1,
        "requested_families": ["invariance"], "n_refs": 1, "refs": [],
        "checks": [{"garment": "g1", "family": "invariance", "check": "pn", "ok": True,
                    "value": 1.0, "tolerance": 2.0}],
        "records": [], "notes": ["unicode → ok é"], "scores_by_garment": {},
        "determinism": {}, "performance": {},
        "timing": {"suite_s": 0.0, "scoring_s": 0.0, "per_garment_s": {}},
        "families": [{"family": "invariance", "n": 1, "passed": 1, "failed": 0,
                      "failing_garments": [], "failing_checks": []}],
        "n_failed": 0, "n_checks": 1, "failing_garments": [],
        "n_garments_requested": 1, "n_garments_ok": 1, "n_garments_failed": 0, "run_ok": True,
    }
    written = sanity.write_report(summary)
    assert "unicode → ok é" in Path(written["report_path"]).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# "zero checks ran" / "a worker died" is never a pass
# --------------------------------------------------------------------------- #

def _sanity_args():
    from sartiq_score import cli

    return cli.build_parser().parse_args(["sanity", "--refs-glob", "x/*.jpg",
                                          "--out", "o", "--limit", "1"])


def _patch_suite(monkeypatch, summary):
    from sartiq_score import cli, sanity

    monkeypatch.setattr(sanity, "run_suite", lambda **_: summary)
    monkeypatch.setattr(sanity, "write_report", lambda *_: {})
    return cli


def test_cli_sanity_fails_when_no_checks_ran(monkeypatch):
    zero = {"n_checks": 0, "n_failed": 0, "failing_garments": [], "families": [],
            "n_garments_requested": 1, "n_garments_ok": 0, "n_garments_failed": 1,
            "run_ok": False, "performance": {}}
    cli = _patch_suite(monkeypatch, zero)
    assert cli._cmd_sanity(_sanity_args()) == 1


def test_cli_sanity_fails_on_worker_failure(monkeypatch):
    partial = {"n_checks": 3, "n_failed": 0, "failing_garments": [], "families": [],
               "n_garments_requested": 2, "n_garments_ok": 1, "n_garments_failed": 1,
               "run_ok": False, "performance": {}}
    cli = _patch_suite(monkeypatch, partial)
    assert cli._cmd_sanity(_sanity_args()) == 1


def test_cli_sanity_passes_a_clean_run(monkeypatch):
    good = {"n_checks": 3, "n_failed": 0, "failing_garments": [], "families": [],
            "n_garments_requested": 1, "n_garments_ok": 1, "n_garments_failed": 0,
            "run_ok": True, "performance": {}}
    cli = _patch_suite(monkeypatch, good)
    assert cli._cmd_sanity(_sanity_args()) == 0


def test_console_says_so_when_nothing_ran():
    from sartiq_score import sanity

    text = sanity.render_console({"n_checks": 0, "n_failed": 0, "failing_garments": [],
                                  "families": [], "n_garments_requested": 1,
                                  "n_garments_failed": 1, "performance": {}})
    assert "NOT a pass" in text
    assert "worker failures" in text
