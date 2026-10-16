"""v2.2 regression tests from the pre-event code review.

Pure-I/O and aggregation tests: no model weights needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sartiq_score import batch
from sartiq_score.cli import resolve_ref_mask
from sartiq_score.io import load_image
from sartiq_score.types import InvalidImage, InvalidReason


def _garment(w=400, h=500, colour=(180, 40, 40)) -> Image.Image:
    a = np.full((h, w, 3), 255, np.uint8)
    a[h // 5: 4 * h // 5, w // 5: 4 * w // 5] = colour
    return Image.fromarray(a)


# -- io --------------------------------------------------------------------

def test_animated_gif_rejected(tmp_path):
    p = tmp_path / "a.gif"
    g = _garment()
    g.save(p, save_all=True, append_images=[g.rotate(30)], duration=[1, 1000])
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.ANIMATED


def test_animated_bytes_with_png_extension_rejected(tmp_path):
    p = tmp_path / "front.png"
    g = _garment()
    g.save(p, format="GIF", save_all=True, append_images=[g.rotate(30)])
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.ANIMATED


def test_single_frame_gif_ok(tmp_path):
    p = tmp_path / "s.gif"
    _garment().save(p)
    assert load_image(p).rgb.shape == (500, 400, 3)


def test_16bit_scaling_is_by_bit_depth_not_content(tmp_path):
    a = np.zeros((300, 300), np.uint16)
    a[:, 150:] = 255
    p1 = tmp_path / "a.png"
    Image.fromarray(a).save(p1)
    b = a.copy()
    b[0, 0] = 256
    p2 = tmp_path / "b.png"
    Image.fromarray(b).save(p2)
    r1, r2 = load_image(p1).rgb, load_image(p2).rgb
    # one changed pixel must not change the whole picture
    assert np.abs(r1.astype(int) - r2.astype(int)).max() <= 1
    # 255 / 65535 of full scale is (near) black in 8-bit
    assert r1.max() <= 1


def test_rgb_colour_key_transparency_composited_on_white(tmp_path):
    g = np.asarray(_garment(colour=(10, 120, 200))).copy()
    g[g.sum(2) == 765] = 0  # background black ...
    p = tmp_path / "k.png"
    Image.fromarray(g).save(p, transparency=(0, 0, 0))  # ... and declared transparent
    rgb = load_image(p).rgb
    assert tuple(rgb[2, 2]) == (255, 255, 255)
    assert tuple(rgb[250, 200]) == (10, 120, 200)


def test_file_byte_cap(tmp_path, monkeypatch):
    from sartiq_score import io as sio

    p = tmp_path / "big.png"
    _garment().save(p)
    monkeypatch.setattr(sio, "MAX_FILE_BYTES", 100)
    with pytest.raises(InvalidImage) as e:
        sio.load_image(p)
    assert e.value.reason == InvalidReason.FILE_TOO_LARGE


def test_limits_checked_on_header_before_decode(tmp_path, monkeypatch):
    p = tmp_path / "wide.png"
    Image.new("RGB", (4000, 1000), (200, 10, 10)).save(p)
    called = {"load": 0}
    orig = Image.Image.load

    def spy(self, *a, **k):
        called["load"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(Image.Image, "load", spy)
    with pytest.raises(InvalidImage) as e:
        load_image(p)
    assert e.value.reason == InvalidReason.ASPECT


def test_large_decode_is_downscaled(tmp_path, monkeypatch):
    from sartiq_score import io as sio

    p = tmp_path / "l.png"
    _garment(2000, 2400).save(p)
    monkeypatch.setattr(sio, "WORK_MAX_PIXELS", 1_000_000)
    lo = sio.load_image(p)
    assert lo.rgb.shape[0] * lo.rgb.shape[1] <= 1_000_000
    assert any(w.startswith("downscaled_for_memory") for w in lo.warnings)


# -- batch -----------------------------------------------------------------

def _tree(tmp_path):
    refs = tmp_path / "refs"
    for gid in ("g1", "g2"):
        (refs / gid).mkdir(parents=True)
        _garment().save(refs / gid / "front.png")
    subs = tmp_path / "subs"
    (subs / "teamA" / "g1").mkdir(parents=True)
    _garment().save(subs / "teamA" / "g1" / "front.png")
    return refs, subs


def test_symlinked_submission_scores_invalid(tmp_path):
    refs, subs = _tree(tmp_path)
    (subs / "cheat" / "g1").mkdir(parents=True)
    os.symlink(refs / "g1" / "front.png", subs / "cheat" / "g1" / "front.png")
    tasks = batch.build_tasks(batch.discover_refs(refs), batch.discover_subs(subs), ["cheat"])
    t = next(t for t in tasks if t["garment"] == "g1")
    rec = batch._score_task(t)
    assert rec["status"] == "invalid" and rec["score"] == 0.0
    assert "invalid:symlink_not_allowed" in rec["flags"]


def test_symlinked_team_and_garment_dirs_not_followed(tmp_path):
    refs, subs = _tree(tmp_path)
    os.symlink(refs, subs / "linkteam")
    (subs / "teamB").mkdir()
    os.symlink(refs / "g2", subs / "teamB" / "g2")
    found = batch.discover_subs(subs)
    assert "linkteam" not in found
    assert "g2" not in found.get("teamB", {})


def test_manifest_sub_not_inherited_when_scoring_team_tree(tmp_path):
    refs, subs = _tree(tmp_path)
    explicit = {("g2", "front"): refs / "g2" / "front.png"}
    tasks = batch.build_tasks(batch.discover_refs(refs), batch.discover_subs(subs), ["teamA"], explicit)
    t = next(t for t in tasks if t["garment"] == "g2")
    assert t["sub"] is None  # teamA has no g2 file -> missing -> 0


def test_manifest_sub_used_in_manifest_only_mode(tmp_path):
    refs, _ = _tree(tmp_path)
    explicit = {("g2", "front"): refs / "g2" / "front.png"}
    tasks = batch.build_tasks(batch.discover_refs(refs), {}, ["manifest"], explicit)
    t = next(t for t in tasks if t["garment"] == "g2")
    assert t["sub"] is not None


def test_tiebreak_uses_garment_weighting_and_shared_ranks():
    refs = {"g1": {"front": "x"}, "g2": {"front": "x", "back": "x"}}

    def rec(team, g, v, score, colour):
        return {"team": team, "garment": g, "view": v, "score": score, "missing": False,
                "status": "ok", "subscores": {"colour": colour, "lpips": 0.5}}

    records = [
        rec("A", "g1", "front", 50, 1.0), rec("A", "g2", "front", 50, 0.0), rec("A", "g2", "back", 50, 0.0),
        rec("B", "g1", "front", 50, 0.0), rec("B", "g2", "front", 50, 0.6), rec("B", "g2", "back", 50, 0.6),
        rec("C", "g1", "front", 50, 1.0), rec("C", "g2", "front", 50, 0.0), rec("C", "g2", "back", 50, 0.0),
    ]
    rows = {r["team"]: r for r in batch._aggregate([], records, refs, {})}
    assert rows["A"]["colour"] == pytest.approx(0.5)
    assert rows["B"]["colour"] == pytest.approx(0.3)
    assert rows["A"]["rank"] == rows["C"]["rank"] == 1  # full tie -> shared rank
    assert rows["B"]["rank"] == 3


def test_ref_mask_is_garment_specific(tmp_path):
    md = tmp_path / "masks"
    (md / "g1").mkdir(parents=True)
    Image.new("L", (10, 10), 255).save(md / "g1" / "front.png")
    Image.new("L", (10, 10), 255).save(md / "front.png")  # legacy flat file must not leak
    assert resolve_ref_mask("refs/g1/front.jpg", md, garment="g1", view="front") == str(md / "g1" / "front.png")
    assert resolve_ref_mask("refs/g2/front.jpg", md, garment="g2", view="front") is None
