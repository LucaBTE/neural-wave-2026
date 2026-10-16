"""Tests for scripts/eval_val.py (scorer-heavy; skipped without model weights)."""

from __future__ import annotations

import csv
import json
import os
import sys

import pytest

import _mock

sys.path.insert(0, str(_mock.SCRIPTS))

import eval_val  # noqa: E402  (needs scripts/ on the path)


pytestmark = pytest.mark.scorer


def _holdout_and_preds(info, tmp_path, *, frac="0.5"):
    holdout = tmp_path / "holdout"
    _mock.run_cmd(["scripts/make_holdout.py", "--pairs", info["root"] / "train" / "pairs.jsonl",
                   "--frac", frac, "--seed", "0", "--out", holdout], check=True)
    preds = tmp_path / "preds"
    _mock.run_cmd(["baselines/baseline_identity.py", "--manifest", holdout / "val.jsonl",
                   "--out", preds, "--team", "identity"], check=True)
    return holdout, preds


@_mock.needs_models
def test_eval_scores_a_holdout(dataset, tmp_path):
    info = dataset
    holdout, preds = _holdout_and_preds(info, tmp_path)
    out = tmp_path / "val_out"
    proc = _mock.run_cmd(["scripts/eval_val.py", "--pairs", holdout / "val.jsonl",
                          "--preds", preds, "--team", "identity", "--out", out, "--jobs", "2"])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    pairs_csv = (out / "pairs.csv").read_text(encoding="utf-8").splitlines()
    assert len(pairs_csv) == 4  # header + 3 val pairs
    assert (out / "review.csv").is_file()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["team"] == "identity"
    assert summary["pairs"] == 3
    assert summary["missing"] == 0
    assert 0.0 <= summary["mean_score"] <= 100.0


@_mock.needs_models
def test_missing_prediction_scores_zero(dataset, tmp_path):
    info = dataset
    holdout, _ = _holdout_and_preds(info, tmp_path)
    empty = tmp_path / "empty_preds"
    empty.mkdir()
    out = tmp_path / "val_out_missing"
    proc = _mock.run_cmd(["scripts/eval_val.py", "--pairs", holdout / "val.jsonl",
                          "--preds", empty, "--team", "identity", "--out", out])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing"] == summary["pairs"] == 3
    assert summary["mean_score"] == 0.0


@_mock.needs_models
def test_symlinked_prediction_is_rejected(dataset, tmp_path):
    """A symlinked front file is never scored (the scorer treats it as invalid)."""
    info = dataset
    holdout = tmp_path / "holdout"
    _mock.run_cmd(["scripts/make_holdout.py", "--pairs", info["root"] / "train" / "pairs.jsonl",
                   "--frac", "0.5", "--seed", "0", "--out", holdout], check=True)
    val = [json.loads(line) for line in
           (holdout / "val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    pid = val[0]["pair_id"]

    preds = tmp_path / "preds"
    real = _mock.synth_image(tmp_path / "real" / "front.png", 5)
    d = preds / "identity" / pid
    d.mkdir(parents=True)
    os.symlink(real, d / "front.png")

    out = tmp_path / "val_out_sym"
    proc = _mock.run_cmd(["scripts/eval_val.py", "--pairs", holdout / "val.jsonl",
                          "--preds", preds, "--team", "identity", "--out", out])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "symlink" in proc.stdout
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing"] >= 1


# --------------------------------------------------------------------------- #
# R4 regressions: discovery mirrors the scorer; summary uses the official row
# --------------------------------------------------------------------------- #


def test_discovery_matches_scorer_case_and_symlinks(tmp_path):
    """front.PNG is found; a symlinked team folder is never followed."""
    preds = tmp_path / "preds"
    (preds / "identity" / "d001").mkdir(parents=True)
    _mock.synth_image(preds / "identity" / "d001" / "front.PNG", 1)

    real = tmp_path / "real_team"
    (real / "d002").mkdir(parents=True)
    _mock.synth_image(real / "d002" / "front.png", 2)
    os.symlink(real, preds / "ghost")

    files, notes = eval_val.team_view_files(preds, "identity")
    assert files["d001"]["front"].name == "front.PNG"  # case-insensitive, like the scorer
    assert notes == []

    ghost_files, ghost_notes = eval_val.team_view_files(preds, "ghost")
    assert ghost_files == {}
    assert any("symlink" in n for n in ghost_notes)


def test_summary_uses_official_garment_weighted_row():
    """The mean comes from the official aggregate row, not a naive average over views."""
    # two views for garment "a" (100, 100) and one for "b" (0): naive view mean = 66.67,
    # but the official garment-weighted score is (100 + 0) / 2 = 50.0.
    records = [
        {"garment": "a", "view": "front", "score": 100.0, "subscores": {}},
        {"garment": "a", "view": "back", "score": 100.0, "subscores": {}},
        {"garment": "b", "view": "front", "score": 0.0, "subscores": {}},
    ]
    rows = [{"team": "manifest", "score": 50.0, "pairs_scored": 3, "pairs_missing": 0,
             "n_review": 0, "n_invalid": 0, "garments_scored": 2, "garments_missing": 0}]
    summary = eval_val.summary_from_aggregate(rows, records, "identity")
    assert summary["mean_score"] == 50.0
    assert summary["scored"] == 3
    assert summary["garments_scored"] == 2
    assert summary["team"] == "identity"


def test_leaderboard_unicode_team_name_round_trips(tmp_path):
    rows = [{"rank": 1, "team": "Équipe,東京", "score": 12.34, "colour": 0.1, "lpips": 0.2,
             "clip": 0.3, "geometry": 0.4, "ssim": 0.5, "pairs_scored": 1, "pairs_missing": 0,
             "garments_scored": 1, "garments_missing": 0, "n_review": 0, "n_invalid": 0}]
    path = tmp_path / "leaderboard.csv"
    eval_val.write_leaderboard_utf8(rows, path)
    with open(path, newline="", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["team"] == "Équipe,東京"
    assert got[0]["rank"] == "1"


# --------------------------------------------------------------------------- #
# v2.3: fail before scoring when the dataset root or the model cache is wrong
# --------------------------------------------------------------------------- #

def test_missing_target_exits_2_naming_root_and_path(dataset, tmp_path):
    """A wrong --dataset must exit 2 with the resolved root, not score zeros."""
    info = dataset
    holdout, preds = _holdout_and_preds(info, tmp_path)
    empty = tmp_path / "empty_dataset"
    empty.mkdir()
    proc = _mock.run_cmd(["scripts/eval_val.py", "--pairs", holdout / "val.jsonl",
                          "--preds", preds, "--team", "identity", "--out", tmp_path / "o",
                          "--dataset", empty, "--limit", "1"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "dataset target missing" in proc.stderr
    assert "resolved dataset root" in proc.stderr
    assert str(empty.resolve()) in proc.stderr
    assert not (tmp_path / "o" / "summary.json").exists()


def test_scorer_setup_failure_exits_2(dataset, tmp_path, monkeypatch, capsys):
    """A model-cache/proxy failure must abort with exit 2 and explain why."""
    info = dataset
    holdout, preds = _holdout_and_preds(info, tmp_path)

    def boom():
        raise RuntimeError("no route to host")

    monkeypatch.setattr(eval_val, "_import_scorer_setup",
                        lambda: (boom, tmp_path / "models_cache"))
    rc = eval_val.main(["--pairs", str(holdout / "val.jsonl"), "--preds", str(preds),
                        "--team", "identity", "--out", str(tmp_path / "o"), "--limit", "1"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Scorer setup failed" in err
    assert "no route to host" in err
    assert "Model cache:" in err
    assert "No results were produced." in err

