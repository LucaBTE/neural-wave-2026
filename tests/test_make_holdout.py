"""Tests for scripts/make_holdout.py."""

from __future__ import annotations

import json

import _mock


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_split_sizes_schema_and_meta(dataset, tmp_path):
    info = dataset
    out = tmp_path / "holdout"
    proc = _mock.run_cmd([
        "scripts/make_holdout.py",
        "--pairs", info["root"] / "train" / "pairs.jsonl",
        "--frac", "0.5", "--seed", "0", "--out", out,
    ])
    assert proc.returncode == 0, proc.stderr
    train = _read_jsonl(out / "train.jsonl")
    val = _read_jsonl(out / "val.jsonl")
    assert len(train) + len(val) == len(info["pairs"])
    assert len(val) == 3
    for row in train + val:
        assert set(row) == {"pair_id", "inputs", "target"}  # schema-identical to train/pairs.jsonl
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert (out / meta["dataset_root"]).resolve() == info["root"].resolve()


def test_deterministic_for_a_seed(dataset, tmp_path):
    info = dataset
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--pairs", info["root"] / "train" / "pairs.jsonl", "--frac", "0.5", "--seed", "7"]
    assert _mock.run_cmd(["scripts/make_holdout.py", *common, "--out", a]).returncode == 0
    assert _mock.run_cmd(["scripts/make_holdout.py", *common, "--out", b]).returncode == 0
    assert (a / "val.jsonl").read_bytes() == (b / "val.jsonl").read_bytes()
    assert (a / "train.jsonl").read_bytes() == (b / "train.jsonl").read_bytes()


def test_warns_about_repeated_targets(dataset_shared, tmp_path):
    info = dataset_shared
    proc = _mock.run_cmd([
        "scripts/make_holdout.py",
        "--pairs", info["root"] / "train" / "pairs.jsonl",
        "--frac", "0.5", "--seed", "0", "--out", tmp_path / "holdout",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "warning:" in proc.stdout
    assert "--group-by-target" in proc.stdout


def test_group_by_target_keeps_shared_pairs_together(dataset_shared, tmp_path):
    info = dataset_shared
    out = tmp_path / "holdout"
    proc = _mock.run_cmd([
        "scripts/make_holdout.py",
        "--pairs", info["root"] / "train" / "pairs.jsonl",
        "--frac", "0.5", "--seed", "0", "--out", out, "--group-by-target",
    ])
    assert proc.returncode == 0, proc.stderr
    train = {row["pair_id"] for row in _read_jsonl(out / "train.jsonl")}
    val = {row["pair_id"] for row in _read_jsonl(out / "val.jsonl")}
    # pairs[0] and pairs[1] share a byte-identical target -> same side
    shared = {info["pairs"][0]["pair_id"], info["pairs"][1]["pair_id"]}
    assert shared <= train or shared <= val


def test_group_by_target_off_ignores_targets(dataset_shared, tmp_path):
    info = dataset_shared
    plain = tmp_path / "plain"
    grouped = tmp_path / "grouped"
    common = ["--pairs", info["root"] / "train" / "pairs.jsonl", "--frac", "0.5", "--seed", "0"]
    _mock.run_cmd(["scripts/make_holdout.py", *common, "--out", plain])
    _mock.run_cmd(["scripts/make_holdout.py", *common, "--out", grouped, "--group-by-target"])
    assert not (plain / "meta.json").read_text() == "" and json.loads(
        (plain / "meta.json").read_text())["group_by_target"] is False
    assert json.loads((grouped / "meta.json").read_text())["group_by_target"] is True


def test_group_by_target_fails_on_unreadable_target(dataset, tmp_path):
    """An unreadable target must be a hard error under --group-by-target, not a singleton."""
    info = dataset
    bad = tmp_path / "bad_pairs.jsonl"
    rows = [
        {"pair_id": "t0001", "inputs": ["train/t0001/input_1.jpg"],
         "target": "train/t0001/target.jpg"},
        {"pair_id": "t9999", "inputs": ["train/t0001/input_1.jpg"],
         "target": "train/t0001/missing_target.jpg"},
    ]
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    proc = _mock.run_cmd(["scripts/make_holdout.py", "--pairs", bad, "--frac", "0.5",
                          "--seed", "0", "--out", tmp_path / "h",
                          "--group-by-target", "--dataset", info["root"]])
    assert proc.returncode != 0, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "t9999" in out and "missing_target.jpg" in out, out

    # without --group-by-target the same manifest still runs (the missing target is a warning-free
    # concern only when you asked to group by it)
    ok = _mock.run_cmd(["scripts/make_holdout.py", "--pairs", bad, "--frac", "0.5",
                        "--seed", "0", "--out", tmp_path / "h2", "--dataset", info["root"]])
    assert ok.returncode == 0, ok.stdout + ok.stderr
