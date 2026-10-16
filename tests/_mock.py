"""Shared helpers for the repo-level script tests.

Builds a throwaway participant dataset with the exact contract layout from small
synthetic PIL images, and runs the CLI scripts the way a participant would (a
subprocess, from the repo root, with the repo's scorer on PYTHONPATH).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scorer"
SCRIPTS = REPO / "scripts"
BASELINES = REPO / "baselines"


# --------------------------------------------------------------------------- #
# synthetic images / dataset
# --------------------------------------------------------------------------- #

def synth_image(path: Path, seed: int, size: tuple[int, int] = (512, 640)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    w, h = size
    color = (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200))
    img = Image.new("RGB", (w, h), (246, 246, 244))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([w * 0.22, h * 0.16, w * 0.78, h * 0.84], radius=int(w * 0.06), fill=color)
    draw.rectangle([w * 0.44, h * 0.05, w * 0.56, h * 0.22], fill=color)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, "JPEG", quality=95)
    else:
        img.save(path)
    return path


def build_mock_dataset(root: Path, *, n_pairs: int = 6, n_dev: int = 2, n_test: int = 2,
                       shared_targets: int = 0, seed: int = 0) -> dict:
    """Write a contract-layout dataset under ``root`` and return ids/paths."""
    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)

    pairs: list[dict] = []
    for i in range(n_pairs):
        pid = f"t{i + 1:04d}"
        d = root / "train" / pid
        synth_image(d / "input_1.jpg", seed * 10 + i * 2 + 1)
        synth_image(d / "input_2.jpg", seed * 10 + i * 2 + 2)
        synth_image(d / "target.jpg", seed * 10 + i * 2 + 3)
        pairs.append({
            "pair_id": pid,
            "inputs": [f"train/{pid}/input_1.jpg", f"train/{pid}/input_2.jpg"],
            "target": f"train/{pid}/target.jpg",
        })
    # make consecutive pairs share byte-identical targets (same garment, repeated product)
    for k in range(min(shared_targets, max(0, n_pairs - 1))):
        shutil.copyfile(root / pairs[k]["target"], root / pairs[k + 1]["target"])

    with open(root / "train" / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row) + "\n")

    (root / "dev").mkdir(exist_ok=True)
    dev_ids: list[str] = []
    with open(root / "dev" / "inputs.jsonl", "w", encoding="utf-8") as fh:
        for i in range(n_dev):
            gid = f"d{i + 1:03d}"
            dev_ids.append(gid)
            synth_image(root / "dev" / gid / "input_1.jpg", seed * 100 + i + 1)
            fh.write(json.dumps({"garment_id": gid, "view": "front",
                                 "inputs": [f"dev/{gid}/input_1.jpg"]}) + "\n")

    (root / "test").mkdir(exist_ok=True)
    test_ids: list[str] = []
    with open(root / "test" / "inputs.jsonl", "w", encoding="utf-8") as fh:
        for i in range(n_test):
            gid = f"g{i + 1:02d}"
            test_ids.append(gid)
            synth_image(root / "test" / gid / "input_1.jpg", seed * 200 + i + 1)
            fh.write(json.dumps({"garment_id": gid, "view": "front",
                                 "inputs": [f"test/{gid}/input_1.jpg"]}) + "\n")

    return {"root": root, "pairs": pairs, "dev_ids": dev_ids, "test_ids": test_ids}


def make_submission_team(team_dir: Path, ids: list[str], *, ext: str = ".png",
                         size: tuple[int, int] = (512, 640)) -> Path:
    for i, gid in enumerate(ids):
        synth_image(team_dir / gid / f"front{ext}", 9000 + i, size=size)
    return team_dir


# --------------------------------------------------------------------------- #
# process runner
# --------------------------------------------------------------------------- #

def run_cmd(args: list, *, cwd: Path | None = None, env_extra: dict | None = None,
            check: bool = False) -> subprocess.CompletedProcess:
    """Run a script as a participant would: clean PYTHONPATH, repo's scorer on the path.

    A repo-relative script path (e.g. ``scripts/eval_val.py``) is resolved against the
    repo, so tests can change ``cwd`` without breaking the script path.
    """
    args = list(args)
    if args and isinstance(args[0], str) and not Path(args[0]).is_absolute() and (REPO / args[0]).is_file():
        args[0] = REPO / args[0]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SCORER)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, *[str(a) for a in args]],
        cwd=str(cwd or REPO), env=env, capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


# --------------------------------------------------------------------------- #
# scorer availability
# --------------------------------------------------------------------------- #

def models_available() -> bool:
    if os.environ.get("SARTIQ_SKIP_MODEL_TESTS") == "1":
        return False
    model_dir = os.environ.get("SCORER_MODEL_DIR")
    if not model_dir:
        return False
    root = Path(model_dir)
    if not root.is_dir():
        return False
    return any(root.rglob("*.onnx")) or any(root.rglob("*.pth"))


needs_models = pytest.mark.skipif(
    not models_available(),
    reason="scorer weights unavailable (set SCORER_MODEL_DIR, or SARTIQ_SKIP_MODEL_TESTS=1 to skip)",
)
