#!/usr/bin/env python3
"""Scoring smoke test: exercise the documented ``pair`` command end to end.

This is a CI helper, not part of the participant-facing repo.  It builds three
small synthetic images with PIL, then runs the command the README documents
(``python -m sartiq_score.cli pair``) and asserts the scorer behaves:

* the reference against a byte-identical copy scores ~100 (status ``ok``);
* a clearly different garment scores far lower;
* a flat, blank image is rejected as ``invalid`` with score 0.

Run from the repository root with the quickstart venv active::

    python .github/smoke_score.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def _garment(path: Path, colour: tuple[int, int, int], shape: str = "rect") -> Path:
    """A flat-lay-ish rectangle/circle garment on a light background (>= 256 px)."""
    w, h = 512, 640
    img = Image.new("RGB", (w, h), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    if shape == "rect":
        draw.rounded_rectangle([w * 0.22, h * 0.16, w * 0.78, h * 0.84], radius=int(w * 0.06), fill=colour)
        draw.rectangle([w * 0.44, h * 0.05, w * 0.56, h * 0.22], fill=colour)
    else:
        draw.ellipse([w * 0.18, h * 0.20, w * 0.82, h * 0.80], fill=colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _blank(path: Path) -> Path:
    Image.new("RGB", (512, 640), (250, 250, 250)).save(path)
    return path


def _score(ref: Path, sub: Path) -> dict:
    """Run the documented CLI and return the parsed JSON report.

    The LPIPS/OpenCLIP packages print a banner on stdout before the report, so
    the JSON is recovered from the tail of stdout (from the first ``{`` line).
    ``env -u PYTHONPATH`` in the README is emulated by dropping PYTHONPATH here,
    which is the only portable equivalent on Windows.
    """
    cmd = [sys.executable, "-m", "sartiq_score.cli", "pair",
           "--ref", str(ref), "--sub", str(sub), "--json"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    # exit 0 = ok/review, 1 = invalid, anything else is a real failure
    if proc.returncode not in (0, 1):
        raise AssertionError(
            f"`sartiq_score.cli pair` exited {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    lines = proc.stdout.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "{"]
    if not starts:
        raise AssertionError(f"no JSON report on stdout:\n{proc.stdout}\n{proc.stderr}")
    report = json.loads("\n".join(lines[starts[0]:]))
    report["_returncode"] = proc.returncode
    return report


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ref = _garment(tmpdir / "ref.png", (74, 96, 138))
        same = _garment(tmpdir / "same.png", (74, 96, 138))
        diff = _garment(tmpdir / "diff.png", (210, 180, 40), shape="ellipse")
        blank = _blank(tmpdir / "blank.png")

        identical = _score(ref, same)
        different = _score(ref, diff)
        flat = _score(ref, blank)

        print(f"identical : score={identical['score']:.2f} status={identical['status']} "
              f"flags={identical['flags']}")
        print(f"different : score={different['score']:.2f} status={different['status']} "
              f"flags={different['flags']}")
        print(f"blank     : score={flat['score']:.2f} status={flat['status']} "
              f"flags={flat['flags']}")
        print(f"scorer    : {identical['versions'].get('package')} "
              f"torch={identical['versions'].get('torch')}")

        failures: list[str] = []
        if not (identical["score"] >= 99.0 and identical["status"] == "ok"):
            failures.append(
                f"identical pair should score ~100/ok, got {identical['score']}/{identical['status']}")
        if not (different["score"] <= 60.0 and different["score"] <= identical["score"] - 25.0):
            failures.append(
                f"different pair should score far lower, got {different['score']} "
                f"(identical {identical['score']})")
        if not (flat["score"] == 0.0 and flat["status"] == "invalid"):
            failures.append(
                f"blank pair should be invalid/0, got {flat['score']}/{flat['status']}")

        if failures:
            for line in failures:
                print(f"FAIL: {line}", file=sys.stderr)
            return 1
        print("scoring smoke test: OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
