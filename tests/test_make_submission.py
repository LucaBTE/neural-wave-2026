"""Tests for scripts/make_submission.py (hardened validator, D8/Astra #14)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import _mock
from PIL import Image


def test_valid_submission_passes(dataset, tmp_path):
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001", "d002", "g01"])
    # run from a scratch cwd so --zip does not drop files into the repo
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 error(s)" in proc.stdout
    # missing test ids are listed as warnings
    assert "g02" in proc.stdout
    # coverage is reported per team and per split
    assert "coverage teamA: dev 2/2, test 1/2" in proc.stdout


def test_unknown_id_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["nope1"])
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not in the dev/test inputs" in proc.stdout
    assert "ERROR:" in proc.stdout


def test_missing_manifest_is_an_error_unless_no_dataset(dataset, tmp_path):
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"])
    empty = tmp_path / "empty_dataset"
    empty.mkdir()

    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", empty], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no dev/test inputs.jsonl" in proc.stdout

    ok = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                        "--dataset", empty, "--no-dataset"], cwd=tmp_path)
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_missing_front_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    # a garment folder with only a back view
    _mock.synth_image(subs / "teamA" / "d001" / "back.png", 1)
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "no front view" in proc.stdout


def test_symlinked_front_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    real = _mock.synth_image(tmp_path / "real" / "front.png", 2)
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    os.symlink(real, d / "front.png")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "symlink" in proc.stdout


def test_symlinked_team_root_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    subs.mkdir()
    real_team = tmp_path / "real_team"
    _mock.make_submission_team(real_team, ["d001"])
    os.symlink(real_team, subs / "teamA")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "team folder is a symlink" in proc.stdout


def test_small_image_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"], size=(100, 100))
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "min_side_too_small" in proc.stdout or "short side < 256" in proc.stdout


def test_extreme_aspect_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"], size=(400, 1600))  # 4:1
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "aspect" in proc.stdout


def test_flat_field_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    Image.new("RGB", (512, 640), (200, 200, 200)).save(d / "front.png")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "flat field" in proc.stdout


def test_undecodable_file_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    # PNG magic followed by garbage: open() fails on a full decode
    (d / "front.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"not a real png")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "INVALID" in proc.stdout


def test_disallowed_extension_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    _mock.synth_image(d / "front.bmp", 3)  # saved as PNG bytes under a .bmp name
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "disallowed extension" in proc.stdout


def test_animated_image_is_an_error(dataset, tmp_path):
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    frames = [Image.new("RGB", (512, 640), (i * 40, 80, 120)) for i in range(3)]
    frames[0].save(d / "front.png", save_all=True, append_images=frames[1:], duration=100)
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"]], cwd=tmp_path)
    assert proc.returncode == 1
    assert "animated" in proc.stdout or "multiframe" in proc.stdout


def test_zip_packs_the_team(dataset, tmp_path):
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"])
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", dataset["root"], "--zip"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    zip_path = tmp_path / "teamA.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "teamA/d001/front.png" in names


# --------------------------------------------------------------------------- #
# R4 regressions: manifests, symlinks, per-team ids, ids vs scorer, decoder
# --------------------------------------------------------------------------- #


def test_malformed_manifest_row_is_an_error_with_file_line(dataset, tmp_path):
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"])
    bad = info["root"] / "dev" / "inputs.jsonl"
    bad.write_text('{"garment_id": "d001", "view": "front", "inputs": ["dev/d001/input_1.jpg"]}\n'
                   "{not valid json}\n", encoding="utf-8")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "inputs.jsonl:2" in out, out


def test_empty_manifest_is_an_error_not_a_silent_pass(dataset, tmp_path):
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["ghost99"])  # an id that is not in the dataset
    (info["root"] / "dev" / "inputs.jsonl").write_text("", encoding="utf-8")
    (info["root"] / "test" / "inputs.jsonl").write_text("\n# only a comment\n", encoding="utf-8")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "inputs.jsonl" in out
    assert "0 error(s)" not in proc.stdout


def test_unknown_id_still_rejected_with_one_valid_split(dataset, tmp_path):
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["nope1"])
    # only the test split is present and valid; the unknown id must still be an error
    (info["root"] / "dev" / "inputs.jsonl").unlink()
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not in the dev/test inputs" in proc.stdout


def test_dangling_symlink_front_json_reports_cleanly(dataset, tmp_path):
    info = dataset
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    os.symlink(d / "does_not_exist.png", d / "front.png")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"], "--json"], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    data = json.loads(proc.stdout)  # must still be valid JSON
    assert any("symlink" in e for e in data["errors"])


def test_coverage_and_missing_are_per_team(dataset, tmp_path):
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"])
    _mock.make_submission_team(subs / "teamB", ["d002"])
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "coverage teamA: dev 1/2, test 0/2" in proc.stdout
    assert "coverage teamB: dev 1/2, test 0/2" in proc.stdout
    assert "teamA: garment 'd002'" in proc.stdout
    assert "teamB: garment 'd001'" in proc.stdout


def test_underscore_team_is_checked_like_the_scorer(dataset, tmp_path):
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "_team", ["d001"])
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "_team" in proc.stdout


def test_requests_helper_dir_is_not_a_team(dataset, tmp_path):
    """The hosted baseline's dry run writes <out>/_requests/; it must not be read as a team."""
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"])
    (subs / "_requests" / "d001").mkdir(parents=True)
    (subs / "_requests" / "d001" / "request.json").write_text("{}", encoding="utf-8")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "_requests" not in proc.stdout
    assert "1 team(s)" in proc.stdout
    assert "0 error(s)" in proc.stdout


def test_underscore_team_alongside_reserved_helper_is_checked(dataset, tmp_path):
    """Only the explicitly named helper dirs are reserved; any other underscore team is real."""
    info = dataset
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "_mylab", ["d001"])
    (subs / "_requests").mkdir(parents=True)
    (subs / "_requests" / "payload.bin").write_bytes(b"x")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "_mylab" in proc.stdout
    assert "_requests" not in proc.stdout
    assert "1 team(s)" in proc.stdout


def test_transparent_noise_is_flat_after_white_compositing(dataset, tmp_path):
    """An alpha=0 image is white after compositing, so the scorer calls it flat (invalid).

    A PIL fallback that dropped alpha instead scored it as noisy and passed it.
    """
    info = dataset
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    w, h = 512, 640
    buf = bytearray(random.Random(1).randbytes(w * h * 4))
    buf[3::4] = b"\x00" * (w * h)  # alpha 0 everywhere, noisy RGB underneath
    Image.frombytes("RGBA", (w, h), bytes(buf)).save(d / "front.png")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "flat field" in proc.stdout


def test_black_pixels_with_varying_alpha_pass_the_gate(dataset, tmp_path):
    """Black RGB with a varying alpha is a gradient after white compositing, so it passes."""
    info = dataset
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    w, h = 512, 640
    buf = bytearray(w * h * 4)
    for i in range(w * h):
        buf[i * 4 + 3] = i % 256  # RGB stays 0; alpha ramps
    Image.frombytes("RGBA", (w, h), bytes(buf)).save(d / "front.png")
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_missing_frozen_scorer_fails_clearly(dataset, tmp_path):
    """With sartiq_score unimportable the validator exits 2 with a clear message."""
    subs = tmp_path / "submissions"
    _mock.make_submission_team(subs / "teamA", ["d001"])
    script = _mock.SCRIPTS / "make_submission.py"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import importlib.abc, runpy, sys\n"
        "SCRIPT, SUBS = sys.argv[1], sys.argv[2]\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'sartiq_score' or name.startswith('sartiq_score.'):\n"
        "            raise ImportError('blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "sys.argv = ['make_submission.py', '--submissions', SUBS, '--no-dataset']\n"
        "try:\n"
        "    runpy.run_path(SCRIPT, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    raise SystemExit(exc.code if exc.code is not None else 1)\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(driver), str(script), str(subs)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "frozen scorer" in proc.stderr, proc.stderr


# --------------------------------------------------------------------------- #
# Fairness review (r4): a symlinked folder named front.<ext>, several front.*,
# and the extension precedence the scorer picks must match the scorer exactly.
# --------------------------------------------------------------------------- #


def test_directory_symlink_named_front_is_an_error(dataset, tmp_path):
    """A *directory* symlink named front.png beside a valid front.jpg must fail, not pass.

    The scorer reads the PNG link (its discovery counts ``is_symlink()``), scores the garment
    invalid/0; the checker used to skip it as a directory and exit 0.
    """
    info = dataset
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    _mock.synth_image(d / "front.jpg", 11)
    real_dir = tmp_path / "elsewhere"
    real_dir.mkdir()
    os.symlink(real_dir, d / "front.png")  # a directory symlink with the front stem
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "symlink" in out, out
    assert "front.png" in out, out


def test_two_front_files_are_an_error_naming_the_scorer_pick(dataset, tmp_path):
    """Two front.* with different extensions is an error that names the file the scorer reads."""
    info = dataset
    subs = tmp_path / "submissions"
    d = subs / "teamA" / "d001"
    d.mkdir(parents=True)
    _mock.synth_image(d / "front.jpg", 12)
    _mock.synth_image(d / "front.png", 13)
    proc = _mock.run_cmd(["scripts/make_submission.py", "--submissions", subs,
                          "--dataset", info["root"]], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "multiple front.* files" in out, out
    assert "front.png" in out, out  # png < jpg, so this is the file the scorer would read
    assert "front.jpg" in out, out  # ... and the stale one, so the fix is obvious


def test_front_extension_precedence_matches_the_scorer(dataset, tmp_path):
    """The order used to name the scorer's pick must be the scorer's own IMAGE_EXTS."""
    code = (
        "import pathlib, sys\n"
        "sys.path.insert(0, str(pathlib.Path('scripts')))\n"
        "import make_submission as m\n"
        "from sartiq_score.batch import IMAGE_EXTS\n"
        "assert tuple(m.SCORER_IMAGE_EXTS) == tuple(IMAGE_EXTS), (m.SCORER_IMAGE_EXTS, IMAGE_EXTS)\n"
        "print('|'.join(m.SCORER_IMAGE_EXTS))\n"
    )
    proc = _mock.run_cmd(["-c", code], cwd=_mock.REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == ".png|.jpg|.jpeg|.webp|.heic"
