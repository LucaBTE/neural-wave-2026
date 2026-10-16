#!/usr/bin/env python3
"""Validate a submission folder before handing it in.

Checks the fixed layout ``<submissions>/<team>/<garment_id>/front.<ext>`` against the
dataset ids and the image rules the frozen scorer enforces.

    python scripts/make_submission.py --submissions submissions
    python scripts/make_submission.py --submissions submissions --team ghostbusters --zip

Run it with the same interpreter that has the scorer installed (the repo's
virtualenv interpreter); the error message below prints the exact one to use.

Checks:
  * garment ids match the dev/test ``inputs.jsonl`` ids: an unknown id is an error; a
    dev/test manifest that cannot be found, is empty, or has a malformed row is an error
    (with ``file:line``) unless you pass ``--no-dataset``;
  * the required ``front`` view is present (extra views -> warning);
  * extension is one of png / jpg / jpeg / webp;
  * *any* symlink anywhere in the team tree -- a team folder, a garment folder, a file, or a
    directory symlink -- is an error, and it is detected before any directory is skipped: the
    scorer never follows a symlink and would score the garment 0;
  * exactly one ``front.*`` per garment: several are an error that names the file the frozen
    scorer would actually read, using the scorer's own extension precedence
    (``png < jpg < jpeg < webp < heic``);
  * helper directories the hosted baseline's dry run creates (``_requests/``) are not teams and
    are skipped; any other underscore-prefixed folder (``_mylab``) is an ordinary team;
  * the file decodes fully and passes the frozen scorer's invalid rules: readable,
    <= 100 MB, short side >= 256 px, long side <= 12000 px, aspect <= 3:1, a single
    still frame, and not a flat field (whole-image luma standard deviation < 4/255).

The image checks come from the frozen scorer itself (``sartiq_score.io`` for decoding
and alpha-on-white compositing, ``sartiq_score.aggregate`` for the flat-field gate). The
script refuses to run without that scorer: a hand-rolled decode would disagree with
official scoring, and a validator that disagrees with the scorer is worse than none.

Exit code is 0 when there are no errors (warnings are fine), 1 on validation errors, and
2 when the frozen scorer is not importable. ``--zip`` also packs each checked team into
``<team>.zip`` here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ALLOWED_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
REQUIRED_VIEW = "front"
DEFAULT_DATASET = "../sartiq_neural_wave_2026"
MAX_BYTES = 100 * 1024 * 1024

#: Helper directories that live under the submission root but are not teams. The hosted baseline's
#: ``--dry-run`` writes request payloads to ``<out>/_requests/``; discovery must not mistake that
#: for a team and report errors. Only these explicitly named directories are reserved: any other
#: underscore-prefixed folder (e.g. ``_mylab``) is an ordinary team, matching the scorer.
RESERVED_DIRS: frozenset[str] = frozenset({"_requests"})

_here = Path(__file__).resolve()
for _cand in (_here.parents[1] / "scorer",):  # <repo>/scorer
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

# The validator must apply the SAME rules the frozen scorer applies, so it uses the
# scorer's own decoder and gate. There is deliberately no Pillow fallback: a hand-rolled
# decode disagreed with the scorer on alpha compositing and on missing dependencies.
_scorer_import_error: str | None = None
try:
    from sartiq_score.io import InvalidImage as _InvalidImage  # type: ignore
    from sartiq_score.io import load_image as _load_image  # type: ignore
    from sartiq_score.aggregate import (  # type: ignore
        UNIFORM_LUMA_STD as _UNIFORM_LUMA_STD,
        is_uniform_submission as _is_uniform_submission,
        luma_std as _luma_std,
    )
    # The scorer's own extension precedence, so "which front.* does the scorer read?" is answered
    # by the scorer, not by a hand-copied order that could drift (png < jpg < jpeg < webp < heic).
    from sartiq_score.batch import IMAGE_EXTS as _SCORER_IMAGE_EXTS  # type: ignore
except Exception as _exc:  # noqa: BLE001
    _scorer_import_error = f"{type(_exc).__name__}: {_exc}"
    _load_image = None
    _InvalidImage = Exception
    _UNIFORM_LUMA_STD = 4.0
    _is_uniform_submission = None
    _luma_std = None
    _SCORER_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".heic")

UNIFORM_LUMA_STD = float(_UNIFORM_LUMA_STD)
#: Every extension the frozen scorer recognises for a view, in its own precedence order. A
#: submission may only use :data:`ALLOWED_EXTS`, but a ``front.*`` with a scorer extension this
#: script rejects (e.g. HEIC) still counts when naming the file the scorer would read.
SCORER_IMAGE_EXTS: tuple[str, ...] = tuple(_SCORER_IMAGE_EXTS)


def require_frozen_scorer() -> None:
    """Fail clearly (exit 2) when the frozen scorer cannot be imported."""
    if _scorer_import_error is None:
        return
    print(
        "error: make_submission.py needs the frozen scorer "
        "(sartiq_score.io, sartiq_score.aggregate and sartiq_score.batch) so its checks match "
        f"official scoring, but the import failed: {_scorer_import_error}\n"
        "Run it with the repo's interpreter and the scorer importable, e.g.:\n"
        f'  "{sys.executable}" scripts/make_submission.py ...\n'
        "There is no Pillow fallback on purpose: a decode that disagrees with the "
        "scorer could silently pass files the event rejects.",
        file=sys.stderr,
    )
    raise SystemExit(2)


class ManifestError(Exception):
    """A dev/test ``inputs.jsonl`` that cannot be trusted (empty or malformed)."""


def load_dataset_ids(dataset_root: Path) -> dict[str, set[str]]:
    """Return ``{split: ids}`` from dev/test ``inputs.jsonl``.

    A missing manifest is skipped (the caller decides whether that is fatal). A manifest
    that exists but is empty, or has a malformed row, raises :class:`ManifestError` with
    ``file:line``, so an unusable manifest can never silently disable id checking.
    """
    out: dict[str, set[str]] = {}
    for split in ("dev", "test"):
        ids: set[str] = set()
        manifest = dataset_root / split / "inputs.jsonl"
        if manifest.is_file():
            for lineno, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"{manifest}:{lineno}: invalid JSON ({exc})") from exc
                if not isinstance(row, dict):
                    raise ManifestError(
                        f"{manifest}:{lineno}: expected a JSON object, got {type(row).__name__}"
                    )
                gid = row.get("garment_id") or row.get("pair_id")
                if not gid:
                    raise ManifestError(f"{manifest}:{lineno}: row needs a garment_id (or pair_id)")
                ids.add(str(gid))
            if not ids:
                raise ManifestError(f"{manifest}: manifest exists but declares no garment ids")
        out[split] = ids
    return out


def check_image(path: Path) -> dict:
    """Full-decode check of one submission file, using the frozen scorer's own rules."""
    problems: list[str] = []
    warnings: list[str] = []
    if path.is_symlink():
        return {"size": None, "luma_std": None,
                "problems": ["front is a symlink => INVALID (copy the real file)"], "warnings": []}
    try:
        nbytes = path.stat().st_size
    except OSError as exc:
        return {"size": None, "luma_std": None,
                "problems": [f"front is unreadable ({exc}) => INVALID"], "warnings": []}
    if nbytes > MAX_BYTES:
        problems.append(f"front > {MAX_BYTES // (1024 * 1024)} MB => INVALID (score 0)")

    try:
        loaded = _load_image(path)  # type: ignore[misc]
    except _InvalidImage as exc:  # type: ignore[misc]
        detail_val = getattr(exc, "detail", "")
        detail = f" ({detail_val})" if detail_val else ""
        problems.append(
            f"front is invalid per the scorer ({getattr(exc, 'reason', 'invalid')}{detail}) "
            "=> INVALID (score 0)")
        return {"size": None, "luma_std": None, "problems": problems, "warnings": warnings}
    except Exception as exc:  # noqa: BLE001
        problems.append(f"front is unreadable ({type(exc).__name__}: {exc}) => INVALID")
        return {"size": None, "luma_std": None, "problems": problems, "warnings": warnings}

    size = [int(loaded.width), int(loaded.height)]
    if loaded.alpha_present:
        warnings.append("front has an alpha/transparency channel; it is composited on "
                        "pure white (255,255,255)")
    std = float(_luma_std(loaded.rgb))  # type: ignore[misc]
    if _is_uniform_submission(loaded.rgb):  # type: ignore[misc]
        problems.append(f"front is a flat field (whole-image luma std {std:.2f} < "
                        f"{UNIFORM_LUMA_STD}/255) => INVALID (score 0)")
    return {"size": size, "luma_std": round(std, 3) if std == std else None,
            "problems": problems, "warnings": warnings}


def _tree_symlinks(root: Path) -> list[Path]:
    """Every symlink under ``root`` -- files and directories -- checked before any directory is
    skipped.

    The scorer's discovery reads ``is_symlink() or is_file()``, so a symlink named like an image
    counts even when it points at a directory. ``os.walk(followlinks=False)`` lists a symlinked
    directory without descending into it, so this never follows a link out of the team tree; it
    just records it.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                found.append(path)
    return sorted(found)


def _front_and_extras(directory: Path) -> tuple[list[Path], list[str], list[str]]:
    """Ordered ``front.*`` candidates for one garment dir.

    Returns ``(candidates, extra_view_stems, disallowed_names)``. A candidate is any entry the
    frozen scorer would treat as this view's submission image: a real file or a symlink (the
    scorer reads ``is_symlink() or is_file()``), with an extension it recognises, in the scorer's
    own ``IMAGE_EXTS`` precedence order, so ``candidates[0]`` is exactly the file the scorer
    reads. ``disallowed_names`` are ``front.*`` entries a submission may not use: an extension the
    scorer reads but the rules forbid (HEIC) or one it ignores (.bmp).

    The symlink bit is tested before a directory is skipped, so a directory symlink named
    ``front.png`` is a candidate here (and an error upstream) instead of being hidden as a folder.
    """
    candidates: list[Path] = []
    extras: list[str] = []
    disallowed: list[str] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        if not (entry.is_symlink() or entry.is_file()):
            continue  # a real directory (or other non-image entry): the scorer skips it too
        ext = entry.suffix.lower()
        if entry.stem == REQUIRED_VIEW:
            if ext in ALLOWED_EXTS:
                candidates.append(entry)
            elif ext in SCORER_IMAGE_EXTS:
                candidates.append(entry)  # the scorer reads it; a submission may not submit it
                disallowed.append(entry.name)
            else:
                disallowed.append(entry.name)
        elif ext in ALLOWED_EXTS:
            extras.append(entry.stem)
    candidates.sort(key=lambda p: SCORER_IMAGE_EXTS.index(p.suffix.lower()))
    return candidates, sorted(set(extras)), disallowed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="make_submission.py",
        description="Validate the submission layout (ids + image constraints) before handing in.",
    )
    ap.add_argument("--submissions", required=True, help="submission root (contains <team>/ folders)")
    ap.add_argument("--team", default=None, help="only check this team folder")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"dataset root holding dev/ and test/ (default: {DEFAULT_DATASET})")
    ap.add_argument("--no-dataset", dest="no_dataset", action="store_true",
                    help="skip garment-id checks (use only when the dev/test manifests are absent)")
    ap.add_argument("--zip", dest="do_zip", action="store_true",
                    help="also pack each checked team into <team>.zip (written here)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    require_frozen_scorer()

    root = Path(args.submissions)
    if not root.is_dir():
        raise SystemExit(f"error: --submissions is not a directory: {root}")

    dataset_root = Path(args.dataset)
    errors: list[str] = []
    warnings: list[str] = []

    split_ids: dict[str, set[str]] = {}
    if not args.no_dataset:
        try:
            split_ids = load_dataset_ids(dataset_root)
        except ManifestError as exc:
            errors.append(str(exc))
        except OSError as exc:
            errors.append(f"{dataset_root}: cannot read the dev/test manifests ({exc})")
    dev_ids = split_ids.get("dev", set())
    test_ids = split_ids.get("test", set())
    known_ids = dev_ids | test_ids

    if not args.no_dataset:
        any_manifest = ((dataset_root / "dev" / "inputs.jsonl").is_file()
                        or (dataset_root / "test" / "inputs.jsonl").is_file())
        if not known_ids and not any_manifest:
            errors.append(
                f"no dev/test inputs.jsonl under {dataset_root}; garment-id checks cannot run "
                f"(pass --dataset <root>, or --no-dataset to skip them)"
            )
    ids_checkable = not args.no_dataset and bool(known_ids)

    # ---- teams ---------------------------------------------------------- #
    if args.team:
        candidate = root / args.team
        if candidate.is_symlink():
            errors.append(f"{args.team}: team folder is a symlink => INVALID (use a real folder)")
            team_dirs = [candidate]
        elif candidate.is_dir():
            team_dirs = [candidate]
        else:
            team_dirs = [root]
            warnings.append(f"--team {args.team!r} not found under {root}; checking {root} as a single team")
    else:
        entries = sorted(root.iterdir())
        # sartiq_score.batch.HIDDEN treats only dot-prefixed entries as hidden, and the scorer
        # accepts an underscore-prefixed folder as an ordinary team name, so the validator must
        # check it too. Only dot-prefixed folders are skipped, plus the explicitly named helper
        # directories (e.g. the baseline's _requests/) that are not teams.
        team_dirs = [d for d in entries
                     if d.is_dir() and not d.name.startswith(".") and d.name not in RESERVED_DIRS]
        if not team_dirs:
            raise SystemExit(
                f"error: no team folders under {root}. Expected {root}/<team>/<garment_id>/front.png"
            )
        loose = [f.name for f in entries if f.is_file() and f.suffix.lower() in ALLOWED_EXTS]
        if loose:
            errors.append(
                f"{len(loose)} image file(s) directly under {root} (e.g. {loose[0]}): "
                "the layout is <team>/<garment_id>/front.<ext>"
            )

    # ---- walk ------------------------------------------------------------ #
    report: dict[str, dict] = {}
    submitted_by_team: dict[str, set[str]] = {}
    n_images = 0
    zips: list[Path] = []
    for team_dir in team_dirs:
        team = team_dir.name
        garments: dict[str, dict] = {}
        team_submitted: set[str] = set()
        submitted_by_team[team] = team_submitted
        if team_dir.is_symlink():
            errors.append(f"{team}: team folder is a symlink => INVALID (use a real folder)")
            report[team] = {"garments": garments, "n_garments": 0, "n_images": 0,
                            "dev_submitted": 0, "test_submitted": 0}
            continue
        # Reject any symlink anywhere in the team tree, before the walk below skips directories:
        # a directory symlink named like an image would otherwise be hidden, while the scorer reads
        # it (is_symlink() counts) and scores the garment 0.
        for link in _tree_symlinks(team_dir):
            rel = link.relative_to(team_dir).as_posix()
            errors.append(
                f"{team}/{rel}: symlink => INVALID (copy the real file; the scorer never follows "
                f"a symlink)"
            )
        garment_dirs = [d for d in sorted(team_dir.iterdir())
                        if not d.name.startswith(".") and (d.is_dir() or d.is_symlink())]
        if not garment_dirs:
            errors.append(f"{team}: no garment folders")
        for gdir in garment_dirs:
            gid = gdir.name
            entry: dict = {"views": {}, "problems": []}
            if ids_checkable and gid not in known_ids:
                errors.append(
                    f"{team}/{gid}: garment id is not in the dev/test inputs => ERROR "
                    f"(scores 0 in the event)"
                )
            if gdir.is_symlink():
                entry["problems"].append("garment folder is a symlink => INVALID (copy the real files)")
                garments[gid] = entry
                errors.extend(f"{team}/{gid}: {p}" for p in entry["problems"])
                continue
            candidates, extras, disallowed = _front_and_extras(gdir)
            if disallowed:
                entry["problems"].append(
                    f"front view has a disallowed extension ({', '.join(disallowed)}) "
                    f"=> INVALID (use {'/'.join(e.lstrip('.') for e in ALLOWED_EXTS)})"
                )
            if not candidates:
                entry["problems"].append("no front view => INVALID (front.<png|jpg|jpeg|webp> required)")
            else:
                picked = candidates[0]  # the file the frozen scorer itself would read
                if len(candidates) > 1:
                    order = " < ".join(e.lstrip(".") for e in SCORER_IMAGE_EXTS)
                    entry["problems"].append(
                        f"multiple front.* files ({', '.join(p.name for p in candidates)}); "
                        f"the scorer reads {picked.name} (extension order {order}) "
                        f"=> ERROR (keep only {picked.name})"
                    )
                if picked.is_symlink():
                    # Symlink check before any stat(): a dangling link must report cleanly.
                    entry["problems"].append("front is a symlink => INVALID (copy the real file)")
                else:
                    checked = check_image(picked)
                    info: dict = {"file": picked.name}
                    try:
                        info["bytes"] = picked.stat().st_size
                    except OSError as exc:
                        info["bytes"] = None
                        entry["problems"].append(f"front is unreadable ({exc}) => INVALID")
                    if checked["size"]:
                        info["size"] = checked["size"]
                    if checked["luma_std"] is not None:
                        info["luma_std"] = checked["luma_std"]
                    entry["problems"].extend(checked["problems"])
                    warnings.extend(f"{team}/{gid}: {w}" for w in checked["warnings"])
                    entry["views"][REQUIRED_VIEW] = info
                    team_submitted.add(gid)
                    n_images += 1
            if extras:
                warnings.append(f"{team}/{gid}: extra view(s) {', '.join(extras)} ignored (front is required)")
            garments[gid] = entry
            for problem in entry["problems"]:
                errors.append(f"{team}/{gid}: {problem}")
        report[team] = {
            "garments": garments, "n_garments": len(garments),
            "n_images": sum(len(g["views"]) for g in garments.values()),
            "dev_submitted": len(team_submitted & dev_ids),
            "test_submitted": len(team_submitted & test_ids),
        }

        if args.do_zip:
            zip_path = Path(f"{team}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(team_dir.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        zf.write(path, path.relative_to(root).as_posix())
            zips.append(zip_path)

    if ids_checkable:
        for team, submitted in sorted(submitted_by_team.items()):
            for gid in sorted(known_ids - submitted):
                warnings.append(
                    f"{team}: garment {gid!r} from the dev/test inputs was not submitted (scores 0)"
                )

    # ---- output ---------------------------------------------------------- #
    if args.json:
        print(json.dumps({
            "teams": report, "errors": errors, "warnings": warnings,
            "dataset": {
                "root": str(dataset_root),
                "checked": not args.no_dataset,
                "dev_total": len(dev_ids), "test_total": len(test_ids),
            },
        }, indent=2, sort_keys=True))
    else:
        for team, rep in sorted(report.items()):
            print(f"{team}: {rep['n_garments']} garments, {rep['n_images']} images")
            for gid, g in sorted(rep["garments"].items()):
                flag = "!!" if g["problems"] else "ok"
                extra = "  " + "; ".join(g["problems"]) if g["problems"] else ""
                print(f"  [{flag}] {gid}: {', '.join(sorted(g['views'])) or '-'}{extra}")
        print()
        for team, rep in sorted(report.items()):
            if not args.no_dataset and (dev_ids or test_ids):
                print(f"coverage {team}: dev {rep['dev_submitted']}/{len(dev_ids)}, "
                      f"test {rep['test_submitted']}/{len(test_ids)}")
        for w in warnings:
            print(f"warning: {w}")
        for e in errors:
            print(f"ERROR:   {e}")
        print()
        print(f"make_submission.py: {len(report)} team(s), {n_images} image(s), "
              f"{len(errors)} error(s), {len(warnings)} warning(s)")
        for zip_path in zips:
            print(f"packed: {zip_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
