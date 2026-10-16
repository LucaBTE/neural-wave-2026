#!/usr/bin/env python3
"""Self-score predictions against a holdout (or any pairs manifest) with the official scorer.

Reads pairs that carry a ``target``, finds your prediction for each, and runs the frozen
scorer over the whole set. The output is the same ``leaderboard.csv`` / ``scores.jsonl`` /
``review.csv`` an official run produces, plus a per-pair ``pairs.csv`` and a
``summary.json`` whose mean comes from the official, garment-weighted aggregate.

    # 1. carve a holdout out of the example pairs (see scripts/make_holdout.py)
    python scripts/make_holdout.py --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout

    # 2. predict for the holdout and score it
    python baselines/baseline_identity.py --manifest holdout/val.jsonl --out predictions --team identity
    python scripts/eval_val.py --pairs holdout/val.jsonl --preds predictions --team identity --out val_out

Run both commands with the *same* interpreter that has the scorer installed
(the repo's virtualenv interpreter). Any terminal prints the exact interpreter to
use, so copy the command it prints rather than retyping the path.

Required views are ``front`` unless a row carries a ``view`` field. A pair with no
prediction is scored as a missing slot = 0, exactly like the real event. ``--dataset``
overrides where the dataset root is; by default it comes from a sibling ``meta.json``
(written by make_holdout.py) or from the manifest's parent's parent.

Prediction discovery reuses the frozen scorer's own rules
(``sartiq_score.batch.discover_subs``): the file extension is matched case-insensitively,
``.heic`` is accepted, dot-prefixed entries are hidden, and a symlinked team folder or
garment folder is never followed. A symlinked prediction file is rejected too (the scorer
treats a symlinked submission as invalid, score 0), so it is never scored here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

DEFAULT_VIEW = "front"
_SPLIT_DIRS = {"train", "dev", "test"}

#: Column order of the frozen scorer's leaderboard.csv.
LEADERBOARD_COLS = (
    "rank", "team", "score", "colour", "lpips", "clip", "geometry", "ssim",
    "pairs_scored", "pairs_missing", "garments_scored", "garments_missing",
    "n_review", "n_invalid",
)


def _scorer_dir() -> Path:
    env = os.environ.get("SCORER_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "scorer"


def _ensure_scorer_path() -> str:
    """Put the frozen scorer on ``sys.path`` (idempotent), or exit with guidance."""
    scorer_dir = str(_scorer_dir())
    if scorer_dir not in sys.path:
        sys.path.insert(0, scorer_dir)
    return scorer_dir


def _import_batch():
    """Import the frozen batch module, failing with a clear message when it is absent."""
    _ensure_scorer_path()
    try:
        from sartiq_score.batch import discover_subs, run_batch  # type: ignore

        return discover_subs, run_batch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"error: could not import the scorer from {_scorer_dir()} ({exc}).\n"
            "Run this script with the scorer environment and a clean import path, e.g.:\n"
            f'  "{sys.executable}" scripts/eval_val.py ...'
        )


def _import_scorer_setup():
    """``(build_scorer, MODEL_DIR)`` from the frozen scorer, or exit with guidance."""
    _ensure_scorer_path()
    try:
        from sartiq_score import MODEL_DIR  # type: ignore
        from sartiq_score.cli import build_scorer  # type: ignore

        return build_scorer, MODEL_DIR
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"error: could not import the scorer from {_scorer_dir()} ({exc}).\n"
            "Run this script with the scorer environment and a clean import path, e.g.:\n"
            f'  "{sys.executable}" scripts/eval_val.py ...'
        )


def resolve_dataset_root(pairs: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    meta = pairs.parent / "meta.json"
    if meta.is_file():
        try:
            recorded = json.loads(meta.read_text(encoding="utf-8-sig")).get("dataset_root")
        except (OSError, ValueError):
            recorded = None
        if recorded:
            root = Path(str(recorded))
            return root if root.is_absolute() else (meta.parent / root)
    if pairs.parent.name in _SPLIT_DIRS:
        return pairs.parent.parent
    return pairs.parent


def load_pairs(pairs: Path, dataset_root: Path) -> list[dict]:
    """Return ``[{pair_id, view, target}, ...]`` from a pairs manifest (must have targets)."""
    if not pairs.is_file():
        raise SystemExit(f"error: --pairs not found: {pairs}")
    rows: list[dict] = []
    for lineno, line in enumerate(pairs.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: {pairs}:{lineno}: invalid JSON ({exc})") from exc
        pid = row.get("pair_id") or row.get("garment_id")
        target = row.get("target")
        if not pid or not target:
            raise SystemExit(f"error: {pairs}:{lineno}: needs pair_id/garment_id and target")
        tpath = Path(str(target))
        rows.append({
            "pair_id": str(pid),
            "view": str(row.get("view") or DEFAULT_VIEW),
            "target": tpath if tpath.is_absolute() else dataset_root / tpath,
        })
    if not rows:
        raise SystemExit(f"error: {pairs} declared no pairs")
    return rows


def team_view_files(subs_root: Path, team: str) -> tuple[dict[str, dict[str, Path]], list[str]]:
    """``(garment -> view -> path, notes)`` for one team, via the frozen discovery.

    Uses ``sartiq_score.batch.discover_subs`` so ``front.PNG`` and other case/extension
    variants are found exactly as official scoring finds them, and symlinked team or
    garment folders are never followed.
    """
    discover_subs, _run_batch = _import_batch()
    try:
        subs = discover_subs(subs_root)
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {exc}")
    notes: list[str] = []
    team_dir = subs_root / team
    if team_dir.is_symlink():
        notes.append(
            f"{team}: team folder is a symlink; the scorer never follows it, so nothing is scored"
        )
    elif not team_dir.is_dir():
        notes.append(f"{team}: no such team folder under {subs_root}")
    return subs.get(team, {}), notes


def write_pairs_csv(records: list[dict], path: Path) -> None:
    cols = ["pair_id", "view", "status", "score", "colour", "lpips", "clip", "geometry",
            "ssim", "flags", "prediction"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            subs = rec.get("subscores") or {}
            writer.writerow({
                "pair_id": rec.get("garment", ""),
                "view": rec.get("view", ""),
                "status": rec.get("status", ""),
                "score": f"{float(rec.get('score') or 0.0):.4f}",
                "colour": f"{float(subs.get('colour') or 0.0):.6f}",
                "lpips": f"{float(subs.get('lpips') or 0.0):.6f}",
                "clip": f"{float(subs.get('clip') or 0.0):.6f}",
                "geometry": f"{float(subs.get('geometry') or 0.0):.6f}",
                "ssim": f"{float(subs.get('ssim') or 0.0):.6f}",
                "flags": "|".join(map(str, rec.get("flags") or [])),
                "prediction": rec.get("sub") or "",
            })


def write_leaderboard_utf8(rows: list[dict], path: Path) -> None:
    """Rewrite the leaderboard with the frozen columns, as UTF-8 csv (never text replace)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_from_aggregate(rows: list[dict], records: list[dict], team: str) -> dict:
    """Build the summary from the official (garment-weighted, rounded) aggregate row.

    Falls back to counting records only when no aggregate row is available.
    """
    row = None
    for candidate in rows:
        if candidate.get("team") in (team, "manifest"):
            row = candidate
            break
    if row is None and rows:
        row = rows[0]

    if row is not None:
        scored = int(row.get("pairs_scored") or 0)
        review = int(row.get("n_review") or 0)
        invalid = int(row.get("n_invalid") or 0)
        return {
            "team": team,
            "pairs": len(records),
            "scored": scored,
            "missing": int(row.get("pairs_missing") or 0),
            "review": review,
            "invalid": invalid,
            "ok": max(0, scored - review - invalid),
            "garments_scored": int(row.get("garments_scored") or 0),
            "garments_missing": int(row.get("garments_missing") or 0),
            "mean_score": float(row.get("score") or 0.0),
        }

    counts = {"ok": 0, "review": 0, "invalid": 0, "missing": 0}
    for rec in records:
        status = "missing" if rec.get("missing") else str(rec.get("status") or "ok")
        counts[status] = counts.get(status, 0) + 1
    scores = [float(rec.get("score") or 0.0) for rec in records]
    mean = sum(scores) / len(scores) if scores else 0.0
    return {
        "team": team,
        "pairs": len(records),
        "scored": len(records) - counts["missing"],
        "missing": counts["missing"],
        "review": counts["review"],
        "invalid": counts["invalid"],
        "ok": counts["ok"],
        "garments_scored": 0,
        "garments_missing": 0,
        "mean_score": round(mean, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval_val.py",
        description="Score predictions against a pairs manifest (a holdout) with the official scorer.",
    )
    ap.add_argument("--pairs", required=True,
                    help="pairs manifest with targets, e.g. holdout/val.jsonl or train/pairs.jsonl")
    ap.add_argument("--preds", required=True, help="prediction root (holds <team>/<id>/<view>.<ext>)")
    ap.add_argument("--team", required=True, help="team folder inside --preds")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--dataset", default=None,
                    help="dataset root (folder holding train/, dev/, test/); default: the pairs manifest's root")
    ap.add_argument("--jobs", type=int, default=1, help="parallel scorer processes (try 4)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--debug-sheets", action="store_true", help="write per-pair contact sheets")
    args = ap.parse_args(argv)

    _discover_subs, run_batch = _import_batch()
    build_scorer, model_dir = _import_scorer_setup()

    pairs_path = Path(args.pairs)
    dataset_root = resolve_dataset_root(pairs_path, args.dataset)
    pairs = load_pairs(pairs_path, dataset_root)
    if args.limit is not None:
        pairs = pairs[: max(0, args.limit)]

    # Fail loudly *before* any scoring when the dataset root is wrong: a missing
    # target would otherwise waste a long model run and read as legitimate zeros.
    missing = [p["target"] for p in pairs if not Path(p["target"]).is_file()]
    if missing:
        print(
            f"error: dataset target missing: {Path(missing[0]).resolve()}\n"
            f"resolved dataset root: {dataset_root.resolve()}\n"
            f'pass --dataset "/path/to/sartiq_neural_wave_2026" (that directory must '
            f"directly contain train/, dev/ and test/; check for an extra ZIP folder).\n"
            f"{len(missing)} of {len(pairs)} target(s) were not found.",
            file=sys.stderr,
        )
        return 2

    # Build the scorer once, in the parent, so a broken model cache / proxy fails
    # here with exit 2 instead of turning into a run of invalid zero scores.
    try:
        build_scorer()
    except Exception as exc:  # noqa: BLE001
        print(
            f"Scorer setup failed: {type(exc).__name__}: {exc}. "
            f"Model cache: {Path(model_dir).expanduser().resolve()}. "
            "Check network/proxy or write permission. No results were produced.",
            file=sys.stderr,
        )
        return 2

    preds_root = Path(args.preds)
    if not preds_root.is_dir():
        raise SystemExit(f"error: --preds is not a directory: {preds_root}")

    team_files, notes = team_view_files(preds_root, args.team)
    for note in notes:
        print(f"  ! {note}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "val_manifest.jsonl"

    found = 0
    symlinked = 0
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            gid = pair["pair_id"]
            view = pair["view"]
            pred = team_files.get(gid, {}).get(view)
            row = {"garment_id": gid, "view": view, "ref": str(pair["target"])}
            if pred is None:
                print(f"  ! {gid}/{view}: no prediction -> scores 0 (missing slot)")
            elif pred.is_symlink():
                symlinked += 1
                print(f"  ! {gid}/{view}: prediction is a symlink -> rejected "
                      f"(invalid, scores 0)")
            else:
                row["sub"] = str(pred)
                found += 1
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"pairs            : {len(pairs)}")
    print(f"predictions found: {found}" + ("" if found == len(pairs)
          else f"  ({len(pairs) - found} score 0 as a missing/rejected slot)"))
    if symlinked:
        print(f"rejected         : {symlinked} symlinked prediction(s) (score 0)")
    print(f"manifest         : {manifest_path}")

    summary = run_batch(
        manifest=str(manifest_path),
        out_dir=str(out),
        debug_sheets=args.debug_sheets,
        jobs=args.jobs,
    )

    records = summary["records"]
    write_pairs_csv(records, out / "pairs.csv")

    summary_obj = summary_from_aggregate(summary["rows"], records, args.team)
    (out / "summary.json").write_text(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")

    # the scorer labels manifest-driven runs "manifest"; show the team the user asked for
    for row in summary["rows"]:
        if row.get("team") == "manifest":
            row["team"] = args.team
    lb = Path(summary["leaderboard"])
    if lb.exists():
        write_leaderboard_utf8(summary["rows"], lb)

    print()
    print(f"mean score : {summary_obj['mean_score']:.2f}  ({summary_obj['scored']} scored, "
          f"{summary_obj['missing']} missing, {summary_obj['review']} review, "
          f"{summary_obj['invalid']} invalid)")
    print(f"per-pair   : {out / 'pairs.csv'}")
    print(f"summary    : {out / 'summary.json'}")
    print(f"leaderboard: {summary['leaderboard']}")
    print(f"scores     : {summary['scores_jsonl']}")
    print(f"review     : {summary['review_csv']}  ({summary['n_review']} entries)")
    if summary["n_review"]:
        print("             ^ status=review: score kept, a human checks these before publishing")
    if summary["n_invalid"]:
        print(f"note: {summary['n_invalid']} invalid submission(s) scored 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
