#!/usr/bin/env python3
"""Carve a self-evaluation holdout out of the example pairs.

The example pairs (``train/``) are the only split with targets, so it is the only one
you can self-score on. Split it into a keep set and a val (holdout) set, then score your
predictions on the val side with ``scripts/eval_val.py``:

    python scripts/make_holdout.py --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout

Run it with the same interpreter that has the scorer installed (the repo's
virtualenv interpreter); the command printed at the end shows the exact one.

writes, keeping the exact ``train/pairs.jsonl`` row schema (``{pair_id, inputs[],
target}``):

    holdout/train.jsonl    the pairs to tune on (the example pairs minus the holdout)
    holdout/val.jsonl      the pairs you did not tune on
    holdout/meta.json      records the dataset root so eval_val.py can resolve paths

A random split can put the same garment on both sides (the pool repeats products and
even targets), which flatters your self-score. ``--group-by-target`` keeps every pair
that shares an identical target file on the same side; repeated garments ship
byte-identical targets, so this catches the obvious share, but it is a light help, not a
full de-duplication of garments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

DEFAULT_DATASET = "../sartiq_neural_wave_2026"
_SPLIT_DIRS = {"train", "dev", "test"}
_ROW_KEYS = ("pair_id", "inputs", "target")


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pairs(pairs: Path) -> list[dict]:
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
        row = {k: row[k] for k in _ROW_KEYS if k in row}
        if not row.get("pair_id") or not row.get("inputs"):
            raise SystemExit(f"error: {pairs}:{lineno}: a pair needs pair_id and non-empty inputs[]")
        rows.append(row)
    if not rows:
        raise SystemExit(f"error: {pairs} declared no pairs")
    return rows


def _target_key(row: dict, dataset_root: Path, hashes: dict[str, int], *, strict: bool) -> str:
    target = row.get("target")
    if target:
        path = Path(target)
        path = path if path.is_absolute() else dataset_root / path
        try:
            sha = _sha256(path)
            hashes[sha] = hashes.get(sha, 0) + 1
            return sha
        except OSError as exc:
            if strict:
                raise SystemExit(
                    f"error: --group-by-target: cannot read the target of pair "
                    f"{row.get('pair_id')!r}: {path} ({type(exc).__name__}: {exc})"
                ) from exc
    elif strict:
        raise SystemExit(
            f"error: --group-by-target: pair {row.get('pair_id')!r} has no target to group by"
        )
    # no target (or, when not grouping strictly, an unreadable one): a singleton group
    return f"id:{row['pair_id']}"


def split_rows(rows: list[dict], dataset_root: Path, frac: float, seed: int,
               group_by_target: bool) -> tuple[list[dict], list[dict], int]:
    """Return ``(train_rows, val_rows, n_repeated_targets)``."""
    n = len(rows)
    if frac <= 0:
        return list(rows), [], 0
    n_val = max(1, int(round(frac * n)))
    if n > 1:
        n_val = min(n_val, n - 1)
    elif n == 1:
        return list(rows), [], 0

    hashes: dict[str, int] = {}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        # strict grouping only when --group-by-target is on: an unreadable target is then a
        # hard error, never a silent singleton that defeats the grouping.
        key = _target_key(row, dataset_root, hashes, strict=group_by_target)
        groups.setdefault(key, []).append(row)
    repeated = sum(count for count in hashes.values() if count > 1)

    rng = random.Random(seed)
    if not group_by_target:
        order = list(range(n))
        rng.shuffle(order)
        val_idx = set(order[:n_val])
        train = [row for i, row in enumerate(rows) if i not in val_idx]
        val = [row for i, row in enumerate(rows) if i in val_idx]
        return train, val, repeated

    keys = sorted(groups)
    rng.shuffle(keys)
    train, val = [], []
    for key in keys:
        if len(val) < n_val:
            val.extend(groups[key])
        else:
            train.extend(groups[key])
    return train, val, repeated


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="make_holdout.py",
        description="Carve a train/val holdout out of train/pairs.jsonl for self-scoring.",
    )
    ap.add_argument("--pairs", required=True, help="train/pairs.jsonl (the only split with targets)")
    ap.add_argument("--frac", type=float, default=0.1, help="fraction held out for val (default: 0.1)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    ap.add_argument("--out", default="holdout", help="output folder (default: holdout)")
    ap.add_argument("--dataset", default=None,
                    help="dataset root (folder holding train/, dev/, test/); default: the pairs manifest's root")
    ap.add_argument("--group-by-target", action="store_true",
                    help="keep pairs that share an identical target file on the same side (off by default)")
    args = ap.parse_args(argv)

    if not 0.0 <= args.frac <= 1.0:
        raise SystemExit("error: --frac must be between 0 and 1")

    pairs = Path(args.pairs)
    dataset_root = resolve_dataset_root(pairs, args.dataset)
    rows = read_pairs(pairs)
    train, val, repeated = split_rows(rows, dataset_root, args.frac, args.seed, args.group_by_target)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    try:
        root_str = os.path.relpath(str(dataset_root.resolve()), str(out.resolve()))
    except ValueError:  # different drives (Windows) - keep what we were given
        root_str = str(dataset_root)
    meta = {
        "dataset_root": root_str,
        "seed": args.seed,
        "frac": args.frac,
        "n_pairs": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "group_by_target": bool(args.group_by_target),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if repeated and not args.group_by_target:
        print(f"warning: {repeated} pair(s) share their target with another pair; a random split can "
              f"place the same garment on both sides (see --group-by-target).")
    print(f"holdout: {len(train)} train / {len(val)} val pair(s) -> {out}")
    print(f"dataset root: {root_str}  (recorded in {out / 'meta.json'})")
    print(f'score with:  "{sys.executable}" scripts/eval_val.py '
          f"--pairs {out / 'val.jsonl'} --preds predictions --team <team> --out val_out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
