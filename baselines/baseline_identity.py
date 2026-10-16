#!/usr/bin/env python3
"""Identity baseline: the floor.

Copies the first input of every pair straight through as the "prediction", into the
official submission layout:

    <out>/<team>/<id>/<view>.<ext>

Nothing is re-encoded, so this measures exactly what a bare, unprocessed capture
scores under the official scorer. Any real pipeline has to beat it.

It reads either manifest the dataset ships, and accepts both:

    train/pairs.jsonl        one row per pair:    {pair_id, inputs[], target}
    dev|test/inputs.jsonl    one row per garment: {garment_id, view, inputs[]}

Paths inside a row are relative to the dataset root (the folder that holds
``train/``, ``dev/`` and ``test/``). ``--dataset`` overrides where that root is; by
default it comes from the manifest: a sibling ``meta.json`` when one is present
(scripts/make_holdout.py writes one), otherwise the manifest's parent's parent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".heic")
DEFAULT_VIEW = "front"
_SPLIT_DIRS = {"train", "dev", "test"}


# --------------------------------------------------------------------------- #
# manifest loading (shared with baseline_nanobanana.py)
# --------------------------------------------------------------------------- #

def resolve_dataset_root(manifest: Path, explicit: str | None) -> Path:
    """Dataset root for a manifest: explicit flag, sibling meta.json, or split parent."""
    if explicit:
        return Path(explicit)
    meta = manifest.parent / "meta.json"
    if meta.is_file():
        try:
            recorded = json.loads(meta.read_text(encoding="utf-8-sig")).get("dataset_root")
        except (OSError, ValueError):
            recorded = None
        if recorded:
            root = Path(str(recorded))
            return root if root.is_absolute() else (meta.parent / root)
    if manifest.parent.name in _SPLIT_DIRS:
        return manifest.parent.parent
    return manifest.parent


def _rel(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def load_rows(manifest: Path, dataset_root: Path) -> list[dict]:
    """Return ``[{id, view, inputs[Path], target Path|None}, ...]`` for a manifest.

    Works for ``train/pairs.jsonl`` (``pair_id``) and for ``dev``/``test``
    ``inputs.jsonl`` (``garment_id``). Target is optional: dev/test rows do not have one.
    """
    if not manifest.is_file():
        raise SystemExit(f"error: manifest not found: {manifest}")
    rows: list[dict] = []
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: {manifest}:{lineno}: invalid JSON ({exc})") from exc
        pid = row.get("pair_id") or row.get("garment_id")
        inputs = row.get("inputs") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        if not pid or not inputs:
            raise SystemExit(
                f"error: {manifest}:{lineno}: a row needs pair_id/garment_id and a non-empty inputs[]"
            )
        rows.append({
            "id": str(pid),
            "view": str(row.get("view") or DEFAULT_VIEW),
            "inputs": [_rel(str(x), dataset_root) for x in inputs],
            "target": _rel(str(row["target"]), dataset_root) if row.get("target") else None,
        })
    if not rows:
        raise SystemExit(f"error: {manifest} declared no rows")
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="baseline_identity.py",
        description="Identity baseline: copy the first input of every pair through as the prediction.",
    )
    ap.add_argument("--manifest", required=True,
                    help="a dataset manifest: train/pairs.jsonl or dev|test/inputs.jsonl")
    ap.add_argument("--out", required=True, help="prediction root (the submission layout is written here)")
    ap.add_argument("--team", required=True, help="team folder name, e.g. identity")
    ap.add_argument("--dataset", default=None,
                    help="dataset root (folder holding train/, dev/, test/); default: the manifest's root")
    ap.add_argument("--view", default=DEFAULT_VIEW,
                    help=f"view name for rows without a 'view' field (default: {DEFAULT_VIEW})")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    dataset_root = resolve_dataset_root(manifest, args.dataset)
    try:
        rows = load_rows(manifest, dataset_root)
    except SystemExit as exc:
        # helpful hint when the dataset root was not found
        print(exc, file=sys.stderr)
        return 2

    out_root = Path(args.out) / args.team
    written = 0
    for row in rows:
        src = row["inputs"][0]
        if not src.is_file():
            raise SystemExit(f"error: input image not found: {src}\n"
                             f"hint: pass --dataset <root> if the dataset is not next to the repo")
        view = row["view"] if row["view"] != DEFAULT_VIEW else args.view
        ext = src.suffix.lower() or ".png"
        dst = out_root / row["id"] / f"{view}{ext}"
        if args.dry_run:
            print(f"{src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written += 1

    if args.dry_run:
        print(f"\n(dry run) {len(rows)} file(s) would be written under {out_root}")
        return 0
    print(f"identity baseline: {written} prediction(s) -> {out_root}")
    print(f'score them with:  "{sys.executable}" scripts/eval_val.py '
          f"--pairs <val manifest> --preds {args.out} --team {args.team} --out val_out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
