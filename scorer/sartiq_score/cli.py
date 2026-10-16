#!/usr/bin/env python3
"""Command-line interface for the Sartiq reference-similarity scorer (v2).

Usage
-----
::

    python -m sartiq_score.cli pair  --ref REF --sub SUB [--json] [--view VIEW]
    python -m sartiq_score.cli batch --refs REFS_DIR --subs SUBS_DIR --out OUT_DIR
    python -m sartiq_score.cli sanity --refs-glob 'GLOB' --out OUT_DIR [--limit N]
    python -m sartiq_score.cli debug --ref REF --sub SUB --out SHEET.jpg

``SCORER/sartiq_scorer.py`` is the v1-compatible wrapper around ``pair``.

The CLI only talks to the frozen public surface: ``api.score_pair``,
``api.Scorer`` and ``types.ScoreReport``.  Everything here is presentation and
event plumbing; the scoring policy lives in ``aggregate``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Presentation order of the per-family sub-scores.  Weights are printed for
#: humans but are *owned* by ``aggregate`` (SPEC weights are frozen policy).
SUBSCORE_ORDER: tuple[str, ...] = ("colour", "lpips", "clip", "geometry", "ssim")
SUBSCORE_WEIGHTS: dict[str, float] = {
    "colour": 0.40,
    "lpips": 0.25,
    "clip": 0.15,
    "geometry": 0.15,
    "ssim": 0.05,
}

_SCALAR_KEYS = ("score", "status", "subscores", "raw", "flags", "versions", "timing_s")


# ---------------------------------------------------------------------------
# Report plumbing (shared with batch.py / sanity.py / debug_sheet.py)
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    """Best-effort conversion of numpy / dataclass-ish values to JSON types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalars / 0-d arrays
        try:
            return jsonable(obj.item())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "tolist"):  # small numpy arrays
        try:
            return jsonable(obj.tolist())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


def report_to_dict(report: Any) -> dict:
    """Serialise a :class:`~sartiq_score.types.ScoreReport` to a plain dict."""
    if report is None:
        return {}
    if isinstance(report, dict):
        return jsonable(report)
    out: dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        if hasattr(report, key):
            out[key] = jsonable(getattr(report, key))
    if not out and hasattr(report, "__dict__"):
        out = jsonable(dict(vars(report)))
    return out


# ---------------------------------------------------------------------------
# Scorer access (the api module)
# ---------------------------------------------------------------------------


def _api_module():
    """Return the ``sartiq_score.api`` module.

    ``sartiq_score/__init__.py`` publishes a module-level ``api = None``
    placeholder for its lazy PEP-562 import; that attribute shadows the
    submodule, so ``from . import api`` yields ``None``.  Import explicitly.
    """
    import importlib  # noqa: PLC0415

    return importlib.import_module("sartiq_score.api")


def build_scorer():
    """Create a model-caching :class:`~sartiq_score.api.Scorer`, or ``None``.

    ``Scorer`` loads LPIPS / CLIP / segmentation once; reuse one instance for a
    whole batch.  When the class is unavailable (or fails to construct) callers
    transparently fall back to :func:`score_with`, which uses the module-level
    ``api.score_pair``.
    """
    try:
        api = _api_module()
    except Exception:
        return None
    scorer_cls = getattr(api, "Scorer", None)
    if scorer_cls is None:
        return None
    try:
        return scorer_cls()
    except Exception:
        return None


def score_with(scorer, ref: str | Path, sub: str | Path, view: str | None = None,
               ref_mask_path: str | Path | None = None):
    """Score one pair, reusing ``scorer`` when it provides ``score_pair``."""
    ref_s, sub_s = str(ref), str(sub)
    fn = None
    if scorer is not None:
        candidate = getattr(scorer, "score_pair", None)
        if callable(candidate):
            fn = candidate
    if fn is None:
        fn = _api_module().score_pair
    return _invoke(fn, ref_s, sub_s, view, ref_mask_path)


def _invoke(fn, ref: str, sub: str, view: str | None, ref_mask_path=None):
    kwargs: dict = {}
    if view is not None:
        kwargs["view"] = view
    if ref_mask_path is not None:
        kwargs["ref_mask_path"] = str(ref_mask_path)
    if not kwargs:
        return fn(ref, sub)
    try:
        return fn(ref, sub, **kwargs)
    except TypeError:
        # older / narrower signature: drop the optional extras and retry
        kwargs.pop("ref_mask_path", None)
        try:
            return fn(ref, sub, **kwargs) if kwargs else fn(ref, sub)
        except TypeError:
            return fn(ref, sub)


def resolve_ref_mask(ref_path: str | Path, mask_dir: str | Path | None,
                     garment: str | None = None, view: str | None = None) -> str | None:
    """Find a hand-fixed reference mask inside *mask_dir*.

    v2.2: ``<mask_dir>/<garment>/<view>.<ext>`` is the only lookup when the garment is known
    (batch mode), so one garment's ``front`` mask can never apply to another garment.  The
    flat ``<mask_dir>/<stem>.<ext>`` form is kept for single-pair CLI use only.
    """
    if not mask_dir:
        return None
    exts = (".png", ".jpg", ".jpeg", ".webp")
    if garment is not None:
        v = view or Path(ref_path).stem
        for ext in exts:
            candidate = Path(mask_dir) / garment / f"{v}{ext}"
            if candidate.is_file():
                return str(candidate)
        return None
    stem = Path(ref_path).stem
    for ext in exts:
        candidate = Path(mask_dir) / f"{stem}{ext}"
        if candidate.is_file():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def format_report_text(report: Any, ref: str | None = None, sub: str | None = None) -> str:
    """Render a score report for a terminal."""
    d = report_to_dict(report)
    score = d.get("score", 0.0)
    status = d.get("status", "?")
    lines: list[str] = []
    lines.append(f"Reference similarity : {_num(score)} / 100   [{status}]")
    if ref:
        lines.append(f"  reference : {ref}")
    if sub:
        lines.append(f"  submission: {sub}")
    subs = d.get("subscores") or {}
    raw = d.get("raw") or {}
    if subs:
        lines.append("  sub-scores")
        seen: set[str] = set()
        for name in SUBSCORE_ORDER + tuple(sorted(subs)):
            if name in seen or name not in subs:
                continue
            seen.add(name)
            weight = SUBSCORE_WEIGHTS.get(name)
            head = f"    {name:<9s} {_num(subs[name]):>7s}"
            if weight is not None:
                head += f"   w={weight:.2f}"
            detail = _raw_detail(name, raw)
            lines.append(head + (f"   {detail}" if detail else ""))
    if raw:
        extra = {k: v for k, v in raw.items() if not isinstance(v, (list, dict))}
        if extra:
            lines.append("  raw")
            for key in sorted(extra):
                lines.append(f"    {key:<22s} {_num(extra[key])}")
    flags = d.get("flags") or []
    lines.append(f"  flags   : {', '.join(map(str, flags)) if flags else 'none'}")
    if d.get("timing_s"):
        lines.append(f"  timing  : {_num(d['timing_s'])} s")
    return "\n".join(lines)


def _raw_detail(name: str, raw: dict) -> str:
    keys = {
        "colour": (
            "colour_global_D",
            "colour_spatial_dE",
            "colour_global_d",
            "colour_global_D",
        ),
        "lpips": ("lpips_d", "lpips"),
        "clip": ("clip_d", "clip"),
        "geometry": ("geometry_iou", "iou", "geometry"),
        "ssim": ("ssim",),
    }.get(name, ())
    parts = [f"{k}={_num(raw[k])}" for k in keys if k in raw]
    return " ".join(parts)


def _num(value: Any) -> str:
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def _cmd_pair(args: argparse.Namespace) -> int:
    scorer = None if args.no_cache else build_scorer()
    mask_path = resolve_ref_mask(args.ref, getattr(args, "ref_mask_dir", None))
    try:
        report = score_with(scorer, args.ref, args.sub, getattr(args, "view", None), mask_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = report_to_dict(report)
        payload["reference"] = str(args.ref)
        payload["submission"] = str(args.sub)
        if getattr(args, "view", None):
            payload["view"] = args.view
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_report_text(report, str(args.ref), str(args.sub)))
    return 0 if report_to_dict(report).get("status") != "invalid" else 1


def _cmd_batch(args: argparse.Namespace) -> int:
    from . import batch  # noqa: PLC0415

    if not args.refs and not args.manifest:
        print("error: give either --refs <dir> or --manifest <file.jsonl>", file=sys.stderr)
        return 2
    if args.refs and args.manifest:
        print("error: --refs and --manifest are mutually exclusive", file=sys.stderr)
        return 2
    if not args.subs and not args.manifest:
        print("error: --subs is required unless the manifest carries submissions", file=sys.stderr)
        return 2
    summary = batch.run_batch(
        refs_dir=args.refs,
        subs_dir=args.subs,
        out_dir=args.out,
        manifest=args.manifest,
        debug_sheets=args.debug_sheets,
        limit=args.limit,
        jobs=args.jobs,
        ref_mask_dir=getattr(args, "ref_mask_dir", None),
    )
    print(f"leaderboard: {summary['leaderboard']}")
    print(f"scores     : {summary['scores_jsonl']}")
    print(f"review     : {summary['review_csv']}  ({summary['n_review']} entries)")
    print(f"garments   : {summary['garments_csv']}")
    print(
        f"teams={summary['n_teams']} garments={summary['n_garments']} "
        f"pairs_scored={summary['n_pairs_scored']} pairs_missing={summary['n_pairs_missing']}"
    )
    for row in summary["rows"]:
        print(
            f"  {row['rank']:>2}. {row['team']:<24s} {row['score']:>10.2f}   "
            f"colour={row['colour']:.4f} lpips={row['lpips']:.4f}"
        )
    return 0


def _cmd_sanity(args: argparse.Namespace) -> int:
    from . import sanity  # noqa: PLC0415

    families = None
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    summary = sanity.run_suite(
        refs_glob=args.refs_glob,
        out_dir=args.out,
        limit=args.limit,
        seed=args.seed,
        jobs=args.jobs,
        families=families,
        wrong_per_garment=args.wrong,
        max_side=args.max_side,
        timeout_s=args.timeout,
        keep_variants=False if args.no_keep_variants else None,
    )
    sanity.write_report(summary)
    print(sanity.render_console(summary))
    if not summary.get("run_ok", summary["n_failed"] == 0):
        if not summary["n_checks"]:
            print("error: the sanity suite produced no checks -- this is not a pass "
                  "(see the notes in the report)", file=sys.stderr)
        if summary.get("n_garments_failed"):
            print(f"error: {summary['n_garments_failed']}/"
                  f"{summary.get('n_garments_requested')} garments failed to run",
                  file=sys.stderr)
        return 1
    return 0


def _cmd_debug(args: argparse.Namespace) -> int:
    from . import debug_sheet  # noqa: PLC0415

    out = debug_sheet.render_debug_sheet(
        ref=args.ref, sub=args.sub, out=args.out, view=getattr(args, "view", None)
    )
    print(f"debug sheet: {out}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sartiq-score",
        description="Sartiq reference-similarity scorer (v2).",
    )
    parser.add_argument("--version", action="version", version="sartiq-score 2.2.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pair = sub.add_parser("pair", help="score one (reference, submission) pair")
    p_pair.add_argument("--ref", required=True, help="reference image path")
    p_pair.add_argument("--sub", required=True, help="submission image path")
    p_pair.add_argument("--view", default=None, help="view name (front/back/...)")
    p_pair.add_argument("--json", action="store_true", help="emit the full report as JSON")
    p_pair.add_argument("--ref-mask-dir", default=None,
                        help="directory of hand-fixed reference masks (PNG, same stem)")
    p_pair.add_argument("--no-cache", action="store_true", help="skip Scorer model caching")
    p_pair.set_defaults(func=_cmd_pair)

    p_batch = sub.add_parser("batch", help="score a submission tree into a leaderboard")
    p_batch.add_argument("--refs", default=None,
                         help="reference root <refs>/<garment>/<view>.<ext>")
    p_batch.add_argument("--manifest", "--refs-manifest", dest="manifest", default=None,
                         help="JSONL validation manifest (garment_id, view, ref) instead of --refs")
    p_batch.add_argument("--subs", default=None,
                         help="submission root <subs>/<team>/<garment>/<view>.<ext>")
    p_batch.add_argument("--out", required=True, help="output directory")
    p_batch.add_argument("--debug-sheets", action="store_true", help="also write per-pair contact sheets")
    p_batch.add_argument("--limit", type=int, default=None, help="only score the first N garments")
    p_batch.add_argument("--jobs", type=int, default=1, help="parallel scoring processes")
    p_batch.add_argument("--ref-mask-dir", default=None,
                         help="directory of hand-fixed reference masks (PNG, same stem)")
    p_batch.set_defaults(func=_cmd_batch)

    p_san = sub.add_parser("sanity", help="run the acceptance suite")
    p_san.add_argument("--refs-glob", required=True, help="reference glob(s), comma separated")
    p_san.add_argument("--out", required=True, help="output directory")
    p_san.add_argument("--limit", type=int, default=None, help="use only the first N references")
    p_san.add_argument("--seed", type=int, default=20260216, help="fixed RNG seed")
    p_san.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p_san.add_argument("--families", default=None, help="comma separated subset of families")
    p_san.add_argument("--wrong", type=int, default=5, help="wrong-garment negatives per garment")
    p_san.add_argument("--max-side", type=int, default=0, help="downscale references (0 = keep)")
    p_san.add_argument("--timeout", type=float, default=0.0, help="per-garment timeout seconds")
    p_san.add_argument("--no-keep-variants", action="store_true",
                       help="delete generated variant images after each garment")
    p_san.set_defaults(func=_cmd_sanity)

    p_dbg = sub.add_parser("debug", help="render one debug contact sheet")
    p_dbg.add_argument("--ref", required=True)
    p_dbg.add_argument("--sub", required=True)
    p_dbg.add_argument("--out", required=True, help="output image path")
    p_dbg.add_argument("--view", default=None)
    p_dbg.set_defaults(func=_cmd_debug)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
