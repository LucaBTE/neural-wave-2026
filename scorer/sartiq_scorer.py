#!/usr/bin/env python3
"""v1-compatible entry point for the Sartiq reference-similarity scorer (v2).

Keeps the original flags (``--reference`` / ``--generated``) so existing
scripts and participant habits keep working, but prints the full v2 report
(per-family sub-scores, raw metrics, review flags, versions).

    python sartiq_scorer.py --reference ref.jpg --generated sub.png
    python sartiq_scorer.py --reference ref.jpg --generated sub.png --json
    python sartiq_scorer.py --reference ref.jpg --generated sub.png --view front

The result is called **reference similarity** (0-100), not product fidelity.
The authoritative entry point is ``python -m sartiq_score.cli`` (or the
``sartiq-score`` console script).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python path/to/sartiq_scorer.py`` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sartiq_score.cli import build_scorer, format_report_text, report_to_dict, score_with  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sartiq_scorer.py",
        description=(
            "Reference similarity between a studio reference packshot and a submitted "
            "generated packshot (scorer v2). v1-compatible flags."
        ),
    )
    parser.add_argument("--reference", "--ref", dest="reference", required=True,
                        help="reference packshot")
    parser.add_argument("--generated", "--sub", "--submission", dest="generated", required=True,
                        help="submitted / generated packshot")
    parser.add_argument("--view", default=None, help="view name (front, back, ...)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--no-cache", action="store_true", help="do not reuse cached models")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scorer = None if args.no_cache else build_scorer()
    report = score_with(scorer, args.reference, args.generated, args.view)
    payload = report_to_dict(report)

    if args.json:
        payload["reference"] = args.reference
        payload["submission"] = args.generated
        if args.view:
            payload["view"] = args.view
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_report_text(report, args.reference, args.generated))
        print()
        print("Reference similarity, not product fidelity. See SPEC.md for the frozen policy.")
    return 0 if payload.get("status") != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
