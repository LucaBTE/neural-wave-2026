#!/usr/bin/env python3
"""Run the original baseline using dev/btc-sartiq/backend/.env.

All original CLI arguments are forwarded unchanged. No keys are printed or saved.
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path
import runpy
import sys

# Importing the original baseline must not create __pycache__ outside dev.
sys.dont_write_bytecode = True
DEV_ROOT = Path(__file__).resolve().parent
BACKEND = DEV_ROOT / 'btc-sartiq' / 'backend'
BASELINE = DEV_ROOT.parent / 'baselines' / 'baseline_nanobanana.py'


def load_backend_environment():
    try:
        from dotenv import load_dotenv
    except ImportError:
        # /bin/python3 may not contain the backend dependencies. Reuse its venv
        # automatically, forwarding arguments without involving a shell.
        python = (DEV_ROOT / '.score-venv' / 'bin' / 'python' if '--score' in sys.argv
                  else BACKEND / '.venv' / 'bin' / 'python')
        if python.is_file() and os.path.abspath(sys.executable) != os.path.abspath(python):
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit('Install python-dotenv in the virtual environment used for this launcher.')
    load_dotenv(BACKEND / '.env', override=False)
    # The backend accepts both names; the original fal baseline reads FAL_KEY.
    if not os.environ.get('FAL_KEY') and os.environ.get('FAL_API_KEY'):
        os.environ['FAL_KEY'] = os.environ['FAL_API_KEY']


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--score-out', default=str(DEV_ROOT / 'scores'))
    options, baseline_arguments = parser.parse_known_args()
    scoring_arguments = None
    if options.score:
        sys.path.insert(0, str(DEV_ROOT))
        import eval_val
        eval_val.use_scorer_environment()
        inputs = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        inputs.add_argument('--manifest', required=True)
        inputs.add_argument('--out', required=True)
        inputs.add_argument('--team', required=True)
        inputs.add_argument('--dataset')
        inputs.add_argument('--limit', type=int)
        inputs.add_argument('--dry-run', action='store_true')
        args, _ = inputs.parse_known_args(baseline_arguments)
        if args.dry_run:
            raise SystemExit('--score cannot be combined with --dry-run: there is no generated image to score.')
        # Fail before paid generation on targetless dev/test manifests or missing dependencies.
        eval_val.prepare_scoring(args.manifest, args.dataset, args.limit)
        scoring_arguments = ['--pairs', args.manifest, '--preds', args.out,
                             '--team', args.team, '--out', options.score_out, '--jobs', '1']
        if args.dataset:
            scoring_arguments += ['--dataset', args.dataset]
        if args.limit is not None:
            scoring_arguments += ['--limit', str(args.limit)]
    load_backend_environment()
    if not BASELINE.is_file():
        raise SystemExit(f'Baseline script not found: {BASELINE}')
    sys.path.insert(0, str(BASELINE.parent))
    sys.argv[:] = [str(BASELINE), *baseline_arguments]
    try:
        runpy.run_path(str(BASELINE), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    if scoring_arguments is not None:
        print('\nScoring with the official local Sartiq evaluator…', flush=True)
        raise SystemExit(eval_val.main(scoring_arguments))
    if '-h' in baseline_arguments or '--help' in baseline_arguments:
        print('\nDev launcher: --score runs the official local evaluator after generation; '
              '--score-out DIR selects its report folder (default: dev/scores).')


if __name__ == '__main__':
    main()
