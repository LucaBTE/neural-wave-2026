#!/usr/bin/env python3
"""Launch the repository's official local evaluator, without changing its policy."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
DEV_ROOT = Path(__file__).resolve().parent
OFFICIAL_EVALUATOR = DEV_ROOT.parent / 'scripts' / 'eval_val.py'
SCORER_PYTHON = DEV_ROOT / '.score-venv' / 'bin' / 'python'


def use_scorer_environment():
    if SCORER_PYTHON.is_file() and os.path.abspath(sys.executable) != os.path.abspath(SCORER_PYTHON):
        os.execv(str(SCORER_PYTHON), [str(SCORER_PYTHON), *sys.argv])


def prepare_scoring(pairs, dataset=None, limit=None):
    # Reuse the official manifest/root logic, including holdout meta.json.
    evaluator = runpy.run_path(str(OFFICIAL_EVALUATOR), run_name='official_eval_helpers')
    pairs_path = Path(pairs)
    dataset_root = evaluator['resolve_dataset_root'](pairs_path, dataset)
    rows = evaluator['load_pairs'](pairs_path, dataset_root)
    if limit is not None:
        rows = rows[:max(0, limit)]
    if not rows:
        raise SystemExit('Scoring requires at least one pair with a target image.')
    missing = [row['target'] for row in rows if not row['target'].is_file()]
    if missing:
        raise SystemExit(f'Scoring target not found: {missing[0]}. Check --dataset.')
    dependencies = ['numpy', 'PIL', 'pillow_heif', 'scipy', 'skimage', 'torch',
                    'torchvision', 'lpips', 'open_clip', 'rembg', 'onnxruntime', 'tqdm']
    missing_dependencies = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing_dependencies:
        raise SystemExit('Official local scorer dependencies are missing: ' +
                         ', '.join(missing_dependencies) +
                         '. Follow the setup in dev/doc/how-to-score.md. No generation was started.')
    # Keep downloaded weights and runtime caches under dev, without editing scorer/.
    model_dir = DEV_ROOT / '.models'
    for name, path in {
        'SCORER_MODEL_DIR': model_dir, 'TORCH_HOME': model_dir / 'torch',
        'HF_HOME': model_dir / 'hf', 'HUGGINGFACE_HUB_CACHE': model_dir / 'hf' / 'hub',
        'U2NET_HOME': model_dir / 'u2net', 'XDG_CACHE_HOME': DEV_ROOT / '.cache',
    }.items():
        os.environ[name] = str(path)
    return evaluator


def main(argv=None):
    use_scorer_environment()
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--limit', type=int)
    # Let the original evaluator display its own --help without loading models.
    if '-h' in arguments or '--help' in arguments:
        evaluator = runpy.run_path(str(OFFICIAL_EVALUATOR), run_name='official_eval_helpers')
    else:
        options, _ = parser.parse_known_args(arguments)
        evaluator = prepare_scoring(options.pairs, options.dataset, options.limit)
    return evaluator['main'](arguments)


if __name__ == '__main__':
    raise SystemExit(main())
