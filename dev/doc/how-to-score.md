# Official scoring

Run commands from the repository root. The supplied `dev` and `test` splits have
no reference targets; use `train/pairs.jsonl` for local scoring.

## Scoring environment — first setup only

```bash
python3 -m venv dev/.score-venv
dev/.score-venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
dev/.score-venv/bin/python -m pip install ./scorer python-dotenv
```

Weights are cached in `dev/.models/`. Keep an existing environment and weights.
The backend key is read from `dev/btc-sartiq/backend/.env`.

## Compare Studio 0.1 and the baseline on the same five products

```bash
/bin/python3 dev/compare_pipelines.py \
  --dataset /home/luca/Downloads/sartiq_neural_wave_2026 \
  --split train --count 5 --jobs 4
```

Each product uses its first input photograph. Targets are used only by the
unchanged official scorer, never sent to a generation model. The command prints
a comparison and saves predictions, diagnostics and scores under the printed
`dev/comparisons/` folder. `--dry-run` checks selection without model calls;
`--baseline-from <previous-comparison>` reuses matching saved baseline outputs.

## Score existing predictions without generation

```bash
/bin/python3 dev/eval_val.py \
  --pairs /home/luca/Downloads/sartiq_neural_wave_2026/train/pairs.jsonl \
  --preds dev/predictions --team studio --out dev/scores --limit 5 --jobs 1
```

Expected layout: `<preds>/<team>/<garment>/front.png`. Read `leaderboard.csv` for
aggregate results and `scores.jsonl` for individual metrics and flags. Missing
submissions within the selected subset count as zero.

For the original baseline alone, `dev/baseline_nanobanana.py` forwards the authors'
CLI options and optionally supports `--score` after generation.

The final score is `0.40 colour + 0.25 LPIPS + 0.15 CLIP + 0.15 geometry + 0.05 SSIM`,
with all reported similarities scaled to 0–100. LPIPS similarity is higher-is-better;
its underlying raw distance is lower-is-better. See [the official specification](../../scorer/README.md).
