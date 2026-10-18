# Launching the hosted baseline

The authors' original instructions are in [baselines/README.md](../../baselines/README.md)
(hosted image-edit baseline), the [repository Quickstart](../../README.md#quickstart)
and [scorer/README.md](../../scorer/README.md). Their baseline generates predictions;
their separate `scripts/eval_val.py` evaluates them against target images. The
`--score` option below is only a launcher shortcut for these two original steps.

From the repository root:

```bash
/bin/python3 dev/baseline_nanobanana.py \
  --manifest /home/luca/Downloads/sartiq_neural_wave_2026/dev/inputs.jsonl \
  --out dev/predictions \
  --team btc \
  --limit 1 \
  --dry-run
```

The launcher loads `dev/btc-sartiq/backend/.env` independently of the working
directory. Both `FAL_KEY` and `FAL_API_KEY` are supported; existing environment
variables take precedence over the same names in the file. The original baseline
still requires `--manifest`, `--out` and `--team`. No credential is printed or saved.
If the system Python lacks python-dotenv, the launcher automatically uses the
backend virtual environment, when available.

Remove `--dry-run` to make paid API calls. This runs the original hosted baseline,
not the LangGraph studio pipeline; model and provider behavior remain unchanged.
The original `baselines/baseline_nanobanana.py` is not modified. Run the launcher
under `dev` to load the backend `.env`; invoking the original script directly does
not load it.

## Official local scoring

The generated `dev` sample `d001` has no target in the supplied manifest, so it
cannot receive an official local score. Use `train/pairs.jsonl` (or a holdout
manifest with targets) for generation AND scoring. Do not compare a `d001`
prediction to an unrelated `t0001` target. Studio quality warnings are not the
official score.

One-time setup, from the repository root, with Python 3.11+:

```bash
/bin/python3 -m venv dev/.score-venv
PIP_CACHE_DIR="$PWD/dev/.cache/pip" dev/.score-venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
PIP_CACHE_DIR="$PWD/dev/.cache/pip" dev/.score-venv/bin/python -m pip install ./scorer python-dotenv
```

Generate and score one TRAIN example:

```bash
/bin/python3 dev/baseline_nanobanana.py \
  --manifest /home/luca/Downloads/sartiq_neural_wave_2026/train/pairs.jsonl \
  --out dev/predictions \
  --team btc \
  --limit 1 \
  --score
```

`--score` uses the dedicated scoring environment when present and checks targets
and dependencies BEFORE spending fal credits. It then invokes the unchanged
official `scripts/eval_val.py`, which uses the repository's `sartiq_score`, metrics,
weights and aggregation. It prints `mean score` (0–100), and writes:

- `dev/scores/leaderboard.csv`: aggregate official score and metric families.
- `dev/scores/pairs.csv`: score and status for each evaluated image.
- `dev/scores/summary.json`: aggregate score and counts.
- `dev/scores/scores.jsonl`: detailed official reports.

Use `--score-out dev/scores/another-run` to keep separate reports. The same
`--limit` applies to generation and scoring; images not generated outside that
limit do not accidentally count as missing. Existing missing slots within the
evaluated subset retain the official zero-score policy.

Score already generated TRAIN predictions without new fal requests:

```bash
/bin/python3 dev/eval_val.py \
  --pairs /home/luca/Downloads/sartiq_neural_wave_2026/train/pairs.jsonl \
  --preds dev/predictions --team btc --out dev/scores --limit 1 --jobs 1
```

Scorer weights are downloaded on the first real evaluation and cached in
`dev/.models`; this can take time. No scorer source files are modified. The
integrated command evaluates the baseline, not the LangGraph studio pipeline.
The same standalone evaluator can score studio outputs saved in the required
`<preds>/<team>/<pair_id>/front.png` layout; see `how-to-score.md`.
