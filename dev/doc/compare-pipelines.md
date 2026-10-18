# Baseline versus Studio: the same five garments

From the repository root:

```bash
/bin/python3 dev/compare_pipelines.py \
  --dataset /home/luca/Downloads/sartiq_neural_wave_2026
```

Default: the first five distinct IDs of `train/pairs.jsonl`, currently
**t0001–t0005**. Both methods receive exactly the FIRST input photograph for each
ID. Additional views are deliberately not supplied to Studio in this comparison.
Target images are used only by the official evaluator, never sent to either image
generation pipeline. The baseline model/prompt remain the originals; Studio uses
the same configured LangGraph pipeline and `.env` as the application.

The script generates ten outputs (five per method), then runs the unchanged
official `scripts/eval_val.py` through `dev/eval_val.py` twice with the SAME selected
manifest. Final values come directly from its garment-weighted leaderboard rows;
they are not a separate formula or a new average of the pair scores.

## Rerun Studio against the existing baseline

To evaluate the simplified pipeline without paying to regenerate the baseline:

```bash
/bin/python3 dev/compare_pipelines.py \
  --dataset /home/luca/Downloads/sartiq_neural_wave_2026 \
  --baseline-from dev/comparisons/20261017-170416-256045
```

This copies the existing baseline predictions into the new report directory,
generates only five new Studio images, and scores both sets using the official
evaluator. It checks product IDs, views, input paths and target paths before any
generation. Missing/empty baseline predictions are rejected. Previous manifests do
not contain source hashes: keep the dataset files unchanged when reusing a run.
Baseline randomness cannot change the comparison because its actual images are
reused. The historic result was baseline 59.38 versus old Studio 47.03; it is not
a measurement of the new pipeline.

Studio defaults to `faithful`: each source photo once, classification and short
description in one query, one normal Nano Banana 2.1 edit, a single-product check,
and a diagnostic-only final matte. Six cold calls per product. A positively detected
duplicate permits one correction plus a mandatory recheck (eight calls maximum).
An unresolved duplicate becomes a failed prediction, not a valid final result. An
explicit model/profile in backend `.env` takes precedence. See [pipeline.md](pipeline.md).

The terminal prints the aggregate comparison, including score, colour, LPIPS,
CLIP, geometry and SSIM. A new folder under `dev/comparisons/<timestamp>/` contains:

- `selected.jsonl`: exact IDs, inputs and targets used.
- `predictions/baseline/<id>/front.png` and `predictions/studio/<id>/front.png`.
- `scores/baseline/` and `scores/studio/`: original official reports.
- `comparison.csv`: both official aggregates, native metric units.
- `comparison.md`: aggregate comparison with all metrics displayed out of 100.
- `per_garment.csv`: the two scores and their difference for every selected ID.
- `diagnostics/`: Studio prompts, classification, quality, models, timings and traces.
- `diagnostics/<id>/front_generated.png`: generated RGB before local resize/padding,
  for checking whether a difference appeared during generation or final delivery.
- `diagnostics/<id>/front_before_color.png`: exact delivery frame immediately
  before local color refinement, including framing in the optional staged profile.
- `diagnostics/<id>/front_source_mask.png` and `front_result_mask.png`: actual
  source SAM and generated BiRefNet masks, when available, for reproducible local
  color replays. Missing masks are not fabricated or replaced with a target mask.
- `generation_status.json`: generation subprocess outcomes.

No existing baseline/studio predictions or previous reports are overwritten.
Failures are kept in the comparison as missing/invalid slots according to official
scorer rules, rather than dropping difficult garments from the mean. A run with
generation failures exits nonzero after writing available reports. No failed output
is replaced with the original photograph. Run fresh commands after fixing a failure.

Prerequisites: `dev/.score-venv` with the official scorer and python-dotenv,
the existing backend `.venv` with Studio dependencies, and the fal key in
`dev/btc-sartiq/backend/.env`. Generation consumes fal credits. Scoring is local;
weights can be downloaded on first evaluation. See [baseline.md](baseline.md).

Preview selection without generation or scoring:

```bash
/bin/python3 dev/compare_pipelines.py --dataset /home/luca/Downloads/sartiq_neural_wave_2026 --dry-run
```

Optional flags: `--count 5`, `--out dev/comparisons/my-run` (must not exist),
`--split train|dev|test`, `--manifest /path/to/pairs-with-targets.jsonl`,
`--baseline-from dev/comparisons/previous-run`.

The actual `test` directory contains g01–g15 and NO reference targets. To generate
its first five images for both methods without pretending to calculate a score:

```bash
/bin/python3 dev/compare_pipelines.py --dataset /home/luca/Downloads/sartiq_neural_wave_2026 --split test --generate-only
```

Without `--generate-only`, targetless manifests are rejected before paid calls.
When official test targets become available, supply their manifest explicitly.
