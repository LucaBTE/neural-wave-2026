# Sartiq scorer: reference similarity

Automated leaderboard scorer for the Sartiq Neural Wave 2026 challenge. It compares a submitted
packshot against a reference packshot of the same garment and returns a reference similarity score
from 0 to 100.

The score is reference similarity, not product fidelity and not a jury rating. Weights, anchors and
status rules are frozen before submissions open and are never fitted to submitted work. `SPEC.md` is
the frozen contract and `ACCEPTANCE.md` summarises how the scorer was tested.

## 1. Install

Python 3.11 or newer. Use the repository quickstart environment: a CPU-only venv at the repository
root, `.venv`. Run every command in this file from the repository root.

```bash
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e 'scorer[test]'
```

To use the scorer on its own, a venv at `scorer/.venv` works too, created the same way with
`uv venv .venv --python 3.12` and `uv pip install -e '.[test]'` from `scorer/`.

Always run with a clean import path so the package resolves to this checkout. After the quickstart
the first example pair is on disk, so this command runs as written:

```bash
env -u PYTHONPATH .venv/bin/python -m sartiq_score.cli pair \
    --ref ../sartiq_neural_wave_2026/train/t0001/target.jpg \
    --sub ../sartiq_neural_wave_2026/train/t0001/input_1.jpg
```

### Model bundle

All weights live in `scorer/.models/` and are addressed through `TORCH_HOME`, `HF_HOME`,
`HUGGINGFACE_HUB_CACHE`, `U2NET_HOME` and `XDG_CACHE_HOME`, which the scorer sets on first use.
`SCORER_MODEL_DIR` overrides the location.

| component | model | cache |
|---|---|---|
| perceptual | LPIPS `alex` v0.1 | bundled with the `lpips` package |
| backbone | torchvision AlexNet IMAGENET1K_V1 | `.models/torch/hub/checkpoints/` |
| semantic | OpenCLIP `ViT-B-32` / `openai` | `.models/open_clip/` |
| segmentation | rembg `u2net` (ONNX, CPU) | `.models/u2net/` |

The first `Scorer()` construction downloads anything missing (about 1.2 GB). After that the directory
is self-contained and can be copied to an offline judging box. `ScoreReport.versions` carries the
package version, the model ids and the SHA-256 of each weight file, so a published score can be tied
to exact weights.

```json
{"clip_model": "ViT-B-32/openai", "clip_sha256": "e6d1bd...",
 "lpips_model": "alex@v0.1", "segmenter": "rembg:u2net", "threads": 4}
```

## 2. Commands

Syntax for each sub-command below. These are templates, not literal commands: `REF`, `SUB`, `DIR`,
`GLOB`, `OUT`, `SHEET.jpg`, `N` and the like are placeholders. For commands you can paste and run,
see section 1, section 3 and section 5.

```
python -m sartiq_score.cli pair   --ref REF --sub SUB [--view VIEW] [--json]
                                  [--ref-mask-dir DIR] [--no-cache]
python -m sartiq_score.cli batch  (--refs DIR | --manifest FILE.jsonl) [--subs DIR]
                                  --out OUT [--jobs N] [--debug-sheets] [--limit N]
                                  [--ref-mask-dir DIR]
python -m sartiq_score.cli sanity --refs-glob 'GLOB[,GLOB]' --out OUT [--limit N]
                                  [--seed N] [--jobs N] [--families LIST] [--wrong N]
                                  [--max-side N] [--timeout N] [--no-keep-variants]
python -m sartiq_score.cli debug  --ref REF --sub SUB --out SHEET.jpg [--view VIEW]
```

`uv` also installs a `sartiq-score` console script with the same sub-commands.

The v1 wrapper `sartiq_scorer.py` keeps the old flags and prints the v2 report:

```bash
env -u PYTHONPATH .venv/bin/python scorer/sartiq_scorer.py \
    --reference ../sartiq_neural_wave_2026/train/t0001/target.jpg \
    --generated ../sartiq_neural_wave_2026/train/t0001/input_1.jpg --view front
# add --json to print the full report (the LPIPS banner prints before it)
```

`debug` renders a one-row contact sheet, `ref | sub | canon ref | canon sub | masks (R=ref, G=sub) |
per-cell deltaE00 heat map`, and prints the score breakdown.

## 3. Submission layout

```
references/<garment_id>/<view>.<png|jpg|jpeg|webp>     # front (this challenge: front only)
submissions/<team>/<garment_id>/<view>.<png|jpg|jpeg|webp>
```

- The required view is `front`; the references do not publish another view unless the organisers
  announce one.
- Garment score = mean over the required views; a missing submission view scores 0.
- Team score = mean over all reference garments; a missing garment scores 0.
- Scores are rounded to 2 decimals; ties break on the garment-weighted colour sub-score, then the
  garment-weighted LPIPS sub-score.
- `<references>/garments.json` may declare per-garment metadata
  (`{"<garment_id>": {"target_style": "flat_lay" | "ghost"}}`). Every target in this challenge is
  flat-lay, which is the default when the file is absent.

### Validation manifest (alternative to the reference folder)

A JSONL manifest can replace the reference tree, one object per line. Paths are relative to where
you run the command (the repository root, so the dataset next to the repo is `../`):

```jsonl
{"garment_id": "t0001", "view": "front", "ref": "../sartiq_neural_wave_2026/train/t0001/target.jpg"}
{"garment_id": "t0002", "view": "front", "ref": "../sartiq_neural_wave_2026/train/t0002/target.jpg",
 "submission": "predictions/t0002/front.png"}
```

- `garment_id` (aliases `garment`, `id`, `gid`, `name`) and `ref` (aliases `reference`, `ref_path`,
  `path`, `target`) are required; `view` defaults to the reference filename stem.
- A per-row `sub` or `submission` (aliases `generated`, `input`) is used only when the submission tree
  has no file for that slot, and only in manifest-only mode.
- With a manifest that declares every submission, `--subs` may be omitted; all rows are then scored
  as a single team named `manifest`.

```bash
env -u PYTHONPATH .venv/bin/python -m sartiq_score.cli batch \
    --manifest val_out/val_manifest.jsonl --out val_out
```

`val_out/val_manifest.jsonl` is written by `scripts/eval_val.py` and already has the
`garment_id`/`ref`/`sub` shape the batch mode wants; pointing batch at a raw `holdout/val.jsonl`
fails, because that file has `pair_id` rows and no `ref`.

### Outputs (batch)

| file | contents |
|---|---|
| `leaderboard.csv` | rank, team, score, per-family aggregates, coverage counts |
| `scores.jsonl` | one full `ScoreReport` per (team, garment, view) slot, including missing slots |
| `review.csv` | every submission the scorer flagged `review` |
| `garments.csv` | garment, view, reference path, target style |
| `debug/<team>/<garment>_<view>.jpg` | contact sheets (with `--debug-sheets`) |

Any exception while scoring a single file is caught and recorded as `status="invalid"` with a
`scorer_error:*` flag, so one bad file never aborts a batch.

## 4. How the score works

Both images are decoded (EXIF applied, ICC converted to sRGB, alpha composited on white), segmented
with one frozen model (rembg `u2net`, two-pass since v2.1: the full image, then a re-segment of the
mask bounding box plus 8 % margin), and canonicalised: the mask bounding box plus 2 % is cropped,
uniformly scaled so its binding dimension occupies 90 % of a 512x384 canvas, and centred on flat
white. There is no rotation, flip, anisotropic scaling or refinement. Before scoring, the submission
canvas is translated by at most 1 % of each side onto the reference using masks only (identity
preferred; reported in `raw.align_dy` and `raw.align_dx`). LPIPS and SSIM are compared after a
symmetric Gaussian blur of sigma 1 (`raw.antialias_sigma`). `SPEC.md` has the full pipeline.

| family | metric | raw value | similarity | weight |
|---|---|---|---|---|
| colour | `colour_global`: mask-weighted Lab quantile distance `D` over `q` in {0.05...0.95} and `c` in {L, a, b} | `D` | `2^(-D/6.0)` | (see below) |
| | `colour_spatial`: 4x4 grid, deltaE00 between masked mean L\*a\*b\* per cell, weighted by reference area, averaged over both directions | `dE00` | `2^(-dE00/5.0)` | (see below) |
| | `colour = 0.6 x colour_global + 0.4 x colour_spatial` | | | 0.40 |
| perceptual | LPIPS `alex` v0.1 on canonical RGB (inputs in [-1, 1]) | `d` | `2^(-d/0.20)` | 0.25 |
| semantic | OpenCLIP `ViT-B-32/openai`, canonical RGB letterboxed to a square (the garment is never centre-cropped) | `d = 1 - cos` | `2^(-d/0.10)` | 0.15 |
| geometry | soft IoU of the two canonical masks, sum of min over sum of max | `IoU` | as-is | 0.15 |
| structure | SSIM on the L\* channel (data_range 100, Gaussian sigma 1.5), clipped | `SSIM` | as-is | 0.05 |

```
score = 100 x (0.40 x colour + 0.25 x lpips + 0.15 x clip + 0.15 x geometry + 0.05 x ssim)
```

Anchors are the metres-to-similarity conversion only. They are provisional, published and frozen.
They are not fitted to submitted work.

### Status rules (no learned hard caps)

| status | when |
|---|---|
| `invalid` | unreadable or corrupt file, min side under 256 px, max side over 12000 px, aspect over 3:1, decompression bomb (header over 150 MPix), file over 100 MB, animated or multi-frame, a symlinked file in a team tree, or a flat-field submission (whole-image luma standard deviation under 4/255, reason `blank_or_uniform`). Score 0. |
| `review` | any of `mask_empty`, `mask_suspect`, `truncated` on the submission, `multi_component` on the submission while the reference is single, or a clip sub-score under 0.30 |
| `ok` | none of the above |

`review` entries keep their score; a human checks them before publishing. Model uncertainty is never
a hard cap, only malformed input is.

## 5. Acceptance suite

The sanity suite spawns `pair` subprocesses with the working directory set to `scorer/`, so a
relative reference glob resolves in the parent but not in those subprocesses. Pass an absolute glob:

```bash
env -u PYTHONPATH .venv/bin/python -m sartiq_score.cli sanity \
    --refs-glob "$(realpath ../sartiq_neural_wave_2026)/train/*/target.jpg" \
    --out sanity_out --limit 15 --jobs 6
```

This is a diagnostic, not a pass/fail gate. On the frozen release the suite reports a small set of
expected invariance and monotonicity failures (the reasons are in `ACCEPTANCE.md`), so the command
exits non-zero and `sanity_report.md` lists every failing case by name. Use it to compare two
scorers or to inspect one transform; do not read a non-zero exit as a broken install.

The suite runs over reference packshots with fixed seeds. It tests monotonicity within families and a
few published orderings, never absolute cross-family thresholds, and reports every failing case by
name.

- Invariance, within 2 points: PNG, JPEG q95, EXIF-rotated, RGBA with a transparent background, 0.5x
  resize, 20 % padding, 10 % translation without clipping, background 240 to 255.
- Monotone: non-increasing within 0.5 points per step, plus a family-level total drop where the
  distortion is unambiguously destructive (colour_shift at least 10, blur 3, JPEG q80 to q10 2,
  removal 8, crop_away 10; hue_rotate has no drop floor because neutral garments barely move). The
  transforms are Lab a\*/b\* colour shift deltaE {2, 5, 10, 20} (garment only), hue rotate {10, 20,
  40} degrees, blur sigma {1, 2, 4, 8}, JPEG q {80, 40, 10}, garment-area removal {5, 15, 30} %,
  crop-away {5, 15, 30} % of garment height. References that are themselves truncated at the frame
  edge are excluded from removal and crop_away; the exclusion is printed in the report.
- Ordering: self at least 99.5; wrong garments below self minus 25 and below the deltaE5 shift; blank,
  colour blob and noise below 20; swapped-halves collage below self minus 15; text overlay below self
  minus 5.
- Robustness: corrupted file, 1x1 image, 20000x20000 header bomb give an explicit `invalid`; CMYK
  JPEG, 16-bit PNG, P-mode with transparency never crash.
- Determinism: the same pair twice in separate processes and in different batch orders, within 0.01.
- Performance: cold-start and warm per-pair time, peak RSS, at 4 threads.

Outputs `sanity_report.md` (per-family table, every failing check, determinism and performance, all
raw variant scores), `sanity_results.jsonl` (one line per variant plus one per check) and the
generated variants under `variants/<garment>/` (`--no-keep-variants` deletes them).

`--limit N` selects a deterministic, evenly spread subset of the references; `--jobs N` parallelises
across processes; `--max-side N` downscales inputs for faster iteration.

## 6. Caveats

- Reference similarity is not fidelity. The reference photograph measures rendered appearance, not
  physical fabric reflectance. No exposure or white-balance normalisation is applied.
- Style matters. Silhouette, LPIPS and SSIM all penalise a ghost-versus-flat-lay difference. All
  targets here are flat-lay, so score like against like.
- Weights and anchors are frozen policy, not a fitted model. Synthetic augmentations test invariance
  and monotonicity only; they cannot establish human preference.
- One segmenter for everyone. rembg `u2net` is not garment-specific; white, cream, lace and
  thin-strap garments are the known weak spot. Reference masks can be hand-fixed with `--ref-mask-dir`
  (PNG with the same stem).
- CPU speed. A single pair costs seconds warm; a full reference set times several teams is minutes,
  not seconds, so use `--jobs`.
- Cross-platform repeatability is not bit-exact. Threads are pinned (`torch.set_num_threads(4)`),
  models run in eval plus `inference_mode`, and deterministic algorithms are requested. The guarantee
  is repeatability within this pinned environment, with numerical tolerance elsewhere.
- Presentation is partly normalised on purpose. A transparent-background submission composites on
  white, a bounded mask-only micro-alignment (1 % of the canvas) plus a symmetric LPIPS/SSIM blur
  absorb the resampling phase of a padded, translated or half-resolution upload. Those are framing
  effects, not fidelity. Cropping, occlusion, colour and content changes are unaffected. `SPEC.md`
  has the details.
