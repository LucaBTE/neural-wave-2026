# Scoring, reference similarity, exactly

The official scorer lives in `scorer/`. This document is the contract: what it computes, what
makes a submission invalid or flagged, how garments and teams are averaged, and what the jury
looks at. If this document and the code ever disagree, the code wins (and tell Felipe).

Read the name carefully: the number is reference similarity, not product fidelity, not true
colour, not a jury rating. It measures how close your output is to a specific studio photograph
of the same garment. A well-framed, clean shot that differs from the reference still loses
points. That is why the automated score is a gate that picks finalists, and the jury decides the
winner.

---

## 1. Preprocessing: framing is free, everything else is not

Both images go through the same pipeline before any metric runs:

1. Decode with Pillow, apply EXIF rotation, convert ICC to sRGB (on failure: warn, assume sRGB).
   Palette, LA, RGBA, CMYK and 16-bit modes are handled. Alpha is composited onto pure white
   `(255,255,255)`; RGB hidden under `alpha = 0` is ignored, so you cannot smuggle content under
   transparency.
2. Segment the garment with one frozen model, `rembg` `u2net` (ONNX, CPU), yielding a soft mask.
   Holes are kept, and every component covering at least 0.5% of the image is kept.
3. Canonicalise: crop the mask bounding box (+2% margin), scale uniformly so the binding
   dimension occupies 90% of a 512x384 canvas (3:4, portrait), and centre it on pure white.
4. Micro-align: translate the submission canvas by at most 1% of each side onto the reference, on
   the masks only (the identity wins ties). Then, before LPIPS and SSIM only, blur both canonical
   images with a Gaussian of sigma 1.0.

Consequences you can plan around:

* Normalisation reduces framing sensitivity: background, canvas size, margin and framing have
  little effect, and a tight crop and a wide shot of the same garment canonicalise to the same
  image. It is not absolute. The background and any shadow are inputs to segmentation, so a
  background that confuses the segmenter changes the mask and the score. The task is the packshot,
  not the composition.
* Aspect ratio is not free if it changes how the mask is cropped. The mask bbox is what gets
  fitted, so a garment cut off at the frame edge is a real error.
* Nothing is rotated, flipped or scaled anisotropically. If your input is sideways, rotate it
  yourself. The output is compared as-is against an upright reference.

---

## 2. The five metrics

Let `s = 2^(-d/a)` be the similarity for a distance `d` with a published anchor `a`.

| family | weight | metric | raw | similarity |
|---|---|---|---|---|
| colour | 0.40 | `colour_global`: Lab (L\*a\*b\*, D65) on each image's own mask; `D` = mean over channels `{L*,a*,b*}` and quantiles `{0.05..0.95}` of `abs(Q_ref - Q_sub)` | `D` | `2^(-D/6.0)` |
| | | `colour_spatial`: 4x4 grid; for each cell with >= 5% reference support, `2^(-dE00/5.0)` on the masked mean Labs, or 0 when the submission has < 5% support in that cell, weighted by reference area; the reverse direction (cells with submission support) is computed the same way and the two directions are averaged | per-cell `dE00` | `2^(-dE00/5.0)` |
| | | `colour = 0.6 * colour_global + 0.4 * colour_spatial` | | |
| perceptual | 0.25 | LPIPS `alex` v0.1 on the canonical RGB pair (inputs in `[-1,1]`) | `d` | `2^(-d/0.20)` |
| semantic | 0.15 | OpenCLIP `ViT-B-32` / `openai`; canonical RGB letterboxed to a square (the garment is never centre-cropped) | `d = 1 - cos` | `2^(-d/0.10)` |
| geometry | 0.15 | soft IoU of the two canonical masks, `sum(min) / sum(max)` | `IoU` | as-is |
| structure | 0.05 | SSIM on the `L*` channel (data_range 100, gaussian sigma 1.5), clipped to `[0,1]` | `SSIM` | as-is |

```
score = 100 * (0.40*colour + 0.25*lpips + 0.15*clip + 0.15*geometry + 0.05*ssim)
```

Reported to 2 decimals. The anchors (`6.0`, `5.0`, `0.20`, `0.10`) are pure unit conversions from
distance to similarity. They are published and frozen, not fitted to submitted work.
`ScoreReport.versions` carries the model ids and the SHA-256 of each weight file, so a published
score can be tied to the exact weights that produced it.

Why the weights are what they are: the two things the challenge says must be captured honestly
are light and colour, so colour is 40%, split between a global distribution comparison (does the
fabric read the same overall) and a spatial one (is the colour in the right place, so a red
collar on a blue jacket cannot win on histogram alone). LPIPS catches material rendering and
local texture. CLIP anchors high-level semantics (it is still the same kind of garment).
Silhouette IoU punishes wrong shape or missing pieces. SSIM is a cheap structural tiebreaker.

---

## 3. Status: ok, review, invalid

| status | when | effect on score |
|---|---|---|
| invalid | the file cannot be read; file > 100 MB; min side < 256 px; max side > 12000 px; aspect ratio > 3:1; decompression bomb (> 150 MPix); animated or multi-frame image; the file is a symlink; a flat-field image (whole-image luma standard deviation < 4/255) | score 0 |
| review | any of: `mask_empty` (mask < 1% of image), `mask_suspect` (> 92%), `truncated` (mask touches the border over > 10% of that side) on the reference or your submission, `multi_component` on your submission while the reference is a single component, or `clip` sub-score < 0.30 | score kept, a human looks at it |
| ok | none of the above | |

`review` is not a penalty and not a rejection. It means the scorer is not fully confident and a
person checks the pair before the leaderboard is published. Model uncertainty never becomes a
hard cap; only malformed input does. If one of your files throws an unexpected error, it is
recorded as `invalid` with a `scorer_error:*` flag. One bad file never aborts the event.

When a reference is wrong (the segmenter mis-segments a garment), organisers correct the reference
side only, never a submission: a hand-made reference mask for that garment. The corrected reference
applies to every team unchanged, every affected garment is re-scored, and the change and the re-run
are logged and announced. No score is ever edited by hand.

---

## 4. Submission format

```
submissions/<team>/<garment_id>/front.<png|jpg|jpeg|webp>
```

The scorer is generic over views, but in this challenge the only required view is `front`, so every
reference and submission uses `front.<png|jpg|jpeg|webp>`.

dev ids and test ids never collide, so one team folder can hold both. PNG is the safe default
(lossless). Full rules and a validator are in [docs/SUBMISSION.md](SUBMISSION.md).
---

## 5. Aggregation: garment scores, team scores, ties

* The required view is `front`, the only view in the references.
* Garment score = mean over the required views. A missing view scores 0 (it stays in the mean).
* Team score = mean over all reference garments. A garment you did not submit scores 0. Equal
  weight per garment.
* Rounded to 2 decimals. Ties break on the aggregate colour sub-score, then aggregate LPIPS, both
  averaged per garment like the main score. Teams equal on all three share a rank.

A wrong guess is almost always better than an empty slot, so never submit an empty folder on
purpose.

On dev the leaderboard reports team totals only: gaps under about 3 points are within run-to-run
noise, and in rehearsal final test scores landed about 10 to 13 points below the dev numbers.

An official run writes `leaderboard.csv`, `scores.jsonl` (one full `ScoreReport` per slot,
including missing ones), `review.csv`, `garments.csv`, and optional `debug/<team>/<garment>_<view>.jpg`
contact sheets (`ref | sub | canon ref | canon sub | masks | per-cell dE00 heat map`).

---

## 6. How to self-evaluate

There is no public validation split. Carve a holdout out of the example pairs (`train/`) and score
pairs you did not tune on, with the same scorer the organisers use. First make the split, keeping
pairs that share a target on the same side:

```bash
.venv/bin/python scripts/make_holdout.py --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout/ --group-by-target
```

Then predict on the holdout inputs and score the predictions:

```bash
.venv/bin/python baselines/baseline_identity.py --manifest holdout/val.jsonl --out predictions --team identity --dataset ../sartiq_neural_wave_2026
.venv/bin/python scripts/eval_val.py --pairs holdout/val.jsonl --preds predictions --team identity --out val_out --jobs 1
```

`--jobs 1` is the safe default: each worker loads its own copy of the models (about 1.4 GB of RAM
each). Raise it only after the models are cached.

`eval_val.py` writes `val_out/val_manifest.jsonl` (exactly what the scorer saw) and
`val_out/leaderboard.csv`. The scorer accepts a JSONL manifest as an alternative to a reference
tree. One row looks like this (the ids are your holdout's; `ref` is the target, `sub` is your
prediction):

```jsonl
{"garment_id": "t0001", "view": "front", "ref": "../sartiq_neural_wave_2026/train/t0001/target.jpg", "sub": "predictions/identity/t0001/front.jpg"}
```

```bash
.venv/bin/python -m sartiq_score.cli batch --manifest val_out/val_manifest.jsonl --out val_out
```

Two things the scorer enforces, both verified:

* a row must carry `garment_id` (aliases `garment`, `id`, `gid`, `name`). A row with only
  `pair_id` is silently skipped, and if every row is skipped the run fails.
* a row without `sub`, with no submission tree, is scored as a missing slot with score 0. You
  cannot hide a missing prediction.

Watch one pair first, it is faster to iterate with. The first line of `holdout/val.jsonl` gives you
a pair id. Bash / zsh / macOS / WSL2:

```bash
PAIR=$(.venv/bin/python -c "import json,pathlib; print(json.loads(pathlib.Path('holdout/val.jsonl').read_text().splitlines()[0])['pair_id'])")
.venv/bin/python -m sartiq_score.cli pair --ref ../sartiq_neural_wave_2026/train/$PAIR/target.jpg --sub predictions/identity/$PAIR/front.jpg --json
```

PowerShell (native Windows):

```powershell
$PAIR = (Get-Content -LiteralPath "holdout/val.jsonl" -Encoding UTF8 -TotalCount 1 | ConvertFrom-Json).pair_id
.venv\Scripts\python.exe -m sartiq_score.cli pair --ref "../sartiq_neural_wave_2026/train/$PAIR/target.jpg" --sub "predictions/identity/$PAIR/front.jpg" --json
```

`--json` does not make stdout pure JSON. The scorer prints the LPIPS model banner before the
report, so `json.load(sys.stdin)` on the whole stream fails. For machine-readable output, read
`val_out/scores.jsonl`, which `batch` writes (one `ScoreReport` per slot, including missing ones).
If you must parse stdout yourself, drop the banner and parse from the first line that is a lone `{`
to the end of output; that is what the scorer does internally.

and render a contact sheet when a number surprises you. Bash / zsh / macOS / WSL2:

```bash
.venv/bin/python -m sartiq_score.cli debug --ref ../sartiq_neural_wave_2026/train/$PAIR/target.jpg --sub predictions/identity/$PAIR/front.jpg --out sheet_$PAIR.jpg
```

PowerShell (native Windows):

```powershell
.venv\Scripts\python.exe -m sartiq_score.cli debug --ref "../sartiq_neural_wave_2026/train/$PAIR/target.jpg" --sub "predictions/identity/$PAIR/front.jpg" --out "sheet_$PAIR.jpg"
```

Timing measured here: about 1.5 to 4 s per pair warm on a 4-core CPU (up to about 4 s when the CPU
is busy), with a one-off startup and a one-off ~1 GB model download on first use.

If you build your own reference tree (`<refs>/<garment_id>/front.<ext>`, the view is `front`), you can
score it with the batch form instead of the manifest form. Syntax template, the bracketed names are
placeholders, not literal:

```
.venv/bin/python -m sartiq_score.cli batch --refs <refs> --subs submissions --out my_scores --debug-sheets --jobs 1
```

---

## 7. What the jury looks at

The automated ranking picks the finalists (number announced at the event). Teams tied at the finalist
cutoff all advance, so the jury round can hold more teams than that. The Sartiq panel then
judges the finalists on commercial usability, criteria unchanged from
[the challenge document](Sartiq_Neural_Wave_2026_Challenge.pdf):

* colour and material fidelity to the real garment,
* silhouette and geometry correctness,
* background and lighting cleanliness,
* absence of artifacts (ghosting, warped seams, hallucinated detail),
* consistency across a product's shots.

Bonus points (unchanged): how well the capture protocol generalises beyond the example pairs,
efficiency (time per garment), and degree of automation.

---

## 8. Known limitations

* Reference similarity is not fidelity. A photograph measures rendered appearance, not physical
  fabric reflectance. No exposure or white-balance normalisation is applied to your submission.
  A colour cast in your capture shows up as a colour penalty. That is part of the problem.
* Style is not neutral. Ghost-mannequin output is penalised against flat-lay references by
  silhouette, LPIPS and SSIM. Every target in this challenge is flat-lay.
* One segmenter for everyone. `rembg u2net` is not garment-specific. White, cream, lace and
  thin-strap garments on a light background are its known weak spot, which is why `mask_empty`
  and `mask_suspect` produce `review` rather than a silent bad score. If your garment silhouettes
  poorly, shoot it against a background that separates it.
* Anchors are a declared policy, not a fitted model. They were chosen from development
  perturbations and frozen. No human-preference labels were used.
* Not bit-exact across machines. Threads are pinned to 4, models run in eval mode with
  deterministic algorithms requested. The guarantee is repeatability within the pinned judging
  environment, with numerical tolerance elsewhere.
* You get detailed metrics only on data you can score yourself, which is your own holdout. The
  dev leaderboard reports team totals. The dev and test references stay private before and after
  the event.
