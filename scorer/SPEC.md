# Scorer specification

Scorer version 2.2, package `sartiq-score 2.2.0`. The policy below is frozen before submissions
open. Weights and anchors are published values and are not fitted to submitted work.

The scorer takes a reference packshot and a submitted packshot of the same garment and returns a
reference similarity score from 0 to 100.

## What the score is

```
score = 100 x (0.40 x colour + 0.25 x lpips + 0.15 x clip + 0.15 x geometry + 0.05 x ssim)
```

Both images go through the same pipeline: decode, check limits, segment the garment, crop and
rescale to a common canvas, then compare the two canvases with five metrics. Reference similarity
measures how close the submission looks to the reference image. It is not product fidelity and not
a jury rating.

## Preprocessing

Each image is decoded on its own.

- EXIF orientation is applied. ICC profiles are converted to sRGB; on failure the image is treated
  as sRGB with a warning. Palette, greyscale plus alpha, RGBA, CMYK and 16-bit modes are handled.
  The alpha channel is composited on white and RGB under alpha 0 is ignored. White is
  `BG_RGB = (255, 255, 255)`, the same fill used for the canvas and the letterbox.
- 16-bit samples are scaled by bit depth (divide by 257). PNG `tRNS` transparency on RGB or L images
  becomes an alpha channel and is composited on white.
- Limits, read from the header before decoding where possible: minimum side 256 px, maximum side
  12000 px, aspect ratio at most 3.0. A file over 100 MB is refused without being read. A header over
  150 MPix (`Image.MAX_IMAGE_PIXELS`) is treated as a decompression bomb. Animated or multi-frame
  images are refused; a single-frame GIF is accepted.
- A decode larger than 50 MPix is downscaled before any float work, with a `downscaled_for_memory`
  warning.

### Segmentation

One model for everyone: rembg u2net (ONNX, CPU), one session created once. The composited RGB is fed
in.

The pass runs twice. Pass 1 runs on the full image. Pass 2 runs on a crop of the pass-1 mask
bounding box plus 8 % margin (`REFINE_MARGIN`), and its contour is pasted back conservatively as
`min(dilate(pass1, 1 px), pass2)`. Pass 2 is skipped when the crop covers 98 % or more of the frame
(`REFINE_MAX_AREA_FRAC`) or when a crop side is under 48 px (`REFINE_MIN_SIDE`).

The output is a soft mask. A 0.5 threshold is used for the bbox, the components and the area only.
Holes are kept, and all components of at least 0.5 % of the image area are kept (secondary
components are not dropped).

Flags: `mask_empty` when the mask area is under 1 %, `mask_suspect` when it is over 92 %,
`multi_component` when there is more than one component. An empty mask makes the canonical crop use
the whole image and the pair is set to review.

The single-pair CLI accepts a `--ref-mask-dir` override (PNG with the same stem) to hand-fix a
reference mask. Batch mode looks up `<mask-dir>/<garment>/<view>.<ext>`; the flat
`<mask-dir>/<stem>.<ext>` form is for single-pair use.

### Canonical canvas

The mask bounding box plus 2 % margin (`CANON_MARGIN = 0.02`) is cropped, uniformly scaled so its
binding dimension occupies 90 % of the canvas (`CANON_FILL = 0.90`), and centred on a flat white
background. The canvas is 512 high by 384 wide, a 3:4 portrait. The soft mask is transformed the same
way and is 0 outside the garment. There is no rotation, flip, anisotropic scaling or refinement.

### Pair micro-alignment

Before the metrics run, the submission canvas is translated by at most 1 % of each side
(`MICRO_ALIGN_FRAC = 0.01`, so at most 4 px horizontally and 5 px vertically on the 384x512 canvas)
onto the shift that maximises soft-mask IoU. The identity wins ties, and any gain under
`MICRO_ALIGN_MIN_GAIN = 1e-4` is rejected, so identical inputs stay identical. The shift is published
as `raw.align_dy` and `raw.align_dx`. Only the masks drive the shift; colour, LPIPS, CLIP and SSIM
never influence it.

## Metrics

All metrics run on the canonical pair (reference, submission).

| family | metric | raw value | similarity | weight |
|---|---|---|---|---|
| colour | `colour_global`: Lab (D65) masks, quantile distance `D` | `D` | `2^(-D/6.0)` | (see below) |
| colour | `colour_spatial`: 4x4 grid, deltaE00 between masked mean L\*a\*b\* | `dE00` | `2^(-dE00/5.0)` | (see below) |
| colour | `colour = 0.6 x colour_global + 0.4 x colour_spatial` | | | 0.40 |
| perceptual | LPIPS `alex` v0.1 on canonical RGB, inputs in [-1, 1] | `d` | `2^(-d/0.20)` | 0.25 |
| semantic | OpenCLIP `ViT-B-32 / openai`, canonical RGB letterboxed to a square | `d = 1 - cos` | `2^(-d/0.10)` | 0.15 |
| geometry | soft IoU of the two canonical masks, sum of min over sum of max | `IoU` | as-is | 0.15 |
| structure | SSIM on the L\* channel of the canonical image, clipped to [0, 1] | `SSIM` | as-is | 0.05 |

- `colour_global` uses each image's own mask (weights are the soft mask, pixels above 0.5, erosion of
  at most 1 px). `D` is the mean over `c` in {L, a, b} and `q` in {0.05, 0.10, ..., 0.95} of
  `|Qref - Qsub|` using weighted quantiles. Anchor `A_CG = 6.0`.
- `colour_spatial` uses a 4x4 grid on the canvas. For each cell with reference support (at least 5 %
  of the cell masked): if the submission support is at least 5 %, the per-cell score is
  `2^(-dE00/5.0)` using the masked mean Labs, else 0. Cells are weighted by reference area. The
  reverse direction (cells with submission support, reference weights from the submission) is computed
  the same way, and the two directions are averaged. Anchor `A_CS = 5.0`.
- LPIPS inputs are the canonical RGB (background already flat white) in [-1, 1]. Anchor `A_LPIPS = 0.20`.
- SSIM uses the L\* channel, `data_range 100`, Gaussian sigma 1.5.
- CLIP letterboxes the canonical RGB to a square on the white background so the garment is never
  centre-cropped. `d = 1 - cos`. Anchor `A_CLIP = 0.10`.

Before LPIPS and SSIM are computed, both canonical images are blurred with a Gaussian of
`sigma = 1.0` (`CANON_ANTIALIAS_SIGMA`). Only those two metrics use the blur; the value is published
as `raw.antialias_sigma`.

### Status rules

No learned hard caps.

- `invalid` (score 0): unreadable or corrupt file, min side under 256 px, max side over 12000 px,
  aspect over 3:1, header over 150 MPix, file over 100 MB, animated or multi-frame, a symlinked file
  inside a team tree (batch mode), or a flat-field submission. A flat-field submission has a
  whole-image Rec.601 luma standard deviation under 4.0 on the 0..255 scale (`UNIFORM_LUMA_STD`); its
  reason is `blank_or_uniform`, and `raw` carries `invalid_reason` and `invalid_detail`.
- `review` (score kept, a human checks it): `mask_empty`, `mask_suspect`, `truncated` on the
  reference or the submission (a truncated mask touches a border over more than 10 % of that side),
  `multi_component` on the submission while the reference is single, or a clip sub-score under 0.30
  (`CLIP_REVIEW_THRESHOLD`).
- `ok`: none of the above.

Raw values and every sub-score are always reported.

## Batch and event rules

Reference layout: `<refs>/<garment_id>/<view>.<ext>`. Submission layout:
`<subs>/<team>/<garment_id>/<view>.<ext>`. The scorer is generic over views: `<view>` is the file
stem, and the required views for a garment are the views present in its reference folder. The Neural
Wave 2026 event uses one view, `front`, so every reference and submission in the event is
`front.<png|jpg|jpeg|webp>`.

- Garment score = mean over the required views. A missing submission view scores 0.
- Team score = mean over all reference garments. A missing garment scores 0.
- Scores are rounded to 2 decimals. Ties break on the garment-weighted colour sub-score, then the
  garment-weighted LPIPS sub-score. Teams tied on score, colour and LPIPS share a rank; alphabetical
  order is display only.
- Symlinked team folders, garment folders and files are never followed. A symlinked file in a team
  tree scores `invalid` with `symlink_not_allowed`.
- Any exception while scoring one file is caught and recorded as `invalid` with a `scorer_error:*`
  flag. One bad file never aborts the batch.
- A manifest can replace the reference tree: one JSON object per line with `garment_id` (aliases
  `garment`, `id`, `gid`, `name`), `ref` (aliases `reference`, `ref_path`, `path`, `target`) and an
  optional `view` (default: the reference filename stem). A per-row `sub` or `submission` is used only
  in manifest-only mode, that is when `--subs` is absent. With a team tree, a missing file scores 0.

Outputs (`batch`): `leaderboard.csv` (rank, team, score, per-family aggregates, coverage),
`scores.jsonl` (one `ScoreReport` per team, garment and view, including missing slots), `review.csv`
(every review entry), `garments.csv` (garment, view, reference path, target style), and
`debug/<team>/<garment>_<view>.jpg` contact sheets with `--debug-sheets`.

`<refs>/garments.json` may declare per-garment metadata
(`{"<garment_id>": {"target_style": "flat_lay" | "ghost"}}`). Every target in this challenge is
flat-lay, which is the default when the file is absent.

## Determinism and versions

The `Scorer` loads the models once, sets torch threads to 4 (`torch.set_num_threads(4)`), requests
deterministic algorithms with `warn_only`, and runs under `inference_mode` in eval mode. A score is
repeatable within this pinned environment to within 0.01 across separate processes and batch
orderings. Cross-platform repeatability is not bit-exact.

`ScoreReport.versions` carries the package version, the model ids and the SHA-256 of each weight
file, so a published score can be tied to exact weights.

## v1 compatibility

`scorer/sartiq_scorer.py` is a thin wrapper with the old `--reference --generated` flags. It prints
the v2 report.
