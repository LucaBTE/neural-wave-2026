# Color fidelity

This update changes only color handling. Models, selection, poses, reference photos,
generation count, resolution and graph nodes stay unchanged. No target photograph
is passed to the generation or correction code.

## Findings from the latest five-garment comparison

Source run: `dev/comparisons/20261017-174403-808028`.

The baseline scored 59.12 overall and 68.41/100 for color. Studio scored 61.11
overall and 62.95/100 for color. Approximately 89% of the net color gap comes
from t0001: color 90.76 for baseline versus 66.30 for Studio. Its auxiliary caption
calls the garment dark blue. Studio also lifts the dark fabric substantially.
t0004 is described as olive green, another questionable color prior. These
observations motivate the changes; they do not prove which prompt phrase caused
any individual output.

## Changes

The generator receives construction observations with inferred color adjectives
removed, including those in serialized target labels. The original classification,
its detection hints and cache keys remain intact. Quoted lettering such as
`"RED WHITE BLUE"` or `'Black Sabbath'` is kept literally. The actual photographs
remain the color authority. Generation instructions require consistent removal of
lighting casts across both fabric and print, preserve intentional cream material,
and prohibit creative grading, added vibrancy and lifted blacks.

`color.preserve_dark_tones` adds a bounded local guard after generation:

- The central original must contain consistently dark, low-chroma material, with
  agreement across at least six of nine sample tiles. These are sampling regions,
  not a verified segmentation or calibrated gray reference.
- The generated foreground must also be predominantly dark and low-chroma, with
  similar chroma. Bright/colored products, mixed evidence, missing masks and
  already sufficiently dark outputs are left unchanged.
- A common linear RGB exposure gain limits an excessive lift to at most four
  L* units and never drops below 0.55. No channel-specific cast is inferred from
  walls or floors. The conversion uses OpenCV's
  [documented RGB/Lab ranges](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html).
- Soft matte weights and lightness/chroma falloffs protect background pixels,
  bright labels, colored prints and highlights. The guard does not blur texture,
  map source wrinkle patterns, resize the garment or desaturate navy into gray.

Decisions are recorded in `quality.color_fidelity` and the final trace. A **Color
fidelity** intermediate appears only when a correction is applied. The algorithm
is uncalibrated: underexposure cannot always be distinguished from a truly darker
material. It deliberately avoids universal white balancing or full histogram
matching, which could reintroduce the source lighting or force cream fabric white.

## Offline verification

The guard was replayed against the existing five Studio outputs, using source
photos only for correction. Foreground mattes were reconstructed locally with the
official scorer's U2Net because the original BiRefNet mattes were not saved. The
reconstructed mattes only support the replay; production still uses its existing
BiRefNet matte and adds no segmentation call. Targets were read only by the
unchanged official evaluator, never by the correction function.

| Metric (0–100) | Existing Studio | Color-only replay |
|---|---:|---:|
| Overall | 61.11 | 61.99 |
| Color | 62.95 | 64.85 |
| t0001 color | 66.30 | 75.85 |
| t0001 overall | 67.44 | 71.84 |

All four other output images remain pixel-identical. The replay's official
geometry aggregate is 85.8363 versus 85.8297 originally: the pixels are not moved,
but the scorer reruns segmentation on RGB and can react slightly to changed tones.
Reports are in `dev/.cache/color-analysis/replay_scores/` for this workspace.

This is a small exploratory replay, not an independent validation set or a score
for newly generated images. Its color result still trails the baseline. The updated
prompt has not been evaluated through live generation. To measure both changes
using the same baseline images:

```bash
/bin/python3 dev/compare_pipelines.py \
  --dataset /home/luca/Downloads/sartiq_neural_wave_2026 \
  --split train --count 5 \
  --baseline-from dev/comparisons/20261017-174403-808028
```

This regenerates only the five Studio outputs and runs the official evaluator for
both teams. Additional garments should also be checked before claiming the color
gap is solved generally.
