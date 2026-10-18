# Color fidelity

Color refinement is local, after generation. Nano Banana, prompts, pose handling,
resolution, graph nodes and inference counts are unchanged. Production correction
receives source RGB, generated RGB and available foreground masks; it cannot read
dataset IDs, targets, baseline images or scores. It uses existing Pillow, NumPy
and OpenCV dependencies.

## Why the previous guard was insufficient

The latest saved comparison (`20261017-184047-273968`) scored 63.05/100 for Studio
color against 67.06 for the baseline. The old final correction only handled dark,
nearly neutral products. It skipped colored products and bright garments. Generated
colors were sometimes brighter or more saturated than supported by the source.
The striped dress and green trousers illustrate this; the printed T-shirt also
retained a warm cast.

Unconditional source histogram matching made the warm-lit printed T-shirt worse
in local experiments. Unconditional source room white balancing previously turned
a black sweater blue. Neither approach is used by the faithful profile.

## Production behavior

1. **Color-managed input.** Apply EXIF orientation and transform an embedded ICC
   profile to sRGB with Pillow/LittleCMS, rather than treating camera/display RGB
   values as sRGB. Remove stale profile metadata after conversion. Malformed
   profiles fall back to RGB decoding without discarding the photograph. See
   [Pillow ImageCms](https://pillow.readthedocs.io/en/stable/reference/ImageCms.html).
2. **Generated studio neutrality.** Use bright, unclipped background samples from
   at least four peripheral tiles across multiple rows and columns, excluding the
   generated product and its boundary. Require agreement within and between tiles.
   The generator was explicitly asked for a neutral studio surface; this assumption
   applies to that surface, never to the original room or cream fabric. If a small
   consistent cast is present, apply bounded linear-RGB channel gains to the frame.
   Warm, cool, green and magenta casts are treated alike. Compress highlights with
   one common factor per pixel to avoid independent channel clipping. Conflicting
   regions, clipped backgrounds and excessive casts are left alone.
3. **Foreground material anchor.** Sample eroded source/generated mask interiors
   on previews up to 512 px. Reject empty/tiny masks and narrow fragments, excessive
   chroma disagreement, incompatible material distributions and ambiguous bright
   materials. Compare robust log-RGB exposure centers, clipping extreme samples;
   restore half an unsupported lightness lift, capped at four L* units. Never
   lift blacks. This retains the generated lighting range rather than copying
   source shadows or wrinkles.
4. **Bounded chroma correction.** For supported chromatic materials, restore half
   the median Lab chroma offset, capped at four units in the a*/b* plane. Large
   hue disagreement is rejected. For low-chroma materials, preserve generated hue
   and adjust chroma radially instead; near-neutral hue is sensitive to noise.
   Blend through the generated alpha and protect bright labels/hardware/highlights.
   Outside the alpha, this material adjustment preserves pixels exactly. Lab uses
   the [OpenCV documented ranges](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html).
5. **Missing-source-mask fallback.** Keep the conservative dark-material guard.
   First require spatial agreement in the original central region. Then, when
   supported, include its contiguous surrounding material so a central lighting
   hotspot does not determine the whole garment's exposure. Disconnected regions
   and incompatible chroma cannot supply this estimate. These are color samples,
   not a replacement SAM segmentation. Only a common linear exposure gain is
   applied; bright details and colored prints remain protected.

No pixels are moved, no fabric is blurred, and no source texture, target palette
or per-garment settings are transferred. The generator still receives each actual
source photograph once, with scene context intact. Original room illuminant
estimation remains available only in the optional `staged` profile.
That profile anchors against its already balanced source reference, so the final
refinement does not intentionally reintroduce the original capture cast.

Decisions and numerical adjustments are recorded under `quality.color_fidelity`
(`backdrop`, `material`, `dark_fallback`) and the final execution trace. A **Color
fidelity** intermediate appears when a correction is applied. Benchmark runs now
save actual source/result masks alongside the generated RGB for exact future
replays, whenever those masks exist.

## Official local evaluation

Replayed the same correction on the five existing Studio outputs, then ran
`dev/eval_val.py`, which delegates to the **unchanged official scorer**. All five
slots are included; no images were dropped. Original predictions and reports were
preserved. No paid generation was performed.

Original SAM/BiRefNet mattes were not saved in these older runs. Replay mattes were
reconstructed with local U2Net, restricted to the recorded source selection bounds.
An unavailable source mask remains unavailable and uses the production fallback;
the narrow T-shirt print selection is rejected as a material reference. These
reconstructed masks are an approximation, not the actual production model outputs.
Targets were read **only by the evaluator**, never by the correction function.

Latest run, `20261017-184047-273968`, metrics out of 100:

| Metric | Baseline | Saved Studio | Studio color replay |
|---|---:|---:|---:|
| Overall score | 61.02 | 61.69 | 65.53 |
| Color | 67.06 | 63.05 | 71.56 |
| LPIPS similarity | 30.83 | 37.03 | 37.43 |
| CLIP similarity | 75.40 | 72.55 | 74.63 |
| Geometry | 79.01 | 86.53 | 86.57 |
| SSIM | 66.62 | 66.96 | 67.32 |

An earlier generation of the **same five garments**, `20261017-174403-808028`:

| Metric | Baseline | Saved Studio | Studio color replay |
|---|---:|---:|---:|
| Overall score | 59.12 | 61.11 | 63.63 |
| Color | 68.41 | 62.95 | 68.56 |

The latest color gain is **8.51 points over saved Studio and 4.50 over baseline**.
The earlier run improves too, with a much smaller margin over its baseline.
RGB changes can slightly change the scorer's independently recomputed segmentation
and thus its geometry number, despite unchanged pixel coordinates. Some individual
metrics/garments can regress; these aggregate results do not prove universal gains.

Official replay reports for this workspace:

- `dev/.cache/color-analysis/material-replay/latest/final-scores/`
- `dev/.cache/color-analysis/material-replay/previous/final-scores/`

These are exploratory replays on development examples used during diagnosis,
**not independent validation or fresh end-to-end results**. Actual SAM/BiRefNet
masks and new random generations can change the result. Confirm on more garments
before claiming a general advantage. Without a calibrated reference, reflectance,
exposure and illumination cannot always be distinguished; the correction reports
`calibrated: false` and skips ambiguous cases instead of inventing true colors.

To regenerate Studio and compare against the same latest baseline images:

```bash
/bin/python3 dev/compare_pipelines.py \
  --dataset /home/luca/Downloads/sartiq_neural_wave_2026 \
  --split train --count 5 \
  --baseline-from dev/comparisons/20261017-184047-273968
```

This incurs five new Studio generations, reuses the saved baseline predictions,
and scores both with the official evaluator. For a broader check, run the same
command with a larger `--count` and omit `--baseline-from` (both methods regenerate).
