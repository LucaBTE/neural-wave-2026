# Catalogue Studio pipeline

The default `faithful` profile uses LangGraph and **one normal generative edit**,
with one corrective edit only for positively detected duplicate products. Upload
1–5 photographs of the same product and press **Generate photo**. The main photo
sets the preferred view; if it is a close-up, a complete additional photograph
provides the silhouette. All photos supply construction and artwork details.

| Step | Purpose | Model / libraries |
|---|---|---|
| Image preparation | Apply EXIF orientation, transform embedded ICC profiles to sRGB, limit the longest edge to 2048 px. | Python + Pillow/LittleCMS; local |
| Identify and describe | Select the foreground product from all supplied views in one request and describe its visible construction. Return a physical category, not a brand or printed title. | `fal-ai/moondream3-preview/query`, reasoning disabled; local multi-view board |
| Locate | Get native bounding boxes for the classified category and select the central candidate. | `fal-ai/moondream3-preview/detect` |
| Validate segmentation | Segment the whole selected product; reject unrelated objects and small printed fragments. Keep the mask for previews, source material color samples and diagnostics. | `fal-ai/sam-3/image` + NumPy/OpenCV |
| Prepare references | Send the full main scene and each additional photograph once. Do not add a second crop of the same product. Preserve lighting context. | Pillow; local, cached uploads and URL deduplication |
| Create catalogue photograph | One 2K edit: faithful colors and artwork, front-facing flat lay, sleeves outward, complete straps/belts, steamed fabric, near-white surface and subtle contact shadow. | `google/nano-banana-2.1/edit` — Nano Banana, also used by the official baseline |
| Check for duplicate products | Inspect the generated photograph for overlapping/ghosted copies, including a plain shirt behind a printed one. One short query on a preview up to 768 px. | `fal-ai/moondream3-preview/query`, reasoning disabled, timeout at most 30 s |
| Correct and recheck, only if duplication is detected | Edit only the generated candidate to retain one product; recheck it independently. One repair maximum. A replacement not confirmed SINGLE is rejected before final delivery. | Same configured Nano Banana endpoint, then Moondream Query |
| Matte and delivery framing | Estimate foreground alpha for local color refinement and quality checks. Resize/pad the complete generated RGB frame to 1200 × 1800; alpha never cuts out delivered details. | `fal-ai/birefnet/v2` + Pillow |
| Color fidelity and final diagnostics | Remove supported studio backdrop casts, bound foreground exposure/chroma drift against the source, protect highlights and use a conservative dark-material fallback when source masks are unavailable. Record each decision. | NumPy + OpenCV; local, no new inference |

**Six inference requests cold; three when preparatory work is cached.** Classification
and description share one request. There is no separate analysis timeout and no
mandatory second image edit. Requests, uploads, downloads and HTTP status polls are
different counts. Polling remains at two seconds. A detected duplicate adds exactly
two inference requests (one edit and one recheck), capped at eight cold/five warm.
There are no automatic retries for generic provider failures.

## Single-product output

`num_images=1` requests one image canvas, not one physical object. Generation and
multi-view prompts explicitly prohibit stacked garments, second shirt bodies,
duplicate necklines and ghost copies. Each unique full source photo is sent once;
the automatically added primary crop has been removed. Additional views are detail
evidence for the same item and must not become additional rendered products.

The independent generated-image check accepts only SINGLE, DUPLICATE or UNCERTAIN.
Inner collar seams, linings, printed portraits, sleeves, lenses and a natural pair
of shoes must not be mistaken for duplicate products. The query adds no upload;
its input is a bounded JPEG preview. Successful checked pixels are reused for final
processing, avoiding a second download.

A positive DUPLICATE verdict routes LangGraph through one Nano Banana correction
and another check. If the recheck is DUPLICATE, inconclusive or unavailable, the job
ends with an error and no `result_url`; its intermediate previews remain inspectable.
An inconclusive initial check keeps the generated candidate available as
`review_required` with `quality.single_product.status=unverified` and a **Review**
label, never a passed check or **Ready** label. This preserves recovery for helper
outages without claiming certainty. A model-based check can miss hallucinations;
it is not a guarantee of semantic correctness. Live detection on the supplied
overlapping-shirt screenshot has not been benchmarked; tests simulate the verdicts.

## When a helper cannot recognize the product

A close-up of a shoulder, print or care label is not a full-product view. The
classifier now sees a locally assembled board of all supplied photographs in its
existing request. Its cache includes these views. Generation explicitly uses the
most complete photograph when the main photo is a detail.

Classification, detection, segmentation and diagnostic matting are best effort.
Unusable classifications, empty/off-center detections, rejected masks and helper
timeouts no longer stop generation. The pipeline continues from the unchanged
source photographs and skips helpers that require unavailable identity or boxes.
No bounding box, foreground mask or confidence is invented. The staged profile
also recovers from an unavailable separate description.

Nano Banana still performs a real edit; the original is never returned as a
successful generated result. Failed BiRefNet diagnostics cannot discard that edit.
Missing foreground checks produce no numerical quality scores and keep an output
available with `review_required`. `quality.recovery` and execution traces record
`fallback`/`skipped` steps and actual inference counts. No helper retry is
introduced. Invalid input, unavailable uploads/downloads and generation-provider
errors can still prevent delivery; this is not a guarantee of model recognition.

## Why the references changed

The five-garment comparison at `dev/comparisons/20261017-170416-256045` scored
59.38 for the baseline and 47.03 for the previous Studio pipeline. It exposed
placeholder descriptions copied from the classification schema, overly long prompts,
and a blue shift in the black sweater after automatic white balance.

The new prompt omits schema examples, treats captions as auxiliary evidence, and
keeps image evidence authoritative. Known placeholder responses are not accepted
as factual descriptions or automatically approved from geometry alone. The main
scene remains visible to the image generator, without a duplicate crop. SAM still
validates selection, but **the generator does not receive an inpainting mask**.
Its input is no longer restricted to whitened product pixels.

Scene-aware generative relighting retains the original room context. The original
room is not automatically treated as a white reference. Final local color refinement
uses the requested neutral generated studio backdrop and reliable foreground source
samples; it does not assume that cream fabric must become white. BiRefNet alpha
controls color blending and diagnostics, never removal of pale straps or other RGB
details. Color refinement never changes shape or composites another garment.

The unchanged official scorer measured improvements in local replays of two saved
five-garment runs. Original mattes were unavailable, so these replays reconstruct
masks locally; they are not fresh end-to-end generation results or an independent
validation set. Targets are read only by the evaluator. See
[color-correction.md](color-correction.md) for measurements and limitations.

## Editable prompts

- [generation.txt](../btc-sartiq/backend/prompts/generation.txt): garment generation,
  with short construction observations and explicit single-instance rules.
- [product_generation.txt](../btc-sartiq/backend/prompts/product_generation.txt):
  accessories and other objects, including glasses.
- [classification.txt](../btc-sartiq/backend/prompts/classification.txt): main-product
  identity and one-sentence description.
- [multiview.txt](../btc-sartiq/backend/prompts/multiview.txt): additional references,
  separating front/back details and avoiding averaging different illumination casts.
- [single_product_check.txt](../btc-sartiq/backend/prompts/single_product_check.txt)
  and [single_product_repair.txt](../btc-sartiq/backend/prompts/single_product_repair.txt):
  multiplicity check and the bounded corrective edit.

Generation templates retain `{{LIGHTING_INSTRUCTIONS}}` and
`{{GARMENT_DESCRIPTION}}` exactly once. They are loaded on each job. Actual prompts,
classification, model IDs, libraries and timings are available in job metadata and
**Execution details**. Intermediate images remain available in **View all steps**;
progress follows the actual graph path: eight nodes normally, ten after a duplicate
repair in the faithful profile. Trace entries record both check verdicts and models.

## Configuration and comparison

Defaults require no change to an existing `.env` unless it pins another model:

```env
FAL_PIPELINE_PROFILE=faithful
FAL_IMAGE_MODEL=google/nano-banana-2.1/edit
FAL_IMAGE_RESOLUTION=2K
```

`FAL_IMAGE_MODEL=fal-ai/nano-banana-pro/edit` remains supported, with the same
single-edit workflow. `fal-ai/nano-banana-2/edit` is also accepted. Explicit existing
model overrides take precedence; check `/api/config` for the actual endpoint.

`FAL_PIPELINE_PROFILE=staged` retains the previous graph structure for comparison:
local white balance, separate description, two edits for garments and cutout-based
composition, plus the same multiplicity gate (eight cold/four warm requests;
non-garments seven/three, plus two only if a duplicate repair is needed). The prompts
are the current edited templates, so this is not a byte-for-byte replay of a historic
run. `COLOR_CORRECTION_STRENGTH` and `FAL_ANALYSIS_MODEL` affect that profile;
`FAL_ANALYSIS_TIMEOUT` also bounds the default identification/localization calls.

See [compare-pipelines.md](compare-pipelines.md) to rerun t0001–t0005 with official
scores, optionally reusing the previous baseline predictions. Studio diagnostics
also save the generated image before local resizing as `front_generated.png`.

Model schemas: [Nano Banana 2.1](https://fal.ai/models/google/nano-banana-2.1/edit/api),
[Moondream Query](https://fal.ai/models/fal-ai/moondream3-preview/query/api),
[Moondream Detect](https://fal.ai/models/fal-ai/moondream3-preview/detect/api),
[SAM 3](https://fal.ai/models/fal-ai/sam-3/image/api),
[BiRefNet](https://fal.ai/models/fal-ai/birefnet/v2/api).
