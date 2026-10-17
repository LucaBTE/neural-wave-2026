# Sartiq Catalogue Studio

## Selecting the centered garment

SAM receives a positive point at the image center and returns up to three candidate
masks in a single inference. Local OpenCV/NumPy selection ranks actual foreground
coverage in the central 40% of the image, with a centroid-distance penalty. It keeps
one candidate rather than unioning unrelated garments. Detached corner components
such as the holder's sleeve are excluded while the complete connected central garment,
including sleeves extending toward the image edges, is retained. Selection does not
require foreground at the exact center pixel, so a gap between trouser legs is valid.

A mask containing only peripheral clothing or almost the entire frame stops the job
before analysis/generation. `target_selection` in job data and the preparation trace record
the candidate index, candidate count, central coverage and selected bounds. Centering
is a geometric prior, not semantic certainty: touching/overlapping garments can remain
ambiguous and the live mask preview should be checked.

The selected mask is used for color estimation, the analysis crop, both generation inputs
and the source side of the quality comparison. Generation receives a close crop of the
isolated product and an isolated reference at the source framing; it no longer receives
the full unmasked scene with competing garments. The image generator does not receive
an inpainting mask: product isolation is enforced in its reference images. Previously,
it could infer the intended product from the full scene despite an incorrect mask,
which explained a correct final garment alongside a wrong sleeve analysis.

There are five model inferences per uncached run, including dedicated pressing. Up to three mask downloads are
needed for selection. Two selected-product references are uploaded concurrently; total
image upload operations are now four (original for SAM, two isolated views, final PNG).
The extra isolated crop replaces unsafe reuse of the unmasked original as a generation
reference when white balance is unchanged. No automatic paid retry is added.

## Models and libraries used per step

Each step prints start/completion lines in the backend terminal with job ID, model
endpoint (or `local`), installed library versions, execution status and duration.
Image preparation is included when running through the API. LangGraph steps record
their orchestration library too; Python's version is included in structured metadata.
No prompt text, image data, image URLs or credentials are printed by this trace.

The same trace is streamed to the browser in `execution_trace` and can be expanded
under **Execution details**, during processing or after completion. It shows the
actual configured analysis/generation models, cached analysis without a new model
request, analysis timeouts, and failed steps. Local color correction and OpenCV
quality checks are explicitly labeled as local processing. Intermediate snapshots
are views of these steps, not additional model executions.

The validation script also saves `<input-name>_trace.json`. Step library versions
refer to locally installed packages; remote models are identified by endpoint ID,
since provider-side model weight versions are not available here.

`backend/.env.example` is an intentionally trackable configuration template.
The project-level `.gitignore` overrides the root `.env.*` rule for this file only;
real `.env` and other environment variants remain ignored. No force-add is necessary.

## Required garment analysis

The default is **`fal-ai/moondream3-preview/query`**, replacing SA2VA 8B for
description. The request explicitly disables extended reasoning (`reasoning=false`)
and sets `temperature=0`. Analysis uses a tight crop of the isolated,
color-corrected garment with 5% padding, bounded to a 1280px longest side without
upscaling. The generator receives two selected-product references retaining source garment detail.
The English analysis prompt is editable in `backend/prompts/description.txt` and
requests at most 120 words covering type, apparent fit, construction, texture,
colors and print details. This is a prompt instruction, not an enforced token cap.

Only this descriptive crop is encoded as quality-95 JPEG with 4:4:4 chroma (no
color subsampling) and sent as a data URI, adding no storage upload. Generator
references remain lossless and retain source garment detail. The analysis snapshot
shows the crop before encoding. Successful descriptions are cached for one hour
in a process-local LRU capped at 128 entries. Keys include the exact crop bytes,
model and complete prompt including the lighting note; changed input/prompt/model
cannot reuse old context. Only text and hashes are cached. Repeated generation of
an identical garment can therefore use 4 model calls instead of 5; concurrent first
requests are not coalesced. Caches are lost on restart, and failures are not cached.

Analysis retains its own 90s SDK client timeout, rather than sharing generation's 300s
budget. This is a safety ceiling, not a target duration. Reduced work comes from the
different model, disabled reasoning, shorter answer and lighter transfer. The SDK
attempts cancellation on timeout; cancellation and billing cessation are best effort.
There is no automatic paid retry or fallback-model call. Analysis is mandatory:
timeouts, empty/malformed output, explicitly truncated responses and provider errors
stop the job before generation. The former `FAL_ANALYSIS_REQUIRED=false` setting
cannot bypass analysis; no placeholder description is substituted.

`analysis` in job data records model, successful status, cache hit, input size,
JPEG byte size, reasoning mode, timeout and duration. The execution trace includes
these request settings and the actual model. To tune settings, edit `.env` and restart:

```env
FAL_ANALYSIS_MODEL=fal-ai/moondream3-preview/query
FAL_ANALYSIS_TIMEOUT=90
FAL_ANALYSIS_MAX_SIDE=1280
```

Allowed timeout is 15–300s and max side is 768–2048px. Legacy SA2VA 8B/4B endpoints
remain selectable for comparison and receive only their supported `image_url` and
`prompt` fields. Existing explicit model overrides must be updated to use the new
default. Neither latency nor equivalent description quality has been benchmarked
against live fal for this change. Provider cold starts, queue load and outages can
still cause delays or timeouts; zero timeouts cannot be guaranteed by client code.
See the [Moondream 3 query API](https://fal.ai/models/fal-ai/moondream3-preview/query/api).

## Live previews and intermediate images

The right-hand image updates during processing and ends with the final output.
At completion, **View all steps** opens a gallery of the original, object mask,
isolated garment, color correction, generation reference, garment analysis input,
generated photo, steam-pressed photo, transparent cutout, white-background composition and final output.
Click any thumbnail to open it at its available resolution. Analysis does not change
pixels, so that step shows its cropped input rather than inventing an edited output. If a job fails,
**View available steps** opens the snapshots completed before the error.

Progress snapshots contain `current_image` and `images` metadata (`id`, `label`, `url`).
Local snapshots are served by `GET /api/jobs/{job_id}/images/{image_id}` as PNG previews
bounded to 1280 pixels on the longest edge. Full processing images are not resized by
snapshot capture. Existing fal image/reference URLs are reused where available.
SSE carries image URLs rather than image bytes; previews require no additional fal
inference or storage uploads. In-memory local previews share the job lifetime and
are lost when the backend restarts. The final image URL remains the full result.

## Request volume and latency

The installed fal Python SDK defaults to polling inference status every 0.1s.
Those HTTP status requests do not start new model inferences. The pipeline now
explicitly polls every 2s, reducing the nominal polling rate from 10/sec to 0.5/sec
(95% fewer while waiting). This can delay observing a completed inference by up to
about 2s per call; it reduces request volume, not model compute time. SDK retries,
uploads, downloads and result retrieval are additional HTTP traffic.

The browser uses one SSE connection instead of repeated requests every 1.8s. Keep
proxy buffering disabled for this endpoint (`X-Accel-Buffering: no` is sent).
The backend still polls fal because the SDK's `subscribe` method is not a push stream.

Output SAM is omitted; dedicated pressing adds one generation edit, for five paid
model invocations per uncached run (four with cached analysis). Selected-product reference
uploads run concurrently. There are four image upload operations, versus five previously.
The original upload is used only for SAM, preserving consistent product identity across
the subsequent stages. A storage upload can itself require multiple HTTP requests.

After completion, `metrics.stages_seconds` and `metrics.total_seconds` in the job
show measured backend stage durations (including provider wait and file transfers).
The total is the sum of timed stages; it excludes the initial local worker-queue wait.
The validation script writes `<input-name>_timings.json`. Live before/after latency
and the new output matte quality still need benchmarking with representative photos;
no runtime or quality improvement percentage is claimed from mocked tests.

Pro at 2K remains the default. To compare a different generation budget, edit backend
`.env` and restart:

```env
FAL_IMAGE_MODEL=fal-ai/nano-banana-2/edit
FAL_IMAGE_RESOLUTION=1K
```

Supported models are `fal-ai/nano-banana-pro/edit` and `fal-ai/nano-banana-2/edit`;
resolutions are `1K`, `2K`, `4K`. Settings are validated before paid work. A 1K
generation is still delivered on a 1200×1800 canvas, which does not restore missing
detail. Compare print/color fidelity and timings before choosing a new default.
No automatic fallback/retry adds paid inferences.

References: [fal Python SDK polling implementation](https://github.com/fal-ai/fal/blob/main/projects/fal_client/src/fal_client/client.py),
[Nano Banana 2 input schema](https://fal.ai/models/fal-ai/nano-banana-2/edit/api).

## Default generation prompt

Edit **`backend/prompts/generation.txt`** to change the photographic direction.
This is the actual English template sent to Nano Banana Pro, read afresh for every
job. It specifies overhead flat lay photography, natural balanced positioning,
fully pressed fabric panels without accidental folds, consistent construction, soft neutral
lighting, white background, and preservation of colors, prints and lettering.

Keep `{{LIGHTING_INSTRUCTIONS}}` and `{{GARMENT_DESCRIPTION}}` exactly once each.
The pipeline fills them with the lighting assessment and the visual object description.
The description is inserted as a JSON string of auxiliary observations; image
references take precedence over model guesses. A missing or invalid template stops
the job before image generation. Template edits need no backend restart.

The exact assembled prompt is available as `generation_prompt` from
`GET /api/jobs/{job_id}`, including during generation. The validation script also
saves it as `validation/<input-name>_prompt.txt`. The interface stays a single-click
flow. Styling directions guide generation but do not
guarantee perfectly reproduced prints or the removal of every crease.

The mandatory follow-up edit uses **`backend/prompts/pressing.txt`**, reloaded for
every job and exposed as `pressing_prompt` in the job API. It receives only the
generated photo, with no wrinkled original references. It explicitly removes
accidental wrinkle geometry and shading, including on knitwear, while preserving
fine stitches, ribbing, seams, intentional pleats, colors and artwork. This adds
one model call at the configured generation resolution. A pressing error stops
the job; the initial generated photo is never silently used as the final result.
The stage appears as **Removing wrinkles** in progress and **Steam-pressed photo**
in the gallery, with its own model/version trace and timing. The validation script
saves `<input-name>_pressing_prompt.txt`. OpenCV does not verify wrinkle removal;
inspect smoothness and identity visually. Live pressing quality remains unvalidated.

Upload one garment photograph and press **Generate photo**. The app produces a centered 1200×1800 studio photograph on white, requesting completely pressed panels while preserving colors, graphics and construction. Live previews and an optional intermediate-image gallery show progress.

## Start

Backend (Python 3.10+), from this directory:

```bash
cd backend
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

On first setup create the virtual environment with `python -m venv .venv`, and copy `.env.example` to `.env`. Set `FAL_API_KEY` or `FAL_KEY`; keep an existing `.env` intact.

Frontend, in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. Restart the backend after upgrading. No additional dependencies beyond the previous LangGraph pipeline are needed.

## Pipeline

LangGraph runs the entire workflow without user intervention:

1. **SAM 3** (`fal-ai/sam-3/image`) identifies the centered garment using a center point and local candidate-mask selection. A local illumination-normalization node then prepares two views of the selected, isolated product.
2. **Moondream 3 Query** (`fal-ai/moondream3-preview/query`), with extended reasoning disabled, describes the isolated color reference: category, apparent fit, construction, fabric appearance, colors, prints/lettering and uncertainties. It is asked for text only. The API's `output` is cleaned of markup and bounded to 6000 characters. Analysis must succeed before generation begins.
3. **Nano Banana Pro Edit** (`fal-ai/nano-banana-pro/edit`) receives both references plus the visual description as auxiliary observations. Image evidence and color-preservation rules take precedence over the description. The prompt requests flat lay catalogue arrangement, complete pressing, mannequin removal, neutral lighting and preservation of lettering/artwork. One 2K portrait image is requested.
4. **Dedicated pressing**, using the same configured image model and resolution, edits only the generated photograph to remove remaining accidental wrinkles and signs of use. It preserves fine fabric texture and product details; it adds one inference without automatic retries.
5. **BiRefNet v2** (`fal-ai/birefnet/v2`) directly isolates the garment from the *pressed* image URL. Pillow uses its alpha to place the pressed image's RGB on a real white 1200×1800 canvas, centered without distorting proportions. No second SAM call or intermediate upload is needed. Visually check edges and stray objects.
6. **OpenCV** checks color drift against the color reference, colored print/detail regions separately, and background whiteness. Shape/structure metrics remain diagnostic. It does not check wrinkle removal. The final PNG is uploaded and returned.

Each uncached run uses **five paid fal inference calls** (SAM, Moondream, Nano Banana twice and BiRefNet), plus uploads. Reusing a cached description reduces this to four. There are no automatic paid retries or fallback to the original photograph. The target examples are not uploaded or used as references, avoiding copying their garment designs.

The former Bria/Kontext route and its fixed-position/preserve-silhouette instructions have been removed. Nano Banana is selected for instruction-based multi-reference restyling, not on the basis of a measured benchmark against these samples. Its live output quality still needs validation.

The opaque background is composited locally. Cutout quality still depends on segmentation, particularly for white garments. A failed segmentation stops the job. Composition preserves pixels inside the pressed garment; it performs no additional fabric smoothing.

## API

- `POST /api/studio`: multipart `file` (JPG/PNG/WebP, max 12 MB), returns HTTP 202 with `job_id`. No mask confirmation or additional render call.
- `GET /api/jobs/{job_id}`: `status`, `result_url`, `quality`, `garment_description`, `error`. The description is available for inspection via the API, without adding a UI step. Internal masks/references are not included.
- `GET /api/jobs/{job_id}/events`: SSE progress stream used by the frontend. It sends the current job on connection and updates when state changes, plus idle keepalive comments. It closes after a terminal state; reconnects receive the latest snapshot.
- `GET /api/config`: model IDs and output dimensions.

Statuses: `queued → preparing → balancing_color → describing → generating → pressing → finishing → checking_quality → complete | review_required`; exceptions produce `error`. Old `/api/segment` and `/api/render` routes are replaced by `/api/studio`.

## Validation

```bash
backend/.venv/bin/python -m pytest -q tests
cd frontend && npm run build
```

Tests execute real LangGraph/FastAPI code with simulated fal responses and verify reference handoff, output composition, error propagation, quality behavior and the single-request API. No API key is required for tests.

To validate actual output after authorizing image uploads and fal charges:

```bash
backend/.venv/bin/python scripts/validate_studio.py /path/to/input_1.jpg
```

This writes the generated PNG and quality report into `validation/`. It does not read target images. Live validation on the two supplied examples has not yet run: automatic approval review requires explicit consent for uploading those photos and consuming fal credits.

## Limits

Generative restyling can alter exact lettering, prints and texture. The prompt requires preservation, but OpenCV does not verify semantic identity or exact text. A passing report is only a color/background check; visually compare these details before publication. Neutralizing warm indoor lighting can legitimately alter observed RGB colors.

Jobs live in process RAM and disappear on restart. This prototype has no persistent checkpointer, authentication or quotas. fal receives uploaded photographs; keep API keys server-side. `reference/` and `backend/reused/` retain the extracted project's code and attribution obligations.

## Model documentation

- [Nano Banana Pro Edit: multi-image input, resolution and aspect ratio](https://fal.ai/models/fal-ai/nano-banana-pro/edit/api)
- [SAM 3](https://fal.ai/models/fal-ai/sam-3/image/api)
- [BiRefNet v2](https://fal.ai/models/fal-ai/birefnet/v2/api)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api)

## Color fidelity under different lighting

White-balance strength defaults to `1.05`, a modest increase in chromatic correction;
the neutral-cast threshold is 1.5%. Set `COLOR_CORRECTION_STRENGTH=1.0` in backend
`.env` for the former gain strength, or tune within 0.5–1.5. Gain and clipping limits,
background-reference checks and mixed-light rejection still apply. Strength is recorded
in `quality.illumination.strength`. Stronger correction is a visual tuning choice,
not a guarantee of more accurate physical colors; it can slightly overcompensate a cast.

Before generative restyling, `backend/color.py` estimates RGB channel gains from bright, unclipped, low-chroma candidates in the **original** scene. It requires spatially distributed background candidates outside an expanded object mask. The object itself does not need neutral pixels and its color does not influence the light estimate. A linear-RGB transform handles warm, cool, green and magenta casts without repainting prints or making a red-specific hue substitution. No additional fal inference call is needed.

This is an uncalibrated heuristic: low-chroma pixels are possible neutral references, not proof of a white/gray material. Consistent colored surfaces can fool it. It does not promise recovery under every light. Missing reference candidates, strongly colored light, potential clipping, or regional disagreement cause it to leave pixels unchanged and return `uncertain`. Regional disagreement can indicate mixed lighting **or different surface colors**; the algorithm does not assert which. Mixed light is not corrected with a fabricated global average. The restyling prompt covers these conditions conservatively, and a job with uncertain illumination always requires color review.

The generator receives two views of the corrected, isolated garment. The complete source photograph remains available locally for background-based illumination estimation. When correction was applied, generation is instructed not to perform a second global white balance. The quality report compares against the corrected source masked to the selected garment rather than rewarding the original cast. `quality.illumination` records status, RGB gains, evidence and rejection reason. `quality.color_reference` identifies whether the reference is estimated white balance or uncorrected input. `print_color_distance` tests saturated details separately so a large white shirt does not hide print drift; it is not OCR or an exact identity check.

No target photo is used for correction. The one-button UI uses five fal inferences without cached analysis. Exact color recovery under mixed, narrow-spectrum or clipped illumination requires better capture/reference information. Synthetic tests cover warm/cool/green/magenta, mixed light, neutral and colored surfaces, and small-print drift; live fal fidelity has not been validated.

Background on the neutral-pixel assumption: [OpenCV GrayworldWB documentation](https://docs.opencv.org/4.1.2/javadoc/org/opencv/xphoto/GrayworldWB.html). This implementation adds spatial agreement/rejection and linear-RGB processing; it does not call the OpenCV contrib algorithm.

## Visual description stage

Graph order: `isolate → balance → describe → restyle → press → finish → verify`.
The description is English to match the generation prompt, and explicitly qualifies apparent fit, uncertain colors and unverified material composition. Existing graphics are described, not redesigned; unclear text must be marked unreadable. Descriptions can still contain model errors, so they are contextual hints rather than verified product metadata. The generator is instructed to treat them as data, not executable instructions, and defer to reference images on conflicts.

An absent/empty/non-text `output`, timeout or provider error stops the job before paid image generation. There is no automatic paid retry. No new packages or user inputs are needed. The integration sends `image_url`, `prompt`, `reasoning=false`, `temperature=0` and reads `output`: [official API schema](https://fal.ai/models/fal-ai/moondream3-preview/query/api).

### Background-based illuminant extraction

The estimator now votes across background tiles, excludes a band around the garment, and rejects internally inconsistent tiles, clipped highlights, saturated surfaces and dark pixels. Agreeing reference tiles yield an estimated light RGB vector. Its inverse channel gains are applied in linear RGB to the original photo, including the object, before the background is replaced. Garment color is not averaged toward gray: solid red/blue/green and cream garments can be corrected using independent scene references.

`quality.illumination` includes `estimated_light_rgb` (relative linear RGB), `estimated_light_srgb` (display swatch, not temperature), `reference_tiles`, `background_candidate_fraction` and the explicit assumption behind the estimate. Neutral scene light also tells the generator to avoid a second global correction.

This estimates illumination only if candidate surfaces are actually neutral and share the object's lighting. Beige walls, colored reflections and separate lighting on the product can violate that assumption. Mixed/conflicting background estimates are flagged rather than averaged. Exact recovery still needs a known neutral/color reference photographed under the object's light; no method can infer missing spectral/color information reliably from every single JPEG. Tests cover colored garments under warm/cool/green/magenta illumination and rejection of saturated backgrounds. No live fal calls were made for this change.
