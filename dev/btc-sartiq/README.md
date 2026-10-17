# Sartiq Catalogue Studio

## Desktop and mobile

The desktop UI uses file selection and drag and drop, without requesting webcam
access, including when the browser window is narrow. On phones and iPads, the
rear-camera view opens initially; **Photos** selects existing images. Capture,
close and permission failures release any active camera stream. If live capture
is unavailable, **Take photo** uses the device camera picker. Live camera access
requires HTTPS or localhost; the native picker remains available on other origins.

Both layouts use the same dark Sartiq palette and multi-photo workflow. On small
screens, **Original / Result** switches previews. Progress, intermediate images,
model/library traces and the before/after comparison remain available.

Frontend verification: `npm run build && npm run test:ui` from `frontend`.
The browser check requires Node 22+ and Chrome (`CHROME_BIN` can override the
binary), and simulates the camera and generation API. It covers desktop,
narrow desktop, mobile capture and denied permission, without calling fal.

## Multiple input photos

Upload **1–5 photos of the same product** (JPG/PNG/WebP, up to 12 MB each,
48 MB total). Click a thumbnail to make it the **Main** photo, use **Add** for
details or other angles, and remove unwanted photos with the × button. Then press
**Generate photo** once.

The main photo drives classification, detection, segmentation and white balance.
The existing analysis call receives a locally assembled board containing the
isolated main view and all additional views. Nano Banana Pro receives the two
isolated main references plus the additional photos in the same generation call
(up to six references); dedicated pressing still edits only the generated image.
No per-view segmentation or extra inference is added. Each extra view needs at
most one cached input upload. Additional views keep their original backgrounds
and lighting; the editable `backend/prompts/multiview.txt` restricts them to detail
evidence, preserves the primary palette and separates front/back artwork. Supply
the same product in every photo; cross-view identity is not automatically certified.

The API remains compatible with single-photo clients: send multipart `file` for
the main photo and repeated `additional_files` fields for optional views.
`/api/config` exposes limits in `inputs`; progress includes source views and the
analysis board. No paid fal validation has been run for the multi-view feature.

## Classifying and selecting the main product

Before SAM, **Moondream 3 Query** examines the full scene with extended reasoning
disabled. `backend/prompts/classification.txt` asks for one deliberately presented
foreground product near the center, rather than the largest region or whatever
covers the exact center pixel. It returns an English label, kind (`garment`,
`accessory`, `other`) and confidence only. Text-generated coordinates are ignored.
For sunglasses on a table in front of a person, the target should be
sunglasses, not the person's sweater. The prompt asks for a best-supported product
name even when its exact subtype is unclear: e.g. "patterned garment" or "printed
textile item" instead of an empty identity or refusal. Low confidence no longer
blocks generation. Scores below 0.6, missing/invalid confidence, or an uncertain kind
set `classification.review_required=true`; the final output is marked for review,
including when the classification is reused from cache. Common kind aliases and
normal punctuation/Unicode in product names are accepted. Missing/unknown kinds use
`other` styling with review, rather than assuming clothing. An unusable identity
(empty name, "unknown", malformed JSON), an API failure, empty detection or invalid
segmentation can still stop the job. No extra inference or automatic retry is added,
and the original photo is never presented as a successfully generated final image.

**Moondream Detect** (`fal-ai/moondream3-preview/detect`) then localizes that label
using native `objects` output with normalized `x_min`, `y_min`, `x_max`, `y_max`.
If multiple instances are found, the one whose box center is nearest the image
center is selected. Invalid, empty or truncated detection stops before SAM.
Detection is cached separately for one hour by preview pixels, model and label.
The selected-product preview uses this detected box, never a text-generated guess.

SAM receives the classified label and detected pixel bounding box, with NO forced
foreground point. A guessed point may land on the table through a lens or between
temples, or on a background sweater, and incorrectly force SAM toward that region.
The earlier query-generated box/point could misguide SAM and make valid masks fail
the geometry checks; the screenshot showed an inaccurate box. SAM returns
up to three candidates in one inference. Local OpenCV/NumPy selection requires
agreement with the detected target region, excludes detached distractors, rejects
large unrelated components and retains holes and disconnected product parts. If no
mask matches, the job stops instead of choosing another central garment. Spatial
agreement is not a semantic guarantee; inspect the target and mask previews,
especially where objects overlap or classification is itself wrong.

`classification` in job data and the classification trace record the chosen target.
`target_selection` records the actual selected SAM candidate and bounds. The gallery
shows **Selected product** with an annotated box; that annotation is never used as
a model reference. Classification results are cached for one hour (128 entries),
keyed by full-scene preview pixels, prompt and model. Failures are not cached.

The selected mask is used for color estimation, the analysis crop, both generation inputs
and the source side of the quality comparison. Generation receives a close crop of the
isolated product and an isolated reference at the source framing; it no longer receives
the full unmasked scene with competing garments. The image generator does not receive
an inpainting mask: product isolation is enforced in its reference images. Previously,
it could infer the intended product from the full scene despite an incorrect mask,
which explained a correct final garment alongside a wrong sleeve analysis.

The restored default for garments uses Nano Banana Pro at 2K twice: catalogue
generation, then dedicated pressing. Garments use seven uncached inferences,
or three with all preprocessing cached. Other products keep one Nano Banana call,
six uncached inferences or two with preprocessing cached. SAM and BiRefNet request inline
lossless PNG outputs, avoiding separate mask/cutout CDN downloads when honored by fal.
Two selected-product references are uploaded concurrently; at most four upload
operations are needed on a cold run (original for SAM, two isolated views, final PNG).
Byte-identical inputs share an upload. A warm regeneration within the reference-cache
window needs only the new final PNG upload.
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
cannot reuse old context. The description cache stores text and byte counts only.
Classification, detection, description and validated SAM results all reuse identical
work, including concurrent first requests. With all four cached, regeneration uses
3 model calls instead of 7 for garments, 2 instead of 6 for other products.
Caches are lost on restart; failed work is not cached or
automatically retried. Callers receive independent copies of cached values.

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
At completion, **View all steps** opens a gallery of the original, classified target,
object mask, isolated product, color correction, generation reference, analysis input,
generated photo, steam-pressed photo (garments), transparent cutout, white-background composition and final output.
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

Output SAM is omitted. Classification and native detection each add one inference.
Generation and dedicated garment pressing are separate Nano Banana Pro edits.
Uncached totals are seven for garments and six for other products. Each cached classification/detection/description
saves one call; cached SAM saves another. Selected-product reference
uploads run concurrently and identical PNGs share one upload.
The original upload is used only for SAM, preserving consistent product identity across
the subsequent stages. A storage upload can itself require multiple HTTP requests.

After completion, `metrics.stages_seconds` and `metrics.total_seconds` in the job
show measured backend stage durations (including provider wait and file transfers).
The total is the sum of timed stages; it excludes the initial local worker-queue wait.
The validation script writes `<input-name>_timings.json`. Live before/after latency
and the new output matte quality still need benchmarking with representative photos;
no runtime or quality improvement percentage is claimed from mocked tests.

`metrics.inference_requests`, `metrics.reused_inferences` and `metrics.storage_uploads`
count logical operations initiated/reused by that job, not individual HTTP polls or
a provider billing audit. Per-step details include `cache_status` (`computed`,
`cached`, `shared`) and `model_invoked`. A shared request waits for work already
running in another job; it does not submit a duplicate inference. The trace marks
cached/shared steps as `cached` and retains the endpoint that produced the result.

Validated SAM masks are cached for one hour, at most 32 entries and approximately
64 MiB of mask pixels. Keys include exact source PNG bytes, model, selected label,
detected box and selection revision. Upload URLs are cached for ten minutes, at most
256 entries, keyed by exact PNG representation (including color metadata) and
credential fingerprint. Reusable input uploads explicitly expire after one hour,
well beyond the reuse window and inference budgets. Final output uploads retain
the previous default storage policy. Reference previews are stored locally with
the job, so temporary input expiry does not remove them from the history gallery.

BiRefNet inference starts concurrently with the final candidate download (pressed
image for garments, initial generation for other products). Its RGB and refined
alpha are then composed locally, with proportions preserved. Both smaller and
larger subjects now fit catalogue framing: at most 78% canvas width and 72% height.
The former thumbnail-only resize left small generated objects surrounded by excessive
white space. Resizing corrects framing; it does not restore absent source detail.
All model choices, analysis resolution, reference PNGs, generation resolution and
quality checks are retained. Generation, pressing and final matting are never cached, so
**Regenerate** still creates a fresh result. The SDK must be `fal-client>=1.0.3`
for explicit input-storage lifetimes; install the updated backend requirements.

Inline-output schemas: [SAM](https://fal.ai/models/fal-ai/sam-3/image/api),
[BiRefNet](https://fal.ai/models/fal-ai/birefnet/v2/api).

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

For garments, **`backend/prompts/generation.txt`** combines catalogue restyling,
ideal shape, full wrinkle removal and preservation of colors, artwork and fabric
texture in the initial Nano Banana Pro request. The restored second Pro edit uses
**`backend/prompts/pressing.txt`**, with the previous dedicated pressing instructions.
It receives only the new studio image, not the wrinkled source photograph, and
preserves existing colors, artwork and fine fabric texture while removing wrinkles.
The same configured model/resolution is used for both edits (default Pro, 2K).
**Removing wrinkles** appears after **Creating photo**, with a separate intermediate
image, trace and timing. An error in pressing stops the job instead of silently
returning the initial generation as final. The exact instructions are available in
`pressing_prompt` and saved by the validation script. BiRefNet uses the pressed image.
OpenCV does not verify wrinkle removal;
inspect smoothness and identity visually. Live output quality remains unvalidated.

Accessories and other products use **`backend/prompts/product_generation.txt`**
instead of the garment template. It preserves category, rigid geometry, materials,
finish and hardware; eyewear instructions explicitly preserve frame, lenses, tint,
bridge, hinges and temples, without garment-specific pressing instructions. Each
accessory/other product uses one Nano Banana call and records pressing as skipped.
Classification is passed as data to both the
isolated-product analysis and the generation prompt. The generator never receives
the full scene containing competing products.

Upload one to five photographs of the same product and press **Generate photo**. The app produces a centered 1200×1800 studio photograph on white. Garments receive generation and dedicated pressing; other products use category-specific generation. Live previews and an optional intermediate-image gallery show progress.

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

For mobile, keep the backend running, connect the phone to the same Wi-Fi as
the computer and open the **Mobile** URL printed by `npm run dev`, for example
`http://192.168.1.42:5173/`. There is no separate mobile server: the interface
adapts to the device. Vite listens on the LAN and proxies `/api` to the local
backend, so port 8000 does not need to be exposed to the phone. The printed URL
uses the actual port, including if Vite selects another port because 5173 is busy.

On a LAN HTTP address, use **Take photo** for the native camera or **Photos**
for existing images. The embedded live camera preview requires HTTPS (localhost
on the phone refers to the phone itself, not this computer). If the page cannot
be reached, check that the devices share a network without client isolation and
that the computer allows incoming connections on the printed frontend port.

## Pipeline

LangGraph runs the entire workflow without user intervention:

1. **Moondream 3 Query** (`fal-ai/moondream3-preview/query`) classifies the main product; **Moondream Detect** (`fal-ai/moondream3-preview/detect`) then locates that category using native object boxes. These are separately traced calls.
2. **SAM 3** (`fal-ai/sam-3/image`) receives that label and detected box, without a forced foreground point, followed by local candidate-mask validation. Local illumination normalization prepares two isolated product views.
3. **Moondream 3 Query** describes the isolated color reference with classification context. Analysis must succeed before generation begins.
4. **Nano Banana Pro Edit** (`fal-ai/nano-banana-pro/edit`) receives both isolated references, classification and description. The template depends on product kind. One 2K portrait generation is requested.
5. **Dedicated Nano Banana Pro Edit** uses only that generated image for complete garment pressing while preserving identity and colors. Non-garments skip this edit.
6. **BiRefNet v2** (`fal-ai/birefnet/v2`) supplies alpha for the pressed garment or generated accessory. Pillow composes its RGB on a white 1200×1800 canvas, fitting the object to catalogue framing. Visually check edges and stray objects.
7. **OpenCV** checks color drift and background whiteness. Shape/structure metrics remain diagnostic. It does not check wrinkle removal or semantic product identity. The final PNG is uploaded and returned.

Uncached runs use **seven paid fal calls for garments**, **six for other products** (classification, native detection, SAM, description, generation, garment pressing, BiRefNet). Each preprocessing cache hit removes one call: warm regeneration uses **three for garments**, **two for other products**. Concurrent identical preprocessing still shares existing work. There are no automatic paid retries. No target examples are uploaded or used as references.

The former Bria/Kontext route and its fixed-position/preserve-silhouette instructions have been removed. Nano Banana is selected for instruction-based multi-reference restyling, not on the basis of a measured benchmark against these samples. Its live output quality still needs validation.

The opaque background is composited locally. Cutout quality still depends on segmentation, particularly for white garments. A failed segmentation stops the job. Composition preserves generated product pixels; it performs no additional fabric smoothing.

## API

- `POST /api/studio`: multipart `file` plus up to four repeated `additional_files` (JPG/PNG/WebP, max 12 MB each, 48 MB total), returns HTTP 202 with `job_id`. No mask confirmation or additional render call.
- `GET /api/jobs/{job_id}`: `status`, `result_url`, `quality`, `classification`, `target_selection`, `garment_description` (legacy field name, also used for accessories), prompts, images, trace and errors.
- `GET /api/jobs/{job_id}/events`: SSE progress stream used by the frontend. It sends the current job on connection and updates when state changes, plus idle keepalive comments. It closes after a terminal state; reconnects receive the latest snapshot.
- `GET /api/config`: model IDs and output dimensions.

Statuses: `queued → classifying → locating → preparing → balancing_color → describing → generating → pressing → finishing → checking_quality → complete | review_required`; API preprocessing can initially report `preparing`. Exceptions produce `error`. Old `/api/segment` and `/api/render` routes are replaced by `/api/studio`.

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

No target photo is used for correction. Uncached runs use seven fal inferences for garments, six for other products. Exact color recovery under mixed, narrow-spectrum or clipped illumination requires better capture/reference information. Synthetic tests cover warm/cool/green/magenta, mixed light, neutral and colored surfaces, and small-print drift; live fal fidelity has not been validated.

Background on the neutral-pixel assumption: [OpenCV GrayworldWB documentation](https://docs.opencv.org/4.1.2/javadoc/org/opencv/xphoto/GrayworldWB.html). This implementation adds spatial agreement/rejection and linear-RGB processing; it does not call the OpenCV contrib algorithm.

## Visual description stage

Graph order: `classify → locate → isolate → balance → describe → restyle → press → finish → verify` (pressing skips non-garments).
The description is English to match the generation prompt, and explicitly qualifies apparent fit, uncertain colors and unverified material composition. Existing graphics are described, not redesigned; unclear text must be marked unreadable. Descriptions can still contain model errors, so they are contextual hints rather than verified product metadata. The generator is instructed to treat them as data, not executable instructions, and defer to reference images on conflicts.

An absent/empty/non-text `output`, timeout or provider error stops the job before paid image generation. There is no automatic paid retry. No new packages or user inputs are needed. The integration sends `image_url`, `prompt`, `reasoning=false`, `temperature=0` and reads `output`: [official API schema](https://fal.ai/models/fal-ai/moondream3-preview/query/api).

### Background-based illuminant extraction

The estimator now votes across background tiles, excludes a band around the garment, and rejects internally inconsistent tiles, clipped highlights, saturated surfaces and dark pixels. Agreeing reference tiles yield an estimated light RGB vector. Its inverse channel gains are applied in linear RGB to the original photo, including the object, before the background is replaced. Garment color is not averaged toward gray: solid red/blue/green and cream garments can be corrected using independent scene references.

`quality.illumination` includes `estimated_light_rgb` (relative linear RGB), `estimated_light_srgb` (display swatch, not temperature), `reference_tiles`, `background_candidate_fraction` and the explicit assumption behind the estimate. Neutral scene light also tells the generator to avoid a second global correction.

This estimates illumination only if candidate surfaces are actually neutral and share the object's lighting. Beige walls, colored reflections and separate lighting on the product can violate that assumption. Mixed/conflicting background estimates are flagged rather than averaged. Exact recovery still needs a known neutral/color reference photographed under the object's light; no method can infer missing spectral/color information reliably from every single JPEG. Tests cover colored garments under warm/cool/green/magenta illumination and rejection of saturated backgrounds. No live fal calls were made for this change.
