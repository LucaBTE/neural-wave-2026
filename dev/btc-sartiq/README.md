# Sartiq Catalogue Studio

Upload 1–5 photographs of the same product and press **Generate photo**.
The default LangGraph pipeline uses one Nano Banana edit for a complete studio
photograph, including product restaging, steaming and scene-aware relighting.

## Start

From `dev/btc-sartiq`, start the backend:

```bash
cd backend
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

First setup: create `.venv` with `python3 -m venv .venv` and copy `.env.example`
to `.env`. Set `FAL_API_KEY` or `FAL_KEY`. Preserve an existing `.env`; it is loaded
by the backend and benchmark worker. The example file is trackable by Git while
real environment files remain ignored.

In another terminal, from `dev/btc-sartiq`:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. For mobile, connect the phone to the same Wi-Fi and
open the **Mobile** URL printed by `npm run dev`. No separate mobile server is
needed: Vite exposes the frontend on the LAN and proxies `/api` to the backend.
The printed URL includes the actual selected port.

Desktop uses file selection/drag and drop without requesting webcam access,
including narrow desktop windows. Mobile offers the rear camera and **Photos**.
Embedded live camera access needs HTTPS or localhost; on a LAN HTTP address,
**Take photo** opens the native device camera picker. Capture, cancellation and
permission failures release camera streams.

## Default pipeline

1. **Moondream Query** identifies the foreground product and describes visible
   construction in one request, without copying a JSON example or guessing fibers.
2. **Moondream Detect** locates its physical category. **SAM 3** segments and
   validates the whole product rather than its printed graphic or background clothing.
3. **Pillow** prepares the unchanged original scene. Each actual photo is sent once,
   without an automatically added duplicate crop. No RGB
   correction is inferred from uncalibrated colored walls or floors.
4. **Nano Banana 2.1** (`google/nano-banana-2.1/edit`, 2K) receives these references,
   optional extra photographs, and a short prompt. It creates a front-facing flat
   lay with sleeves outward, steamed fabric and preserved colors, graphics and parts.
5. **Moondream Query** checks the generated photograph for duplicated or overlapping
   products, including a plain shirt behind a printed shirt. A detected duplicate
   triggers at most one Nano Banana correction and a mandatory recheck. An unresolved
   duplicate is not delivered as a final result. An initially inconclusive check
   keeps the output available with **Review**, without claiming validation.
6. **BiRefNet v2** supplies a diagnostic matte. Delivery preserves the generated
   RGB frame and shadows, resized/padded to 1200 × 1800; the matte cannot remove
   pale straps or other generated details.
7. **OpenCV/NumPy** remove supported residual casts from the generated neutral
   studio, then bound material exposure/chroma drift using reliable source foreground
   samples. Bright details are protected; missing source masks retain a conservative
   dark-material fallback. Decisions are recorded with color/background diagnostics,
   distinct from
   the official competition score and do not certify text, identity or wrinkle removal.

Six model requests cold, three when preprocessing is cached. A positive duplicate
check adds one corrective edit and one recheck: at most eight cold/five warm requests.
No mandatory second generation or separate description call. No generic paid retry.
The generation model
is the same endpoint as the official baseline; targeted context and photographic
instructions are the intended improvement, not an assumed model advantage.

The main photo sets the preferred view; click a thumbnail to make it **Main**. **Add** supplies
construction or artwork details (up to five photographs, 12 MB each/48 MB total).
Classification sees all photos in one locally assembled board. Extra photos also
go directly to the generator; detection and segmentation use the main view. If the
main image is a close-up or label, the most complete view supplies the silhouette.
Use the same product in all photos. Details
from the back must not move onto the front.

Live progress follows actual graph nodes. **Execution details** shows model IDs,
installed library versions, durations and cache reuse. **View all steps** opens the
intermediate images; failed jobs expose available snapshots. The isolated-mask
preview is diagnostic: the generator receives each scene photo once, not an inpainting
mask. Successful geometry checks cannot certify semantic product identity.

Unusable classifications, empty detections, rejected masks and helper timeouts
continue through source-reference generation. Helpers that need unavailable
identity or boxes are skipped, without inventing masks or confidence. Failed
diagnostic matting also preserves the generated image. `quality.recovery` and the
trace show recovery; missing checks produce `review_required` with an available
output. This does not override a positively detected duplicate: correction must be
confirmed before final delivery. The check can miss hallucinations; it is not a
semantic guarantee. Its preview is bounded to 768 px, with reasoning disabled and
a timeout capped at 30 seconds, and requires no additional upload.

Full step/model table, rationale and limitations: [pipeline.md](../doc/pipeline.md).
Official scoring and the five-garment comparison: [compare-pipelines.md](../doc/compare-pipelines.md).

## Prompts and configuration

Edit `backend/prompts/generation.txt` for garments,
`product_generation.txt` for accessories/other objects,
`classification.txt` for product identification, and `multiview.txt` for extra photos.
`single_product_check.txt` and `single_product_repair.txt` define the multiplicity
check and bounded corrective edit. Normal generation still uses one image edit.
Generation templates preserve `{{LIGHTING_INSTRUCTIONS}}` and
`{{GARMENT_DESCRIPTION}}` exactly once. They reload for each job; assembled prompts
are recorded in job metadata. Captions remain short, auxiliary evidence.
Model-inferred color adjectives are removed from the generation caption and target
labels; quoted artwork text is retained. Classification and detection themselves
are unchanged. Photographs establish colors. Embedded input ICC profiles are
transformed to sRGB before upload and numeric color processing. Final studio and
material color decisions are recorded in `quality.color_fidelity`. See
[color-correction.md](../doc/color-correction.md).

Defaults:

```env
FAL_PIPELINE_PROFILE=faithful
FAL_IMAGE_MODEL=google/nano-banana-2.1/edit
FAL_IMAGE_RESOLUTION=2K
```

Explicit `.env` model overrides are respected. Nano Banana Pro
(`fal-ai/nano-banana-pro/edit`) and `fal-ai/nano-banana-2/edit` remain supported;
resolutions are 1K/2K/4K. Restart the backend after environment changes.

The optional `staged` profile retains local white balance, separate analysis,
garment pressing and cutout composition for comparison, with the same multiplicity
gate: eight cold/four warm requests for garments, seven/three for other products,
plus two only for a detected duplicate. It uses the current edited
prompts. To select the previous generation endpoint as well, explicitly set
`FAL_IMAGE_MODEL=fal-ai/nano-banana-pro/edit`. The default profile ignores
`COLOR_CORRECTION_STRENGTH` and the separate `FAL_ANALYSIS_MODEL`; identification
and localization still use `FAL_ANALYSIS_TIMEOUT` (90 seconds by default).

## API and caches

- `POST /api/studio`: multipart `file` and optional repeated `additional_files`;
  returns 202 with `job_id`.
- `GET /api/jobs/{job_id}`: output, quality diagnostics, classification, actual
  stages, prompts, images, model/library trace and metrics.
- `GET /api/jobs/{job_id}/events`: SSE progress stream, including image URLs.
- `GET /api/jobs/{job_id}/images/{image_id}`: local preview, longest edge 1280 px.
- `GET /api/config`: actual model IDs, profile, input limits and inference counts.

Classification, detection and validated SAM masks are cached for one hour;
byte-identical input uploads are reused for ten minutes. Concurrent jobs share
preprocessing. Changed pixels/prompts invalidate their respective caches. Generated
outputs and diagnostic mattes are fresh. Metrics count logical inference/upload
operations; HTTP polling, downloads and provider retries are additional traffic.
fal status polling uses a two-second interval. Caches and job previews live in RAM.

## Verification

From this directory:

```bash
backend/.venv/bin/python -m pytest -q tests
cd frontend
npm run build
npm run test:ui
```

Tests use mocked fal responses and exercise real graph/API code. Browser checks
require Node 22+ and Chrome (`CHROME_BIN` override) and simulate camera access.
They do not measure photographic quality. Benchmark the changed pipeline with the
unchanged official scorer; target photographs are never model inputs. New live
scores for the simplified default have not yet been measured.

Generative editing can still alter text, print or texture. Exact color recovery
requires calibrated lighting evidence; neutral-looking room surfaces are insufficient.
The diagnostic matte and source-vs-relit color comparison can produce false warnings.
Invalid inputs, transfer failures and generation-provider errors can still stop
the job; the original is never reported as a generated output. This prototype keeps jobs in RAM and has no
persistent checkpointer, authentication or quotas. Keys remain server-side.
`reference/` and `backend/reused/` retain their code and attribution obligations.
