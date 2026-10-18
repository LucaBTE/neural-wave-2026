# Sartiq Catalogue Studio

Turns 1–5 photos of one product into a 1200×1800 catalogue photo on a white background.
Every model runs on [fal.ai](https://fal.ai), so each run costs fal credits.

## Setup

Backend (Python 3.10+):

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp -n .env.example .env   # then set FAL_API_KEY (-n keeps an existing .env)
```

Frontend (only for the web app):

```bash
cd frontend
npm install
```

## Run from the CLI

From `dev/btc-sartiq/`:

```bash
# one photo
backend/.venv/bin/python scripts/validate_studio.py path/to/photo.jpg

# several photos of the SAME product (first one is the main view, max 5)
backend/.venv/bin/python scripts/validate_studio.py --views main.jpg side.jpg detail.jpg
```

Without `--views`, each photo is processed as a separate product.

Results go to `validation/<folder>_<name>_studio.png`. The same folder gets the prompt, quality report, timings, trace and classification as `.txt`/`.json` files.

To generate the whole competition submission, see [`../doc/how-to-submit.md`](../doc/how-to-submit.md).

## Run the web app (FastAPI + frontend)

Two terminals:

```bash
# 1. backend, on :8000
cd backend
.venv/bin/uvicorn server:app --reload --port 8000

# 2. frontend, on :5173 (proxies /api to :8000)
cd frontend
npm run dev
```

Open http://127.0.0.1:5173, upload the photos and press **Generate photo**.
The page shows live progress, and **View all steps** opens the intermediate images.

## Configuration

`backend/.env` (restart the backend after changing it):

| Variable | Default | Notes |
|---|---|---|
| `FAL_API_KEY` (or `FAL_KEY`) | — | required |
| `FAL_IMAGE_MODEL` | `fal-ai/nano-banana-pro/edit` | or `fal-ai/nano-banana-2/edit` |
| `FAL_IMAGE_RESOLUTION` | `2K` | `1K`, `2K`, `4K` |
| `FAL_ANALYSIS_MODEL` | `fal-ai/moondream3-preview/query` | |
| `FAL_ANALYSIS_TIMEOUT` | `90` | seconds, 15–300 |
| `FAL_ANALYSIS_MAX_SIDE` | `1280` | px, 768–2048 |
| `COLOR_CORRECTION_STRENGTH` | `1.05` | 0.5–1.5 |

The prompts are plain text files in `backend/prompts/`. They are re-read for every job, so editing them needs no restart.
`generation.txt` must contain `{{LIGHTING_INSTRUCTIONS}}` and `{{GARMENT_DESCRIPTION}}` exactly once each.

## How it works

A LangGraph pipeline in `backend/pipeline.py`:

```
classify → locate → isolate → balance → describe → restyle → press → finish → verify
```

| Step | What it does | Tool |
|---|---|---|
| classify | identifies the main product (garment, accessory, other) | Moondream 3 Query |
| locate | finds its bounding box | Moondream 3 Detect |
| isolate | segments the product | SAM 3 + OpenCV |
| balance | white-balances from the background | local (`color.py`) |
| describe | writes a short visual description | Moondream 3 Query |
| restyle | generates the catalogue photo | Nano Banana Pro Edit |
| press | removes wrinkles (garments only) | Nano Banana Pro Edit |
| finish | cuts out the product and places it on white | BiRefNet v2 + Pillow |
| verify | checks colour drift and the background | local (`quality.py`) |

A garment needs 7 fal calls, any other product 6. Rerunning the same photo uses fewer, because classification, detection, description and segmentation are cached in memory for one hour.
Any error stops the job, and nothing is retried automatically.

Step-by-step details: [`../doc/pipeline.md`](../doc/pipeline.md).

## API

- `POST /api/studio`: multipart `file` (main photo) plus up to 4 `additional_files`. Accepts JPG/PNG/WebP, max 12 MB each and 48 MB in total. Returns `job_id`.
- `GET /api/jobs/{job_id}`: status, result URL, quality report, prompts, trace.
- `GET /api/jobs/{job_id}/events`: the same data as an SSE stream (used by the frontend).
- `GET /api/config`: models, limits, output size.

A job ends as `complete`, `review_required` (output produced, but needs a visual check) or `error`.

## Tests

The tests use simulated fal responses, so they need no API key and cost nothing.

```bash
backend/.venv/bin/python -m pytest -q tests
cd frontend && npm run build
```

## Limits

- Generation can alter lettering, prints and texture. The quality check only looks at colour and the background, so inspect results by eye.
- White balance is a heuristic. Mixed or strongly coloured light marks the result `review_required`.
- Jobs live in memory and are lost when the backend restarts. There is no authentication or quota.
- `reference/` and `backend/reused/` contain code from the original project, with its attribution obligations.
