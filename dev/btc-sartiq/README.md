# Sartiq Catalogue Studio — Hackathon extraction

A focused end-to-end MVP derived from `magic-eraser-main`. Includes a **new runnable React/Vite frontend**, a **FastAPI Fal orchestration backend**, and a `reference/` directory with the directly relevant original source files. Does **not** require the source project's Supabase authentication, Postgres database or project-management models.

## Quick start

**Backend (terminal A)**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and enter FAL_API_KEY
uvicorn server:app --reload --port 8000
```

**Frontend (terminal B)**
```bash
cd frontend
npm install
npm run dev
```
Open **http://127.0.0.1:5173** (or localhost:5173). Avoid committing `.env`.

## Pipeline

1. User uploads phone photo and supplies object description (e.g. `red sweater`). Image is EXIF-corrected, resized max 2048 px, and uploaded to Fal from server.
2. **SAM 3.1** (`fal-ai/sam-3-1/image`) segments the garment using `apply_mask: false`. Returned mask(s) are converted to a binary garment mask. Frontend shows the result.
3. Pipeline **inverts** the mask: white = background to regenerate, black = protected garment. A 0–12 px protected-edge dilation option helps protect garment boundaries.
4. User chooses curated studio/editorial/dark prompts and may add art direction; **FLUX.1 [pro] Fill** (`fal-ai/flux-pro/v1/fill`) inpaints only the background.
5. Frontend displays source, garment mask, actual fill mask, and generated result. Open resulting Fal URL to save image.

**Important limitation:** An inpainting model does **not** produce different views, new camera angles, re-draped garments, invisible garment details, or an actual multi-image professional catalogue. This MVP replaces the *background while keeping the original captured garment view*. Review outputs for seams, logos, colors and fidelity. For different angles or on-model shots, a separate garment-conditioned generative workflow with explicit consistency checks is required.

## What came from the original project

- `reference/original-backend/`: 14 useful original source files covering Fal clients, adapters, gateways, SAM/Moondream segmentation, FLUX/Bria filling, process execution and masks.
- `reference/original-frontend/`: 14 useful original components, pipeline/laboratory pages, hooks and API utilities. They are **reference code** and not wired directly into this simplified frontend because the source components depend on authentication, database and project abstractions.
- `backend/reused/fal_client.py`: **verbatim** Fal API request wrapper from source (reference implementation). `backend/pipeline.py` implements simplified standalone integration with the same Fal `subscribe` pattern, eliminating source-specific database dependencies.

## API

- `GET /api/config` → presets and model IDs
- `POST /api/segment` → multipart `file`, `garment` → `job_id`
- `GET /api/jobs/{job_id}` → status and image/mask URLs or data URIs
- `POST /api/render` → JSON `{job_id,preset,prompt,edge_protection}` → `job_id`

## Tests

```bash
cd backend && pip install -r requirements.txt
cd .. && pytest -q tests
cd frontend && npm run typecheck
```

## Hackathon caveats / next steps

- **Not tested against a live Fal account** (no API key available here); endpoint input/output schemas may evolve.
- The `fal_client.upload()` call uploads local images; Fal storage URLs are externally accessible and uploaded photos are shared with the provider. Verify privacy/consent and storage policy.
- Mask response semantics must be visually verified with the selected SAM model before demonstrations. Semi-transparent mask previews are not reliably equivalent to binary masks; thresholding may need tuning for individual response formats.
- For production, move jobs from in-process RAM/thread-pool to Redis/Celery or similar; avoid leaking base64 images through repeated polling, add authentication, per-user quotas, input validation, retries, cost controls and persistent result storage.
- There is no capture/lighting hardware control. For the Sartiq challenge, consider adding guided framing/exposure, white-balance and light-intensity presets, along with sharpness and exposure checks, before segmentation.
- `reference/` contains source code from the uploaded project. Retain original license obligations/attribution before redistribution; no license was assumed.
