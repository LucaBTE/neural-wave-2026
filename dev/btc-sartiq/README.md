# Sartiq Catalogue Studio

Upload one garment photograph and press **Crea foto da catalogo**. The app produces a centered 1200×1800 studio photograph on white, allowing the garment's pose, folds and drape to change while instructing the model to preserve its colors, graphics and construction. Only the original and final image are shown.

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

1. **SAM 3** (`fal-ai/sam-3/image`) identifies clothing automatically. The original photo and an isolated white-background reference are uploaded.
2. **Nano Banana Pro Edit** (`fal-ai/nano-banana-pro/edit`) receives both references. The original is the authority for garment details; the isolated version identifies the product. The prompt requests a new catalogue arrangement, perspective correction, natural folds, mannequin removal, neutral lighting and preservation of existing lettering/artwork. One 2K portrait image is requested.
3. **SAM 3 + BiRefNet v2** (`fal-ai/birefnet/v2`) isolate the *generated* garment, refine its alpha, and discard the generated background. Pillow places the garment on a real RGB-white 1200×1800 canvas, centered with generous margins and without distorting its proportions. No original garment pixels are pasted back over the new pose.
4. **OpenCV** checks color drift and background whiteness. Shape/structure metrics remain diagnostic: they do not reject intentional pose or fold changes. The final PNG is uploaded and returned.

Each run uses **four paid fal inference calls** (SAM twice, Nano Banana once, BiRefNet once), plus uploads. There are no automatic paid retries or fallback to the original photograph. The target examples are not uploaded or used as references, avoiding copying their garment designs.

The former Bria/Kontext route and its fixed-position/preserve-silhouette instructions have been removed. Nano Banana is selected for instruction-based multi-reference restyling, not on the basis of a measured benchmark against these samples. Its live output quality still needs validation.

The opaque background is composited locally, so it does not rely solely on the generator following the white-background prompt. Cutout quality still depends on segmentation, particularly for white garments. A failed segmentation stops the job instead of silently returning the original room. The final background has no synthetic contact shadow; dimensional folds and studio shading are retained within the garment.

## API

- `POST /api/studio`: multipart `file` (JPG/PNG/WebP, max 12 MB), returns HTTP 202 with `job_id`. No mask confirmation or additional render call.
- `GET /api/jobs/{job_id}`: `status`, `result_url`, `quality`, `error`. Internal masks/references are not included.
- `GET /api/config`: model IDs and output dimensions.

Statuses: `queued → preparing → generating → finishing → checking_quality → complete | review_required`; exceptions produce `error`. The UI presents a single processing state. Old `/api/segment` and `/api/render` routes are replaced by `/api/studio`.

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
