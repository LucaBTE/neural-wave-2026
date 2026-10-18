# Sartiq Catalogue Studio — pipeline 0.1

One studio photo from 1–5 photographs of the same product. The pipeline uses
LangGraph, one Nano Banana 2.1 edit at 2K, the original RGB context crop and bounded
local color correction. Models, prompts and active image processing match 0.1.

## Backend

From the repository root:

```bash
cd dev/btc-sartiq/backend
python3 -m venv .venv                    # first setup only
.venv/bin/pip install -r requirements.txt
# First setup: copy .env.example to .env and set FAL_API_KEY or FAL_KEY.
.venv/bin/python -m uvicorn server:app --reload --port 8000
```

Keep your existing `.env`. Credentials stay on the backend.

## Frontend

In another terminal, from the repository root:

```bash
cd dev/btc-sartiq/frontend
npm install
npm run dev
```

Use the local or mobile URL printed by Vite. The frontend proxies `/api` to the
backend; there is no separate mobile server.

## Run one product

From the repository root:

```bash
dev/btc-sartiq/backend/.venv/bin/python -B dev/btc-sartiq/scripts/validate_studio.py \
  --views /path/to/product/input_1.jpg /path/to/product/input_2.jpg
```

The first photograph is the main view. Results, prompt, model trace and quality
information are saved in `dev/btc-sartiq/validation/`.

- [Pipeline and model table](../doc/pipeline.md)
- [Generate a submission](../doc/how-to-submit.md)
- [Official scoring and baseline comparison](../doc/how-to-score.md)

## Essential tests

```bash
cd dev/btc-sartiq
backend/.venv/bin/pip install -r backend/requirements-dev.txt
backend/.venv/bin/python -B -m pytest -q tests
```

Tests mock remote models. They cover the current graph, source selection, colors,
multiple views, duplicate rejection, API compatibility and version restoration.

## Version 0.1

The original source archive and SHA-256 manifest are in `dev/versions/0.1/`.
From the repository root:

```bash
python3 dev/pipeline_version.py verify 0.1
python3 dev/pipeline_version.py restore 0.1
```

Restore backs up replaced files inside `dev` and preserves `.env`. Restart the
backend afterwards. The archived source includes the older compatibility code;
the active codebase removes those unused branches without changing the 0.1 path.
