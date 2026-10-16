from __future__ import annotations
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pipeline import MODELS, build_pipeline, normalize_photo

load_dotenv(Path(__file__).with_name('.env'))
app = FastAPI(title="Sartiq Catalogue Studio", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
pool = ThreadPoolExecutor(max_workers=3)
lock = threading.Lock()
jobs: dict[str, dict] = {}


def update(job_id: str, **fields):
    with lock:
        jobs[job_id].update(fields)


def studio_task(job_id: str, data: bytes):
    try:
        state = build_pipeline(lambda **fields: update(job_id, **fields)).invoke(
            {"image": normalize_photo(data), "garment": "clothing"})
        update(job_id, status='complete' if state['quality']['passed'] else 'review_required',
               result_url=state['result_url'], quality=state['quality'])
    except Exception as exc:
        update(job_id, status='error', error=str(exc))


@app.get('/api/config')
def config():
    return {'models': MODELS, 'output': {'width': 1200, 'height': 1800, 'background': 'white'}}


@app.post('/api/studio', status_code=202)
async def studio(file: UploadFile = File(...)):
    if file.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(400, 'Carica una foto PNG, JPG o WebP')
    data = await file.read(12_000_001)
    if len(data) > 12_000_000:
        raise HTTPException(413, 'Dimensione massima: 12 MB')
    try:
        normalize_photo(data)
    except Exception:
        raise HTTPException(400, 'Immagine non valida')
    job_id = uuid.uuid4().hex
    with lock:
        jobs[job_id] = {'status': 'queued', 'error': None, 'result_url': None}
    pool.submit(studio_task, job_id, data)
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, 'Elaborazione non trovata')
        return {'job_id': job_id, **job}
