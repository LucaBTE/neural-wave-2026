from __future__ import annotations
import threading
import uuid
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pipeline import MODELS, build_pipeline, normalize_photo, generation_settings, pipeline_profile, png_bytes, CLASSIFICATION_MODEL
from vision import analysis_settings
from execution_trace import begin_step, end_step

load_dotenv(Path(__file__).with_name('.env'))
app = FastAPI(title="Sartiq Catalogue Studio", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
pool = ThreadPoolExecutor(max_workers=3)
lock = threading.Lock()
changes = threading.Condition(lock)
jobs: dict[str, dict] = {}
TERMINAL = {'complete', 'review_required', 'error'}


def update(job_id: str, **fields):
    with changes:
        jobs[job_id].update(fields)
        jobs[job_id]['revision'] = jobs[job_id].get('revision', 0) + 1
        changes.notify_all()


def public_job(job_id: str) -> dict:
    """Called under the job lock; image bytes are served separately, never in SSE."""
    return {'job_id': job_id, **{key: value for key, value in jobs[job_id].items()
                                if not key.startswith('_')}}


def record_image(job_id: str, key: str, label: str, image: Image.Image | str):
    data = None
    if isinstance(image, Image.Image):
        preview = image.copy()
        preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        data = png_bytes(preview)
        url = f'/api/jobs/{job_id}/images/{key}'
    else:
        url = image  # An existing fal output/reference URL: no extra upload.
    entry = {'id': key, 'label': label, 'url': url}
    with changes:
        job = jobs[job_id]
        if data is not None:
            job.setdefault('_images', {})[key] = data
        history = [item for item in job.get('images', []) if item['id'] != key]
        job.update(images=[*history, entry], current_image=entry,
                   revision=job.get('revision', 0) + 1)
        changes.notify_all()


def studio_task(job_id: str, data: bytes, additional_data: list[bytes] | None = None):
    started = perf_counter()
    preparation = begin_step('prepare_image', 'Image preparation', None, {'Pillow': 'pillow'}, job_id)
    prepared = False
    try:
        update(job_id, status='preparing', execution_trace=[preparation])
        image = normalize_photo(data)
        additional_images = [normalize_photo(item) for item in (additional_data or [])]
        preparation = end_step(preparation, perf_counter() - started)
        prepared = True
        update(job_id, execution_trace=[preparation])
        state = build_pipeline(lambda **fields: update(job_id, **fields),
                               lambda key, label, image: record_image(job_id, key, label, image),
                               run_id=job_id).invoke(
            {"image": image, 'additional_images': additional_images,
             'execution_trace': [preparation]})
        update(job_id, status='complete' if state['quality']['passed'] else 'review_required',
               result_url=state['result_url'], quality=state['quality'],
               garment_description=state['garment_description'],
               generation_prompt=state['generation_prompt'], metrics=state['metrics'], analysis=state['analysis'],
               pressing_prompt=state['pressing_prompt'],
               classification=state['classification'],
               execution_trace=state['execution_trace'], target_selection=state['target_selection'])
    except Exception as exc:
        if not prepared:
            preparation = end_step(preparation, perf_counter() - started, 'error', error_type=type(exc).__name__)
            update(job_id, execution_trace=[preparation])
        update(job_id, status='error', error=str(exc))


@app.get('/api/config')
def config():
    model, resolution = generation_settings()
    analysis = analysis_settings()
    return {'models': {**MODELS, 'restyling': model, 'pressing': None,
                       'description': CLASSIFICATION_MODEL},
            'pipeline_profile':pipeline_profile(), 'generation_resolution': resolution,
            'inference_requests_per_run': 6,
            'inference_requests_cached_run': 3,
            'inference_requests_max_per_run':8,
            'inference_requests_max_cached_run':5,
            'duplicate_repair':{'max_attempts':1,'extra_inferences':2,'recheck_required':True},
            'reference_cache_seconds': 600,
            'inference_requests_by_kind': {'garment':6, 'accessory':6, 'other':6},
            'inference_requests_cached_by_kind': {'garment':3, 'accessory':3, 'other':3},
            'analysis': {**analysis, 'required':False,
                         'model':CLASSIFICATION_MODEL,
                         'source':'classification'},
            'inputs': {'max_photos': 5, 'max_file_bytes': 12_000_000, 'max_total_bytes': 48_000_000},
            'output': {'width': 1200, 'height': 1800, 'background': 'white'}}


@app.post('/api/studio', status_code=202)
async def studio(file: UploadFile = File(...), additional_files: list[UploadFile] = File(default=[])):
    photos = [file, *additional_files]
    if len(photos) > 5:
        raise HTTPException(400, 'Maximum 5 photos of the same product')
    payloads = []
    total = 0
    for photo in photos:
        if photo.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
            raise HTTPException(400, 'Upload PNG, JPG or WebP photos')
        data = await photo.read(12_000_001)
        total += len(data)
        if len(data) > 12_000_000 or total > 48_000_000:
            raise HTTPException(413, 'Maximum 12 MB per photo and 48 MB in total')
        try:
            normalize_photo(data)
        except Exception:
            raise HTTPException(400, 'Invalid image')
        payloads.append(data)
    job_id = uuid.uuid4().hex
    with lock:
        jobs[job_id] = {'status': 'queued', 'error': None, 'result_url': None, 'input_count': len(photos)}
    pool.submit(studio_task, job_id, payloads[0], payloads[1:])
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, 'Job not found')
        return public_job(job_id)


@app.get('/api/jobs/{job_id}/images/{image_id}')
def get_image(job_id: str, image_id: str):
    with lock:
        image = jobs.get(job_id, {}).get('_images', {}).get(image_id)
    if image is None:
        raise HTTPException(404, 'Image not found')
    return Response(content=image, media_type='image/png',
                    headers={'Cache-Control': 'private, max-age=3600'})


def job_events(job_id: str):
    """One connection: snapshots on changes, comments to keep idle streams alive.

    Reconnecting receives the current snapshot, so missed events are harmless.
    No lock is held while sending data to a slow client.
    """
    revision = -1
    while True:
        with changes:
            changed = changes.wait_for(
                lambda: jobs[job_id].get('revision', 0) != revision, timeout=15)
            snapshot = public_job(job_id) if changed else None
            if snapshot is not None:
                revision = snapshot.get('revision', 0)
        if snapshot is None:
            yield ': keepalive\n\n'
            continue
        yield 'data: ' + json.dumps(snapshot) + '\n\n'
        if snapshot['status'] in TERMINAL:
            return


@app.get('/api/jobs/{job_id}/events')
def events(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, 'Job not found')
    return StreamingResponse(job_events(job_id), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
