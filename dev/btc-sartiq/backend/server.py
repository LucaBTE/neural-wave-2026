from __future__ import annotations
import base64
import io
import os
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from pipeline import PRESETS, background_mask, garment_mask_from_sam, generate, normalize_photo, upload_png, png_bytes

load_dotenv(Path(__file__).with_name('.env'))
app = FastAPI(title="Sartiq Catalogue Studio", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
pool = ThreadPoolExecutor(max_workers=3)
lock = threading.Lock()
jobs: dict[str,dict] = {}

class RenderRequest(BaseModel):
    job_id: str
    preset: str = 'studio'
    prompt: str = ''
    edge_protection: int = 2

def as_uri(im: Image.Image) -> str:
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(im)).decode()

def update(job_id: str, **fields):
    with lock: jobs[job_id].update(fields)

def segment_task(job_id: str, data: bytes, garment: str):
    try:
        image = normalize_photo(data)
        image_url = upload_png(image)
        update(job_id, status='segmenting', image=as_uri(image), image_url=image_url)
        mask = garment_mask_from_sam(image, image_url, garment)
        update(job_id, status='mask_ready', garment_mask=as_uri(mask), background_mask=as_uri(background_mask(mask,2)), mask_url=upload_png(mask))
    except Exception as exc: update(job_id, status='error', error=str(exc))

def generate_task(job_id: str, req: RenderRequest):
    try:
        with lock: job = jobs[job_id].copy()
        uri = job['garment_mask'].split(',',1)[1]
        mask = Image.open(io.BytesIO(base64.b64decode(uri))); mask.load()
        bg = background_mask(mask, req.edge_protection)
        mask_url = upload_png(bg)
        update(job_id, status='generating', background_mask=as_uri(bg))
        result_url = generate(job['image_url'],mask_url,req.preset,req.prompt)
        update(job_id, status='complete', result_url=result_url)
    except Exception as exc: update(job_id,status='error',error=str(exc))

@app.get('/api/config')
def config(): return {'presets':PRESETS, 'models':{'segmentation':'fal-ai/sam-3-1/image','generation':'fal-ai/flux-pro/v1/fill'}}

@app.post('/api/segment')
async def segment(file: UploadFile=File(...), garment: str=Form('garment')):
    if file.content_type not in ('image/png','image/jpeg','image/webp'): raise HTTPException(400,'Please upload PNG, JPG or WebP')
    data=await file.read(12_000_001)
    if len(data)>12_000_000: raise HTTPException(413,'Max file size is 12 MB')
    try: normalize_photo(data)
    except Exception: raise HTTPException(400,'Invalid image')
    job_id=uuid.uuid4().hex
    update_entry={'status':'queued','error':None,'result_url':None}
    with lock: jobs[job_id]=update_entry
    pool.submit(segment_task,job_id,data,garment.strip() or 'garment')
    return {'job_id':job_id}

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with lock: job=jobs.get(job_id)
    if not job: raise HTTPException(404,'Job not found')
    return {'job_id':job_id, **job}

@app.post('/api/render')
def render(req: RenderRequest):
    if req.preset not in PRESETS: raise HTTPException(400,'Unknown preset')
    if not 0 <= req.edge_protection <= 12: raise HTTPException(400,'Edge protection must be 0..12')
    with lock:
        job=jobs.get(req.job_id)
        if not job: raise HTTPException(404,'Job not found')
        if job['status'] not in ('mask_ready','complete'): raise HTTPException(409,'Mask is not ready')
        job.update(status='queued_generation',result_url=None,error=None)
    pool.submit(generate_task,req.job_id,req)
    return {'job_id':req.job_id}
