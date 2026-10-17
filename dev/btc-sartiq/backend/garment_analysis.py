"""Required garment description; generation retains full image references."""
import base64
import hashlib
import io
import os
import threading
from collections import OrderedDict
from pathlib import Path
from time import monotonic, perf_counter

from PIL import Image

DEFAULT_MODEL = 'fal-ai/moondream3-preview/query'
ANALYSIS_MODELS = (DEFAULT_MODEL, 'fal-ai/sa2va/8b/image', 'fal-ai/sa2va/4b/image')
PROMPT_PATH = Path(__file__).with_name('prompts') / 'description.txt'
_cache = OrderedDict()
_lock = threading.Lock()
CACHE_TTL = 3600
CACHE_SIZE = 128


def analysis_settings():
    model = os.getenv('FAL_ANALYSIS_MODEL', DEFAULT_MODEL).strip()
    if model not in ANALYSIS_MODELS:
        raise ValueError('FAL_ANALYSIS_MODEL must be one of: ' + ', '.join(ANALYSIS_MODELS))
    timeout = int(os.getenv('FAL_ANALYSIS_TIMEOUT', '90'))
    max_side = int(os.getenv('FAL_ANALYSIS_MAX_SIDE', '1280'))
    if not 15 <= timeout <= 300:
        raise ValueError('FAL_ANALYSIS_TIMEOUT must be 15..300 seconds')
    if not 768 <= max_side <= 2048:
        raise ValueError('FAL_ANALYSIS_MAX_SIDE must be 768..2048 pixels')
    # Analysis is mandatory, including deployments with the former false setting.
    return {'model': model, 'timeout': timeout, 'max_side': max_side, 'required': True,
            'reasoning': False if model == DEFAULT_MODEL else None}


def prepare_analysis_image(image, mask, max_side):
    mask = mask.convert('L').resize(image.size, Image.Resampling.NEAREST)
    box = mask.point(lambda p: 255 if p > 127 else 0).getbbox()
    if box is None:
        raise ValueError('Cannot analyze an empty object mask')
    padding = max(4, round(max(box[2] - box[0], box[3] - box[1]) * .05))
    box = (max(0, box[0] - padding), max(0, box[1] - padding),
           min(image.width, box[2] + padding), min(image.height, box[3] + padding))
    isolated = Image.composite(image.convert('RGB'), Image.new('RGB', image.size, 'white'), mask)
    cropped = isolated.crop(box)
    cropped.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return cropped


def clear_analysis_cache():
    with _lock:
        _cache.clear()


def analyze_garment(image, mask, illumination, run_model, extract_text, encode_png):
    settings = analysis_settings()
    started = perf_counter()
    cropped = prepare_analysis_image(image, mask, settings['max_side'])
    prompt = PROMPT_PATH.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('The garment description prompt is empty')
    prompt += ('\nWhite balance was estimated; describe observed colors without further correction.'
               if illumination['applied'] else '\nColor observations remain lighting-dependent.'
               if illumination['status'] == 'uncertain' else '\nScene light was assessed as neutral.')
    # High-quality 4:4:4 JPEG reduces transfer size without chroma subsampling.
    # Only the descriptive view is compressed, never generation references.
    encoded = io.BytesIO()
    cropped.save(encoded, format='JPEG', quality=95, subsampling=0, optimize=True)
    data = encoded.getvalue()
    # Hash lossless pixels so compression cannot alias distinct source details.
    key = hashlib.sha256(settings['model'].encode() + prompt.encode() +
                         encode_png(cropped)).hexdigest()
    metadata = {'model': settings['model'], 'status': 'complete', 'cache_hit': False,
                'input_size': list(cropped.size), 'timeout_seconds': settings['timeout'],
                'input_format': 'jpeg', 'input_bytes': len(data), 'required': True,
                'reasoning': settings['reasoning']}
    with _lock:
        cached = _cache.get(key)
        if cached and monotonic() - cached[0] < CACHE_TTL:
            _cache.move_to_end(key)
            metadata.update(cache_hit=True, seconds=round(perf_counter() - started, 3))
            return cached[1], metadata, cropped
        _cache.pop(key, None)
    # One inference, no paid retries and no placeholder description on failure.
    arguments = {'image_url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii'),
                 'prompt': prompt}
    if settings['model'] == DEFAULT_MODEL:
        arguments.update(reasoning=False, temperature=0)
    response = run_model(settings['model'], arguments)
    description = extract_text(response)
    if settings['model'] == DEFAULT_MODEL:
        payload = response.get('data') if isinstance(response.get('data'), dict) else response
        reason = payload.get('finish_reason')
        if reason not in (None, 'stop'):
            raise RuntimeError('Garment analysis did not finish normally: ' + str(reason))
    with _lock:
        _cache[key] = (monotonic(), description)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    metadata['seconds'] = round(perf_counter() - started, 3)
    return description, metadata, cropped
