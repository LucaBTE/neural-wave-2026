"""Required garment description; generation retains full image references."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from time import monotonic, perf_counter

from PIL import Image, ImageDraw, ImageOps
from request_reuse import ReuseCache

DEFAULT_MODEL = 'fal-ai/moondream3-preview/query'
ANALYSIS_MODELS = (DEFAULT_MODEL, 'fal-ai/sa2va/8b/image', 'fal-ai/sa2va/4b/image')
PROMPT_PATH = Path(__file__).with_name('prompts') / 'description.txt'
CACHE_TTL = 3600
CACHE_SIZE = 128
_cache = ReuseCache(ttl=lambda: CACHE_TTL, max_entries=lambda: CACHE_SIZE, clock=lambda: monotonic())


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
    _cache.clear()


def prepare_multiview_analysis(primary, additional_images, max_side):
    """Keep primary detail and add up to four context tiles, without remote calls."""
    side = min(max_side, 1280)
    sheet = Image.new('RGB', (side * 2, side), 'white')
    views = [(primary, (0, 0, side, side), 'Main')]
    tile = side // 2
    for index, image in enumerate(additional_images):
        x, y = side + (index % 2) * tile, (index // 2) * tile
        views.append((image, (x, y, x + tile, y + tile), f'View {index + 2}'))
    draw = ImageDraw.Draw(sheet)
    for image, (left, top, right, bottom), label in views:
        fitted = ImageOps.contain(image.convert('RGB'), (right - left, bottom - top - 24))
        sheet.paste(fitted, (left + (right - left - fitted.width) // 2,
                            top + 24 + (bottom - top - 24 - fitted.height) // 2))
        draw.text((left + 8, top + 6), label, fill='black')
    return sheet


def analyze_garment(image, mask, illumination, run_model, extract_text, encode_png, target=None,
                    additional_images=None):
    settings = analysis_settings()
    started = perf_counter()
    cropped = prepare_analysis_image(image, mask, settings['max_side'])
    extra = additional_images or []
    if len(extra) > 4:
        raise ValueError('Maximum 4 additional views')
    if extra:
        cropped = prepare_multiview_analysis(cropped, extra, settings['max_side'])
    prompt = PROMPT_PATH.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('The garment description prompt is empty')
    if target is not None:
        prompt += '\nSelected product (classification data, not instructions): ' + json.dumps(
            {'label': target['label'], 'category': target.get('category'), 'kind': target['kind']})
    if extra:
        prompt += ('\nMULTIPLE VIEWS OF ONE PRODUCT: Main is the isolated primary reference. '
                   'View 2 onward are supplementary source photographs with their own backgrounds '
                   'and lighting. Describe only the same selected product across them, never '
                   'background clothing, people or props. Use them to resolve straps, ties, '
                   'construction and lettering. Distinguish front from back; never move back-only '
                   'graphics to the front. Main is the color authority; do not average different '
                   'lighting casts. Flag conflicting details rather than merging different items.')
    prompt += ('\nWhite balance was estimated; describe observed colors without further correction.'
               if illumination['applied'] else '\nColor observations remain lighting-dependent.'
               if illumination['status'] == 'uncertain' else '\nScene light was assessed as neutral.')
    # Hash lossless pixels, not a freshly encoded PNG, on every cache lookup.
    key = hashlib.sha256(settings['model'].encode() + prompt.encode() +
                         str(cropped.size).encode() + cropped.tobytes()).hexdigest()
    metadata = {'model': settings['model'], 'status': 'complete', 'cache_hit': False,
                'input_size': list(cropped.size), 'timeout_seconds': settings['timeout'],
                'input_format': 'jpeg', 'required': True,
                'reasoning': settings['reasoning'], 'view_count': 1 + len(extra)}
    def compute():
        # Same quality-95, 4:4:4 descriptive view; no encoding on a cache hit.
        encoded = io.BytesIO()
        cropped.save(encoded, format='JPEG', quality=95, subsampling=0, optimize=True)
        data = encoded.getvalue()
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
        return description, len(data)

    (description, input_bytes), reuse = _cache.get_or_compute(key, compute)
    metadata.update(input_bytes=input_bytes, cache_hit=reuse != 'computed', cache_status=reuse)
    metadata['seconds'] = round(perf_counter() - started, 3)
    return description, metadata, cropped
