from __future__ import annotations
import io
import base64
import binascii
import hashlib
import html
import json
import re
import os
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Callable, TypedDict
from urllib.request import urlopen

import fal_client
from fal_client.client import FalClientError
from httpx import HTTPError
from langgraph.graph import END, START, StateGraph
from quality import compare_product
from color import restore_color_fidelity, normalize_color_profile
from generation_prompt import build_generation_prompt
from vision import analysis_settings
from execution_trace import begin_step, end_step
from target_selection import select_centered_mask, assess_product_selection
from product_classification import CLASSIFICATION_MODEL, LOCALIZATION_MODEL, classify_product, locate_product
from product_validation import (VALIDATION_MODEL, CHECK_PREFIX, CHECK_TIMEOUT,
                                check_single_product, single_product_repair_prompt)
from request_reuse import ReuseCache
from studio_fidelity import context_reference, clean_detached_fragment
from PIL import Image, ImageDraw, ImageOps

SAM_MODEL = "fal-ai/sam-3/image"
BIREF_MODEL = "fal-ai/birefnet/v2"
FAITHFUL_MODEL = "google/nano-banana-2.1/edit"
MODELS = {"description": CLASSIFICATION_MODEL, "segmentation": SAM_MODEL, "restyling": FAITHFUL_MODEL,
          "classification": CLASSIFICATION_MODEL,
          "localization": LOCALIZATION_MODEL,
          "background_removal": BIREF_MODEL, "product_validation":VALIDATION_MODEL,
          "quality": "Python + OpenCV"}
_segmentation_cache = ReuseCache(ttl=3600, max_entries=32, max_bytes=64 * 1024 * 1024)
_upload_cache = ReuseCache(ttl=600, max_entries=256)
# Optional understanding/matting must not prevent the image editor from working
# directly from photographs. Configuration and programming errors still surface.
REFERENCE_ERRORS = (RuntimeError, ValueError, OSError, FalClientError, HTTPError)


def clear_reuse_caches():
    _segmentation_cache.clear()
    _upload_cache.clear()


def _cache_scope():
    return hashlib.sha256((os.getenv('FAL_KEY') or os.getenv('FAL_API_KEY') or '').encode()).hexdigest()


def _image_key(image):
    # Include exactly the PNG representation sent to fal, including color metadata.
    return hashlib.sha256(png_bytes(image)).hexdigest()

def extract_description(response: Any) -> str:
    """fal returns text in output, potentially with HTML/segmentation tokens."""
    if not isinstance(response, dict):
        raise RuntimeError("Garment analysis did not return a text description")
    payload = response.get("data") if isinstance(response.get("data"), dict) else response
    output = payload.get("output")
    if not isinstance(output, str):
        raise RuntimeError("Garment analysis did not return a text description")
    output = html.unescape(output)
    output = re.sub(r"<[^>]*>|\[SEG\]", " ", output)
    output = " ".join(output.split())
    if not output:
        raise RuntimeError("Garment analysis returned an empty garment description")
    return output[:6000]


def _url(item: Any) -> str | None:
    if isinstance(item, str) and item.strip(): return item.strip()
    if isinstance(item, dict) and isinstance(item.get("url"), str): return item["url"]
    return None

def extract_urls(response: Any, masks: bool = False) -> list[str]:
    if not isinstance(response, dict): return []
    for item in (response, response.get("data")):
        if not isinstance(item, dict): continue
        if masks and isinstance(item.get("masks"), list):
            urls = [url for x in item["masks"] if (url := _url(x))]
            if urls: return urls
        keys = ("mask", "mask_image", "image", "mask_url", "output_url", "image_url", "url") if masks else ("images", "image", "output_url", "image_url", "url")
        for key in keys:
            value = item.get(key)
            if isinstance(value, list):
                urls = [url for x in value if (url := _url(x))]
                if urls: return urls
            elif (url := _url(value)): return [url]
    return []

def fetch_image(url: str) -> Image.Image:
    # Inline lossless masks avoid a separate CDN download for each model output.
    if re.match(r'^data:image/(?:png|jpeg|webp);base64,', url):
        payload = url.split(',', 1)[1]
        if len(payload) > ((20_000_000 + 2) // 3) * 4:
            raise ValueError('Remote image too large')
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('Invalid inline image from fal') from exc
    else:
        if not url.startswith("https://"):
            raise ValueError("Expected an HTTPS or base64 image URL from fal")
        from urllib.request import Request
        req = Request(url, headers={"User-Agent": "CatalogueStudio/1.0"})
        with urlopen(req, timeout=45) as response:
            data = response.read(20_000_001)
    if len(data) > 20_000_000: raise ValueError("Remote image too large")
    im = Image.open(io.BytesIO(data)); im.load(); return im

def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO(); image.save(buffer, format="PNG"); return buffer.getvalue()

def upload_png(image: Image.Image, *, expires_in=None) -> str:
    configure_fal()
    options = {'lifecycle': fal_client.StorageSettings(expires_in=expires_in)} if expires_in else {}
    return fal_client.upload(png_bytes(image), content_type="image/png", **options)


def upload_reference(image):
    # The provider lifetime exceeds both the 10-minute reuse TTL and inference
    # wait budgets. Final deliverables keep the existing default retention.
    key = (_cache_scope(), _image_key(image))
    def compute():
        url = upload_png(image, expires_in='1h')
        if not isinstance(url, str) or not url.startswith('https://'):
            raise RuntimeError('fal storage did not return an HTTPS reference URL')
        return url
    return _upload_cache.get_or_compute(key, compute)

def configure_fal() -> None:
    key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not key: raise RuntimeError("Set FAL_API_KEY or FAL_KEY in backend/.env")
    os.environ["FAL_KEY"] = key

def call(model: str, arguments: dict) -> dict:
    configure_fal()
    timeout = analysis_settings()['timeout'] if model in (CLASSIFICATION_MODEL, LOCALIZATION_MODEL) else 300
    if model == VALIDATION_MODEL and arguments.get('prompt','').startswith(CHECK_PREFIX):
        timeout = min(timeout,CHECK_TIMEOUT)
    # The SDK defaults to 0.1s: ten status HTTP requests/sec for ONE inference.
    return fal_client.subscribe(model, arguments=arguments, with_logs=False,
                                interval=2.0, client_timeout=timeout)


def pipeline_profile():
    return 'faithful'


def generation_settings() -> tuple[str, str]:
    model = os.getenv("FAL_IMAGE_MODEL", FAITHFUL_MODEL).strip()
    resolution = os.getenv("FAL_IMAGE_RESOLUTION", "2K").strip()
    if model != FAITHFUL_MODEL:
        raise ValueError("Pipeline 0.1 uses google/nano-banana-2.1/edit")
    if resolution not in ("1K", "2K", "4K"):
        raise ValueError("FAL_IMAGE_RESOLUTION must be 1K, 2K or 4K")
    return model, resolution

def normalize_photo(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data)); image.load()
    image = normalize_color_profile(ImageOps.exif_transpose(image))
    if max(image.size) > 2048:
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image

def garment_mask_from_sam(image: Image.Image, image_url: str, garment: str, target=None) -> Image.Image:
    arguments = {"image_url": image_url, "prompt": garment, "apply_mask": False,
        "point_prompts": [{"x": image.width // 2, "y": image.height // 2, "label": 1}],
        "return_multiple_masks": True, "max_masks": 3, "output_format": "png", "sync_mode": True}
    if target is not None:
        left, top, right, bottom = target['bbox']
        arguments.update(prompt=target['category'],
            point_prompts=[],
            box_prompts=[{'x_min': int(left * image.width), 'y_min': int(top * image.height),
                          'x_max': min(image.width, ceil(right * image.width)),
                          'y_max': min(image.height, ceil(bottom * image.height))}])
    response = call(SAM_MODEL, arguments)
    urls = extract_urls(response, masks=True)
    if not urls: raise RuntimeError(f"SAM did not return a mask. Response keys: {list(response) if isinstance(response,dict) else type(response)}")
    layers = []
    for url in urls:
        mask = fetch_image(url)
        # SAM masks expected white foreground, black background.
        if mask.mode == "RGBA":
            alpha = mask.getchannel("A")
            layer = alpha if alpha.getextrema() != (255,255) else ImageOps.grayscale(mask.convert("RGB"))
        else: layer = ImageOps.grayscale(mask.convert("RGB"))
        layers.append(layer.resize(image.size, Image.Resampling.NEAREST).point(lambda px: 255 if px >= 127 else 0))
    garment_mask, selection = select_centered_mask(layers, target)
    garment_mask.info['target_selection'] = selection
    return garment_mask


def output_url(model: str, arguments: dict) -> str:
    urls = extract_urls(call(model, arguments))
    if not urls:
        raise RuntimeError(f"{model} did not return an image")
    return urls[0]


def fetch_background_matte(image_url: str) -> Image.Image:
    # The generated photo contains a single garment on white. BiRefNet supplies
    # its mask directly: no second SAM inference or re-upload is necessary.
    url = output_url(BIREF_MODEL, {"image_url": image_url,
        "output_format": "png", "refine_foreground": True, "sync_mode": True})
    return fetch_image(url)


def remove_background(image: Image.Image, image_url: str, matte=None) -> Image.Image:
    cutout = fetch_background_matte(image_url) if matte is None else matte
    if "A" not in cutout.getbands():
        raise RuntimeError("BiRefNet did not return a transparent cutout")
    cutout = cutout.convert("RGBA").resize(image.size, Image.Resampling.LANCZOS)
    alpha = cutout.getchannel("A")
    if not alpha.point(lambda p: 255 if p > 127 else 0).getbbox():
        raise RuntimeError("BiRefNet returned an empty garment")
    # Preserve source RGB; use BiRefNet only to refine the matte.
    result = image.convert("RGBA")
    result.putalpha(alpha)
    return result


class PipelineState(TypedDict, total=False):
    classification: dict
    image: Image.Image
    additional_images: list[Image.Image]
    additional_reference_urls: list[str]
    generated_image: Image.Image
    image_url: str
    garment: str
    mask: Image.Image | None
    target_selection: dict
    color_reference: Image.Image
    color_url: str
    illumination: dict
    reference_url: str
    garment_description: str
    analysis: dict
    generation_prompt: str
    generated_url: str
    pressed_url: str
    pressing_prompt: str
    result: Image.Image
    result_mask: Image.Image | None
    result_url: str
    quality: dict
    metrics: dict
    execution_trace: list[dict]
    segmentation_reuse: str
    source_upload_status: str
    reference_uploads: list[str]
    reference_policy: str
    reference_crop: list[int] | None
    output_cleanup: dict
    direct_reference: bool
    recovery: dict
    inspected_image: Image.Image | None
    product_checks: list[dict]
    needs_product_repair: bool


def build_pipeline(on_progress: Callable[..., None] | None = None,
                   on_image: Callable[[str, str, Image.Image | str], None] | None = None,
                   run_id: str | None = None):
    """Single-click workflow, with at most one edit for detected duplicate products."""
    model, resolution = generation_settings()  # Validate before any paid call.
    analysis_config = analysis_settings()
    metrics = {"stages_seconds": {}, "total_seconds": 0.0}
    pillow = {'Pillow': 'pillow'}
    remote = {'fal-client': 'fal-client', 'LangGraph': 'langgraph'}
    vision = {'NumPy': 'numpy', 'OpenCV': 'opencv-python-headless', **pillow}
    steps = {
        'classify': ('Identifying product', 'classifying', CLASSIFICATION_MODEL, {**pillow, **remote},
                     {'reasoning': False, 'max_side': 1280, 'required': False,
                      'timeout_seconds': analysis_config['timeout']}),
        'locate': ('Locating product', 'locating', LOCALIZATION_MODEL, {**pillow, **remote},
                   {'max_side': 1280, 'timeout_seconds': analysis_config['timeout']}),
        'isolate': ('Preparing', 'preparing', SAM_MODEL, {**vision, **remote}, {}),
        'balance': ('Preparing references', 'balancing_color', None, {**pillow, **remote}, {}),
        'restyle': ('Creating photo', 'generating', model, remote, {'resolution': resolution}),
        'inspect': ('Checking product', 'checking_product', VALIDATION_MODEL, {**pillow, **remote},
                    {'timeout_seconds':min(analysis_config['timeout'],CHECK_TIMEOUT), 'max_side':768}),
        'repair_product': ('Correcting duplicate product', 'correcting_product', model, remote, {'resolution':resolution}),
        'inspect_repair': ('Verifying correction', 'rechecking_product', VALIDATION_MODEL, {**pillow, **remote},
                           {'timeout_seconds':min(analysis_config['timeout'],CHECK_TIMEOUT), 'max_side':768}),
        'finish': ('Refining', 'finishing', BIREF_MODEL, {**vision, **remote}, {}),
        'verify': ('Final check', 'checking_quality', None, {**vision, **remote}, {}),
    }

    def report(stage, **fields):
        if on_progress:
            on_progress(status=stage, **fields)

    def snapshot(key, label, image):
        if on_image:
            on_image(key, label, image)

    def progress_stages(include_repair=False):
        return [{'status':steps[name][1], 'label':steps[name][0]} for name,_ in nodes
                if include_repair or name not in ('repair_product','inspect_repair')]

    def recovered(state, stage, exc=None, reason='source_reference_fallback', **fields):
        return {**fields, 'recovery': {**state.get('recovery', {}), stage: {
            'reason': reason, 'error_type': type(exc).__name__ if exc else None}}}

    def classify(state):
        report('classifying')
        image = state["image"]
        if len(state.get('additional_images', [])) > 4:
            raise ValueError('Maximum 4 additional views')
        snapshot("original", "Original", image)
        for index, extra in enumerate(state.get('additional_images', []), start=2):
            snapshot(f'original_{index}', f'Original · View {index}', extra)
        recovery = {}
        try:
            target = classify_product(image, call, extract_description,
                                      additional_images=state.get('additional_images'))
        except REFERENCE_ERRORS as exc:
            # No guessed category or confidence: the image editor can recognize
            # the product from the original views even when the helper cannot.
            target = {'label':'Product shown in the reference photographs', 'category':'product',
                      'kind':'other', 'confidence':None, 'review_required':True,
                      'identity_supported':False, 'cache_hit':False, 'cache_status':'unavailable'}
            recovery = recovered(state, 'classify', exc, direct_reference=True)
        report('classifying', classification=target,
               stages=progress_stages())
        result = {**recovery, 'classification': target}
        description = target.get('description') or (target['category'] + ': ' + target['label'])
        result.update(garment_description=description, pressing_prompt='', analysis={
            'source':'classification', 'model':CLASSIFICATION_MODEL,
            'status':'complete' if target.get('identity_supported', True) else 'unavailable',
            'cache_hit':target['cache_hit'], 'view_count':1 + len(state.get('additional_images', [])),
            'required':False, 'identity_supported':target.get('identity_supported', True),
            'separate_model_request':False})
        return result

    def locate(state):
        report('locating')
        image = state['image']
        original_target = state['classification']
        attempted = original_target.get('identity_supported', True)
        try:
            if not attempted:
                raise RuntimeError('No supported category to localize')
            target = locate_product(image, original_target, call)
        except REFERENCE_ERRORS as exc:
            target = {**original_target, 'bbox':None, 'review_required':True,
                      'localization_cache_hit':False,
                      'localization_cache_status':'fallback' if attempted else 'skipped',
                      'localization_invoked':attempted}
            report('locating', classification=target)
            snapshot('target', 'Source reference · location unavailable', image)
            return recovered(state, 'locate', exc, classification=target, direct_reference=True)
        report('locating', classification=target)
        preview = image.copy()
        left, top, right, bottom = target['bbox']
        draw = ImageDraw.Draw(preview)
        draw.rectangle((int(left * image.width), int(top * image.height),
                        min(image.width - 1, int(right * image.width)),
                        min(image.height - 1, int(bottom * image.height))),
                       outline='#c3d8a4', width=max(2, image.width // 250))
        snapshot('target', 'Selected product: ' + target['category'], preview)
        return {'classification': target}

    def isolate(state):
        report("preparing")
        image = state['image']
        target = state['classification']
        transferred = {'source_upload_status':'skipped', 'image_url':None}
        def without_mask(exc=None):
            selection = {'strategy':'source_references', 'bbox':None, 'verified':False}
            classification = {**target, 'review_required':True,
                              'selection_validation':{'approved':False, 'method':'unavailable'}}
            report('preparing', target_selection=selection, classification=classification)
            return recovered(state, 'isolate', exc, **transferred, mask=None,
                classification=classification, target_selection=selection, direct_reference=True,
                segmentation_reuse='fallback' if target.get('bbox') else 'skipped')
        if not target.get('bbox'):
            return without_mask()
        key = (_cache_scope(), SAM_MODEL, 'whole-product-v2', _image_key(image),
               target['category'], json.dumps(target['bbox']))
        def compute():
            url, uploaded = upload_reference(image)
            transferred.update(image_url=url, source_upload_status=uploaded)
            mask = garment_mask_from_sam(image, url, target['category'], target)
            return {'image_url': url, 'mask': mask, 'source_upload_status': uploaded}
        try:
            prepared, reuse = _segmentation_cache.get_or_compute(
                key, compute, weight=lambda value: value['mask'].width * value['mask'].height)
        except REFERENCE_ERRORS as exc:
            return without_mask(exc)
        url, mask = prepared['image_url'], prepared['mask']
        selection = mask.info['target_selection']
        validation = assess_product_selection(target, selection, image.size)
        target = {**target, 'selection_validation': validation,
                  'review_required': bool(target.get('review_required')) and not validation['approved']}
        report('preparing', target_selection=selection, classification=target)
        snapshot("mask", "Object mask", mask)
        snapshot("segmented", "Isolated product", Image.composite(
            image, Image.new("RGB", image.size, "white"), mask))
        return {**prepared, 'classification': target,
                'target_selection': selection, 'segmentation_reuse': reuse}

    def balance(state):
        report("balancing_color")
        corrected = state['image'].copy()
        illumination = {'status':'deferred', 'applied':False, 'gains_rgb':[1.,1.,1.],
                        'reason':'original_scene_retained_for_relighting', 'calibrated':False,
                        'method':'scene_context', 'strength':0.}
        snapshot("color", "Source colors", corrected)
        reference = corrected
        crop = None
        if not state.get('direct_reference'):
            reference, crop = context_reference(reference, state['classification'], state.get('mask'))
        # Deduplicate source photos before replacing the primary with a crop;
        # otherwise uploading the same photo again would create two references.
        seen = {_image_key(state['image'])}
        extra_images = []
        for extra in state.get('additional_images', []):
            key = _image_key(extra)
            if key not in seen:
                seen.add(key)
                extra_images.append(extra)
        with ThreadPoolExecutor(max_workers=3) as uploads:
            reference_upload = uploads.submit(upload_reference, reference)
            additional_uploads = [uploads.submit(upload_reference, extra)
                                  for extra in extra_images]
            reference_url, reference_reuse = reference_upload.result()
            additional = [future.result() for future in additional_uploads]
        # Keep a local preview even after temporary input storage expires.
        snapshot("reference", "Generation reference", reference)
        return {"color_reference": corrected, "illumination": illumination,
                "color_url": reference_url, "reference_url": reference_url,
                'reference_crop': crop,
                'reference_policy': 'source_photographs' if state.get('direct_reference') else
                    'context_crop' if crop else 'original_scene',
                'additional_reference_urls': [url for url, _ in additional],
                'reference_uploads': [reference_reuse, *[reuse for _, reuse in additional]]}


    def restyle(state):
        # One primary reference, with each distinct source photo supplied once.
        # The context crop replaces the primary; it is never a second reference.
        references = list(dict.fromkeys([state['reference_url'], *state.get('additional_reference_urls', [])]))
        prompt = build_generation_prompt(state["garment_description"], state["illumination"],
                                         state['classification'], additional_count=len(references)-1)
        report("generating", generation_prompt=prompt,
               generation_settings={"model": model, "resolution": resolution})
        url = output_url(model, {"image_urls": references,
            "prompt": prompt, "aspect_ratio": "2:3", "resolution": resolution,
            "output_format": "png", "num_images": 1})
        snapshot("generated", "Generated photo", url)
        result = {"generated_url": url, "generation_prompt": prompt}
        result.update(pressed_url=url)
        return result


    def inspect(state, after_repair=False):
        phase = 'rechecking_product' if after_repair else 'checking_product'
        report(phase)
        image = fetch_image(state['pressed_url']).convert('RGB')
        check = {'model':VALIDATION_MODEL, 'after_repair':after_repair}
        try:
            check['verdict'] = check_single_product(image,call,extract_description)
        except REFERENCE_ERRORS as exc:
            check.update(verdict='uncertain',error_type=type(exc).__name__)
        checks = [*state.get('product_checks',[]),check]
        report(phase, single_product={'checks':checks, 'status':check['verdict']})
        # Once duplication is positively detected, never publish a replacement
        # unless the independent recheck confirms one product. No retry loop.
        if after_repair and check['verdict'] != 'single':
            raise RuntimeError('A single product could not be verified after correction. Please regenerate.')
        duplicate = check['verdict'] == 'duplicate'
        if duplicate:
            report(phase,stages=progress_stages(include_repair=True))
        return {'inspected_image':image, 'product_checks':checks, 'needs_product_repair':duplicate}

    def repair_product(state):
        report('correcting_product')
        url = output_url(model, {'image_urls':[state['pressed_url']],
            'prompt':single_product_repair_prompt(), 'aspect_ratio':'2:3', 'resolution':resolution,
            'output_format':'png', 'num_images':1})
        snapshot('product_repaired','Corrected single product',url)
        return {'pressed_url':url, 'generated_url':url, 'inspected_image':None}

    def finish(state):
        report("finishing")
        # The quality gate already downloaded this exact generated candidate.
        with ThreadPoolExecutor(max_workers=2) as transfers:
            image_future = (None if state.get('inspected_image') is not None else
                            transfers.submit(fetch_image,state['pressed_url']))
            matte_future = transfers.submit(fetch_background_matte, state['pressed_url'])
            generated = (state['inspected_image'].copy() if image_future is None else
                         image_future.result().convert('RGB'))
            try:
                matte = matte_future.result()
                cutout = remove_background(generated, state["pressed_url"], matte=matte)
            except REFERENCE_ERRORS as exc:
                # A missing diagnostic matte must not discard a successful edit.
                result = ImageOps.pad(generated, (1200,1800), method=Image.Resampling.LANCZOS,
                                      color=(245,245,245))
                snapshot('composed', 'Studio photograph', result)
                return recovered(state, 'finish', exc, result=result, result_mask=None,
                                 generated_image=generated)
        snapshot("cutout", "Product matte (diagnostic)", cutout)
        # Preserve the model's RGB, shadows and framing. The matte is for
        # color blending/diagnostics; recompositing can erase pale straps and prints.
        result = ImageOps.pad(generated, (1200,1800), method=Image.Resampling.LANCZOS,
                              color=(245,245,245))
        result_mask = ImageOps.pad(cutout.getchannel('A'), (1200,1800),
                                   method=Image.Resampling.NEAREST, color=0)
        cleanup = {'applied':False, 'removed_boxes':[]}
        result, result_mask, cleanup = clean_detached_fragment(
            result, result_mask, state['classification']['category'])
        if cleanup['applied']:
            snapshot('cleanup', 'Clean studio photograph', result)
        snapshot("composed", "Studio composition", result)
        return {"result": result, "result_mask": result_mask, 'generated_image':generated,
                'output_cleanup':cleanup}

    def verify(state):
        report("checking_quality")
        result, color_fidelity = restore_color_fidelity(
            state['color_reference'], state['result'], state.get('mask'), state.get('result_mask'))
        if color_fidelity['applied']:
            snapshot('color_fidelity', 'Color fidelity', result)
        try:
            if state.get('mask') is None or state.get('result_mask') is None:
                raise ValueError('Foreground comparison unavailable')
            quality = compare_product(state["color_reference"], state["mask"], result, state["result_mask"])
        except (ValueError, RuntimeError):
            quality = {'status':'review_required', 'passed':False, 'metrics':{}, 'thresholds':{},
                       'warnings':['Foreground comparison unavailable; the photo was generated from the source references.'],
                       'note':'No mask or diagnostic scores were invented.'}
        quality['recovery'] = state.get('recovery', {})
        quality['color_fidelity'] = color_fidelity
        quality['output_cleanup'] = state.get('output_cleanup', {'applied':False, 'removed_boxes':[]})
        checks = state['product_checks']
        single = checks[-1]['verdict'] == 'single'
        quality['single_product'] = {'status':'passed' if single else 'unverified',
                                     'repair_attempted':len(checks)>1, 'checks':checks}
        if not single:
            quality['warnings'].append('The single-product check was inconclusive; verify that no duplicate garment is visible.')
            quality.update(status='review_required',passed=False)
        quality["illumination"] = state["illumination"]
        quality['color_reference'] = 'original_capture_lighting_dependent'
        quality['pipeline_profile'] = 'faithful'
        if state['classification'].get('review_required'):
            quality['warnings'].append('Main-product classification is uncertain; verify that the output matches the intended product.')
            quality.update(status='review_required', passed=False)
        result_url = upload_png(result)
        snapshot("final", "Final output", result_url)
        return {"quality": quality, "result_url": result_url, 'result':result}

    graph = StateGraph(PipelineState)
    nodes = [("classify", classify), ("locate", locate), ("isolate", isolate), ("balance", balance),
             ("restyle", restyle), ('inspect',inspect),
             ('repair_product',repair_product), ('inspect_repair',lambda state:inspect(state,after_repair=True)),
             ("finish", finish), ("verify", verify)]
    for name, node in nodes:
        def measured(state, operation=node, stage=name):
            started = perf_counter()
            label, phase, step_model, libraries, details = steps[stage]
            previous_trace = state.get('execution_trace', [])
            entry = begin_step(stage, label, step_model, libraries, run_id, **details)
            report(phase, execution_trace=[*previous_trace, entry])
            try:
                result = operation(state)
            except Exception as exc:
                entry = end_step(entry, perf_counter() - started, 'error', error_type=type(exc).__name__)
                report(phase, execution_trace=[*previous_trace, entry])
                raise
            outcome = 'complete'
            used = {}
            if stage == 'classify':
                target = result['classification']
                outcome = 'cached' if target['cache_hit'] else 'complete'
                if target.get('review_required') and not target['cache_hit']:
                    outcome = 'review_required'
                used = {**target, 'model_invoked': not target['cache_hit'],
                        'pipeline_profile':'faithful'}
            elif stage == 'locate':
                target = result['classification']
                hit = target['localization_cache_hit']
                invoked = target.get('localization_invoked', not hit)
                if target['bbox'] is None:
                    outcome = 'fallback' if invoked else 'skipped'
                else:
                    outcome = 'cached' if hit else 'complete'
                used = {'bbox': target['bbox'], 'cache_hit': hit, 'model_invoked': invoked}
                used['cache_status'] = target['localization_cache_status']
            elif stage == 'isolate':
                # Resolve the preliminary warning once independent localization
                # and segmentation support it, including on cached executions.
                target = result['classification']
                if target['selection_validation']['approved']:
                    previous_trace = [
                        {**step, 'status': 'cached' if target['cache_hit'] else 'complete',
                         'details': {**step['details'], **target}}
                        if step['id'] == 'classify' else step for step in previous_trace]
                reuse = result['segmentation_reuse']
                outcome = reuse if reuse in ('skipped', 'fallback') else ('cached' if reuse != 'computed' else 'complete')
                used = {**result['target_selection'], 'cache_status': reuse,
                        'cache_hit': reuse in ('cached', 'shared'), 'model_invoked': reuse in ('computed', 'fallback'),
                        'storage_uploads': int(result['source_upload_status'] == 'computed' and reuse in ('computed', 'fallback'))}
            elif stage == 'restyle':
                used = {'input_image_count': len(set([state['reference_url'], *state.get('additional_reference_urls', [])])),
                        'reference_policy':state['reference_policy'], 'reference_crop':state.get('reference_crop')}
            elif stage == 'finish':
                used = {'output_cleanup':result.get('output_cleanup', {'applied':False, 'removed_boxes':[]})}
            elif stage in ('inspect','inspect_repair'):
                used = result['product_checks'][-1]
                outcome = 'complete' if used['verdict'] == 'single' else 'review_required'
            elif stage == 'repair_product':
                used = {'input_image_count':1, 'reason':'duplicate_product_detected', 'max_attempts':1}
            elif stage == 'balance':
                used = {'correction_applied': result['illumination']['applied'],
                        'strength': result['illumination'].get('strength'),
                        'storage_uploads': result['reference_uploads'].count('computed'),
                        'reused_uploads': sum(item != 'computed' for item in result['reference_uploads'])}
            elif stage == 'verify':
                used = {'storage_uploads': 1, 'color_fidelity':result['quality']['color_fidelity']}
            recovery = result.get('recovery', {}).get(stage)
            if recovery:
                if outcome != 'skipped':
                    outcome = 'fallback'
                used.update(recovery)
            entry = end_step(entry, perf_counter() - started, outcome, **used)
            durations = metrics['stages_seconds']
            durations[stage] = entry['seconds']
            metrics['total_seconds'] = round(sum(durations.values()), 3)
            trace = [*previous_trace, entry]
            metrics['inference_requests'] = sum(step['model'] is not None and
                step['details'].get('model_invoked', True) for step in trace)
            metrics['reused_inferences'] = sum(step['model'] is not None and
                step['details'].get('model_invoked') is False and step['status'] != 'skipped' for step in trace)
            metrics['storage_uploads'] = sum(step['details'].get('storage_uploads', 0) for step in trace)
            snapshot_metrics = {**metrics, 'stages_seconds': dict(durations)}
            report(phase, execution_trace=trace, metrics=snapshot_metrics)
            return {**result, 'metrics': snapshot_metrics, 'execution_trace': trace}

        graph.add_node(name, measured)
    main_path = [name for name,_ in nodes if name not in ('repair_product','inspect_repair')]
    graph.add_edge(START,main_path[0])
    for previous,name in zip(main_path,main_path[1:]):
        if previous == 'inspect':
            graph.add_conditional_edges(previous,
                lambda state:'repair_product' if state['needs_product_repair'] else 'finish',
                {'repair_product':'repair_product','finish':'finish'})
        else:
            graph.add_edge(previous,name)
    graph.add_edge('repair_product','inspect_repair')
    graph.add_edge('inspect_repair','finish')
    graph.add_edge(main_path[-1],END)
    return graph.compile()
