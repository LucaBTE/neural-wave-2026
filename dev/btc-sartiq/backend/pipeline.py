from __future__ import annotations
import io
import html
import re
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Callable, TypedDict
from urllib.request import urlopen

import fal_client
from langgraph.graph import END, START, StateGraph
from quality import compare_product
from color import normalize_illumination
from generation_prompt import build_generation_prompt, build_pressing_prompt
from garment_analysis import ANALYSIS_MODELS, DEFAULT_MODEL, analyze_garment, analysis_settings, prepare_analysis_image
from execution_trace import begin_step, end_step
from target_selection import select_centered_mask
from PIL import Image, ImageChops, ImageFilter, ImageOps

SAM_MODEL = "fal-ai/sam-3/image"
BIREF_MODEL = "fal-ai/birefnet/v2"
DESCRIPTION_MODEL = DEFAULT_MODEL
RESTYLE_MODEL = "fal-ai/nano-banana-pro/edit"
MODELS = {"description": DESCRIPTION_MODEL, "segmentation": SAM_MODEL, "restyling": RESTYLE_MODEL,
          "background_removal": BIREF_MODEL, "quality": "Python + OpenCV"}

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
    # Fal output URLs or data URIs only; do not expose arbitrary user URL fetching.
    if not url.startswith("https://"):
        raise ValueError("Expected an HTTPS image URL from fal")
    from urllib.request import Request
    req = Request(url, headers={"User-Agent": "CatalogueStudio/1.0"})
    with urlopen(req, timeout=45) as response:
        data = response.read(20_000_001)
    if len(data) > 20_000_000: raise ValueError("Remote image too large")
    im = Image.open(io.BytesIO(data)); im.load(); return im

def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO(); image.save(buffer, format="PNG"); return buffer.getvalue()

def upload_png(image: Image.Image) -> str:
    configure_fal()
    return fal_client.upload(png_bytes(image), content_type="image/png")

def configure_fal() -> None:
    key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not key: raise RuntimeError("Set FAL_API_KEY or FAL_KEY in backend/.env")
    os.environ["FAL_KEY"] = key

def call(model: str, arguments: dict) -> dict:
    configure_fal()
    timeout = analysis_settings()['timeout'] if model in ANALYSIS_MODELS else 300
    # The SDK defaults to 0.1s: ten status HTTP requests/sec for ONE inference.
    return fal_client.subscribe(model, arguments=arguments, with_logs=False,
                                interval=2.0, client_timeout=timeout)


def generation_settings() -> tuple[str, str]:
    model = os.getenv("FAL_IMAGE_MODEL", RESTYLE_MODEL).strip()
    resolution = os.getenv("FAL_IMAGE_RESOLUTION", "2K").strip()
    if model not in (RESTYLE_MODEL, "fal-ai/nano-banana-2/edit"):
        raise ValueError("FAL_IMAGE_MODEL must be fal-ai/nano-banana-pro/edit or fal-ai/nano-banana-2/edit")
    if resolution not in ("1K", "2K", "4K"):
        raise ValueError("FAL_IMAGE_RESOLUTION must be 1K, 2K or 4K")
    return model, resolution

def normalize_photo(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data)); image.load()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > 2048:
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image

def garment_mask_from_sam(image: Image.Image, image_url: str, garment: str) -> Image.Image:
    response = call(SAM_MODEL, {"image_url": image_url, "prompt": garment, "apply_mask": False,
        "point_prompts": [{"x": image.width // 2, "y": image.height // 2, "label": 1}],
        "return_multiple_masks": True, "max_masks": 3, "output_format": "png"})
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
    garment_mask, selection = select_centered_mask(layers)
    garment_mask.info['target_selection'] = selection
    return garment_mask

def background_mask(garment_mask: Image.Image, feather: int = 0) -> Image.Image:
    # White = regenerate, BLACK = keep (object preserved). Small inward dilation
    # of the kept area helps avoid replacing garment edge pixels.
    protected = garment_mask.convert("L")
    if feather > 0: protected = protected.filter(ImageFilter.MaxFilter(2*feather+1))
    return ImageOps.invert(protected)


def output_url(model: str, arguments: dict) -> str:
    urls = extract_urls(call(model, arguments))
    if not urls:
        raise RuntimeError(f"{model} did not return an image")
    return urls[0]


def remove_background(image: Image.Image, image_url: str) -> Image.Image:
    # The generated photo contains a single garment on white. BiRefNet supplies
    # its mask directly: no second SAM inference or re-upload is necessary.
    url = output_url(BIREF_MODEL, {"image_url": image_url,
        "output_format": "png", "refine_foreground": True})
    cutout = fetch_image(url)
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


def compose_studio(cutout: Image.Image, size=(1200, 1800)) -> tuple[Image.Image, Image.Image]:
    """Deterministic white background; fit without distorting the restyled garment."""
    alpha = cutout.getchannel("A")
    bbox = alpha.point(lambda p: 255 if p > 20 else 0).getbbox()
    if bbox is None:
        raise RuntimeError("No garment left after background removal")
    cropped = cutout.crop(bbox)
    cropped.thumbnail((int(size[0] * .78), int(size[1] * .72)), Image.Resampling.LANCZOS)
    position = ((size[0] - cropped.width) // 2, (size[1] - cropped.height) // 2)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(cropped, position, cropped.getchannel("A"))
    mask = Image.new("L", size, 0)
    mask.paste(cropped.getchannel("A"), position)
    return canvas, mask


class PipelineState(TypedDict, total=False):
    image: Image.Image
    image_url: str
    garment: str
    mask: Image.Image
    target_selection: dict
    color_reference: Image.Image
    color_url: str
    illumination: dict
    reference_url: str
    garment_description: str
    analysis: dict
    generation_prompt: str
    generated_url: str
    pressing_prompt: str
    pressed_url: str
    result: Image.Image
    result_mask: Image.Image
    result_url: str
    quality: dict
    metrics: dict
    execution_trace: list[dict]


def build_pipeline(on_progress: Callable[..., None] | None = None,
                   on_image: Callable[[str, str, Image.Image | str], None] | None = None,
                   run_id: str | None = None):
    """Full single-click workflow. No automatic paid retries or preview stops."""
    model, resolution = generation_settings()  # Validate before any paid call.
    analysis_config = analysis_settings()
    pressing_prompt = build_pressing_prompt()  # Fail before paid work if missing/empty.
    metrics = {"stages_seconds": {}, "total_seconds": 0.0}
    pillow = {'Pillow': 'pillow'}
    remote = {'fal-client': 'fal-client', 'LangGraph': 'langgraph'}
    vision = {'NumPy': 'numpy', 'OpenCV': 'opencv-python-headless', **pillow}
    steps = {
        'isolate': ('Preparing', 'preparing', SAM_MODEL, {**vision, **remote}, {}),
        'balance': ('Balancing colors', 'balancing_color', None, {**vision, **remote}, {}),
        'describe': ('Analyzing garment', 'describing', analysis_config['model'], {**pillow, **remote},
                     {'timeout_seconds': analysis_config['timeout'], 'max_side': analysis_config['max_side'],
                      'required': True, 'reasoning': analysis_config['reasoning']}),
        'restyle': ('Creating photo', 'generating', model, remote, {'resolution': resolution}),
        'press': ('Removing wrinkles', 'pressing', model, remote, {'resolution': resolution}),
        'finish': ('Refining', 'finishing', BIREF_MODEL, {**pillow, **remote}, {}),
        'verify': ('Final check', 'checking_quality', None, {**vision, **remote}, {}),
    }

    def report(stage, **fields):
        if on_progress:
            on_progress(status=stage, **fields)

    def snapshot(key, label, image):
        if on_image:
            on_image(key, label, image)

    def isolate(state):
        report("preparing")
        image = state["image"]
        snapshot("original", "Original", image)
        url = upload_png(image)
        mask = garment_mask_from_sam(image, url, state.get("garment", "clothing"))
        selection = mask.info['target_selection']
        report('preparing', target_selection=selection)
        snapshot("mask", "Object mask", mask)
        snapshot("segmented", "Isolated garment", Image.composite(
            image, Image.new("RGB", image.size, "white"), mask))
        return {"image_url": url, "mask": mask, 'target_selection': selection}

    def balance(state):
        report("balancing_color")
        corrected, illumination = normalize_illumination(state["image"], state["mask"])
        snapshot("color", "Color correction", corrected)
        reference = Image.composite(corrected, Image.new("RGB", corrected.size, "white"), state["mask"])
        # Both generation inputs must contain only the selected product. Sending
        # the full scene allowed the generator to choose a different garment.
        target_detail = prepare_analysis_image(corrected, state['mask'], max(corrected.size))
        with ThreadPoolExecutor(max_workers=2) as uploads:
            color_upload = uploads.submit(upload_png, target_detail)
            reference_upload = uploads.submit(upload_png, reference)
            color_url, reference_url = color_upload.result(), reference_upload.result()
        snapshot("reference", "Generation reference", reference_url)
        return {"color_reference": corrected, "illumination": illumination,
                "color_url": color_url, "reference_url": reference_url}

    def describe(state):
        report("describing")
        description, analysis, cropped = analyze_garment(state["color_reference"], state["mask"],
            state["illumination"], call, extract_description, png_bytes)
        snapshot("description", "Garment analysis", cropped)
        report("describing", analysis=analysis)
        return {"garment_description": description, "analysis": analysis}

    def restyle(state):
        prompt = build_generation_prompt(state["garment_description"], state["illumination"])
        report("generating", generation_prompt=prompt,
               generation_settings={"model": model, "resolution": resolution})
        url = output_url(model, {"image_urls": [state["color_url"], state["reference_url"]],
            "prompt": prompt, "aspect_ratio": "2:3", "resolution": resolution,
            "output_format": "png", "num_images": 1})
        snapshot("generated", "Generated photo", url)
        return {"generated_url": url, "generation_prompt": prompt}

    def press(state):
        report('pressing', pressing_prompt=pressing_prompt)
        # Use only the generated photo: wrinkled source references would reintroduce
        # the surface condition this dedicated edit is meant to replace.
        url = output_url(model, {'image_urls': [state['generated_url']],
            'prompt': pressing_prompt, 'aspect_ratio': '2:3', 'resolution': resolution,
            'output_format': 'png', 'num_images': 1})
        snapshot('pressed', 'Steam-pressed photo', url)
        return {'pressed_url': url, 'pressing_prompt': pressing_prompt}

    def finish(state):
        report("finishing")
        generated = fetch_image(state["pressed_url"]).convert("RGB")
        cutout = remove_background(generated, state["pressed_url"])
        snapshot("cutout", "Background removed", cutout)
        result, result_mask = compose_studio(cutout)
        snapshot("composed", "Studio composition", result)
        return {"result": result, "result_mask": result_mask}

    def verify(state):
        report("checking_quality")
        quality = compare_product(state["color_reference"], state["mask"], state["result"], state["result_mask"])
        quality["illumination"] = state["illumination"]
        quality["color_reference"] = "estimated_white_balance" if state["illumination"]["applied"] else "uncorrected_input"
        if state["illumination"]["status"] == "uncertain":
            quality["warnings"].append("Illumination is ambiguous; true garment colors could not be established.")
            quality.update(status="review_required", passed=False)
        result_url = upload_png(state["result"])
        snapshot("final", "Final output", result_url)
        return {"quality": quality, "result_url": result_url}

    graph = StateGraph(PipelineState)
    nodes = [("isolate", isolate), ("balance", balance), ("describe", describe), ("restyle", restyle), ("press", press), ("finish", finish), ("verify", verify)]
    previous = START
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
            if stage == 'isolate':
                used = result['target_selection']
            elif stage == 'describe':
                analysis = result['analysis']
                outcome = 'cached' if analysis['cache_hit'] else analysis['status']
                used = {'cache_hit': analysis['cache_hit'], 'input_size': analysis['input_size'],
                        'input_format': analysis['input_format'], 'input_bytes': analysis['input_bytes'],
                        'model_invoked': not analysis['cache_hit']}
            elif stage == 'balance':
                used = {'correction_applied': result['illumination']['applied'],
                        'strength': result['illumination'].get('strength')}
            entry = end_step(entry, perf_counter() - started, outcome, **used)
            durations = metrics['stages_seconds']
            durations[stage] = entry['seconds']
            metrics['total_seconds'] = round(sum(durations.values()), 3)
            snapshot_metrics = {**metrics, 'stages_seconds': dict(durations)}
            trace = [*previous_trace, entry]
            report(phase, execution_trace=trace, metrics=snapshot_metrics)
            return {**result, 'metrics': snapshot_metrics, 'execution_trace': trace}

        graph.add_node(name, measured)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()
