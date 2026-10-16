from __future__ import annotations
import io
import os
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import fal_client
from PIL import Image, ImageChops, ImageFilter, ImageOps

SAM_MODEL = "fal-ai/sam-3-1/image"
FILL_MODEL = "fal-ai/flux-pro/v1/fill"
# Presets deliberately describe ONLY the background; garment remains outside inpaint mask.
PRESETS = {
    "studio": "Luxury apparel catalogue product photograph. Seamless warm white cyclorama studio background, soft broad diffused key light at camera left, gentle fill, realistic contact shadow and natural professional color grading. Keep the garment exactly where it is and unchanged. No logos, no text, no props, no people.",
    "editorial": "Premium fashion editorial still life. Elegant warm neutral plaster studio background, soft directional window lighting, subtle authentic shadows, minimal sophisticated composition, fashion catalogue photography. Preserve the original garment perfectly. No people, no text, no extra clothing.",
    "dark": "High end e-commerce apparel photography. Deep charcoal studio background, soft large octabox key light, controlled rim lighting and authentic soft contact shadow. Preserve the garment itself unaltered, including shape, color, seams, texture and logos. No additional objects or text.",
}

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
    # Modern fal-client supports uploading bytes via upload(data, content_type=...)
    return fal_client.upload(png_bytes(image), content_type="image/png")

def call(model: str, arguments: dict) -> dict:
    key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not key: raise RuntimeError("Set FAL_API_KEY or FAL_KEY in backend/.env")
    os.environ["FAL_KEY"] = key
    return fal_client.subscribe(model, arguments=arguments, with_logs=False, client_timeout=180)

def normalize_photo(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data)); image.load()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > 2048:
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image

def garment_mask_from_sam(image: Image.Image, image_url: str, garment: str) -> Image.Image:
    response = call(SAM_MODEL, {"image_url": image_url, "prompt": garment, "apply_mask": False, "return_multiple_masks": False, "output_format": "png"})
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
    garment_mask = layers[0]
    for layer in layers[1:]: garment_mask = ImageChops.lighter(garment_mask,layer)
    if garment_mask.getbbox() is None: raise RuntimeError("No garment pixels found. Try a more specific garment prompt.")
    return garment_mask

def background_mask(garment_mask: Image.Image, feather: int = 0) -> Image.Image:
    # White = regenerate, BLACK = keep (object preserved). Small inward dilation
    # of the kept area helps avoid replacing garment edge pixels.
    protected = garment_mask.convert("L")
    if feather > 0: protected = protected.filter(ImageFilter.MaxFilter(2*feather+1))
    return ImageOps.invert(protected)

def generate(image_url: str, mask_url: str, preset: str, custom_prompt: str = "") -> str:
    prompt = PRESETS[preset] + (" Additional art direction: " + custom_prompt if custom_prompt.strip() else "")
    response = call(FILL_MODEL, {"image_url": image_url, "mask_url": mask_url, "prompt": prompt, "output_format": "png"})
    urls = extract_urls(response)
    if not urls: raise RuntimeError("Fill response did not include an image")
    return urls[0]
