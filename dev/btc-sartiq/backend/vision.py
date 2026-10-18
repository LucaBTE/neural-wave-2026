"""Shared Moondream settings and local multi-view preparation."""
import os
from PIL import Image, ImageDraw, ImageOps

DEFAULT_MODEL = 'fal-ai/moondream3-preview/query'
CACHE_TTL = 3600
CACHE_SIZE = 128


def analysis_settings():
    timeout = int(os.getenv('FAL_ANALYSIS_TIMEOUT', '90'))
    max_side = int(os.getenv('FAL_ANALYSIS_MAX_SIDE', '1280'))
    if not 15 <= timeout <= 300:
        raise ValueError('FAL_ANALYSIS_TIMEOUT must be 15..300 seconds')
    if not 768 <= max_side <= 2048:
        raise ValueError('FAL_ANALYSIS_MAX_SIDE must be 768..2048 pixels')
    return {'model': DEFAULT_MODEL, 'timeout': timeout, 'max_side': max_side,
            'required': True, 'reasoning': False}


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

