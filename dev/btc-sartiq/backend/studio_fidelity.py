"""Keep source RGB intact and remove only well-separated studio fragments."""
import math

import cv2
import numpy as np
from PIL import Image, ImageFilter


def context_reference(image, target, mask):
    """Crop only an independently agreed whole-product box, with lighting context."""
    box = target.get('bbox')
    if (mask is None or not box or
            not target.get('selection_validation', {}).get('approved')):
        return image.copy(), None
    left, top, right, bottom = box
    mx, my = (right - left) * .08, (bottom - top) * .08
    bounds = [max(0, math.floor((left - mx) * image.width)),
              max(0, math.floor((top - my) * image.height)),
              min(image.width, math.ceil((right + mx) * image.width)),
              min(image.height, math.ceil((bottom + my) * image.height))]
    # Include any real segmented parts extending beyond the detector's box.
    actual = mask.getbbox()
    if actual:
        bounds = [min(bounds[0], actual[0]), min(bounds[1], actual[1]),
                  max(bounds[2], actual[2]), max(bounds[3], actual[3])]
    return image.crop(tuple(bounds)), bounds


def clean_detached_fragment(image, mask, category):
    """Remove detached upper-shirt remnants only on a verified flat light surface.

    Keep the main RGB silhouette, omitted pale straps and its contact shadow intact.
    This is not an alpha recomposition or a general-purpose object eraser.
    """
    report = {'applied': False, 'removed_boxes': []}
    if mask is None or category.casefold() not in {
            't-shirt', 'tshirt', 'shirt', 'sweater', 'jumper', 'pullover', 'sweatshirt', 'hoodie'}:
        return image, mask, report
    alpha = np.asarray(mask.convert('L'))
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 100).astype('uint8'), connectivity=8)
    if count < 3:
        return image, mask, report
    primary = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[primary])
    if area < image.width * image.height * .12:
        return image, mask, report
    rgb = np.array(image.convert('RGB'))
    cleaned_mask = mask.copy()
    for index in range(1, count):
        if index == primary:
            continue
        fx, fy, fw, fh, fa = map(int, stats[index])
        if not (.005 * area < fa < .15 * area and fw > .2 * w and
                fh < .35 * h and fy + fh < y - .08 * h):
            continue
        margin = max(8, int(.025 * h))
        l, t = max(0, fx - margin), max(0, fy - margin)
        r, b = min(image.width, fx + fw + margin), min(image.height, fy + fh + margin)
        patch = rgb[t:b, l:r]
        border = np.concatenate([patch[:3].reshape(-1, 3), patch[-3:].reshape(-1, 3),
                                 patch[:, :3].reshape(-1, 3), patch[:, -3:].reshape(-1, 3)])
        surface = np.median(border, axis=0)
        # Refuse textured/colored backgrounds, nonuniform shadows or clipped objects.
        if (surface.min() < 230 or surface.max() - surface.min() > 10 or
                np.max(np.percentile(abs(border.astype(float) - surface), 95, axis=0)) > 3):
            continue
        fill = Image.new('RGB', image.size, tuple(int(v) for v in surface))
        removal = Image.new('L', image.size, 0)
        removal.paste(255, (l + 3, t + 3, r - 3, b - 3))
        removal = removal.filter(ImageFilter.GaussianBlur(1))
        image = Image.composite(fill, image, removal)
        cleaned_mask.paste(0, (l, t, r, b))
        report['removed_boxes'].append([l, t, r, b])
    report['applied'] = bool(report['removed_boxes'])
    return image, cleaned_mask, report
