"""Choose one central product from SAM candidates, never union unrelated garments."""
import cv2
import numpy as np
from PIL import Image


def assess_product_selection(target, selection, image_size):
    """Confirm an uncertain identity using independent detector/SAM geometry.

    Compare bounding boxes, not foreground fill: glasses and straps have holes.
    This confirms spatial consistency, not semantic identity or output fidelity.
    """
    width, height = image_size
    left, top, right, bottom = target['bbox']
    expected = (left * width, top * height, right * width, bottom * height)
    actual = selection['bbox']
    intersection = (max(0, min(expected[2], actual[2]) - max(expected[0], actual[0])) *
                    max(0, min(expected[3], actual[3]) - max(expected[1], actual[1])))
    expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
    actual_area = (actual[2] - actual[0]) * (actual[3] - actual[1])
    iou = intersection / max(1, expected_area + actual_area - intersection)
    width_coverage = (actual[2] - actual[0]) / max(1, expected[2] - expected[0])
    height_coverage = (actual[3] - actual[1]) / max(1, expected[3] - expected[1])
    approved = (selection['strategy'] == 'classified_product' and
                bool(target.get('category')) and target.get('identity_supported', True) and iou >= .7 and
                width_coverage >= .8 and height_coverage >= .8)
    return {'approved': approved, 'method': 'detector_mask_agreement',
            'box_iou': round(iou, 4), 'width_coverage': round(width_coverage, 4),
            'height_coverage': round(height_coverage, 4)}


def select_classified_mask(masks, target):
    """Require spatial agreement with the classified object, not image size."""
    candidates = []
    for index, mask in enumerate(masks):
        binary = (np.asarray(mask.convert('L')) >= 127).astype(np.uint8)
        height, width = binary.shape
        if binary.mean() > .95:
            continue
        left, top, right, bottom = target['bbox']
        # Small tolerance for imperfect visual localization; never use the whole
        # central image region, which may be occupied by a background sweater.
        x0, y0 = max(0, int((left - .03) * width)), max(0, int((top - .03) * height))
        x1, y1 = min(width, int(np.ceil((right + .03) * width))), min(height, int(np.ceil((bottom + .03) * height)))
        region = np.zeros_like(binary)
        region[y0:y1, x0:x1] = 1
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        kept = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area >= 4 and np.count_nonzero((labels == component) & (region > 0)) / area >= .85:
                kept.append(component)
        selected = np.isin(labels, kept)
        if np.count_nonzero(selected) < max(8, width * height * .0001):
            continue
        ys, xs = np.where(selected)
        # Reject a small fragment of the product, even if it contains the point.
        target_w, target_h = (right - left) * width, (bottom - top) * height
        if (xs.max() - xs.min() + 1 < target_w * .55 or
                ys.max() - ys.min() + 1 < target_h * .55):
            continue
        actual_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        expected = np.array([left * width, top * height, right * width, bottom * height])
        error = float(np.abs(np.array(actual_box) - expected).sum() / (width + height))
        score = -error
        details = {'strategy': 'classified_product', 'label': target['label'],
                   'candidate_index': index, 'candidate_count': len(masks),
                   'bbox': actual_box, 'components_kept': len(kept)}
        candidates.append((score, Image.fromarray(selected.astype(np.uint8) * 255), details))
    if not candidates:
        raise RuntimeError('No mask matches the classified product near the image center. Reposition the product and try again.')
    _, selected, details = max(candidates, key=lambda candidate: candidate[0])
    return selected, details


def select_centered_mask(masks, target=None):
    if target is not None:
        return select_classified_mask(masks, target)
    candidates = []
    for index, mask in enumerate(masks):
        binary = (np.asarray(mask.convert('L')) >= 127).astype(np.uint8)
        height, width = binary.shape
        if binary.mean() > .95:
            continue  # A nearly full-frame mask does not isolate a product.
        center = np.zeros_like(binary)
        center[int(height * .30):max(int(height * .70), int(height * .30) + 1),
               int(width * .30):max(int(width * .70), int(width * .30) + 1)] = 1
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < max(16, width * height * .001):
                continue
            central_pixels = int(np.count_nonzero((labels == component) & (center > 0)))
            if central_pixels < max(1, np.count_nonzero(center) * .005):
                continue
            components.append((central_pixels, area, component))
        if not components:
            continue
        _, primary_area, primary = max(components)
        x, y, w, h, _ = stats[primary]
        # Keep complete components near the central product, including disconnected
        # straps/legs. Remove detached corner clothing such as the holder's sleeve.
        kept = [primary]
        for component in range(1, count):
            if component == primary or stats[component, cv2.CC_STAT_AREA] < max(16, primary_area * .01):
                continue
            cx, cy = centroids[component]
            if (.18 * width <= cx <= .82 * width and .18 * height <= cy <= .82 * height and
                    x - .12 * width <= cx <= x + w + .12 * width and
                    y - .12 * height <= cy <= y + h + .12 * height):
                kept.append(component)
        selected = np.isin(labels, kept)
        central_coverage = np.count_nonzero(selected & (center > 0)) / max(1, np.count_nonzero(center))
        ys, xs = np.where(selected)
        distance = np.hypot(xs.mean() / width - .5, ys.mean() / height - .5)
        score = float(central_coverage - .25 * distance)
        details = {'strategy': 'central_product', 'candidate_index': index,
                   'candidate_count': len(masks), 'central_coverage': round(float(central_coverage), 4),
                   'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   'components_kept': len(kept)}
        candidates.append((score, Image.fromarray(selected.astype(np.uint8) * 255), details))
    if not candidates:
        raise RuntimeError('No garment was found near the image center. Center the complete garment and try again.')
    _, selected, details = max(candidates, key=lambda candidate: candidate[0])
    return selected, details
