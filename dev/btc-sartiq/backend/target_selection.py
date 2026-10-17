"""Choose one central product from SAM candidates, never union unrelated garments."""
import cv2
import numpy as np
from PIL import Image


def select_centered_mask(masks):
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
