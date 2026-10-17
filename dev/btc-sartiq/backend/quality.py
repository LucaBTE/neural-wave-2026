"""Heuristic product fidelity checks, not a semantic identity guarantee."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

THRESHOLDS = {"color_distance_max": 0.60, "print_color_distance_max": 0.45, "lightness_shift_max": 0.45,
              "white_background_fraction_min": 0.98}


def _crop(image: Image.Image, mask: Image.Image):
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    binary = np.asarray(mask) > 127
    ys, xs = np.where(binary)
    if len(xs) < 64:
        raise ValueError("Too few garment pixels for quality comparison")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rgb = np.asarray(image.convert("RGB").crop(box).resize((256, 256), Image.Resampling.LANCZOS))
    region = np.asarray(mask.crop(box).resize((256, 256), Image.Resampling.NEAREST)) > 127
    return rgb, region, (box[2] - box[0]) / (box[3] - box[1])


def compare_product(original, original_mask, result, result_mask) -> dict:
    a, ma, aspect_a = _crop(original, original_mask)
    b, mb, aspect_b = _crop(result, result_mask)
    # Compare Lab chromatic channels to reduce sensitivity to studio exposure.
    def histogram(rgb, mask):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        hist = cv2.calcHist([lab], [1, 2], mask.astype(np.uint8) * 255,
                            [32, 32], [0, 256, 0, 256])
        return cv2.normalize(hist, None, norm_type=cv2.NORM_L1)

    color = float(cv2.compareHist(histogram(a, ma), histogram(b, mb), cv2.HISTCMP_BHATTACHARYYA))
    # A small red print must not be hidden by a much larger white fabric region.
    colorful_a = ma & (cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[:, :, 1] > 60)
    colorful_b = mb & (cv2.cvtColor(b, cv2.COLOR_RGB2HSV)[:, :, 1] > 60)
    print_distance = None
    if colorful_a.sum() >= 32:
        print_distance = (float(cv2.compareHist(histogram(a, colorful_a), histogram(b, colorful_b),
                          cv2.HISTCMP_BHATTACHARYYA)) if colorful_b.sum() >= 32 else 1.0)
    iou = float(np.count_nonzero(ma & mb) / max(1, np.count_nonzero(ma | mb)))
    aspect = min(aspect_a, aspect_b) / max(aspect_a, aspect_b)
    overlap = cv2.erode((ma & mb).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    # Local contrast comparison helps detect changed fabric/markings; ignore matte edges.
    ga, gb = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) for x in (a, b)]
    mu_a, mu_b = [cv2.GaussianBlur(x, (11, 11), 1.5) for x in (ga, gb)]
    var_a = np.maximum(0, cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a * mu_a)
    var_b = np.maximum(0, cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b * mu_b)
    covariance = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    structure = (2 * covariance + 58.5225) / (var_a + var_b + 58.5225)
    detail = float(np.clip(structure[overlap].mean(), 0, 1)) if overlap.any() else 0.0
    # Chroma alone cannot distinguish a black garment from a white one.
    lightness_shift = abs(float(np.median(cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[:, :, 0][ma]))
                          - float(np.median(cv2.cvtColor(b, cv2.COLOR_RGB2LAB)[:, :, 0][mb]))) / 255
    warnings = []
    if print_distance is not None and print_distance > THRESHOLDS["print_color_distance_max"]:
        warnings.append("Colors in the print or colored garment details differ from the color reference.")
    if lightness_shift > THRESHOLDS["lightness_shift_max"]:
        warnings.append("Garment lightness differs substantially from the source.")
    if color > THRESHOLDS["color_distance_max"]:
        warnings.append("Garment color differs from the source.")
    # Pose, folds and local structure are deliberately allowed to change.
    # Keep these metrics diagnostic, never reject the requested restyling on them.
    outside = np.asarray(result_mask.resize(result.size)) < 10
    rgb = np.asarray(result.convert("RGB"))
    white_fraction = float(np.mean(np.all(rgb[outside] >= 245, axis=1))) if outside.any() else 0.0
    if white_fraction < THRESHOLDS["white_background_fraction_min"]:
        warnings.append("The background is not uniformly white.")
    return {"status": "review_required" if warnings else "passed", "passed": not warnings,
            "metrics": {"print_color_distance": round(print_distance, 4) if print_distance is not None else None, "lightness_shift": round(lightness_shift, 4), "white_background_fraction": round(white_fraction, 4), "color_distance": round(color, 4), "silhouette_iou": round(iou, 4),
                        "aspect_similarity": round(aspect, 4), "detail_similarity": round(detail, 4)},
            "thresholds": THRESHOLDS.copy(), "warnings": warnings,
            "note": "Color/background checks only; pose and folds may change. Print identity, exact lettering and texture require visual review; passing does not verify them."}
