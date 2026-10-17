"""Conservative, uncalibrated illuminant estimation, not ground-truth color recovery.

Estimate channel gains from spatially distributed, bright low-chroma candidates.
Reject disagreeing regions (possible mixed light), strong casts and insufficient
samples. Work in linear RGB; never selectively repaint a print or match a target.
"""
import cv2
import os
import numpy as np
from PIL import Image


def to_linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def to_srgb(rgb):
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)


def normalize_illumination(image: Image.Image, mask: Image.Image):
    strength = float(os.getenv('COLOR_CORRECTION_STRENGTH', '1.05'))
    if not np.isfinite(strength) or not .5 <= strength <= 1.5:
        raise ValueError('COLOR_CORRECTION_STRENGTH must be 0.5..1.5')
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    region = np.asarray(mask.convert('L').resize(image.size, Image.Resampling.NEAREST)) > 127
    report = {'status': 'uncertain', 'applied': False, 'gains_rgb': [1., 1., 1.],
              'reason': '', 'method': 'background_illuminant_linear_rgb',
              'calibrated': False, 'candidate_fraction': 0., 'strength': strength}

    def unchanged(reason):
        report['reason'] = reason
        return image.convert('RGB').copy(), report

    high, low = rgb.max(axis=2), rgb.min(axis=2)
    saturation = (high - low) / np.maximum(high, 1e-6)
    # Do not use dark folds, saturated prints, clipped highlights, or the white
    # canvas added by the pipeline: this function only receives the original.
    candidates = (high > .42) & (high < .98) & (saturation < .40)
    report['candidate_fraction'] = round(float(candidates.mean()), 4)
    if not region.any():
        return unchanged('empty_object_mask')
    # Exclude a band around the product: matte errors must not let fabric colors
    # influence the scene illuminant. Work on the real scene, never the added white.
    radius = max(1, round(min(image.size) * .015))
    expanded = cv2.dilate(region.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    background_candidates = candidates & ~expanded
    report['background_candidate_fraction'] = round(float(background_candidates.mean()), 4)
    if np.count_nonzero(background_candidates) < 64:
        return unchanged('no_independent_scene_reference')
    linear = to_linear(rgb)
    chroma = linear / np.maximum(linear.mean(axis=2, keepdims=True), 1e-6)
    estimates = []
    positions = []
    rows = np.array_split(np.arange(rgb.shape[0]), 6)
    cols = np.array_split(np.arange(rgb.shape[1]), 6)
    for row, ys in enumerate(rows):
        for col, xs in enumerate(cols):
            chosen = background_candidates[np.ix_(ys, xs)]
            if chosen.sum() >= 16:
                samples = chroma[np.ix_(ys, xs)][chosen]
                estimate = np.median(samples, axis=0)
                # Reject tiles containing multiple incompatible candidate colors.
                spread = np.percentile(np.max(np.abs(np.log(samples / estimate)), axis=1), 90)
                if spread < .12:
                    estimates.append(estimate)
                    positions.append((row, col))
    if len(estimates) < 4 or len({p[0] for p in positions}) < 2 or len({p[1] for p in positions}) < 2:
        return unchanged('insufficient_spatial_support')
    estimates = np.asarray(estimates)
    illuminant = np.median(estimates, axis=0)
    disagreement = float(np.percentile(np.max(np.abs(np.log(estimates / illuminant)), axis=1), 90))
    report['reference_tiles'] = len(estimates)
    report['regional_disagreement'] = round(disagreement, 4)
    if disagreement > .16:
        return unchanged('mixed_light_or_conflicting_surface_colors')
    # Fabric is never forced to neutral or required to agree with the background.
    # This preserves cream, pastels and uniformly colored products.
    report['estimated_light_rgb'] = [round(float(v), 5) for v in illuminant]
    light_preview = np.clip(to_srgb(illuminant / illuminant.max()), 0, 1)
    report['estimated_light_srgb'] = [int(round(float(v) * 255)) for v in light_preview]
    report['assumption'] = 'Background candidates are neutral surfaces under the same light as the object; not calibrated.'
    gains = 1 / illuminant
    # Preserve illuminant luminance instead of boosting exposure to pure white.
    weights = np.array([.2126, .7152, .0722])
    # Modest extra chromatic correction; normalize exposure independently so
    # stronger white balance does not become an arbitrary brightness boost.
    gains = gains ** strength
    gains *= float(np.dot(illuminant, weights) / np.dot(illuminant * gains, weights))
    if gains.min() < .60 or gains.max() > 1.85:
        return unchanged('cast_too_strong_for_reliable_correction')
    if np.max(np.abs(gains - 1)) < .015:
        report.update(status='neutral', reason='no_significant_cast')
        return image.convert('RGB').copy(), report
    corrected = linear * gains
    if float(np.any(corrected[region] > 1, axis=1).mean()) > .01:
        return unchanged('correction_would_clip_garment_colors')
    result = Image.fromarray(np.uint8(np.rint(np.clip(to_srgb(corrected), 0, 1) * 255)))
    report.update(status='estimated', applied=True, reason='consistent_background_references',
                  gains_rgb=[round(float(x), 5) for x in gains])
    return result, report
