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


def preserve_dark_tones(source: Image.Image, generated: Image.Image, generated_mask: Image.Image | None):
    """Bound unwanted lifting of a consistently dark, nearly neutral product.

    No illuminant is guessed from the room and no color is forced to gray. The
    original's central material must agree across six of nine tiles, the generated
    foreground must also be predominantly dark/low-chroma, and hue estimates must
    agree. Only a common linear-RGB exposure factor is applied, fading out on bright
    details and matte edges. This is an uncalibrated guard, not true-color recovery.
    """
    report = {'method':'dark_material_exposure_guard', 'applied':False, 'calibrated':False,
              'gain_linear_rgb':1., 'reason':'', 'reference':'central_source_consensus'}
    def unchanged(reason):
        return generated, {**report, 'reason':reason}
    if generated_mask is None:
        return unchanged('no_generated_mask')
    original = source.convert('RGB').copy()
    original.thumbnail((512,512), Image.Resampling.LANCZOS)
    rgb = np.asarray(original, dtype=np.float32) / 255
    # A close-up is valid; otherwise this region avoids walls, hangers and hands.
    h,w = rgb.shape[:2]
    roi = rgb[int(h*.30):max(int(h*.80),int(h*.30)+1),
              int(w*.30):max(int(w*.70),int(w*.30)+1)]
    if min(roi.shape[:2]) < 9:
        return unchanged('insufficient_source_support')
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    candidates = (lab[...,0] > 3) & (lab[...,0] < 22) & (np.linalg.norm(lab[...,1:],axis=2) < 12)
    if candidates.mean() < .75:
        return unchanged('source_is_not_consistently_dark_neutral')
    tiles = []
    for ys in np.array_split(np.arange(roi.shape[0]),3):
        for xs in np.array_split(np.arange(roi.shape[1]),3):
            chosen = candidates[np.ix_(ys,xs)]
            if chosen.mean() >= .70:
                tiles.append(np.median(lab[np.ix_(ys,xs)][chosen],axis=0))
    if len(tiles) < 6:
        return unchanged('insufficient_spatial_agreement')
    source_lab = np.median(np.asarray(tiles),axis=0)
    if np.percentile(np.abs(np.asarray(tiles)[:,0]-source_lab[0]),90) > 4:
        return unchanged('inconsistent_source_exposure')
    preview = generated.convert('RGB').copy()
    preview.thumbnail((512,512),Image.Resampling.LANCZOS)
    sample = np.asarray(preview,dtype=np.float32) / 255
    alpha = np.asarray(generated_mask.convert('L').resize(preview.size,Image.Resampling.NEAREST))
    core = cv2.erode((alpha > 240).astype(np.uint8),np.ones((3,3),np.uint8)) > 0
    if np.count_nonzero(core) < 64:
        return unchanged('insufficient_generated_support')
    output_lab = cv2.cvtColor(sample,cv2.COLOR_RGB2LAB)
    dark = (output_lab[...,0] < 30) & (np.linalg.norm(output_lab[...,1:],axis=2) < 14)
    if dark[core].mean() < .80:
        return unchanged('generated_palette_is_not_dark_neutral')
    generated_lab = np.median(output_lab[core & dark],axis=0)
    report.update(source_lightness=round(float(source_lab[0]),3),
                  generated_lightness=round(float(generated_lab[0]),3), reference_tiles=len(tiles))
    if np.linalg.norm(source_lab[1:]-generated_lab[1:]) > 6:
        return unchanged('source_and_generated_chroma_disagree')
    if generated_lab[0]-source_lab[0] < 1.5:
        return unchanged('no_significant_dark_tone_lift')
    # Limit the adjustment to four L* units; retain the generated lighting and
    # texture rather than histogram-matching the source shadows and wrinkles.
    target_l = max(float(source_lab[0]),float(generated_lab[0])-4)
    def luminance(l):
        return ((l+16)/116)**3 if l > 8 else l/903.3
    gain = float(np.clip(luminance(target_l)/luminance(float(generated_lab[0])),.55,1.))
    full = np.asarray(generated.convert('RGB'),dtype=np.float32) / 255
    full_lab = cv2.cvtColor(full,cv2.COLOR_RGB2LAB)
    full_l = full_lab[...,0]
    # Smoothly preserve bright labels, prints and highlights exactly. Alpha zero
    # preserves every background pixel, including the studio contact shadow.
    fade = np.clip((45-full_l)/20,0,1)
    fade = fade*fade*(3-2*fade)
    chroma_fade = np.clip((18-np.linalg.norm(full_lab[...,1:],axis=2))/6,0,1)
    fade *= chroma_fade*chroma_fade*(3-2*chroma_fade)
    weight = np.asarray(generated_mask.convert('L').resize(generated.size,Image.Resampling.BILINEAR),
                        dtype=np.float32)/255 * fade
    factor = 1-weight*(1-gain)
    corrected = to_srgb(to_linear(full)*factor[...,None])
    result = Image.fromarray(np.uint8(np.rint(np.clip(corrected,0,1)*255)))
    report.update(applied=True, reason='bounded_dark_tone_lift', gain_linear_rgb=round(gain,5))
    return result, report


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
    # A few bright trims or white motifs must not disable white balance for the
    # entire product. Fit exposure uniformly, then compress only out-of-gamut
    # highlights with one scale per pixel, preserving linear channel ratios.
    # Never clip blue independently: that would leave warm whites yellow.
    peak = np.max(corrected, axis=2)
    exposure_scale = min(1., 1. / max(1., float(np.percentile(peak[region], 99.5))))
    if exposure_scale < .8:
        return unchanged('correction_requires_excessive_exposure_reduction')
    corrected *= exposure_scale
    corrected /= np.maximum(1., corrected.max(axis=2, keepdims=True))
    result = Image.fromarray(np.uint8(np.rint(np.clip(to_srgb(corrected), 0, 1) * 255)))
    report.update(status='estimated', applied=True, reason='consistent_background_references',
                  gains_rgb=[round(float(x), 5) for x in gains],
                  exposure_scale=round(exposure_scale, 5),
                  highlight_fraction=round(float((peak[region] > 1).mean()), 5))
    return result, report
