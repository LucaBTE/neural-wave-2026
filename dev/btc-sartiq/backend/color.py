"""Color-managed decoding and bounded source-based color fidelity refinement.

The faithful profile uses known-neutral generated studio context and foreground
material statistics; it does not estimate white balance from the original room.
No correction reads evaluation targets or claims calibrated color recovery.
"""
import io
import cv2
import numpy as np
from PIL import Image, ImageCms


def normalize_color_profile(image):
    """Transform embedded camera/display profiles to sRGB before numeric RGB work."""
    profile = image.info.get('icc_profile')
    if profile:
        try:
            image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile('sRGB'), outputMode='RGB',
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
        except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
            image = image.convert('RGB')
    else:
        image = image.convert('RGB')
    # Pixels are now sRGB; never forward a stale Display P3/Adobe RGB tag.
    image.info.pop('icc_profile', None)
    return image


def _sample(image, mask=None):
    preview = image.convert('RGB').copy()
    preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
    rgb = np.asarray(preview, dtype=np.float32) / 255
    if mask is None:
        return rgb, None
    alpha = np.asarray(mask.convert('L').resize(preview.size, Image.Resampling.NEAREST)) / 255
    return rgb, alpha


def _studio_cast(image, mask):
    """Use only the explicitly requested neutral studio backdrop, never the room.

    Spatial agreement rejects colored props, mixed lighting and clipped whites.
    This cannot establish the illuminant of the original photograph.
    """
    report = {'applied': False, 'reason': 'no_generated_mask', 'gains_linear_rgb': [1., 1., 1.]}
    if mask is None:
        return image, report
    rgb, alpha = _sample(image, mask)
    # Exclude the product and its boundary, even when it is pale or thin.
    excluded = cv2.dilate((alpha > .05).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    candidates = (~excluded) & (rgb.min(axis=2) > .82) & (rgb.max(axis=2) < .985)
    candidates &= np.ptp(rgb, axis=2) < .06
    tiles, positions = [], []
    for row, ys in enumerate(np.array_split(np.arange(len(rgb)), 4)):
        for col, xs in enumerate(np.array_split(np.arange(rgb.shape[1]), 4)):
            if row not in (0, 3) and col not in (0, 3):
                continue
            pixels = rgb[np.ix_(ys, xs)][candidates[np.ix_(ys, xs)]]
            if len(pixels) < 64:
                continue
            chroma = np.log(to_linear(pixels))
            chroma -= chroma.mean(axis=1, keepdims=True)
            estimate = np.median(chroma, axis=0)
            if np.percentile(np.max(abs(chroma - estimate), axis=1), 90) < .025:
                tiles.append(estimate)
                positions.append((row, col))
    report.update(reference_tiles=len(tiles))
    if len(tiles) < 4 or len({p[0] for p in positions}) < 2 or len({p[1] for p in positions}) < 2:
        return image, {**report, 'reason': 'insufficient_neutral_backdrop'}
    estimate = np.median(tiles, axis=0)
    if np.max(np.abs(np.asarray(tiles) - estimate)) > .025:
        return image, {**report, 'reason': 'inconsistent_backdrop'}
    light = np.exp(estimate)
    gains = np.dot(light, [.2126, .7152, .0722]) / light
    if np.max(abs(gains - 1)) < .008:
        return image, {**report, 'reason': 'neutral_backdrop'}
    if gains.min() < .9 or gains.max() > 1.1:
        return image, {**report, 'reason': 'ambiguous_backdrop_cast'}
    full = to_linear(np.asarray(image.convert('RGB'), dtype=np.float32) / 255) * gains
    # Compress highlights with one common per-pixel factor, never clip channels separately.
    full /= np.maximum(1, full.max(axis=2, keepdims=True))
    result = Image.fromarray(np.uint8(np.rint(np.clip(to_srgb(full), 0, 1) * 255)))
    return result, {**report, 'applied': True, 'reason': 'consistent_neutral_studio_backdrop',
                    'gains_linear_rgb': [round(float(x), 5) for x in gains]}


def _material_anchor(source, generated, source_mask, generated_mask):
    """Bound generative tone/chroma drift using source foreground statistics only.

    A robust log-RGB exposure centre reduces sensitivity to sharp source folds;
    no source texture or geometry is copied. Bounded Lab offsets retain generated
    texture. Low-chroma hue is unreliable, so preserve it and adjust chroma radially.
    This is a source-fidelity constraint, not calibrated reflectance recovery.
    """
    report = {'applied': False, 'reason': 'missing_foreground_mask'}
    if source_mask is None or generated_mask is None:
        return generated, report
    samples, labs = [], []
    for name, image, mask in [('source', source, source_mask), ('generated', generated, generated_mask)]:
        rgb, alpha = _sample(image, mask)
        core = cv2.erode((alpha > .94).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ys, xs = np.where(core)
        # A thin printed strip or tiny fragment cannot establish whole-product color.
        if len(xs) < 128 or core.mean() > .92:
            return generated, {**report, 'reason': name + '_mask_insufficient'}
        if (np.ptp(xs) / rgb.shape[1] < .12 or np.ptp(ys) / rgb.shape[0] < .12):
            return generated, {**report, 'reason': name + '_mask_fragment'}
        pixels = rgb[core]
        samples.append(pixels)
        labs.append(cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0])
    src, dst = [np.median(lab, axis=0) for lab in labs]
    report.update(source_median_lab=[round(float(v), 3) for v in src],
                  generated_median_lab=[round(float(v), 3) for v in dst])
    # A bright neutral material may be white under a cast, cream, or a clipped
    # exposure. Do not infer its reflectance from the source's apparent yellowness.
    if min(src[0], dst[0]) > 72:
        return generated, {**report, 'reason': 'bright_material_illumination_ambiguous'}
    if np.linalg.norm(src[1:] - dst[1:]) > 12:
        return generated, {**report, 'reason': 'source_lighting_or_palette_disagreement'}
    # Reject incompatible multi-material distributions (e.g. front/back or a
    # partial mask). Compare shading-normalized chroma, independent of pose.
    features = [lab[:, 1:] / np.maximum(lab[:, :1] + 16, 24) * 60 for lab in labs]
    spreads = [np.percentile(f, [10, 50, 90], axis=0) for f in features]
    disagreement = float(np.mean(abs((spreads[0] - spreads[0][1]) -
                                     (spreads[1] - spreads[1][1]))))
    report['palette_disagreement'] = round(disagreement, 3)
    if disagreement > 10:
        return generated, {**report, 'reason': 'incompatible_material_distribution'}

    def exposure_center(pixels):
        linear = to_linear(pixels)
        lo, hi = np.percentile(linear, [5, 95], axis=0)
        center = np.exp(np.log(np.clip(linear, lo + 1e-6, hi + 1e-6)).mean(axis=0))
        return float(cv2.cvtColor(to_srgb(center).astype(np.float32).reshape(1, 1, 3),
                                 cv2.COLOR_RGB2LAB)[0, 0, 0])

    source_l, generated_l = [exposure_center(pixels) for pixels in samples]
    # Never lift blacks or amplify a source's shadows/exposure range. Restore
    # only half the excess lightness, capped at four L* units.
    delta_l = float(np.clip((source_l - generated_l) * .5, -4, 0))
    delta_ab = (src[1:] - dst[1:]) * .5
    delta_ab *= min(1., 4. / max(float(np.linalg.norm(delta_ab)), 1e-6))
    source_c, generated_c = [float(np.median(np.linalg.norm(lab[:, 1:], axis=1))) for lab in labs]
    low_chroma = min(source_c, generated_c) < 15
    chroma_gain = float(np.clip(1 + .5 * (source_c / max(generated_c, 1) - 1), .8, 1.2))
    if low_chroma:
        delta_ab[:] = 0
    else:
        # Large hue changes can be a different illuminant, not generator drift.
        hue_cos = np.dot(src[1:], dst[1:]) / max(np.linalg.norm(src[1:]) * np.linalg.norm(dst[1:]), 1e-6)
        if hue_cos < np.cos(np.deg2rad(15)):
            return generated, {**report, 'reason': 'source_hue_illumination_ambiguous'}
        chroma_gain = 1.
    report.update(lightness_offset=round(delta_l, 4), chroma_offset=[round(float(v), 4) for v in delta_ab],
                  chroma_gain=round(chroma_gain, 5), source_exposure_center=round(source_l, 3),
                  generated_exposure_center=round(generated_l, 3))
    if abs(delta_l) < .25 and np.linalg.norm(delta_ab) < .35 and abs(chroma_gain - 1) < .02:
        return generated, {**report, 'reason': 'no_significant_material_drift'}
    rgb = np.asarray(generated.convert('RGB'), dtype=np.float32) / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    corrected = lab.copy()
    corrected[..., 0] = np.maximum(0, corrected[..., 0] + delta_l)
    corrected[..., 1:] = corrected[..., 1:] * chroma_gain + delta_ab
    # Preserve neutral labels, pale hardware and highlights; a bulk material
    # estimate must not recolor these. It is not spatial histogram matching.
    protection = np.clip((85 - lab[..., 0]) / 15, 0, 1)
    protection *= protection * (3 - 2 * protection)
    alpha = np.asarray(generated_mask.convert('L').resize(generated.size, Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255
    weight = alpha * protection
    result_rgb = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
    result = Image.fromarray(np.uint8(np.rint(np.clip(rgb + weight[..., None] * (result_rgb - rgb), 0, 1) * 255)))
    return result, {**report, 'applied': True, 'reason': 'bounded_foreground_material_drift'}


def _central_material_reference(source, generated, generated_mask):
    """Find color samples when semantic segmentation failed, not a new object mask.

    A broad central patch must have consistent material color in at least six of
    nine tiles. The existing material anchor still checks agreement with the
    generated palette. Bright fabrics cannot establish a white reference this way.
    Sampling is bounded and never changes the generation/cropping selection.
    """
    report = {'accepted': False, 'reason': 'no_generated_mask',
              'method': 'central_material_consensus', 'is_segmentation': False}
    if generated_mask is None:
        return None, report
    rgb, _ = _sample(source)
    height, width = rgb.shape[:2]
    top, bottom, left, right = int(height * .30), int(height * .80), int(width * .30), int(width * .70)
    region = rgb[top:bottom, left:right]
    if min(region.shape[:2]) < 12:
        return None, {**report, 'reason': 'insufficient_source_support'}
    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
    center = np.median(lab.reshape(-1, 3), axis=0)
    if not 8 < center[0] < 72:
        return None, {**report, 'reason': 'ambiguous_exposure_or_bright_material'}
    accepted = []
    for row, ys in enumerate(np.array_split(np.arange(len(lab)), 3)):
        for col, xs in enumerate(np.array_split(np.arange(lab.shape[1]), 3)):
            tile = np.median(lab[np.ix_(ys, xs)].reshape(-1, 3), axis=0)
            if abs(float(tile[0] - center[0])) <= 18 and np.linalg.norm(tile[1:] - center[1:]) <= 8:
                accepted.append((row, col))
    report['reference_tiles'] = len(accepted)
    if (len(accepted) < 6 or len({r for r, _ in accepted}) < 3 or
            len({c for _, c in accepted}) < 3):
        return None, {**report, 'reason': 'inconsistent_central_material'}
    # Keep both colors of stripes/patterns within agreeing tiles. Exclude entire
    # inconsistent tiles instead of choosing pixels by the desired output color.
    support = np.zeros((height, width), dtype=np.uint8)
    ys_parts = np.array_split(np.arange(top, bottom), 3)
    xs_parts = np.array_split(np.arange(left, right), 3)
    for row, col in accepted:
        support[np.ix_(ys_parts[row], xs_parts[col])] = 255
    return Image.fromarray(support), {**report, 'accepted': True,
                                      'reason': 'spatial_material_consensus'}


def restore_color_fidelity(source, generated, source_mask=None, generated_mask=None):
    """Local color refinement. No targets, captions, model calls or new dependency."""
    balanced, backdrop = _studio_cast(generated, generated_mask)
    result, material = _material_anchor(source, balanced, source_mask, generated_mask)
    dark = {'applied': False, 'reason': 'material_anchor_available'}
    support_report = {'accepted': False, 'reason': 'not_needed'}
    fallback_material = {'applied': False, 'reason': 'not_needed'}
    if material['reason'] in ('missing_foreground_mask', 'source_mask_insufficient', 'source_mask_fragment'):
        # Preserve the existing conservative fallback when SAM is unavailable.
        result, dark = preserve_dark_tones(source, balanced, generated_mask)
        if not dark['applied']:
            support, support_report = _central_material_reference(source, balanced, generated_mask)
            if support is not None:
                result, fallback_material = _material_anchor(source, balanced, support, generated_mask)
    return result, {'method': 'studio_neutrality_and_foreground_material_anchor', 'calibrated': False,
                    'applied': backdrop['applied'] or material['applied'] or dark['applied'] or fallback_material['applied'],
                    'backdrop': backdrop, 'material': material, 'dark_fallback': dark,
                    'source_support': support_report, 'fallback_material': fallback_material}


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
    # A central flash/window hotspot is not representative of the whole dark
    # fabric. Once the central material is independently confirmed, sample its
    # contiguous surrounding support too. Walls/hangers and disconnected dark
    # furniture cannot enter this estimate. These are color samples, not a
    # substitute for a SAM product mask or an inferred product silhouette.
    wider = cv2.cvtColor(rgb[int(h*.20):max(int(h*.85),int(h*.20)+1),
                            int(w*.15):max(int(w*.85),int(w*.15)+1)], cv2.COLOR_RGB2LAB)
    compatible = ((wider[..., 0] > 3) & (wider[..., 0] < 22) &
                  (np.linalg.norm(wider[..., 1:] - source_lab[1:], axis=2) < 3))
    if compatible.mean() >= .70:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(compatible.astype(np.uint8), 8)
        if count > 1:
            main = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            support = labels == main
            central_support = support[int(h*.30)-int(h*.20):int(h*.80)-int(h*.20),
                                      int(w*.30)-int(w*.15):int(w*.70)-int(w*.15)]
            if support.sum() >= compatible.sum() * .90 and central_support.mean() >= .70:
                surrounding = np.median(wider[support], axis=0)
                if abs(float(surrounding[0] - source_lab[0])) <= 4:
                    report.update(central_lightness=round(float(source_lab[0]), 3),
                                  reference='central_consensus_and_connected_material')
                    source_lab = surrounding
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
