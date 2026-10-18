"""Reference-free production tests with synthetic, known material colors."""
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from color import restore_color_fidelity, to_linear, to_srgb
from pipeline import normalize_photo


def scene(color=(110, 65, 40), background=(240, 240, 240)):
    image = Image.new('RGB', (180, 240), background)
    mask = Image.new('L', image.size)
    image.paste(color, (40, 40, 140, 200))
    mask.paste(255, (40, 40, 140, 200))
    return image, mask


def lab(image):
    return cv2.cvtColor(np.asarray(image, dtype=np.float32) / 255, cv2.COLOR_RGB2LAB)


def shifted(image, mask, delta):
    rgb = np.asarray(image).copy()
    changed = cv2.cvtColor(lab(image) + np.array(delta, dtype=np.float32), cv2.COLOR_LAB2RGB)
    rgb[np.asarray(mask) > 0] = np.uint8(np.rint(np.clip(changed, 0, 1) * 255))[np.asarray(mask) > 0]
    return Image.fromarray(rgb)


@pytest.mark.parametrize('color', [(150, 70, 40), (35, 75, 140), (120, 70, 45), (60, 85, 70)])
def test_restores_material_drift_without_repainting_scene_or_labels(color):
    source, mask = scene(color)
    generated = shifted(source, mask, (5, 2, 2))
    # A white label and small pale details must retain their generated RGB.
    generated.paste((250, 250, 248), (65, 65, 95, 75))
    result, report = restore_color_fidelity(source, generated, mask, mask)
    assert report['material']['applied']
    region = np.zeros((240, 180), bool)
    region[90:180, 50:130] = True
    before = np.linalg.norm(lab(generated)[region] - lab(source)[region], axis=1).mean()
    after = np.linalg.norm(lab(result)[region] - lab(source)[region], axis=1).mean()
    assert after < before * .8
    assert np.array_equal(np.asarray(result)[np.asarray(mask) == 0], np.asarray(generated)[np.asarray(mask) == 0])
    assert result.getpixel((80, 70)) == generated.getpixel((80, 70))
    assert mask.getbbox() == (40, 40, 140, 200)


@pytest.mark.parametrize('cast', [[1.04, 1, .94], [.94, 1, 1.04], [.97, 1.03, .97], [1.02, .97, 1.02]])
def test_neutral_studio_reference_removes_different_casts_across_fabric_and_print(cast):
    expected, mask = scene((170, 45, 40))
    expected.paste((215, 204, 180), (60, 60, 120, 90))  # Genuine cream print.
    expected.paste((30, 30, 32), (60, 100, 120, 130))
    tinted = Image.fromarray(np.uint8(np.rint(np.clip(
        to_srgb(to_linear(np.asarray(expected) / 255) * cast), 0, 1) * 255)))
    result, report = restore_color_fidelity(expected, tinted, None, mask)
    assert report['backdrop']['applied']
    def chroma(im):
        rgb = to_linear(np.asarray(im) / 255)
        return rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 1e-6)
    roi = np.asarray(mask) > 0
    before = abs(chroma(tinted)[roi] - chroma(expected)[roi]).mean()
    after = abs(chroma(result)[roi] - chroma(expected)[roi]).mean()
    # 8-bit quantization of dark fabric limits ratio precision after two encodes.
    assert after < before * .5
    assert result.getpixel((80, 75))[0] > result.getpixel((80, 75))[2]  # Cream stays cream.


def test_correction_preserves_texture_and_does_not_copy_source_folds():
    source, mask = scene()
    generated = np.asarray(shifted(source, mask, (5, 2, 2))).copy()
    # Output weave is distinct from the source. It must survive the color edit.
    for y in range(40, 200):
        generated[y, 40:140] += y % 3
    generated = Image.fromarray(generated)
    result, report = restore_color_fidelity(source, generated, mask, mask)
    assert report['applied']
    before, after = lab(generated)[80:180, 60:120, 0], lab(result)[80:180, 60:120, 0]
    assert np.corrcoef(before.ravel(), after.ravel())[0, 1] > .99
    assert after.std() > before.std() * .85


def test_cream_and_ambiguous_warm_whites_are_not_forced_to_an_invented_color():
    source, mask = scene((242, 214, 175), (215, 190, 150))
    generated, _ = scene((234, 225, 204))
    result, report = restore_color_fidelity(source, generated, mask, mask)
    assert report['material']['reason'] == 'bright_material_illumination_ambiguous'
    assert result.tobytes() == generated.tobytes()


def test_strong_source_cast_is_not_copied_back_into_red_print():
    source, mask = scene((225, 77, 10))
    generated, _ = scene((237, 99, 85))
    result, report = restore_color_fidelity(source, generated, mask, mask)
    assert report['material']['reason'] == 'source_lighting_or_palette_disagreement'
    assert result.tobytes() == generated.tobytes()


def test_mixed_studio_light_is_not_averaged_into_global_white_balance():
    source, mask = scene((150, 65, 35))
    generated = np.asarray(source).copy()
    for xs, cast in [(slice(0, 90), [1.04, 1, .94]), (slice(90, 180), [.94, 1, 1.04])]:
        generated[:, xs] = np.uint8(np.rint(np.clip(to_srgb(to_linear(generated[:, xs] / 255) * cast), 0, 1) * 255))
    generated = Image.fromarray(generated)
    result, report = restore_color_fidelity(source, generated, None, mask)
    assert not report['backdrop']['applied']
    assert result.tobytes() == generated.tobytes()


@pytest.mark.parametrize('mask_type', ['empty', 'fragment', 'missing'])
def test_bad_source_masks_do_not_establish_whole_product_color(mask_type):
    source, mask = scene((225, 77, 10))
    generated, _ = scene((237, 99, 85))
    source_mask = Image.new('L', mask.size)
    if mask_type == 'fragment':
        source_mask.paste(255, (40, 100, 140, 110))
    elif mask_type == 'missing':
        source_mask = None
    result, report = restore_color_fidelity(source, generated, source_mask, mask)
    assert not report['material']['applied']
    assert not report['applied']
    assert result.tobytes() == generated.tobytes()


def test_neutral_output_does_not_use_a_beige_original_room_as_white_reference():
    source, mask = scene((60, 80, 70), (215, 190, 150))
    generated, _ = scene((60, 80, 70))
    result, report = restore_color_fidelity(source, generated, mask, mask)
    assert not report['applied']
    assert result.tobytes() == generated.tobytes()


def test_dark_fallback_uses_connected_material_instead_of_a_central_light_hotspot():
    source = Image.new('RGB', (400, 500), (180, 180, 180))
    source.paste((25, 26, 29), (50, 80, 350, 430))
    source.paste((31, 32, 35), (120, 150, 280, 400))
    generated = Image.new('RGB', source.size, (240, 240, 240))
    generated.paste((34, 35, 39), (50, 80, 350, 430))
    mask = Image.new('L', source.size)
    mask.paste(255, (50, 80, 350, 430))
    result, report = restore_color_fidelity(source, generated, None, mask)
    dark = report['dark_fallback']
    assert dark['applied']
    assert dark['reference'] == 'central_consensus_and_connected_material'
    assert dark['source_lightness'] < dark['central_lightness']
    assert result.getpixel((200, 250))[0] < generated.getpixel((200, 250))[0]
    assert result.getpixel((0, 0)) == generated.getpixel((0, 0))


def test_icc_profile_is_transformed_before_pixel_arithmetic():
    # A real non-RGB embedded profile verifies actual LittleCMS conversion,
    # rather than merely checking whether an ICC tag was removed.
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB'))
    source = Image.new('LAB', (300, 300), (140, 155, 145))
    encoded = io.BytesIO()
    source.save(encoded, format='TIFF', icc_profile=profile.tobytes())
    expected = ImageCms.profileToProfile(source, profile, ImageCms.createProfile('sRGB'),
        outputMode='RGB', renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    result = normalize_photo(encoded.getvalue())
    assert result.tobytes() == expected.tobytes()
    assert 'icc_profile' not in result.info


def test_invalid_profile_keeps_photo_available():
    source, _ = scene()
    encoded = io.BytesIO()
    source.save(encoded, format='PNG', icc_profile=b'invalid ICC')
    result = normalize_photo(encoded.getvalue())
    assert result.tobytes() == source.tobytes()
    assert 'icc_profile' not in result.info
