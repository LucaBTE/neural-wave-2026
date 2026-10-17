import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import numpy as np
import pytest
from PIL import Image
from color import normalize_illumination, to_linear, to_srgb
from quality import compare_product


def scene():
    rgb = np.full((160, 160, 3), .70, dtype=np.float32)
    rgb[35:125, 40:120] = .72
    rgb[65:95, 60:100] = [.65, .12, .08]
    mask = Image.new('L', (160, 160)); mask.paste(255, (35, 35, 125, 125))
    return rgb, mask


def photo(rgb):
    return Image.fromarray(np.uint8(np.rint(np.clip(rgb, 0, 1) * 255)))


@pytest.mark.parametrize('cast', [[1.25, 1., .70], [.75, 1., 1.25], [.85, 1.2, .85], [1.12, .8, 1.12]])
def test_color_casts(cast):
    source, mask = scene()
    tinted = photo(to_srgb(to_linear(source) * cast))
    corrected, info = normalize_illumination(tinted, mask)
    assert info['applied'] and not info['calibrated']
    def chroma(im):
        pixels = to_linear(np.asarray(im) / 255.)
        return pixels / np.maximum(pixels.sum(axis=2, keepdims=True), 1e-6)
    expected = chroma(photo(source))
    assert np.mean(abs(chroma(corrected) - expected)) < np.mean(abs(chroma(tinted) - expected)) * .2


def test_neutral_unchanged():
    source, mask = scene()
    corrected, info = normalize_illumination(photo(source), mask)
    assert info['status'] == 'neutral'
    assert corrected.tobytes() == photo(source).tobytes()


def test_mixed_light_not_averaged():
    source, mask = scene()
    linear = to_linear(source)
    linear[:, :80] *= [1.25, 1., .7]
    linear[:, 80:] *= [.7, 1., 1.25]
    original = photo(to_srgb(linear))
    result, info = normalize_illumination(original, mask)
    assert info['reason'] == 'mixed_light_or_conflicting_surface_colors'
    assert result.tobytes() == original.tobytes()


def test_colored_fabric_not_whitened():
    source = Image.new('RGB', (160, 160), (180, 90, 40))
    result, info = normalize_illumination(source, Image.new('L', source.size, 255))
    assert info['status'] == 'uncertain'
    assert result.tobytes() == source.tobytes()


def test_cream_is_preserved_under_neutral_background_light():
    source, mask = scene()
    source[35:125, 35:125] = [.78, .70, .57]
    result, info = normalize_illumination(photo(source), mask)
    assert not info['applied']
    assert info['status'] == 'neutral'
    assert result.tobytes() == photo(source).tobytes()


def test_small_print_drift():
    image = Image.new('RGB', (200, 200), 'white')
    mask = Image.new('L', image.size); mask.paste(255, (20, 20, 180, 180))
    image.paste((200, 20, 20), (90, 90, 110, 110))
    result = image.copy(); result.paste((240, 110, 15), (90, 90, 110, 110))
    report = compare_product(image, mask, result, mask)
    assert report['metrics']['print_color_distance'] > .45
    assert not report['passed']


@pytest.mark.parametrize('fabric', [[.65,.12,.08], [.08,.12,.65], [.78,.70,.57], [.12,.55,.18]])
@pytest.mark.parametrize('cast', [[1.25,1.,.70], [.75,1.,1.25], [.85,1.2,.85], [1.12,.8,1.12]])
def test_background_corrects_colored_objects_without_neutral_fabric(fabric, cast):
    source, mask = scene()
    foreground = np.asarray(mask) > 127
    source[foreground] = fabric
    tinted = photo(to_srgb(to_linear(source) * cast))
    corrected, report = normalize_illumination(tinted, mask)
    assert report['applied']
    assert 'estimated_light_rgb' in report
    def chroma(im):
        p = to_linear(np.asarray(im) / 255.)[foreground]
        return p / p.sum(axis=1, keepdims=True)
    expected = chroma(photo(source))
    assert abs(chroma(corrected) - expected).mean() < abs(chroma(tinted) - expected).mean() * .25


def test_saturated_background_is_not_used_as_light_reference():
    source, mask = scene()
    source[np.asarray(mask) == 0] = [.8, .1, .1]
    result, report = normalize_illumination(photo(source), mask)
    assert not report['applied']
    assert report['reason'] == 'no_independent_scene_reference'
    assert result.tobytes() == photo(source).tobytes()


def test_bright_garment_does_not_disable_scene_white_balance(monkeypatch):
    monkeypatch.setenv('COLOR_CORRECTION_STRENGTH', '1.10')
    source, mask = scene()
    foreground = np.asarray(mask) > 127
    source[foreground] = [.6, .8, 1.]
    tinted = photo(to_srgb(to_linear(source) * [1.25, 1., .7]))
    before = tinted.tobytes()
    corrected, report = normalize_illumination(tinted, mask)
    assert report['applied']
    assert report['highlight_fraction'] > .01
    assert .8 <= report['exposure_scale'] < 1.
    def chroma(image):
        pixels = to_linear(np.asarray(image) / 255.)[foreground]
        return pixels / pixels.sum(axis=1, keepdims=True)
    error_before = np.abs(chroma(tinted) - chroma(photo(source))).mean()
    assert np.abs(chroma(corrected) - chroma(photo(source))).mean() < error_before * .2
    assert tinted.tobytes() == before


def test_correction_strength_is_adjustable_without_changing_input(monkeypatch):
    source, mask = scene()
    tinted = photo(to_srgb(to_linear(source) * [1.25, 1., .7]))
    before = tinted.tobytes()
    monkeypatch.setenv('COLOR_CORRECTION_STRENGTH', '1.0')
    _, baseline = normalize_illumination(tinted, mask)
    monkeypatch.setenv('COLOR_CORRECTION_STRENGTH', '1.05')
    _, stronger = normalize_illumination(tinted, mask)
    assert baseline['applied'] and stronger['applied']
    assert stronger['strength'] == 1.05
    assert stronger['gains_rgb'][2] / stronger['gains_rgb'][0] > baseline['gains_rgb'][2] / baseline['gains_rgb'][0]
    assert tinted.tobytes() == before


@pytest.mark.parametrize('strength', ['nan', 'inf', '0', '2'])
def test_invalid_correction_strength_rejected(monkeypatch, strength):
    monkeypatch.setenv('COLOR_CORRECTION_STRENGTH', strength)
    source, mask = scene()
    with pytest.raises(ValueError, match='COLOR_CORRECTION_STRENGTH'):
        normalize_illumination(photo(source), mask)
