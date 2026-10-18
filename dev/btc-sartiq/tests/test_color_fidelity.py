"""Reference-free production tests with synthetic, known material colors."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from color import restore_color_fidelity


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


@pytest.mark.parametrize('color', [(150, 70, 40)])
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
