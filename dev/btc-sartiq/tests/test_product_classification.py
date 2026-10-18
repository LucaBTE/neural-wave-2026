import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import product_classification as classification
from target_selection import select_centered_mask


def target():
    return {'label': 'black sunglasses', 'category': 'sunglasses', 'kind': 'accessory',
            'bbox': [.3, .55, .7, .8], 'point': [.4, .7], 'confidence': .95}


@pytest.fixture(autouse=True)
def reset_cache():
    classification.clear_classification_cache()
    yield
    classification.clear_classification_cache()


def scene_masks():
    sweater = Image.new('L', (200, 200))
    sweater.paste(255, (10, 0, 190, 105))
    glasses = Image.new('L', (200, 200))
    glasses.paste(255, (60, 110, 140, 160))
    return sweater, glasses


def test_classified_glasses_win_over_larger_sweater_covering_center():
    sweater, glasses = scene_masks()
    assert sweater.getpixel((100, 100)) == 255
    assert glasses.getpixel((100, 100)) == 0
    for masks in ([sweater, glasses], [glasses, sweater]):
        selected, details = select_centered_mask(masks, target())
        assert selected.tobytes() == glasses.tobytes()
        assert details['strategy'] == 'classified_product'


def test_central_worn_sweatshirt_wins_over_distant_hanging_shirt():
    response = {'objects':[
        {'x_min':.12,'y_min':.18,'x_max':.27,'y_max':.40},
        {'x_min':.28,'y_min':.30,'x_max':.87,'y_max':.76},
    ]}
    assert classification.detection_box(response) == [.28,.30,.87,.76]
