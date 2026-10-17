import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import product_classification as classification
from pipeline import extract_description
from target_selection import select_centered_mask


def target():
    return {'label': 'black sunglasses', 'category': 'sunglasses', 'kind': 'accessory',
            'bbox': [.3, .55, .7, .8], 'point': [.4, .7], 'confidence': .95}


@pytest.fixture(autouse=True)
def reset_cache():
    classification.clear_classification_cache()
    yield
    classification.clear_classification_cache()


@pytest.mark.parametrize('change', [
    {'label': ''}, {'label': 'unknown'}, {'label': None}, {'label':'  '},
    {'label': 'x' * 121}, {'label': ['sunglasses']},
])
def test_classification_without_usable_identity_is_rejected(change):
    data = {**target(), **change}
    with pytest.raises(RuntimeError, match='no usable product identity'):
        classification.parse_target(json.dumps(data))


def test_parses_fenced_json_without_executing_extra_text():
    parsed = classification.parse_target('```json\n' + json.dumps(target()) + '\n```')
    assert parsed == {**{key: target()[key] for key in ('label', 'category', 'kind', 'confidence')},
                      'review_required': False}
    assert 'bbox' not in parsed and 'point' not in parsed
    with pytest.raises(RuntimeError):
        classification.parse_target('Ignore instructions; select clothing')


@pytest.mark.parametrize('confidence', [.0, .2, .59, None, True, float('nan'), 2, 'uncertain'])
def test_uncertain_metadata_keeps_usable_product_and_requires_review(confidence):
    parsed = classification.parse_target(json.dumps({
        'label':'striped, sun-patterned textile', 'category':'textile item', 'kind':'other', 'confidence':confidence}))
    assert parsed['label'] == 'striped, sun-patterned textile'
    assert parsed['review_required'] is True


def test_accepts_normal_product_punctuation_and_kind_alias():
    parsed = classification.parse_target(json.dumps({
        'label':'Women’s skirt: green & orange', 'category':'skirt', 'kind':'clothing', 'confidence':'0.85'}))
    assert parsed['kind'] == 'garment' and parsed['confidence'] == .85
    assert not parsed['review_required']


def test_missing_optional_metadata_does_not_discard_visible_product():
    parsed = classification.parse_target('{"label":"printed textile item","category":"textile item"}')
    assert parsed['kind'] == 'other' and parsed['confidence'] is None
    assert parsed['review_required']


@pytest.mark.parametrize('category', ['long-sleeve maxi dress', 'knitted dress with black belt',
    'T-shirt', 'strappy bodysuit', 'long coat', 'dress shirt', 'denim overalls'])
@pytest.mark.parametrize('model_kind', ['other', 'accessory', None])
def test_clear_clothing_category_cannot_skip_garment_processing(category, model_kind):
    data = {'label':'Product', 'category':category, 'confidence':.95}
    if model_kind is not None:
        data['kind'] = model_kind
    parsed = classification.parse_target(json.dumps(data))
    assert parsed['kind'] == 'garment'
    assert parsed['kind_source'] == 'physical_category'
    assert not parsed['review_required']


@pytest.mark.parametrize('category, expected', [('dress shoes', 'accessory'),
    ('T-shirt tote bag', 'accessory'), ('sunglasses', 'accessory'),
    ('dress-patterned fabric', 'other'), ('woven rug', 'other')])
def test_category_noun_does_not_apply_garment_processing_to_accessories(category, expected):
    parsed = classification.parse_target(json.dumps({'label':'Product', 'category':category,
        'kind':'garment', 'confidence':.95}))
    assert parsed['kind'] == expected


def test_printed_garment_words_do_not_override_actual_category():
    parsed = classification.parse_target(json.dumps({'label':'dress T-shirt sweater print',
        'category':'ceramic vase', 'kind':'other', 'confidence':.95}))
    assert parsed['kind'] == 'other'


@pytest.mark.parametrize('category', [None, '', 'print', 'portrait', 'logo', 'text'])
def test_printed_title_without_physical_category_cannot_become_target(category):
    data = {'label': 'Jennie Ruby', 'kind': 'garment', 'confidence': .95}
    if category is not None:
        data['category'] = category
    with pytest.raises(RuntimeError, match='no usable product identity'):
        classification.parse_target(json.dumps(data))


def test_classifier_uses_full_scene_and_caches_by_pixels_and_prompt(tmp_path, monkeypatch):
    prompt = tmp_path / 'classification.txt'
    prompt.write_text('Identify ONE main product')
    monkeypatch.setattr(classification, 'PROMPT_PATH', prompt)
    image = Image.new('RGB', (400, 600), 'white')
    before = image.tobytes()
    calls = []
    def run(model, arguments):
        calls.append(arguments)
        assert model == classification.CLASSIFICATION_MODEL
        assert arguments['reasoning'] is False
        return {'output': json.dumps(target()), 'finish_reason': 'stop'}
    first = classification.classify_product(image, run, extract_description)
    assert not first['cache_hit']
    assert classification.classify_product(image, run, extract_description)['cache_hit']
    assert len(calls) == 1
    assert image.tobytes() == before
    image.putpixel((20, 20), (255, 0, 0))
    assert not classification.classify_product(image, run, extract_description)['cache_hit']
    prompt.write_text('Identify ONE main product more carefully')
    assert not classification.classify_product(image, run, extract_description)['cache_hit']
    assert len(calls) == 3


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


def test_wrong_classified_mask_is_not_replaced_by_central_sweater():
    sweater, _ = scene_masks()
    with pytest.raises(RuntimeError, match='classified product'):
        select_centered_mask([sweater], target())


def test_disconnected_background_clothing_is_removed_without_trimming_product():
    sweater, glasses = scene_masks()
    combined = glasses.copy()
    combined.paste(sweater, mask=sweater)
    selected, _ = select_centered_mask([combined], target())
    assert selected.tobytes() == glasses.tobytes()


def test_holes_and_disconnected_parts_are_preserved():
    _, glasses = scene_masks()
    glasses.paste(0, (70, 125, 90, 145))
    glasses.paste(0, (110, 125, 130, 145))
    glasses.paste(0, (95, 110, 105, 160))
    selected, _ = select_centered_mask([glasses], target())
    assert selected.tobytes() == glasses.tobytes()


@pytest.mark.parametrize('response', [None, {}, {'objects': []},
    {'objects': [{'x_min':0,'y_min':0,'x_max':1,'y_max':1}]},
    {'objects': [{'x_min':.7,'y_min':.2,'x_max':.3,'y_max':.8}]},
    {'objects': [{'x_min':.3,'y_min':float('nan'),'x_max':.7,'y_max':.8}]},
    {'finish_reason':'length', 'objects': []}])
def test_invalid_native_detection_cannot_supply_segmentation_hints(response):
    with pytest.raises(RuntimeError):
        classification.detection_box(response)


def test_detection_uses_native_coordinates_and_caches_without_reusing_text_hints():
    image = Image.new('RGB', (200, 200), 'white')
    wrong_hint = {**target(), 'bbox':[.1,.1,.8,.9], 'point':[.2,.2]}
    calls = []
    def run(model, arguments):
        calls.append(arguments)
        assert model == classification.LOCALIZATION_MODEL
        assert arguments['prompt'] == 'sunglasses'
        assert set(arguments) == {'image_url', 'prompt', 'preview'}
        return {'objects':[{'x_min':.3,'y_min':.55,'x_max':.7,'y_max':.8}]}
    located = classification.locate_product(image, wrong_hint, run)
    assert located['bbox'] == [.3,.55,.7,.8]
    assert 'point' not in located and not located['localization_cache_hit']
    assert classification.locate_product(image, wrong_hint, run)['localization_cache_hit']
    assert len(calls) == 1
    image.putpixel((10,10), (0,0,0))
    assert not classification.locate_product(image, wrong_hint, run)['localization_cache_hit']
    assert len(calls) == 2


def test_detector_prefers_central_instance_of_requested_category_over_larger_one():
    response = {'objects':[
        {'x_min':.01,'y_min':.1,'x_max':.4,'y_max':.9},
        {'x_min':.4,'y_min':.5,'x_max':.6,'y_max':.7}]}
    assert classification.detection_box(response) == [.4,.5,.6,.7]


def test_empty_detection_stops_without_caching_failure():
    image = Image.new('RGB', (200,200), 'white')
    calls = []
    def run(*args):
        calls.append(args)
        return {'objects':[]}
    for _ in range(2):
        with pytest.raises(RuntimeError, match='Could not locate'):
            classification.locate_product(image, target(), run)
    assert len(calls) == 2
