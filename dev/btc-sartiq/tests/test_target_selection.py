import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import pipeline
from target_selection import select_centered_mask


def masks():
    sleeve = Image.new('L', (200,200))
    sleeve.paste(255, (0,0,90,30))
    garment = Image.new('L', (200,200))
    garment.paste(255, (75,35,135,185))
    return sleeve, garment


def test_selects_centered_product_regardless_of_candidate_order():
    sleeve, garment = masks()
    for candidates in ([sleeve, garment], [garment, sleeve]):
        selected, metadata = select_centered_mask(candidates)
        assert selected.tobytes() == garment.tobytes()
        assert metadata['bbox'] == [75,35,135,185]
        assert metadata['strategy'] == 'central_product'


def test_removes_detached_corner_sleeve_from_combined_mask():
    sleeve, garment = masks()
    combined = garment.copy()
    combined.paste(sleeve, mask=sleeve)
    selected, _ = select_centered_mask([combined])
    assert selected.tobytes() == garment.tobytes()


def test_keeps_complete_wide_sleeves_of_centered_garment():
    garment = Image.new('L', (200,200))
    garment.paste(255, (75,30,125,180))
    garment.paste(255, (0,45,200,75))
    corner = Image.new('L', garment.size)
    corner.paste(255, (0,0,55,20))
    combined = garment.copy(); combined.paste(corner, mask=corner)
    selected, _ = select_centered_mask([combined])
    assert selected.tobytes() == garment.tobytes()


def test_clean_central_candidate_wins_over_connected_distractor():
    sleeve, garment = masks()
    combined = garment.copy()
    combined.paste(sleeve, mask=sleeve)
    combined.paste(255, (75,20,85,45))  # a bridge makes this one component
    selected, metadata = select_centered_mask([combined, garment])
    assert selected.tobytes() == garment.tobytes()
    assert metadata['candidate_index'] == 1


def test_target_does_not_need_to_cover_exact_center_pixel():
    trousers = Image.new('L', (200,200))
    trousers.paste(255, (60,30,140,60))
    trousers.paste(255, (60,60,85,190))
    trousers.paste(255, (115,60,140,190))
    assert trousers.getpixel((100,100)) == 0
    selected, _ = select_centered_mask([trousers])
    assert selected.tobytes() == trousers.tobytes()


@pytest.mark.parametrize('kind', ['peripheral', 'empty', 'full', 'bounding_box_only'])
def test_rejects_masks_without_actual_central_product_pixels(kind):
    sleeve, _ = masks()
    candidate = sleeve
    if kind == 'empty':
        candidate = Image.new('L', sleeve.size)
    elif kind == 'full':
        candidate = Image.new('L', sleeve.size, 255)
    elif kind == 'bounding_box_only':
        candidate = Image.new('L', sleeve.size)
        candidate.paste(255, (0,0,200,15))
        candidate.paste(255, (0,185,200,200))
        candidate.paste(255, (0,0,15,200))
        candidate.paste(255, (185,0,200,200))
    with pytest.raises(RuntimeError, match='image center'):
        select_centered_mask([candidate])


def test_sam_gets_center_hint_and_does_not_union_masks(monkeypatch):
    sleeve, garment = masks()
    calls = []
    def call(model, arguments):
        calls.append(arguments)
        return {'masks':[{'url':'https://sleeve'}, {'url':'https://product'}]}
    monkeypatch.setattr(pipeline, 'call', call)
    monkeypatch.setattr(pipeline, 'fetch_image', lambda url: sleeve if url.endswith('sleeve') else garment)
    selected = pipeline.garment_mask_from_sam(Image.new('RGB', garment.size), 'https://input', 'clothing')
    assert len(calls) == 1
    assert calls[0]['point_prompts'] == [{'x':100, 'y':100, 'label':1}]
    assert calls[0]['return_multiple_masks']
    assert calls[0]['max_masks'] == 3
    assert selected.tobytes() == garment.tobytes()
    assert selected.info['target_selection']['candidate_index'] == 1
