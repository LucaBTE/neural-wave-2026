import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import generation_prompt


def test_pressing_template_is_reloaded_and_empty_instructions_rejected(tmp_path, monkeypatch):
    template = tmp_path / 'pressing.txt'
    monkeypatch.setattr(generation_prompt, 'PRESSING_TEMPLATE_PATH', template)
    template.write_text('Press the garment.')
    assert generation_prompt.build_pressing_prompt() == 'Press the garment.'
    template.write_text('Updated instructions.')
    assert generation_prompt.build_pressing_prompt() == 'Updated instructions.'
    template.write_text(' ')
    with pytest.raises(ValueError, match='must not be empty'):
        generation_prompt.build_pressing_prompt()



def test_template_changes_are_loaded_for_each_job(tmp_path, monkeypatch):
    template = tmp_path / 'generation.txt'
    monkeypatch.setattr(generation_prompt, 'TEMPLATE_PATH', template)
    template.write_text('First {{LIGHTING_INSTRUCTIONS}} {{GARMENT_DESCRIPTION}}')
    lighting = {'status': 'neutral', 'applied': False}
    assert generation_prompt.build_generation_prompt('Oversized shirt', lighting).startswith('First')
    template.write_text('Updated {{LIGHTING_INSTRUCTIONS}} {{GARMENT_DESCRIPTION}}')
    assert generation_prompt.build_generation_prompt('Oversized shirt', lighting).startswith('Updated')


def test_description_is_inserted_once_as_literal_data():
    description = 'Red shirt with "text" and {{LIGHTING_INSTRUCTIONS}} on the print.'
    prompt = generation_prompt.build_generation_prompt(description, {'status': 'uncertain', 'applied': False})
    assert '{{LIGHTING_INSTRUCTIONS}} on the print.' in prompt
    assert '\\"text\\"' in prompt
    assert 'No reliable global correction' in prompt


def test_color_guesses_are_removed_from_auxiliary_caption_and_identity_only():
    prompt = generation_prompt.build_generation_prompt(
        'A dark blue knitted sweater with ribbed cuffs and the text "RED WHITE BLUE".',
        {'status':'deferred', 'applied':False},
        {'label':'dark blue sweater','category':'navy sweater','kind':'garment'})
    assert 'dark blue' not in prompt and 'navy sweater' not in prompt
    assert 'knitted sweater with ribbed cuffs' in prompt
    assert '\\"RED WHITE BLUE\\"' in prompt  # artwork is never rewritten
    assert 'photographs alone establish colors' in prompt


@pytest.mark.parametrize('caption,expected', [
    ('olive green pants with floral embroidery','pants with floral embroidery'),
    ("white tee, printed 'Black Sabbath' in red", "tee, printed 'Black Sabbath' in"),
    ('a cream ribbed sweater','a ribbed sweater'),
    ('Women’s navy jacket with metal buttons','Women’s jacket with metal buttons')])
def test_construction_observations_preserve_text_and_parts(caption, expected):
    assert generation_prompt.construction_observations(caption) == expected


@pytest.mark.parametrize('content', ['Empty', '{{LIGHTING_INSTRUCTIONS}}',
    '{{GARMENT_DESCRIPTION}} {{GARMENT_DESCRIPTION}} {{LIGHTING_INSTRUCTIONS}}',
    '{{GARMENT_DESCRIPTION}} {{LIGHTING_INSTRUCTIONS}} {{UNKNOWN}}'])
def test_invalid_template_fails_before_generation(tmp_path, monkeypatch, content):
    template = tmp_path / 'generation.txt'
    template.write_text(content)
    monkeypatch.setattr(generation_prompt, 'TEMPLATE_PATH', template)
    with pytest.raises(ValueError, match='placeholder'):
        generation_prompt.build_generation_prompt('Shirt', {'status': 'neutral', 'applied': False})


@pytest.mark.parametrize('extras',[0,1,4])
def test_single_product_rule_never_requires_a_nonexistent_second_reference(extras):
    prompt = generation_prompt.build_generation_prompt('Printed T-shirt',
        {'status':'deferred','applied':False},additional_count=extras)
    assert 'Generate exactly ONE physical garment instance' in prompt
    assert 'second plain' in prompt
    if extras:
        assert f'INPUT: {1+extras} reference images for ONE product.' in prompt
        assert 'Images 2 onward' in prompt
    else:
        assert 'Image 2' not in prompt
