import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import generation_prompt


def test_pressing_template_is_reloaded_and_empty_template_rejected(tmp_path, monkeypatch):
    template = tmp_path / 'pressing.txt'
    monkeypatch.setattr(generation_prompt, 'PRESSING_TEMPLATE_PATH', template)
    template.write_text('First pressing instructions')
    assert generation_prompt.build_pressing_prompt() == 'First pressing instructions'
    template.write_text('Updated pressing instructions')
    assert generation_prompt.build_pressing_prompt() == 'Updated pressing instructions'
    template.write_text(' \n')
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


@pytest.mark.parametrize('content', ['Empty', '{{LIGHTING_INSTRUCTIONS}}',
    '{{GARMENT_DESCRIPTION}} {{GARMENT_DESCRIPTION}} {{LIGHTING_INSTRUCTIONS}}',
    '{{GARMENT_DESCRIPTION}} {{LIGHTING_INSTRUCTIONS}} {{UNKNOWN}}'])
def test_invalid_template_fails_before_generation(tmp_path, monkeypatch, content):
    template = tmp_path / 'generation.txt'
    template.write_text(content)
    monkeypatch.setattr(generation_prompt, 'TEMPLATE_PATH', template)
    with pytest.raises(ValueError, match='placeholder'):
        generation_prompt.build_generation_prompt('Shirt', {'status': 'neutral', 'applied': False})
