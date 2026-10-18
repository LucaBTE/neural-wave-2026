import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import generation_prompt


def test_template_changes_are_loaded_for_each_job(tmp_path, monkeypatch):
    template = tmp_path / 'generation.txt'
    monkeypatch.setattr(generation_prompt, 'TEMPLATE_PATH', template)
    template.write_text('First {{LIGHTING_INSTRUCTIONS}} {{GARMENT_DESCRIPTION}}')
    lighting = {'status': 'neutral', 'applied': False}
    assert generation_prompt.build_generation_prompt('Oversized shirt', lighting).startswith('First')
    template.write_text('Updated {{LIGHTING_INSTRUCTIONS}} {{GARMENT_DESCRIPTION}}')
    assert generation_prompt.build_generation_prompt('Oversized shirt', lighting).startswith('Updated')


def test_color_guesses_are_removed_from_auxiliary_caption_and_identity_only():
    prompt = generation_prompt.build_generation_prompt(
        'A dark blue knitted sweater with ribbed cuffs and the text "RED WHITE BLUE".',
        {'status':'deferred', 'applied':False},
        {'label':'dark blue sweater','category':'navy sweater','kind':'garment'})
    assert 'dark blue' not in prompt and 'navy sweater' not in prompt
    assert 'knitted sweater with ribbed cuffs' in prompt
    assert '\\"RED WHITE BLUE\\"' in prompt  # artwork is never rewritten
    assert 'photographs alone establish colors' in prompt
