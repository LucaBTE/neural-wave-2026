"""Build the exact generation prompt from an editable UTF-8 template."""
import json
import re
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name('prompts') / 'generation.txt'
PRESSING_TEMPLATE_PATH = TEMPLATE_PATH.with_name('pressing.txt')


def build_pressing_prompt() -> str:
    """Reload the dedicated finishing instructions for each job."""
    prompt = PRESSING_TEMPLATE_PATH.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('Pressing prompt must not be empty')
    return prompt


def build_generation_prompt(description: str, illumination: dict) -> str:
    # Read on every job so edits take effect without restarting the server.
    template = TEMPLATE_PATH.read_text(encoding='utf-8')
    tokens = re.findall(r'\{\{([A-Z_]+)\}\}', template)
    required = {'LIGHTING_INSTRUCTIONS', 'GARMENT_DESCRIPTION'}
    if set(tokens) != required or any(tokens.count(token) != 1 for token in required):
        raise ValueError('Generation template must contain each required placeholder exactly once: '
                         '{{LIGHTING_INSTRUCTIONS}}, {{GARMENT_DESCRIPTION}}')
    if not isinstance(description, str) or not description.strip():
        raise ValueError('A non-empty garment description is required')
    lighting = (
        'The references have already been white-balanced or assessed as neutral. '
        'Preserve their palette; do not apply a second global color cast correction.'
        if illumination['applied'] or illumination['status'] == 'neutral' else
        'No reliable global correction was applied. Relight conservatively using scene '
        'evidence, including local mixed light; do not assume a material\'s true color.'
    )
    values = {'LIGHTING_INSTRUCTIONS': lighting,
              'GARMENT_DESCRIPTION': json.dumps(description.strip(), ensure_ascii=False)}
    # Single pass: placeholder-like text inside model output remains literal data.
    return re.sub(r'\{\{([A-Z_]+)\}\}', lambda match: values[match.group(1)], template).strip()
