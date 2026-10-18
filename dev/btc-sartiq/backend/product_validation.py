"""Check generated product multiplicity; no reference uploads or silent approval."""
import base64
import io
from pathlib import Path

from vision import DEFAULT_MODEL

VALIDATION_MODEL = DEFAULT_MODEL
CHECK_PREFIX = 'SINGLE PRODUCT CHECK\n'
CHECK_TIMEOUT = 30
PROMPT_PATH = Path(__file__).with_name('prompts') / 'single_product_check.txt'
REPAIR_PATH = PROMPT_PATH.with_name('single_product_repair.txt')


def check_single_product(image, run_model, extract_text):
    prompt = CHECK_PREFIX + PROMPT_PATH.read_text(encoding='utf-8').strip()
    preview = image.convert('RGB').copy()
    preview.thumbnail((768,768))
    buffer = io.BytesIO()
    preview.save(buffer,format='JPEG',quality=85,subsampling=0)
    response = run_model(VALIDATION_MODEL, {
        'image_url':'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'),
        'prompt':prompt, 'reasoning':False, 'temperature':0})
    if not isinstance(response,dict):
        raise RuntimeError('Product check did not return a response')
    payload = response.get('data') if isinstance(response.get('data'),dict) else response
    if payload.get('finish_reason') not in (None,'stop'):
        return 'uncertain'
    verdict = extract_text(response).strip().strip('"\' .').lower()
    return verdict if verdict in ('single','duplicate','uncertain') else 'uncertain'


def single_product_repair_prompt():
    prompt = REPAIR_PATH.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('Single-product repair prompt is empty')
    return prompt
