import base64
import io
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import pipeline
from product_validation import check_single_product, CHECK_PREFIX, VALIDATION_MODEL


@pytest.mark.parametrize('text,expected', [('SINGLE','single'), ('"DUPLICATE".','duplicate'),
    ('UNCERTAIN','uncertain'), ('SINGLE DUPLICATE UNCERTAIN','uncertain'),
    ('There may be two shirts','uncertain')])
def test_only_explicit_verdicts_are_accepted(text,expected):
    image = Image.new('RGB',(1600,2400),'white')
    before = image.tobytes()
    calls = []
    def run(model,arguments):
        calls.append((model,arguments))
        assert model == VALIDATION_MODEL
        assert arguments['prompt'].startswith(CHECK_PREFIX)
        assert arguments['reasoning'] is False
        decoded = Image.open(io.BytesIO(base64.b64decode(arguments['image_url'].split(',')[1])))
        assert max(decoded.size) == 768
        return {'output':text}
    assert check_single_product(image,run,pipeline.extract_description) == expected
    assert len(calls) == 1 and image.tobytes() == before


def test_truncated_response_never_approves_an_image():
    assert check_single_product(Image.new('RGB',(10,10)),
        lambda *a:{'output':'SINGLE','finish_reason':'length'},pipeline.extract_description) == 'uncertain'


def test_product_check_has_its_own_short_timeout_without_reducing_analysis_budget(monkeypatch):
    monkeypatch.setenv('FAL_KEY','fake-key')
    monkeypatch.setenv('FAL_ANALYSIS_TIMEOUT','90')
    seen = []
    monkeypatch.setattr(pipeline.fal_client,'subscribe',lambda *a,**k:seen.append(k) or {})
    pipeline.call(VALIDATION_MODEL,{'prompt':CHECK_PREFIX+'Count products'})
    pipeline.call(VALIDATION_MODEL,{'prompt':'Identify ONE main product'})
    assert [r['client_timeout'] for r in seen] == [30,90]
    assert all(r['interval'] == 2 for r in seen)
