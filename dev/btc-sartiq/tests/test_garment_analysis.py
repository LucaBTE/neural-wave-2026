import sys
import base64
import io
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import garment_analysis as analysis
import pipeline


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for key in ('FAL_ANALYSIS_MODEL', 'FAL_ANALYSIS_TIMEOUT', 'FAL_ANALYSIS_MAX_SIDE', 'FAL_ANALYSIS_REQUIRED'):
        monkeypatch.delenv(key, raising=False)
    analysis.clear_analysis_cache()
    yield
    analysis.clear_analysis_cache()


def inputs():
    image = Image.new('RGB', (2000, 2000), 'gray')
    image.paste('red', (700, 500, 1300, 1500))
    mask = Image.new('L', image.size)
    mask.paste(255, (700, 500, 1300, 1500))
    return image, mask, {'status':'neutral', 'applied':False}


def test_crop_keeps_garment_and_does_not_mutate_generator_references():
    image, mask, _ = inputs()
    before = image.tobytes()
    cropped = analysis.prepare_analysis_image(image, mask, 1536)
    assert cropped.size == (700, 1100)
    assert cropped.getpixel((0,0)) == (255,255,255)
    assert cropped.getpixel((350,550)) == (255,0,0)
    assert image.size == (2000,2000) and image.tobytes() == before


def test_cache_reuses_only_identical_image_prompt_and_model(monkeypatch, tmp_path):
    image, mask, light = inputs()
    calls = []
    prompt = tmp_path / 'description.txt'
    prompt.write_text('Describe the garment.')
    monkeypatch.setattr(analysis, 'PROMPT_PATH', prompt)
    def run(model, arguments):
        calls.append((model, arguments))
        return {'output':'Red long-sleeve sweater with ribbed cuffs.'}
    def describe():
        return analysis.analyze_garment(image, mask, light, run, pipeline.extract_description, pipeline.png_bytes)
    first = describe()
    second = describe()
    assert len(calls) == 1
    assert first[0] == second[0] and second[1]['cache_hit']
    sent = Image.open(io.BytesIO(base64.b64decode(calls[0][1]['image_url'].split(',',1)[1])))
    assert sent.size == first[2].size
    prompt.write_text('Describe the garment more precisely.')
    assert not describe()[1]['cache_hit']
    monkeypatch.setenv('FAL_ANALYSIS_MODEL', 'fal-ai/sa2va/4b/image')
    assert not describe()[1]['cache_hit']
    image.putpixel((1000,1000), (0,255,0))
    assert not describe()[1]['cache_hit']
    assert len(calls) == 4


def test_failed_required_analysis_is_not_cached():
    image, mask, light = inputs()
    calls = []
    def run(*args):
        calls.append(args)
        raise TimeoutError()
    for _ in range(2):
        with pytest.raises(TimeoutError):
            analysis.analyze_garment(image, mask, light, run, pipeline.extract_description, pipeline.png_bytes)
    assert len(calls) == 2


def test_cache_expires_and_is_bounded(monkeypatch):
    image, mask, light = inputs()
    clock = [10.0]
    monkeypatch.setattr(analysis, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(analysis, 'CACHE_SIZE', 1)
    calls = []
    def run(*args):
        calls.append(args)
        return {'output':'A red garment.'}
    def describe():
        return analysis.analyze_garment(image, mask, light, run, pipeline.extract_description, pipeline.png_bytes)
    describe()
    clock[0] += analysis.CACHE_TTL + 1
    assert not describe()[1]['cache_hit']
    image.putpixel((1000,1000), (0,255,0))
    describe()
    assert len(analysis._cache) == 1
    image.putpixel((1000,1000), (255,0,0))
    assert not describe()[1]['cache_hit']
    assert len(calls) == 4


def test_only_analysis_crop_is_downsized():
    image, mask, _ = inputs()
    cropped = analysis.prepare_analysis_image(image, mask, 768)
    assert max(cropped.size) == 768
    assert image.size == mask.size == (2000,2000)


def test_sdk_timeout_is_shorter_only_for_analysis(monkeypatch):
    monkeypatch.setenv('FAL_KEY', 'fake-key')
    calls = []
    monkeypatch.setattr(pipeline.fal_client, 'subscribe', lambda *args, **kwargs: calls.append(kwargs) or {})
    pipeline.call(analysis.DEFAULT_MODEL, {})
    pipeline.call(pipeline.RESTYLE_MODEL, {})
    pipeline.call(pipeline.LOCALIZATION_MODEL, {})
    assert [call['client_timeout'] for call in calls] == [90, 300, 90]


@pytest.mark.parametrize('key,value', [('FAL_ANALYSIS_TIMEOUT','0'), ('FAL_ANALYSIS_MAX_SIDE','10'),
    ('FAL_ANALYSIS_MODEL','unknown'), ('FAL_ANALYSIS_TIMEOUT','301')])
def test_invalid_analysis_settings_fail_before_paid_work(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        pipeline.build_pipeline()


def test_fast_analysis_uses_no_reasoning_and_smaller_jpeg_payload():
    image, mask, light = inputs()
    calls = []
    def run(model, arguments):
        calls.append((model, arguments))
        return {'output': 'Red long-sleeved sweater with ribbed cuffs.', 'finish_reason': 'stop'}
    _, metadata, cropped = analysis.analyze_garment(
        image, mask, light, run, pipeline.extract_description, pipeline.png_bytes)
    model, arguments = calls[0]
    assert model == 'fal-ai/moondream3-preview/query'
    assert arguments['reasoning'] is False and arguments['temperature'] == 0
    assert arguments['image_url'].startswith('data:image/jpeg;base64,')
    sent = Image.open(io.BytesIO(base64.b64decode(arguments['image_url'].split(',', 1)[1])))
    assert sent.format == 'JPEG' and sent.size == cropped.size
    assert max(cropped.size) <= 1280
    assert metadata['required'] is True and metadata['reasoning'] is False


@pytest.mark.parametrize('model', ['fal-ai/sa2va/8b/image', 'fal-ai/sa2va/4b/image'])
def test_legacy_models_keep_their_own_api_contract(monkeypatch, model):
    monkeypatch.setenv('FAL_ANALYSIS_MODEL', model)
    image, mask, light = inputs()
    def run(selected_model, arguments):
        assert selected_model == model
        assert set(arguments) == {'image_url', 'prompt'}
        return {'output': 'A red sweater.'}
    analysis.analyze_garment(image, mask, light, run, pipeline.extract_description, pipeline.png_bytes)


def test_truncated_analysis_is_rejected_and_not_cached():
    image, mask, light = inputs()
    def run(*args):
        return {'output': 'Red sweater with', 'finish_reason': 'length'}
    with pytest.raises(RuntimeError, match='did not finish normally'):
        analysis.analyze_garment(image, mask, light, run, pipeline.extract_description, pipeline.png_bytes)
    assert not analysis._cache
