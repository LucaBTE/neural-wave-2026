import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from PIL import Image
from pipeline import background_mask,extract_urls

def test_mask_inverts_and_protects_garment():
    m=Image.new('L',(20,20),0)
    for x in range(5,15):
        for y in range(5,15): m.putpixel((x,y),255)
    bg=background_mask(m)
    assert bg.getpixel((10,10))==0
    assert bg.getpixel((0,0))==255

def test_sam_masks_and_generation_response():
    assert extract_urls({'masks':[{'url':'https://a'},{'url':'https://b'}]},masks=True)==['https://a','https://b']
    assert extract_urls({'images':[{'url':'https://out'}]})==['https://out']

import numpy as np
import pytest
import pipeline
from quality import compare_product


def product(color='red', size=(80, 100)):
    image = Image.new('RGB', size, 'white')
    mask = Image.new('L', size, 0)
    image.paste(color, (20, 20, 60, 80))
    mask.paste(255, (20, 20, 60, 80))
    return image, mask


@pytest.fixture
def fal_stub(monkeypatch):
    original, mask = product()
    matte = original.convert('RGBA')
    matte.putalpha(mask)
    images = {'https://sam': mask, 'https://biref': matte,
              'https://generated': original}
    calls, uploads = [], []

    def call(model, arguments):
        calls.append((model, arguments))
        if model == pipeline.SAM_MODEL:
            return {'masks': [{'url': 'https://sam'}]}
        if model == pipeline.BIREF_MODEL:
            return {'image': {'url': 'https://biref'}}
        return {'images': [{'url': 'https://generated'}]}

    def upload(image):
        url = f'https://upload/{len(uploads)}'
        uploads.append(image.copy())
        images[url] = image.copy()
        return url

    monkeypatch.setattr(pipeline, 'call', call)
    monkeypatch.setattr(pipeline, 'upload_png', upload)
    monkeypatch.setattr(pipeline, 'fetch_image', lambda url: images[url].copy())
    return original, mask, calls, uploads, images


def test_full_graph_restyles_then_enforces_white_background(fal_stub):
    original, _, calls, uploads, _ = fal_stub
    events = []
    state = pipeline.build_pipeline(lambda **e: events.append(e)).invoke({'image': original})
    assert [c[0] for c in calls] == [pipeline.SAM_MODEL, pipeline.RESTYLE_MODEL,
                                    pipeline.SAM_MODEL, pipeline.BIREF_MODEL]
    assert calls[1][1]['image_urls'] == [state['image_url'], state['reference_url']]
    assert calls[1][1]['resolution'] == '2K'
    assert calls[2][1]['image_url'] == state['generated_url']
    assert state['result'].size == (1200, 1800)
    assert state['result'].getpixel((0, 0)) == (255, 255, 255)
    assert state['quality']['passed']
    assert {e['status'] for e in events} == {'preparing','generating','finishing','checking_quality'}


def test_composition_recenters_without_changing_colors():
    image, mask = product()
    cutout = image.convert('RGBA'); cutout.putalpha(mask)
    result, final_mask = pipeline.compose_studio(cutout)
    assert result.getpixel((600, 900)) == (255, 0, 0)
    box = final_mask.getbbox()
    assert box[0] + box[2] == 1200
    assert box[1] + box[3] == 1800


def test_missing_output_stops_before_next_paid_call(fal_stub, monkeypatch):
    original, _, _, _, _ = fal_stub
    monkeypatch.setattr(pipeline, 'call', lambda *args: {})
    with pytest.raises(RuntimeError, match='SAM did not return a mask'):
        pipeline.build_pipeline().invoke({'image':original, 'garment':'sweater'})


def test_biref_must_return_alpha(fal_stub):
    original, mask, _, _, images = fal_stub
    images['https://biref'] = original
    with pytest.raises(RuntimeError, match='transparent cutout'):
        pipeline.remove_background(original, mask, 2)


def test_quality_ignores_changed_background():
    a, mask = product()
    b = Image.composite(a, Image.new('RGB', a.size, 'black'), mask)
    assert compare_product(b, mask, a, mask)['passed']
    assert not compare_product(a, mask, b, mask)['passed']


def test_quality_detects_color_and_shape_changes():
    a, mask = product()
    b, _ = product('blue')
    assert not compare_product(a, mask, b, mask)['passed']
    assert compare_product(a, mask, b, mask)['metrics']['color_distance'] > .35
    wide_mask = Image.new('L', a.size, 0)
    wide_mask.paste(255, (5, 20, 75, 80))
    report = compare_product(a, mask, a, wide_mask)
    assert report['metrics']['aspect_similarity'] < .85
    assert not any('silhouette' in w for w in report['warnings'])


def test_quality_detects_lost_texture():
    a, mask = product()
    pixels = np.array(a)
    pixels[20:80:4, 20:60] = 0
    textured = Image.fromarray(pixels)
    assert compare_product(textured, mask, a, mask)['metrics']['detail_similarity'] < .6


def test_quality_rejects_empty_mask():
    a, mask = product()
    with pytest.raises(ValueError, match='Too few'):
        compare_product(a, mask, a, Image.new('L', a.size))


def test_single_click_api(fal_stub, monkeypatch):
    from fastapi.testclient import TestClient
    import server
    client = TestClient(server.app)
    pending = []
    monkeypatch.setattr(server.pool, 'submit', lambda *args: pending.append(args))
    original, _, _, _, _ = fal_stub
    response = client.post('/api/studio', files={'file':('photo.png', pipeline.png_bytes(original), 'image/png')})
    assert response.status_code == 202
    job_id = response.json()['job_id']
    task, *args = pending.pop()
    task(*args)
    job = client.get('/api/jobs/' + job_id).json()
    assert job['status'] == 'complete'
    assert job['quality']['passed'] and job['result_url']
    assert 'garment_mask' not in job
    assert client.post('/api/studio', files={'file':('bad.png', b'bad', 'image/png')}).status_code == 400
    with server.lock:
        server.jobs.pop(job_id)


def test_failed_generation_never_returns_original_as_success(fal_stub, monkeypatch):
    import server
    original, _, _, _, _ = fal_stub
    monkeypatch.setattr(pipeline, 'output_url', lambda *args: (_ for _ in ()).throw(RuntimeError('Provider failed')))
    server.jobs['failure'] = {'status':'queued', 'result_url':None}
    server.studio_task('failure', pipeline.png_bytes(original))
    assert server.jobs['failure']['status'] == 'error'
    assert server.jobs['failure']['result_url'] is None
    server.jobs.pop('failure')


def test_quality_detects_black_to_white_identity_change():
    original, mask = product('black')
    result, _ = product('white')
    assert not compare_product(original, mask, result, mask)['passed']


def test_new_pose_is_not_rejected_for_shape_alone():
    original, mask = product('red')
    result = Image.new('RGB', original.size, 'white')
    changed = Image.new('L', original.size, 0)
    result.paste('red', (5, 30, 75, 70))
    changed.paste(255, (5, 30, 75, 70))
    report = compare_product(original, mask, result, changed)
    assert report['metrics']['aspect_similarity'] < .85
    assert report['passed']
