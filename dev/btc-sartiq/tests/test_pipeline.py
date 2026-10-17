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
import threading
import pytest
import pipeline
from garment_analysis import clear_analysis_cache
from quality import compare_product


@pytest.fixture(autouse=True)
def fresh_analysis_cache():
    clear_analysis_cache()
    yield
    clear_analysis_cache()


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
              'https://generated': original, 'https://pressed': original.copy()}
    calls, uploads = [], []
    uploads_lock = threading.Lock()

    def call(model, arguments):
        calls.append((model, arguments))
        if model == pipeline.DESCRIPTION_MODEL:
            return {'output': '<p>Relaxed red knitted sweater with ribbed cuffs.</p><|im_end|>', 'masks': []}
        if model == pipeline.SAM_MODEL:
            return {'masks': [{'url': 'https://sam'}]}
        if model == pipeline.BIREF_MODEL:
            return {'image': {'url': 'https://biref'}}
        url = 'https://pressed' if arguments['image_urls'] == ['https://generated'] else 'https://generated'
        return {'images': [{'url': url}]}

    def upload(image):
        with uploads_lock:
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
    snapshots = []
    state = pipeline.build_pipeline(lambda **e: events.append(e),
                                    lambda key, label, image: snapshots.append((key, label, image))).invoke({'image': original})
    assert [c[0] for c in calls] == [pipeline.SAM_MODEL, pipeline.DESCRIPTION_MODEL, pipeline.RESTYLE_MODEL,
                                    pipeline.RESTYLE_MODEL, pipeline.BIREF_MODEL]
    assert len(uploads) == 4  # original + selected detail + isolated reference + final
    assert [entry[0] for entry in snapshots] == ['original', 'mask', 'segmented', 'color',
        'reference', 'description', 'generated', 'pressed', 'cutout', 'composed', 'final']
    assert snapshots[-1][2] == state['result_url']
    assert snapshots[6][2] == state['generated_url']
    assert snapshots[7][2] == state['pressed_url'] == 'https://pressed'
    assert snapshots[8][2].mode == 'RGBA'
    assert state['color_url'] != state['image_url']
    assert calls[2][1]['image_urls'] == [state['color_url'], state['reference_url']]
    assert calls[1][1]['image_url'].startswith('data:image/jpeg;base64,')
    assert calls[1][1]['reasoning'] is False
    assert state['analysis']['status'] == 'complete'
    assert not state['analysis']['cache_hit']
    assert state['garment_description'] in calls[2][1]['prompt']
    assert state['generation_prompt'] == calls[2][1]['prompt']
    assert any(event.get('generation_prompt') == state['generation_prompt'] for event in events)
    assert '<p>' not in state['garment_description']
    assert calls[2][1]['resolution'] == '2K'
    assert calls[3][1]['image_urls'] == [state['generated_url']]
    assert calls[3][1]['prompt'] == state['pressing_prompt']
    assert calls[3][1]['resolution'] == calls[2][1]['resolution']
    assert any(event.get('pressing_prompt') == state['pressing_prompt'] for event in events)
    assert calls[4][1]['image_url'] == state['pressed_url']
    assert state['result'].size == (1200, 1800)
    assert state['result'].getpixel((0, 0)) == (255, 255, 255)
    assert state['quality']['illumination']['status'] == 'uncertain'
    assert set(state['metrics']['stages_seconds']) == {'isolate', 'balance', 'describe', 'restyle', 'press', 'finish', 'verify'}
    assert state['metrics']['total_seconds'] >= 0
    assert [step['id'] for step in state['execution_trace']] == [
        'isolate', 'balance', 'describe', 'restyle', 'press', 'finish', 'verify']
    assert state['execution_trace'][2]['model'] == pipeline.DESCRIPTION_MODEL
    assert state['execution_trace'][3]['model'] == pipeline.RESTYLE_MODEL
    assert state['execution_trace'][1]['model'] is None
    assert set(state['execution_trace'][1]['libraries']) >= {'NumPy', 'OpenCV', 'Pillow'}
    assert all(step['seconds'] is not None for step in state['execution_trace'])
    assert events[-1]['execution_trace'] == state['execution_trace']
    assert not state['quality']['passed']
    assert {e['status'] for e in events} == {'preparing','balancing_color','describing','generating','pressing','finishing','checking_quality'}


def test_composition_recenters_without_changing_colors():
    image, mask = product()
    cutout = image.convert('RGBA'); cutout.putalpha(mask)
    result, final_mask = pipeline.compose_studio(cutout)
    assert result.getpixel((600, 900)) == (255, 0, 0)
    box = final_mask.getbbox()
    assert box[0] + box[2] == 1200
    assert box[1] + box[3] == 1800


def test_final_composition_uses_pressed_pixels(fal_stub):
    original, _, _, _, images = fal_stub
    pressed, _ = product('blue')
    images['https://pressed'] = pressed
    state = pipeline.build_pipeline().invoke({'image': original})
    assert state['result'].getpixel((600, 900)) == (0, 0, 255)


def test_pressing_failure_stops_before_background_removal(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    events, snapshots = [], []
    def fail_pressing(model, arguments):
        if arguments.get('image_urls') == ['https://generated']:
            raise RuntimeError('Pressing provider failed')
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', fail_pressing)
    graph = pipeline.build_pipeline(lambda **event: events.append(event),
        lambda key, label, image: snapshots.append(key))
    with pytest.raises(RuntimeError, match='Pressing provider failed'):
        graph.invoke({'image': original})
    assert not any(model == pipeline.BIREF_MODEL for model, _ in calls)
    assert snapshots[-1] == 'generated'
    assert 'final' not in snapshots
    assert events[-1]['execution_trace'][-1]['id'] == 'press'
    assert events[-1]['execution_trace'][-1]['status'] == 'error'


def test_gray_corner_sleeve_cannot_reach_analysis_or_generator(fal_stub, monkeypatch):
    import base64
    import io
    original, mask, calls, _, images = fal_stub
    original.paste((128,128,128), (0,0,20,15))
    sleeve = Image.new('L', original.size)
    sleeve.paste(255, (0,0,20,15))
    images['https://sleeve'] = sleeve
    stub = pipeline.call
    def candidates(model, arguments):
        if model == pipeline.SAM_MODEL:
            calls.append((model, arguments))
            return {'masks':[{'url':'https://sleeve'}, {'url':'https://sam'}]}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', candidates)
    state = pipeline.build_pipeline().invoke({'image':original})
    assert state['mask'].tobytes() == mask.tobytes()
    assert state['target_selection']['candidate_index'] == 1
    for url in calls[2][1]['image_urls']:
        assert not np.any(np.all(np.asarray(images[url]) == (128,128,128), axis=2))
    analysis_image = Image.open(io.BytesIO(base64.b64decode(calls[1][1]['image_url'].split(',',1)[1])))
    assert not np.any(np.all(np.asarray(analysis_image) == (128,128,128), axis=2))
    assert len(calls) == 5


def test_peripheral_only_detection_stops_before_analysis(fal_stub):
    original, _, calls, _, images = fal_stub
    mask = Image.new('L', original.size)
    mask.paste(255, (0,0,20,15))
    images['https://sam'] = mask
    with pytest.raises(RuntimeError, match='image center'):
        pipeline.build_pipeline().invoke({'image':original})
    assert [model for model, _ in calls] == [pipeline.SAM_MODEL]


def test_missing_output_stops_before_next_paid_call(fal_stub, monkeypatch):
    original, _, _, _, _ = fal_stub
    monkeypatch.setattr(pipeline, 'call', lambda *args: {})
    with pytest.raises(RuntimeError, match='SAM did not return a mask'):
        pipeline.build_pipeline().invoke({'image':original, 'garment':'sweater'})


def test_biref_must_return_alpha(fal_stub):
    original, mask, _, _, images = fal_stub
    images['https://biref'] = original
    with pytest.raises(RuntimeError, match='transparent cutout'):
        pipeline.remove_background(original, 'https://generated')


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
    assert job['status'] == 'review_required'
    assert job['quality']['illumination']['status'] == 'uncertain' and job['result_url']
    assert 'Relaxed red knitted sweater' in job['garment_description']
    assert job['generation_prompt']
    assert job['pressing_prompt']
    config = client.get('/api/config').json()
    assert config['inference_requests_per_run'] == 5
    assert config['models']['pressing'] == config['models']['restyling']
    assert job['metrics']['total_seconds'] >= 0
    assert job['execution_trace'][0]['id'] == 'prepare_image'
    assert all(step['run_id'] == job_id for step in job['execution_trace'])
    assert job['current_image']['id'] == 'final'
    assert job['images'][-1]['url'] == job['result_url']
    assert '_images' not in job
    mask_url = next(image['url'] for image in job['images'] if image['id'] == 'mask')
    response = client.get(mask_url)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    import io
    saved_mask = Image.open(io.BytesIO(response.content))
    assert saved_mask.getpixel((40,40)) == 255
    assert client.get('/api/jobs/' + job_id + '/images/missing').status_code == 404
    assert client.get('/api/jobs/missing/images/original').status_code == 404
    with client.stream('GET', '/api/jobs/' + job_id + '/events') as stream:
        assert stream.status_code == 200
        import json
        frames = [line for line in stream.iter_lines() if line.startswith('data: ')]
        assert len(frames) == 1
        assert json.loads(frames[0][6:])['status'] == job['status']
        assert json.loads(frames[0][6:])['images'] == job['images']
        assert '_images' not in json.loads(frames[0][6:])
    assert client.get('/api/jobs/missing/events').status_code == 404
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
    assert server.jobs['failure']['execution_trace'][-1]['id'] == 'restyle'
    assert server.jobs['failure']['execution_trace'][-1]['status'] == 'error'
    assert server.jobs['failure']['execution_trace'][-1]['seconds'] is not None
    assert [image['id'] for image in server.jobs['failure']['images']] == [
        'original', 'mask', 'segmented', 'color', 'reference', 'description']
    server.jobs.pop('failure')


def test_local_previews_update_progress_without_uploading_or_mutating_image():
    import json
    import server
    job_id = 'preview-test'
    with server.lock:
        server.jobs[job_id] = {'status': 'preparing', 'revision': 0}
    stream = server.job_events(job_id)
    try:
        assert 'current_image' not in json.loads(next(stream)[6:])
        image = Image.new('RGBA', (1600, 2000), (255, 0, 0, 128))
        server.record_image(job_id, 'cutout', 'Background removed', image)
        snapshot = json.loads(next(stream)[6:])
        assert snapshot['current_image']['id'] == 'cutout'
        assert '_images' not in snapshot
        assert len(snapshot['images']) == 1
        assert image.size == (1600, 2000)
        # Re-recording an existing step replaces its entry; old SSE snapshots stay immutable.
        server.record_image(job_id, 'cutout', 'Background removed', image)
        assert len(server.jobs[job_id]['images']) == 1
        assert len(snapshot['images']) == 1
    finally:
        stream.close()
        with server.lock:
            server.jobs.pop(job_id)


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


def test_corrected_reference_used_by_generator_and_quality(fal_stub, monkeypatch):
    original, mask, calls, _, images = fal_stub
    corrected = Image.new('RGB', original.size, (180, 150, 130))
    monkeypatch.setattr(pipeline, 'normalize_illumination', lambda *args: (corrected, {'status':'estimated', 'applied':True}))
    seen = []
    def compare(source, *args):
        seen.append(source)
        return {'passed': True, 'warnings': []}
    monkeypatch.setattr(pipeline, 'compare_product', compare)
    state = pipeline.build_pipeline().invoke({'image':original})
    expected_detail = pipeline.prepare_analysis_image(corrected, mask, max(corrected.size))
    assert images[calls[2][1]['image_urls'][0]].tobytes() == expected_detail.tobytes()
    assert seen[0].tobytes() == corrected.tobytes()
    assert 'second global' in calls[2][1]['prompt']
    assert state['quality']['color_reference'] == 'estimated_white_balance'


def test_fal_status_requests_are_not_repeated_inferences(monkeypatch):
    monkeypatch.setenv('FAL_KEY', 'fake-key')
    received = []
    monkeypatch.setattr(pipeline.fal_client, 'subscribe', lambda *args, **kwargs: received.append((args, kwargs)) or {'ok':True})
    assert pipeline.call('test/model', {'image_url':'https://test'}) == {'ok':True}
    assert len(received) == 1
    assert received[0][1]['interval'] == 2.0
    assert received[0][1]['with_logs'] is False


def test_generation_model_is_configurable_and_validated(monkeypatch):
    monkeypatch.setenv('FAL_IMAGE_MODEL', 'fal-ai/nano-banana-2/edit')
    monkeypatch.setenv('FAL_IMAGE_RESOLUTION', '1K')
    assert pipeline.generation_settings() == ('fal-ai/nano-banana-2/edit', '1K')
    monkeypatch.setenv('FAL_IMAGE_MODEL', 'unsupported/model')
    with pytest.raises(ValueError, match='FAL_IMAGE_MODEL'):
        pipeline.generation_settings()
    with pytest.raises(ValueError, match='FAL_IMAGE_MODEL'):
        pipeline.build_pipeline()


def test_generated_alpha_preserves_color_and_soft_edges(fal_stub):
    original, mask, calls, uploads, images = fal_stub
    matte = original.convert('RGBA')
    alpha = mask.copy()
    alpha.putpixel((20, 40), 128)
    matte.putalpha(alpha)
    images['https://biref'] = matte
    cutout = pipeline.remove_background(original, 'https://generated')
    assert cutout.getpixel((20,40)) == (255,0,0,128)
    assert cutout.getpixel((0,0))[3] == 0
    assert calls == [(pipeline.BIREF_MODEL, {'image_url':'https://generated', 'output_format':'png', 'refine_foreground':True})]
    assert uploads == []


def test_progress_stream_sends_changes_and_closes_on_completion():
    import json
    import server
    job_id = 'stream-test'
    with server.lock:
        server.jobs[job_id] = {'status':'queued', 'revision':0}
    stream = server.job_events(job_id)
    try:
        assert json.loads(next(stream)[6:])['status'] == 'queued'
        frames = []
        ready = threading.Event()
        def receive():
            ready.set()
            frames.append(next(stream))
        listener = threading.Thread(target=receive)
        listener.start()
        assert ready.wait(2)
        server.update(job_id, status='complete', result_url='https://result')
        listener.join(timeout=2)
        assert not listener.is_alive()
        assert json.loads(frames[0][6:])['result_url'] == 'https://result'
        with pytest.raises(StopIteration):
            next(stream)
        # Reconnect after missing an event: immediately gets the terminal state.
        assert json.loads(next(server.job_events(job_id))[6:])['status'] == 'complete'
    finally:
        stream.close()
        with server.lock:
            server.jobs.pop(job_id)


@pytest.mark.parametrize('response', [{}, {'output':None}, {'output':42}, {'output':'<p> [SEG] </p><|im_end|>'}, []])
def test_invalid_description_rejected(response):
    with pytest.raises(RuntimeError, match='Garment analysis'):
        pipeline.extract_description(response)


def test_description_cleans_wrapped_response_and_limits_length():
    assert pipeline.extract_description({'data': {'output':'<p>Oversized &amp; ribbed</p> [SEG]<|im_end|>'}}) == 'Oversized & ribbed'
    assert len(pipeline.extract_description({'output':'x' * 7000})) == 6000


def test_description_failure_stops_before_generation(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    def fail_description(model, arguments):
        if model == pipeline.DESCRIPTION_MODEL:
            return {'output': ''}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', fail_description)
    with pytest.raises(RuntimeError, match='empty garment description'):
        pipeline.build_pipeline().invoke({'image':original})
    assert not any(model == pipeline.RESTYLE_MODEL for model, _ in calls)


def test_analysis_cannot_be_skipped_by_former_optional_setting(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    def timed_out(model, arguments):
        if model == pipeline.DESCRIPTION_MODEL:
            calls.append((model, arguments))
            raise TimeoutError('Analysis deadline exceeded')
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', timed_out)
    monkeypatch.setenv('FAL_ANALYSIS_REQUIRED', 'false')
    events = []
    with pytest.raises(TimeoutError):
        pipeline.build_pipeline(lambda **event: events.append(event)).invoke({'image': original})
    assert len(calls) == 2
    assert events[-1]['execution_trace'][-1]['id'] == 'describe'
    assert events[-1]['execution_trace'][-1]['status'] == 'error'
    assert not any(model == pipeline.RESTYLE_MODEL for model, _ in calls)


def test_analysis_timeout_strict_mode_stops_generation(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    def timed_out(model, arguments):
        if model == pipeline.DESCRIPTION_MODEL:
            raise TimeoutError('SA2VA deadline exceeded')
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', timed_out)
    monkeypatch.setenv('FAL_ANALYSIS_REQUIRED', 'true')
    with pytest.raises(TimeoutError):
        pipeline.build_pipeline().invoke({'image':original})
    assert not any(model == pipeline.RESTYLE_MODEL for model, _ in calls)


def test_trace_records_configured_model_and_cache_without_another_inference(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    monkeypatch.setenv('FAL_IMAGE_MODEL', 'fal-ai/nano-banana-2/edit')
    pipeline.build_pipeline().invoke({'image':original})
    before = sum(model == pipeline.DESCRIPTION_MODEL for model, _ in calls)
    state = pipeline.build_pipeline(run_id='cached-run').invoke({'image':original})
    entry = state['execution_trace'][2]
    assert entry['model'] == pipeline.DESCRIPTION_MODEL
    assert entry['status'] == 'cached'
    assert entry['details']['model_invoked'] is False
    assert entry['run_id'] == 'cached-run'
    assert state['execution_trace'][3]['model'] == 'fal-ai/nano-banana-2/edit'
    assert state['execution_trace'][4]['model'] == 'fal-ai/nano-banana-2/edit'
    assert sum(model == pipeline.DESCRIPTION_MODEL for model, _ in calls) == before
