import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from PIL import Image


import threading
import pytest
import pipeline
from product_classification import clear_classification_cache


@pytest.fixture(autouse=True)
def fresh_analysis_cache(monkeypatch):
    monkeypatch.setenv('FAL_PIPELINE_PROFILE', 'faithful')
    clear_classification_cache()
    pipeline.clear_reuse_caches()
    yield
    clear_classification_cache()
    pipeline.clear_reuse_caches()


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
        if arguments.get('prompt','').startswith(pipeline.CHECK_PREFIX):
            return {'output':'SINGLE'}
        if model == pipeline.LOCALIZATION_MODEL:
            return {'objects': [{'x_min':.25,'y_min':.2,'x_max':.75,'y_max':.8}]}
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            return {'output': '{"label":"red sweater","category":"sweater","kind":"garment","bbox":[0.25,0.2,0.75,0.8],"point":[0.5,0.5],"confidence":0.95}'}
        if model == pipeline.SAM_MODEL:
            return {'masks': [{'url': 'https://sam'}]}
        if model == pipeline.BIREF_MODEL:
            return {'image': {'url': 'https://biref'}}
        url = 'https://pressed' if arguments['image_urls'] == ['https://generated'] else 'https://generated'
        return {'images': [{'url': url}]}

    def upload(image, **options):
        with uploads_lock:
            url = f'https://upload/{len(uploads)}'
            uploads.append(image.copy())
            images[url] = image.copy()
        return url

    monkeypatch.setattr(pipeline, 'call', call)
    monkeypatch.setattr(pipeline, 'upload_png', upload)
    monkeypatch.setattr(pipeline, 'fetch_image', lambda url: images[url].copy())
    return original, mask, calls, uploads, images


def test_default_profile_keeps_scene_colors_and_uses_one_generation(fal_stub, monkeypatch):
    original, _, calls, _, images = fal_stub
    monkeypatch.delenv('FAL_PIPELINE_PROFILE')
    monkeypatch.delenv('FAL_IMAGE_MODEL', raising=False)
    # A colored wall is useful context, but must not be treated as a gray card.
    original.paste((210, 175, 130), (0, 0, 20, 100))
    events = []
    state = pipeline.build_pipeline(lambda **e: events.append(e)).invoke({'image': original})
    assert pipeline.pipeline_profile() == 'faithful'
    assert [model for model, _ in calls] == [pipeline.CLASSIFICATION_MODEL,
        pipeline.LOCALIZATION_MODEL, pipeline.SAM_MODEL, pipeline.FAITHFUL_MODEL, pipeline.VALIDATION_MODEL, pipeline.BIREF_MODEL]
    assert state['metrics']['inference_requests'] == 6
    assert state['analysis']['source'] == 'classification'
    assert state['analysis']['separate_model_request'] is False
    assert state['pressing_prompt'] == ''
    assert state['color_reference'].tobytes() == original.tobytes()
    generation = calls[3][1]
    assert generation['image_urls'] == [state['reference_url']]
    assert state['reference_policy'] == 'context_crop'
    assert state['reference_crop'] is not None
    assert images[state['reference_url']].tobytes() == original.crop(state['reference_crop']).tobytes()
    # The crop identifies the whole garment, rather than only a graphic on it.
    assert images[state['color_url']].height >= 60
    assert len(generation['prompt'].split()) < 450
    assert generation['resolution'] == '2K'
    assert 'Lay sleeves outward' in generation['prompt']
    assert 'steam and press' in generation['prompt']
    phases = next(e['stages'] for e in events if 'stages' in e)
    assert len(phases) == 8
    assert not {'describing', 'pressing'} & {p['status'] for p in phases}


def test_faithful_finish_preserves_rgb_and_parts_missing_from_diagnostic_matte(fal_stub, monkeypatch):
    original, _, _, _, images = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE', 'faithful')
    # Simulate a thin pale strap that BiRefNet mistakenly omits. It must survive
    # finalization, along with the generated surface and contact shadow.
    generated = Image.new('RGB', (1200, 1800), (245, 245, 245))
    generated.paste((240, 232, 216), (250, 250, 280, 800))
    generated.paste((10, 10, 10), (300, 400, 900, 1400))
    generated.paste((230, 230, 230), (300, 1400, 900, 1450))
    images['https://generated'] = generated
    matte = generated.convert('RGBA')
    alpha = Image.new('L', generated.size)
    alpha.paste(255, (300, 400, 900, 1400))
    matte.putalpha(alpha)
    images['https://biref'] = matte
    state = pipeline.build_pipeline().invoke({'image': original})
    assert state['result'].getpixel((265, 500)) == (240, 232, 216)
    assert state['result'].getpixel((600, 1420)) == (230, 230, 230)
    assert state['result'].getpixel((600, 800)) == (10, 10, 10)
    assert state['result'].getpixel((0, 0)) == (245, 245, 245)
    assert state['result_mask'].getpixel((265, 500)) == 0  # diagnostic only


def test_generator_receives_one_version_of_each_distinct_source_photo(fal_stub, monkeypatch):
    original,_,calls,_,images = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE','faithful')
    detail = Image.new('RGB',(60,90),'purple')
    state = pipeline.build_pipeline().invoke({'image':original,
        'additional_images':[original.copy(),detail,detail.copy()]})
    arguments = next(args for model,args in calls if model == pipeline.FAITHFUL_MODEL)
    assert arguments['image_urls'] == [state['reference_url'],state['additional_reference_urls'][0]]
    assert len(state['additional_reference_urls']) == 1
    assert images[arguments['image_urls'][0]].tobytes() == original.crop(state['reference_crop']).tobytes()
    assert images[arguments['image_urls'][1]].tobytes() == detail.tobytes()
    assert 'INPUT: 2 reference images for ONE product.' in arguments['prompt']
    assert state['execution_trace'][4]['details']['input_image_count'] == 2
    assert state['quality']['single_product']['status'] == 'passed'


@pytest.mark.parametrize('second_verdict',['SINGLE','DUPLICATE'])
def test_overlapping_duplicate_is_corrected_once_and_must_pass_recheck(fal_stub,monkeypatch,second_verdict):
    original,_,calls,_,images = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE','faithful')
    images['https://repaired'] = original.copy()
    stub = pipeline.call
    checks,repairs = [],[]
    def run(model,arguments):
        if arguments.get('prompt','').startswith(pipeline.CHECK_PREFIX):
            calls.append((model,arguments))
            checks.append(arguments)
            if len(checks) == 1:
                return {'output':'DUPLICATE'}
            if second_verdict == 'timeout':
                raise TimeoutError('Check unavailable')
            return {'output':second_verdict}
        if arguments.get('prompt','').startswith('Correct this catalogue photograph'):
            calls.append((model,arguments))
            repairs.append(arguments)
            return {'images':[{'url':'https://repaired'}]}
        return stub(model,arguments)
    monkeypatch.setattr(pipeline,'call',run)
    events,snapshots = [],[]
    graph = pipeline.build_pipeline(lambda **e:events.append(e),lambda key,label,img:snapshots.append(key))
    if second_verdict == 'SINGLE':
        state = graph.invoke({'image':original})
        assert state['result_url'] and state['pressed_url'] == 'https://repaired'
        assert state['quality']['single_product']['status'] == 'passed'
        assert state['quality']['single_product']['repair_attempted']
        assert [r['verdict'] for r in state['product_checks']] == ['duplicate','single']
        assert state['metrics']['inference_requests'] == 8
        assert snapshots[-1] == 'final'
        assert [s['id'] for s in state['execution_trace']][-5:] == [
            'inspect','repair_product','inspect_repair','finish','verify']
    else:
        with pytest.raises(RuntimeError,match='single product could not be verified'):
            graph.invoke({'image':original})
        assert 'final' not in snapshots
        assert not any(m == pipeline.BIREF_MODEL for m,_ in calls)
        assert events[-1]['execution_trace'][-1]['id'] == 'inspect_repair'
    assert len(checks) == 2 and len(repairs) == 1
    assert repairs[0]['image_urls'] == ['https://generated']
    assert repairs[0]['num_images'] == 1
    assert repairs[0]['resolution'] == '2K'
    assert 'product_repaired' in snapshots
    stages = next(e['stages'] for e in events if any(p['status'] == 'correcting_product' for p in e.get('stages',[])))
    assert len(stages) == 10


def test_inconclusive_initial_check_keeps_output_but_never_claims_validation(fal_stub,monkeypatch):
    original,_,calls,_,_ = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE','faithful')
    stub = pipeline.call
    def uncertain(model,arguments):
        if arguments.get('prompt','').startswith(pipeline.CHECK_PREFIX):
            calls.append((model,arguments))
            return {'output':'UNCERTAIN'}
        return stub(model,arguments)
    monkeypatch.setattr(pipeline,'call',uncertain)
    state = pipeline.build_pipeline().invoke({'image':original})
    assert state['result_url'] and state['quality']['status'] == 'review_required'
    assert state['quality']['single_product']['status'] == 'unverified'
    assert not state['quality']['single_product']['repair_attempted']
    assert state['metrics']['inference_requests'] == 6
    assert sum(model == pipeline.FAITHFUL_MODEL for model,_ in calls) == 1


def test_api_never_publishes_a_known_duplicate_as_the_final_result(fal_stub,monkeypatch):
    import server
    original,_,calls,_,_ = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE','faithful')
    stub = pipeline.call
    def duplicate(model,arguments):
        if arguments.get('prompt','').startswith(pipeline.CHECK_PREFIX):
            calls.append((model,arguments))
            return {'output':'DUPLICATE'}
        return stub(model,arguments)
    monkeypatch.setattr(pipeline,'call',duplicate)
    identity = 'duplicate-rejected'
    server.jobs[identity] = {'status':'queued','error':None,'result_url':None}
    try:
        server.studio_task(identity,pipeline.png_bytes(original))
        job = server.jobs[identity]
        assert job['status'] == 'error' and job['result_url'] is None
        assert 'single product could not be verified' in job['error']
        assert all(img['id'] != 'final' for img in job['images'])
        assert job['execution_trace'][-1]['id'] == 'inspect_repair'
    finally:
        server.jobs.pop(identity)


def test_faithful_regeneration_reuses_preprocessing_and_keeps_all_views(fal_stub, monkeypatch):
    original, _, calls, _, images = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE', 'faithful')
    extra = Image.new('RGB', (50, 80), 'green')
    pipeline.build_pipeline().invoke({'image': original, 'additional_images': [extra]})
    first_count = len(calls)
    state = pipeline.build_pipeline().invoke({'image': original, 'additional_images': [extra]})
    assert [m for m, _ in calls[first_count:]] == [pipeline.FAITHFUL_MODEL, pipeline.VALIDATION_MODEL, pipeline.BIREF_MODEL]
    assert state['metrics']['inference_requests'] == 3
    assert state['metrics']['reused_inferences'] == 3
    assert state['metrics']['storage_uploads'] == 1
    urls = calls[first_count][1]['image_urls']
    assert len(urls) == 2 and images[urls[1]].tobytes() == extra.tobytes()
    assert 'MULTI-VIEW REFERENCE RULES' in state['generation_prompt']
    assert state['analysis']['view_count'] == 2


def test_faithful_successful_edit_survives_all_helpers_being_unavailable(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    monkeypatch.setenv('FAL_PIPELINE_PROFILE', 'faithful')
    stub = pipeline.call
    def only_generation(model, arguments):
        if model != pipeline.FAITHFUL_MODEL:
            calls.append((model, arguments))
            return {}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', only_generation)
    state = pipeline.build_pipeline().invoke({'image':original})
    assert state['result_url'] and state['generated_image'] is not None
    assert state['mask'] is None and state['result_mask'] is None
    assert set(state['recovery']) == {'classify','locate','isolate','finish'}
    assert state['metrics']['inference_requests'] == 4
    assert state['metrics']['reused_inferences'] == 0


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
        'original', 'target', 'mask', 'segmented', 'color', 'reference']
    server.jobs.pop('failure')




def test_upload_api_keeps_frontend_contract_and_rejects_too_many_photos(fal_stub,monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import server
    original,_,calls,_,_=fal_stub
    monkeypatch.setattr(server,'pool',SimpleNamespace(submit=lambda fn,*args:fn(*args)))
    photo=('photo.png',pipeline.png_bytes(original),'image/png')
    with TestClient(server.app) as client:
        response=client.post('/api/studio',files={'file':photo})
        assert response.status_code==202
        identity=response.json()['job_id']
        try:
            job=client.get('/api/jobs/'+identity).json()
            assert job['status']=='complete' and job['result_url']
            assert job['pressing_prompt']=='' and job['generation_prompt']
            assert job['images'][-1]['id']=='final'
            assert job['metrics']['inference_requests']==6
            config=client.get('/api/config').json()
            assert config['models']['restyling']==pipeline.FAITHFUL_MODEL
            assert config['models']['pressing'] is None
            count=len(calls)
            too_many=client.post('/api/studio',files=[('file',photo)]+[('additional_files',photo)]*5)
            assert too_many.status_code==400 and len(calls)==count
        finally:
            server.jobs.pop(identity,None)
