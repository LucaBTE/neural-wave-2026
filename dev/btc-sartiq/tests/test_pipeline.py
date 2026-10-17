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
from product_classification import clear_classification_cache
from quality import compare_product


@pytest.fixture(autouse=True)
def fresh_analysis_cache():
    clear_analysis_cache()
    clear_classification_cache()
    pipeline.clear_reuse_caches()
    yield
    clear_analysis_cache()
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
        if model == pipeline.LOCALIZATION_MODEL:
            return {'objects': [{'x_min':.25,'y_min':.2,'x_max':.75,'y_max':.8}]}
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            return {'output': '{"label":"red sweater","category":"sweater","kind":"garment","bbox":[0.25,0.2,0.75,0.8],"point":[0.5,0.5],"confidence":0.95}'}
        if model == pipeline.DESCRIPTION_MODEL:
            return {'output': '<p>Relaxed red knitted sweater with ribbed cuffs.</p><|im_end|>', 'masks': []}
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


def test_full_graph_restyles_then_enforces_white_background(fal_stub):
    original, _, calls, uploads, _ = fal_stub
    events = []
    snapshots = []
    state = pipeline.build_pipeline(lambda **e: events.append(e),
                                    lambda key, label, image: snapshots.append((key, label, image))).invoke({'image': original})
    assert [c[0] for c in calls] == [pipeline.CLASSIFICATION_MODEL, pipeline.LOCALIZATION_MODEL, pipeline.SAM_MODEL, pipeline.DESCRIPTION_MODEL, pipeline.RESTYLE_MODEL,
                                    pipeline.RESTYLE_MODEL, pipeline.BIREF_MODEL]
    assert sum(model == pipeline.RESTYLE_MODEL for model, _ in calls) == 2
    # This synthetic source already has the exact isolated white background, so
    # the full-frame reference can reuse the original's identical PNG upload.
    assert len(uploads) == 3
    assert [entry[0] for entry in snapshots] == ['original', 'target', 'mask', 'segmented', 'color',
        'reference', 'description', 'generated', 'pressed', 'cutout', 'composed', 'final']
    assert snapshots[-1][2] == state['result_url']
    assert snapshots[7][2] == state['generated_url']
    assert snapshots[8][2] == state['pressed_url'] == 'https://pressed'
    assert snapshots[9][2].mode == 'RGBA'
    assert state['color_url'] != state['image_url']
    assert calls[4][1]['image_urls'] == [state['color_url'], state['reference_url']]
    assert calls[3][1]['image_url'].startswith('data:image/jpeg;base64,')
    assert calls[3][1]['reasoning'] is False
    assert state['analysis']['status'] == 'complete'
    assert not state['analysis']['cache_hit']
    assert state['garment_description'] in calls[4][1]['prompt']
    assert state['generation_prompt'] == calls[4][1]['prompt']
    assert any(event.get('generation_prompt') == state['generation_prompt'] for event in events)
    assert '<p>' not in state['garment_description']
    assert calls[4][1]['resolution'] == '2K'
    assert calls[5][1]['image_urls'] == [state['generated_url']]
    assert calls[5][1]['prompt'] == state['pressing_prompt']
    assert calls[5][1]['resolution'] == calls[4][1]['resolution'] == '2K'
    assert calls[6][1]['image_url'] == state['pressed_url']
    assert state['result'].size == (1200, 1800)
    assert state['result'].getpixel((0, 0)) == (255, 255, 255)
    assert state['quality']['illumination']['status'] == 'uncertain'
    assert set(state['metrics']['stages_seconds']) == {'classify', 'locate', 'isolate', 'balance', 'describe', 'restyle', 'press', 'finish', 'verify'}
    assert state['metrics']['total_seconds'] >= 0
    assert [step['id'] for step in state['execution_trace']] == [
        'classify', 'locate', 'isolate', 'balance', 'describe', 'restyle', 'press', 'finish', 'verify']
    assert state['execution_trace'][4]['model'] == pipeline.DESCRIPTION_MODEL
    assert state['execution_trace'][5]['model'] == pipeline.RESTYLE_MODEL
    assert state['execution_trace'][3]['model'] is None
    assert set(state['execution_trace'][3]['libraries']) >= {'NumPy', 'OpenCV', 'Pillow'}
    assert all(step['seconds'] is not None for step in state['execution_trace'])
    assert events[-1]['execution_trace'] == state['execution_trace']
    assert not state['quality']['passed']
    assert {e['status'] for e in events} == {'classifying','locating','preparing','balancing_color','describing','generating','pressing','finishing','checking_quality'}


def test_composition_recenters_without_changing_colors():
    image, mask = product()
    cutout = image.convert('RGBA'); cutout.putalpha(mask)
    result, final_mask = pipeline.compose_studio(cutout)
    assert result.getpixel((600, 900)) == (255, 0, 0)
    box = final_mask.getbbox()
    assert box[0] + box[2] == 1200
    assert box[1] + box[3] == 1800


def test_small_subject_fills_catalogue_frame_without_distorting_proportions():
    image, mask = product()
    cutout = image.convert('RGBA'); cutout.putalpha(mask)
    result, final_mask = pipeline.compose_studio(cutout)
    left, top, right, bottom = final_mask.getbbox()
    assert right - left == 864 and bottom - top == 1296
    assert (right - left) / (bottom - top) == pytest.approx(40 / 60)
    assert result.getpixel((600, 900)) == (255, 0, 0)
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_pressing_failure_never_returns_initial_generation_as_final(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    events, snapshots = [], []
    def fail_pressing(model, arguments):
        if arguments.get('image_urls') == ['https://generated']:
            raise RuntimeError('Pressing failed')
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', fail_pressing)
    with pytest.raises(RuntimeError, match='Pressing failed'):
        pipeline.build_pipeline(lambda **event: events.append(event),
            lambda key, label, image: snapshots.append(key)).invoke({'image': original})
    assert not any(model == pipeline.BIREF_MODEL for model, _ in calls)
    assert snapshots[-1] == 'generated' and 'final' not in snapshots
    assert events[-1]['execution_trace'][-1]['id'] == 'press'
    assert events[-1]['execution_trace'][-1]['status'] == 'error'


def test_regeneration_reuses_validated_inputs_but_generates_fresh_output(fal_stub):
    original, expected_mask, calls, uploads, _ = fal_stub
    first = pipeline.build_pipeline().invoke({'image': original})
    first_calls, first_uploads = len(calls), len(uploads)
    # Editing a caller's returned mask/metadata must not poison shared storage.
    first['mask'].putpixel((40, 40), 0)
    first['target_selection']['bbox'][0] = 999
    second = pipeline.build_pipeline().invoke({'image': original})
    assert [model for model, _ in calls[first_calls:]] == [pipeline.RESTYLE_MODEL, pipeline.RESTYLE_MODEL, pipeline.BIREF_MODEL]
    assert len(uploads) - first_uploads == 1  # fresh final deliverable only
    assert second['color_url'] == first['color_url']
    assert second['reference_url'] == first['reference_url']
    assert second['mask'].tobytes() == expected_mask.tobytes()
    assert second['target_selection']['bbox'][0] == 20
    assert first['metrics']['inference_requests'] == 7
    assert second['metrics']['inference_requests'] == 3
    assert second['metrics']['reused_inferences'] == 4
    assert second['metrics']['storage_uploads'] == 1
    assert second['execution_trace'][2]['details']['model_invoked'] is False
    assert second['generation_prompt'] == first['generation_prompt']
    assert second['result_url'] != first['result_url']


def test_concurrent_jobs_do_not_duplicate_preprocessing(fal_stub):
    from concurrent.futures import ThreadPoolExecutor
    original, _, calls, uploads, _ = fal_stub
    ready = threading.Barrier(2)
    def generate():
        ready.wait(timeout=3)
        return pipeline.build_pipeline().invoke({'image': original.copy()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        states = list(pool.map(lambda _: generate(), range(2)))
    assert len(calls) == 10  # four shared preprocessing + three fresh calls per garment
    assert sum(model == pipeline.SAM_MODEL for model, _ in calls) == 1
    assert sum(model == pipeline.RESTYLE_MODEL for model, _ in calls) == 4
    assert sum(model == pipeline.BIREF_MODEL for model, _ in calls) == 2
    assert len(uploads) == 4  # two identical synthetic references + two deliverables
    assert sum(state['metrics']['inference_requests'] for state in states) == 10
    assert all(state['result_url'] for state in states)


def test_changed_source_invalidates_segmentation_cache(fal_stub):
    original, _, calls, _, _ = fal_stub
    pipeline.build_pipeline().invoke({'image': original})
    original.putpixel((2, 2), (40, 40, 40))
    state = pipeline.build_pipeline().invoke({'image': original})
    assert sum(model == pipeline.SAM_MODEL for model, _ in calls) == 2
    assert state['execution_trace'][2]['details']['model_invoked'] is True


def test_finishing_overlaps_generated_download_and_background_inference(fal_stub, monkeypatch):
    original, _, _, _, _ = fal_stub
    download_started, matte_started = threading.Event(), threading.Event()
    fetch, call = pipeline.fetch_image, pipeline.call
    def fetch_with_overlap(url):
        if url == 'https://pressed':
            download_started.set()
            assert matte_started.wait(3), 'BiRefNet should not wait for the image download'
        return fetch(url)
    def call_with_overlap(model, arguments):
        if model == pipeline.BIREF_MODEL:
            assert download_started.wait(3)
            matte_started.set()
        return call(model, arguments)
    monkeypatch.setattr(pipeline, 'fetch_image', fetch_with_overlap)
    monkeypatch.setattr(pipeline, 'call', call_with_overlap)
    state = pipeline.build_pipeline().invoke({'image': original})
    assert state['result_url'] and matte_started.is_set()


def test_fetch_inline_png_keeps_exact_rgb_and_alpha_without_network(monkeypatch):
    import base64
    image = Image.new('RGBA', (20, 20), (42, 73, 110, 128))
    url = 'data:image/png;base64,' + base64.b64encode(pipeline.png_bytes(image)).decode('ascii')
    def unexpected(*args, **kwargs):
        pytest.fail('Inline output must not trigger an HTTP download')
    monkeypatch.setattr(pipeline, 'urlopen', unexpected)
    result = pipeline.fetch_image(url)
    assert result.mode == image.mode and result.tobytes() == image.tobytes()


@pytest.mark.parametrize('url', ['data:image/png;base64,not-valid!',
    'data:image/svg+xml;base64,PHN2Zz4=', 'http://example.com/image.png'])
def test_fetch_rejects_invalid_inline_images_and_unsupported_schemes(url):
    with pytest.raises(ValueError):
        pipeline.fetch_image(url)


def test_reference_upload_cache_expires_and_preserves_storage_policy(monkeypatch):
    from request_reuse import ReuseCache
    clock, calls = [0], []
    monkeypatch.setattr(pipeline, '_upload_cache', ReuseCache(ttl=600, clock=lambda: clock[0]))
    def upload(image, **kwargs):
        calls.append(kwargs)
        return f'https://reference/{len(calls)}'
    monkeypatch.setattr(pipeline, 'upload_png', upload)
    image = Image.new('RGB', (20,20), 'red')
    first, status = pipeline.upload_reference(image)
    assert status == 'computed' and calls == [{'expires_in':'1h'}]
    assert pipeline.upload_reference(image) == (first, 'cached')
    # Even identical pixels with different color metadata are different inputs.
    profiled = image.copy(); profiled.info['icc_profile'] = b'different profile'
    assert pipeline.upload_reference(profiled)[1] == 'computed'
    clock[0] = 601
    assert pipeline.upload_reference(image)[1] == 'computed'


def test_final_upload_keeps_default_retention(monkeypatch):
    monkeypatch.setenv('FAL_KEY', 'test-key')
    calls = []
    monkeypatch.setattr(pipeline.fal_client, 'upload', lambda data, **kwargs: calls.append(kwargs) or 'https://uploaded')
    image = Image.new('RGB', (20,20), 'red')
    pipeline.upload_png(image)
    pipeline.upload_png(image, expires_in='1h')
    assert 'lifecycle' not in calls[0]
    assert calls[1]['lifecycle'].expires_in == '1h'


def test_final_composition_uses_pressed_pixels(fal_stub):
    original, _, _, _, images = fal_stub
    generated, _ = product('blue')
    images['https://pressed'] = generated
    state = pipeline.build_pipeline().invoke({'image': original})
    assert state['result'].getpixel((600, 900)) == (0, 0, 255)



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
    for url in calls[4][1]['image_urls']:
        assert not np.any(np.all(np.asarray(images[url]) == (128,128,128), axis=2))
    analysis_image = Image.open(io.BytesIO(base64.b64decode(calls[3][1]['image_url'].split(',',1)[1])))
    assert not np.any(np.all(np.asarray(analysis_image) == (128,128,128), axis=2))
    assert len(calls) == 7


def test_peripheral_only_detection_stops_before_analysis(fal_stub):
    original, _, calls, _, images = fal_stub
    mask = Image.new('L', original.size)
    mask.paste(255, (0,0,20,15))
    images['https://sam'] = mask
    with pytest.raises(RuntimeError, match='image center'):
        pipeline.build_pipeline().invoke({'image':original})
    assert [model for model, _ in calls] == [pipeline.CLASSIFICATION_MODEL, pipeline.LOCALIZATION_MODEL, pipeline.SAM_MODEL]


def test_printed_title_never_drives_detection_or_replaces_complete_shirt(fal_stub, monkeypatch):
    import json
    original, whole_shirt, calls, _, images = fal_stub
    print_only = Image.new('L', original.size)
    print_only.paste(255, (30, 35, 50, 50))
    images['https://print'] = print_only
    stub = pipeline.call

    def run(model, arguments):
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            calls.append((model, arguments))
            return {'output': json.dumps({'label':'Jennie Ruby', 'category':'T-shirt',
                                         'kind':'garment', 'confidence':.95})}
        if model == pipeline.SAM_MODEL:
            calls.append((model, arguments))
            return {'masks':[{'url':'https://print'}, {'url':'https://sam'}]}
        return stub(model, arguments)

    monkeypatch.setattr(pipeline, 'call', run)
    state = pipeline.build_pipeline().invoke({'image':original})
    for model, arguments in calls:
        if model in (pipeline.LOCALIZATION_MODEL, pipeline.SAM_MODEL):
            assert arguments['prompt'] == 'T-shirt'
    assert state['mask'].tobytes() == whole_shirt.tobytes()
    assert state['target_selection']['candidate_index'] == 1
    assert len([1 for model, _ in calls if model == pipeline.RESTYLE_MODEL]) == 2


def test_dress_misreported_as_other_uses_garment_prompt_and_two_nano_edits(fal_stub, monkeypatch):
    import json
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    def run(model, arguments):
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            calls.append((model, arguments))
            return {'output':json.dumps({'label':'checked maxi dress with black belt',
                'category':'long-sleeve maxi dress', 'kind':'other', 'confidence':.95})}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', run)
    state = pipeline.build_pipeline().invoke({'image':original})
    assert state['classification']['kind'] == 'garment'
    assert state['classification']['model_kind'] == 'other'
    assert 'MANDATORY FRONT-VIEW RESTAGING' in state['generation_prompt']
    assert state['pressing_prompt']
    assert next(step for step in state['execution_trace'] if step['id'] == 'press')['status'] == 'complete'
    assert len([model for model, _ in calls if model == pipeline.RESTYLE_MODEL]) == 2
    assert state['metrics']['inference_requests'] == 7


def test_print_only_mask_is_rejected_for_complete_shirt_box(fal_stub, monkeypatch):
    original, _, calls, _, images = fal_stub
    print_only = Image.new('L', original.size)
    print_only.paste(255, (30, 35, 50, 50))
    images['https://sam'] = print_only
    with pytest.raises(RuntimeError, match='classified product'):
        pipeline.build_pipeline().invoke({'image':original})
    assert not any(model == pipeline.RESTYLE_MODEL for model, _ in calls)


def test_worn_foreground_sweatshirt_isolated_without_side_rack_shirt(fal_stub, monkeypatch):
    import json
    _, _, calls, _, images = fal_stub
    original = Image.new('RGB', (200, 200), 'gray')
    original.paste('yellow', (24, 36, 54, 80))
    original.paste('lightblue', (56, 60, 174, 152))
    foreground = Image.new('L', original.size)
    foreground.paste(255, (56, 60, 174, 152))
    distant = Image.new('L', original.size)
    distant.paste(255, (24, 36, 54, 80))
    images.update({'https://sam':foreground, 'https://distant':distant})
    stub = pipeline.call
    def run(model, arguments):
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            calls.append((model, arguments))
            return {'output':json.dumps({'label':'light blue sweatshirt worn by main person',
                'category':'light blue sweatshirt', 'kind':'garment', 'confidence':.95})}
        if model == pipeline.LOCALIZATION_MODEL:
            calls.append((model, arguments))
            assert arguments['prompt'] == 'light blue sweatshirt'
            return {'objects':[
                {'x_min':.12,'y_min':.18,'x_max':.27,'y_max':.4},
                {'x_min':.28,'y_min':.30,'x_max':.87,'y_max':.76}]}
        if model == pipeline.SAM_MODEL:
            calls.append((model, arguments))
            assert arguments['prompt'] == 'light blue sweatshirt'
            return {'masks':[{'url':'https://distant'}, {'url':'https://sam'}]}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', run)
    state = pipeline.build_pipeline().invoke({'image':original})
    assert state['mask'].tobytes() == foreground.tobytes()
    assert state['classification']['kind'] == 'garment'
    assert state['classification']['bbox'] == [.28,.30,.87,.76]
    references = next(args['image_urls'] for model, args in calls if model == pipeline.RESTYLE_MODEL)
    for url in references:
        assert not np.any(np.all(np.asarray(images[url]) == (255,255,0), axis=2))
    assert state['metrics']['inference_requests'] == 7


def test_glasses_target_guides_sam_and_skips_clothing_restyling(fal_stub, monkeypatch):
    import json
    _, _, calls, _, images = fal_stub
    original = Image.new('RGB', (200, 200), 'white')
    original.paste('blue', (10, 0, 190, 105))  # sweater covers the exact center
    original.paste('black', (60, 110, 140, 160))
    sweater = Image.new('L', original.size)
    sweater.paste(255, (10, 0, 190, 105))
    glasses = Image.new('L', original.size)
    glasses.paste(255, (60, 110, 140, 160))
    target = {'label': 'black sunglasses', 'category':'sunglasses', 'kind': 'accessory', 'bbox': [.3,.55,.7,.8],
              'point': [.4,.7], 'confidence': .95}
    images.update({'https://sam': glasses, 'https://sweater': sweater})
    generated = Image.new('RGB', original.size, 'white')
    generated.paste('black', (60, 110, 140, 160))
    images['https://generated'] = generated
    matte = generated.convert('RGBA'); matte.putalpha(glasses)
    images['https://biref'] = matte
    stub = pipeline.call
    def run(model, arguments):
        if model == pipeline.LOCALIZATION_MODEL:
            calls.append((model, arguments))
            return {'objects': [{'x_min':.3,'y_min':.55,'x_max':.7,'y_max':.8}]}
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            calls.append((model, arguments))
            return {'output': json.dumps({**target, 'bbox':[.1,.1,.8,.9], 'point':[.2,.2]})}
        if model == pipeline.SAM_MODEL:
            calls.append((model, arguments))
            return {'masks': [{'url':'https://sweater'}, {'url':'https://sam'}]}
        if model == pipeline.DESCRIPTION_MODEL:
            calls.append((model, arguments))
            return {'output': 'Black sunglasses with dark lenses and thick temples.'}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', run)
    snapshots = []
    state = pipeline.build_pipeline(on_image=lambda key, label, image: snapshots.append(key)).invoke({'image': original})
    sam = next(args for model, args in calls if model == pipeline.SAM_MODEL)
    assert sam['prompt'] == 'sunglasses'
    assert sam['point_prompts'] == []
    assert sam['box_prompts'] == [{'x_min':60,'y_min':110,'x_max':140,'y_max':160}]
    assert state['mask'].tobytes() == glasses.tobytes()
    references = next(args['image_urls'] for model, args in calls if model == pipeline.RESTYLE_MODEL)
    for url in references:
        assert not np.any(np.all(np.asarray(images[url]) == (0,0,255), axis=2))
    assert state['classification']['kind'] == 'accessory'
    assert 'For glasses, preserve both lenses' in state['generation_prompt']
    assert 'PRESSING IS MANDATORY' not in state['generation_prompt']
    assert not state['pressing_prompt']
    assert sum(model == pipeline.RESTYLE_MODEL for model, _ in calls) == 1
    press_trace = next(step for step in state['execution_trace'] if step['id'] == 'press')
    assert press_trace['status'] == 'skipped' and press_trace['model'] is None
    assert press_trace['details']['model_invoked'] is False
    assert 'pressed' not in snapshots and snapshots[-1] == 'final'
    assert len(calls) == 6


def test_ambiguous_classification_stops_before_segmentation(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    def ambiguous(model, arguments):
        calls.append((model, arguments))
        return {'output': '{"label":"unknown","kind":"other","bbox":[0.2,0.2,0.8,0.8],"point":[0.5,0.5],"confidence":0.2}'}
    monkeypatch.setattr(pipeline, 'call', ambiguous)
    with pytest.raises(RuntimeError, match='no usable product identity'):
        pipeline.build_pipeline().invoke({'image': original})
    assert len(calls) == 1 and calls[0][0] == pipeline.CLASSIFICATION_MODEL


@pytest.mark.parametrize('consistent', [True, False])
def test_uncertain_classification_is_approved_only_when_detection_and_mask_agree(fal_stub, monkeypatch, consistent):
    import json
    original, _, calls, _, images = fal_stub
    if not consistent:
        fragment = Image.new('L', original.size)
        fragment.paste(255, (28, 32, 52, 68))  # Passes basic selection, lacks full extent.
        images['https://sam'] = fragment
    stub = pipeline.call
    def uncertain(model, arguments):
        if model == pipeline.CLASSIFICATION_MODEL and 'Identify ONE main product' in arguments.get('prompt', ''):
            calls.append((model, arguments))
            return {'output': json.dumps({'label':'patterned garment','category':'garment','kind':'garment','confidence':.2})}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', uncertain)
    monkeypatch.setattr(pipeline, 'normalize_illumination', lambda image, mask:
        (image.copy(), {'status':'neutral', 'applied':False}))
    monkeypatch.setattr(pipeline, 'compare_product', lambda *args: {'passed':True, 'status':'complete', 'warnings':[]})
    state = pipeline.build_pipeline().invoke({'image': original})
    assert state['result_url']
    assert state['classification']['review_required'] is not consistent
    assert state['classification']['selection_validation']['approved'] is consistent
    assert state['classification']['confidence'] == .2  # Preserve the model's actual confidence.
    assert state['quality']['passed'] is consistent
    assert bool(state['quality']['warnings']) is not consistent
    assert state['execution_trace'][0]['status'] == ('complete' if consistent else 'review_required')
    assert len(calls) == 7
    repeated = pipeline.build_pipeline().invoke({'image': original})
    assert repeated['classification']['cache_hit']
    assert repeated['quality']['passed'] is consistent
    assert repeated['classification']['review_required'] is not consistent
    assert repeated['execution_trace'][0]['status'] == 'cached'
    assert len(calls) == 10  # Validation also runs on reused preparation.


def test_missing_detection_stops_before_sam_without_using_text_box(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    def missing(model, arguments):
        if model == pipeline.LOCALIZATION_MODEL:
            calls.append((model, arguments))
            return {'objects': []}
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', missing)
    events = []
    with pytest.raises(RuntimeError, match='Could not locate'):
        pipeline.build_pipeline(lambda **event: events.append(event)).invoke({'image': original})
    assert [model for model, _ in calls] == [pipeline.CLASSIFICATION_MODEL, pipeline.LOCALIZATION_MODEL]
    assert events[-1]['execution_trace'][-1]['id'] == 'locate'
    assert events[-1]['execution_trace'][-1]['status'] == 'error'


def test_missing_output_stops_before_next_paid_call(fal_stub, monkeypatch):
    original, _, _, _, _ = fal_stub
    stub = pipeline.call
    monkeypatch.setattr(pipeline, 'call', lambda model, args: {} if model == pipeline.SAM_MODEL else stub(model, args))
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


@pytest.mark.parametrize('additional_count', [1, 4])
def test_multiview_references_reach_analysis_and_generator_without_more_inferences(fal_stub, additional_count):
    original, _, calls, _, images = fal_stub
    extras = [Image.new('RGB', (120, 180), color)
              for color in ['purple', 'green', 'pink', 'blue'][:additional_count]]
    before = [image.tobytes() for image in extras]
    state = pipeline.build_pipeline().invoke({'image':original, 'additional_images':extras})
    assert len(calls) == 7
    assert state['analysis']['view_count'] == 1 + additional_count
    assert state['analysis']['input_size'] == [2560, 1280]
    generation = [arguments for model, arguments in calls if model == pipeline.RESTYLE_MODEL]
    assert len(generation) == 2
    assert len(generation[0]['image_urls']) == 2 + additional_count
    for index, url in enumerate(generation[0]['image_urls'][2:]):
        assert images[url].tobytes() == before[index]
    assert generation[1]['image_urls'] == ['https://generated']
    assert 'MULTI-VIEW REFERENCE RULES' in state['generation_prompt']
    assert f'INPUT: {2 + additional_count} reference images for ONE product.' in state['generation_prompt']
    assert state['metrics']['inference_requests'] == 7
    assert [image.tobytes() for image in extras] == before
    warm = pipeline.build_pipeline().invoke({'image':original, 'additional_images':extras})
    assert warm['analysis']['cache_hit']
    assert warm['metrics']['inference_requests'] == 3
    extras[0].paste('orange', (0, 0, 120, 180))
    changed = pipeline.build_pipeline().invoke({'image':original, 'additional_images':extras})
    assert not changed['analysis']['cache_hit']
    assert changed['metrics']['inference_requests'] == 4
    assert changed['classification']['cache_hit']


def test_too_many_views_fail_before_any_inference(fal_stub):
    original, _, calls, _, _ = fal_stub
    with pytest.raises(ValueError, match='Maximum 4 additional views'):
        pipeline.build_pipeline().invoke({'image':original, 'additional_images':[original] * 5})
    assert calls == []


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
    assert config['inference_requests_per_run'] == 7
    assert config['inference_requests_cached_run'] == 3
    assert config['inference_requests_by_kind']['garment'] == 7
    assert config['inference_requests_by_kind']['accessory'] == 6
    assert job['classification']['label'] == 'red sweater'
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


def test_multiview_single_click_api(fal_stub, monkeypatch):
    from fastapi.testclient import TestClient
    import server
    client = TestClient(server.app)
    pending = []
    monkeypatch.setattr(server.pool, 'submit', lambda *args: pending.append(args))
    original, _, _, _, _ = fal_stub
    main = ('file', ('main.png', pipeline.png_bytes(original), 'image/png'))
    extra = ('additional_files', ('detail.png', pipeline.png_bytes(Image.new('RGB', (160,160), 'pink')), 'image/png'))
    response = client.post('/api/studio', files=[main, extra])
    assert response.status_code == 202
    job_id = response.json()['job_id']
    try:
        task, *args = pending.pop()
        task(*args)
        job = client.get('/api/jobs/' + job_id).json()
        assert job['result_url'] and job['input_count'] == 2
        assert job['analysis']['view_count'] == 2
        assert any(image['id'] == 'original_2' for image in job['images'])
        assert job['metrics']['inference_requests'] == 7
        assert client.get('/api/config').json()['inputs']['max_photos'] == 5
    finally:
        with server.lock:
            server.jobs.pop(job_id)
    assert client.post('/api/studio', files=[main, *[extra] * 5]).status_code == 400
    bad = ('additional_files', ('bad.png', b'bad', 'image/png'))
    assert client.post('/api/studio', files=[main, bad]).status_code == 400
    unsupported = ('additional_files', ('bad.gif', b'bad', 'image/gif'))
    assert client.post('/api/studio', files=[main, unsupported]).status_code == 400
    assert pending == []


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
        'original', 'target', 'mask', 'segmented', 'color', 'reference', 'description']
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
    assert images[calls[4][1]['image_urls'][0]].tobytes() == expected_detail.tobytes()
    assert seen[0].tobytes() == corrected.tobytes()
    assert 'second global' in calls[4][1]['prompt']
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
    assert calls == [(pipeline.BIREF_MODEL, {'image_url':'https://generated', 'output_format':'png', 'refine_foreground':True, 'sync_mode':True})]
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
        if model == pipeline.DESCRIPTION_MODEL and 'Identify ONE main product' not in arguments.get('prompt', ''):
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
        if model == pipeline.DESCRIPTION_MODEL and 'Identify ONE main product' not in arguments.get('prompt', ''):
            calls.append((model, arguments))
            raise TimeoutError('Analysis deadline exceeded')
        return stub(model, arguments)
    monkeypatch.setattr(pipeline, 'call', timed_out)
    monkeypatch.setenv('FAL_ANALYSIS_REQUIRED', 'false')
    events = []
    with pytest.raises(TimeoutError):
        pipeline.build_pipeline(lambda **event: events.append(event)).invoke({'image': original})
    assert len(calls) == 4
    assert events[-1]['execution_trace'][-1]['id'] == 'describe'
    assert events[-1]['execution_trace'][-1]['status'] == 'error'
    assert not any(model == pipeline.RESTYLE_MODEL for model, _ in calls)


def test_analysis_timeout_strict_mode_stops_generation(fal_stub, monkeypatch):
    original, _, calls, _, _ = fal_stub
    stub = pipeline.call
    def timed_out(model, arguments):
        if model == pipeline.DESCRIPTION_MODEL and 'Identify ONE main product' not in arguments.get('prompt', ''):
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
    entry = state['execution_trace'][4]
    assert entry['model'] == pipeline.DESCRIPTION_MODEL
    assert entry['status'] == 'cached'
    assert entry['details']['model_invoked'] is False
    assert entry['run_id'] == 'cached-run'
    assert state['execution_trace'][5]['model'] == 'fal-ai/nano-banana-2/edit'
    assert state['execution_trace'][6]['model'] == 'fal-ai/nano-banana-2/edit'
    assert sum(model == 'fal-ai/nano-banana-2/edit' for model, _ in calls) == 4  # two per garment
    assert sum(model == pipeline.DESCRIPTION_MODEL for model, _ in calls) == before
