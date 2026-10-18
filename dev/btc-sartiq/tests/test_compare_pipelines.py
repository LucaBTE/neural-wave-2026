import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def comparison():
    source = Path(__file__).resolve().parents[2] / 'compare_pipelines.py'
    spec = importlib.util.spec_from_file_location('compare_pipelines_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path, targets=True):
    rows = []
    for index in range(1, 7):
        identity = f't{index:04}'
        folder = tmp_path / 'train' / identity
        folder.mkdir(parents=True)
        (folder / 'input_1.jpg').touch()
        (folder / 'input_2.jpg').touch()
        (folder / 'target.jpg').touch()
        row = {'pair_id':identity, 'view':'front',
               'inputs':[f'train/{identity}/input_1.jpg', f'train/{identity}/input_2.jpg']}
        if targets:
            row['target'] = f'train/{identity}/target.jpg'
        rows.append(row)
    path = tmp_path / 'train/pairs.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_selects_exact_same_first_inputs_and_keeps_targets_separate(comparison, tmp_path):
    selected = comparison.select_inputs(manifest(tmp_path), tmp_path, 5)
    assert [r['pair_id'] for r in selected] == [f't{i:04}' for i in range(1,6)]
    for row in selected:
        assert row['inputs'] == [str(tmp_path / 'train' / row['pair_id'] / 'input_1.jpg')]
        assert row['target'] not in row['inputs']


def test_missing_targets_fail_before_any_generation(comparison, tmp_path, monkeypatch):
    manifest(tmp_path, targets=False)
    monkeypatch.setattr(comparison, 'run_command', lambda *a, **k:pytest.fail('Unexpected subprocess'))
    with pytest.raises(ValueError, match='official scoring needs its target'):
        comparison.main(['--dataset',str(tmp_path)])


def test_generation_only_does_not_invent_scores(comparison, tmp_path):
    selected = comparison.select_inputs(manifest(tmp_path, targets=False), tmp_path, 5, generate_only=True)
    assert all('target' not in row for row in selected)


def test_baseline_reuse_checks_source_paths_and_missing_predictions(comparison, tmp_path):
    selected = comparison.select_inputs(manifest(tmp_path), tmp_path, 5)
    previous = tmp_path / 'previous'
    previous.mkdir()
    (previous / 'selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
    for row in selected:
        source = previous / 'predictions/baseline' / row['pair_id'] / 'front.png'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'prediction')
    files = comparison.prior_baseline_files(previous, selected)
    assert len(files) == 5 and files[0][1] == Path('t0001/front.png')
    changed = [{**selected[0], 'inputs':[str(tmp_path / 'different.jpg')]}]
    with pytest.raises(ValueError, match='different inputs/target'):
        comparison.prior_baseline_files(previous, changed)
    files[0][0].unlink()
    with pytest.raises(ValueError, match='prediction missing or invalid'):
        comparison.prior_baseline_files(previous, selected)


def test_baseline_reuse_skips_baseline_generation_and_copies_predictions(comparison, tmp_path, monkeypatch):
    selected = comparison.select_inputs(manifest(tmp_path), tmp_path, 5)
    previous = tmp_path / 'previous'
    previous.mkdir()
    (previous / 'selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
    for row in selected:
        source = previous / 'predictions/baseline' / row['pair_id'] / 'front.png'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original prediction')
    monkeypatch.setattr(comparison, 'DEV', tmp_path)
    monkeypatch.setattr(comparison, 'preflight', lambda *args:None)
    commands = []
    monkeypatch.setattr(comparison, 'run_command', lambda command, **kwargs:commands.append(list(map(str,command))) or 0)
    output = tmp_path / 'new'
    assert comparison.main(['--dataset',str(tmp_path),'--generate-only',
                            '--baseline-from',str(previous),'--out',str(output)]) == 0
    assert len(commands) == 1
    assert commands[0][2].endswith('benchmark_studio.py')
    for row in selected:
        assert (output / 'predictions/baseline' / row['pair_id'] / 'front.png').read_bytes() == b'original prediction'


@pytest.mark.parametrize('partial', [False, True])
def test_complete_comparison_uses_one_manifest_and_official_aggregates(comparison, tmp_path, monkeypatch, partial):
    manifest(tmp_path)
    output = tmp_path / 'comparison'
    monkeypatch.setattr(comparison, 'DEV', tmp_path)
    monkeypatch.setattr(comparison, 'preflight', lambda *args:None)
    commands = []
    def run(command, allow_partial=False):
        args = list(map(str, command))
        commands.append(args)
        if str(comparison.DEV / 'eval_val.py') in args:
            team = args[args.index('--team')+1]
            folder = Path(args[args.index('--out')+1])
            folder.mkdir(parents=True)
            row = dict(team=team,score=80 if team=='baseline' else 75,colour=.8,
                lpips=.6,clip=.9,geometry=.85,ssim=.8,pairs_scored=4,pairs_missing=1,
                n_review=0,n_invalid=0)
            with (folder/'leaderboard.csv').open('w',newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            scores = [dict(garment=f't{i:04}',view='front',score=0 if i==5 else row['score'],
                status='missing' if i==5 else 'ok', missing=i==5) for i in range(1,6)]
            (folder/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores))
        return 1 if partial and allow_partial else 0
    monkeypatch.setattr(comparison, 'run_command', run)
    assert comparison.main(['--dataset',str(tmp_path),'--out',str(output)]) == int(partial)
    assert len(commands) == 4
    baseline_manifest = commands[0][commands[0].index('--manifest')+1]
    studio_manifest = commands[1][commands[1].index('--manifest')+1]
    assert baseline_manifest == studio_manifest == str(output / 'selected.jsonl')
    assert all(command[command.index('--pairs')+1] == baseline_manifest for command in commands[2:])
    rows = list(csv.DictReader((output/'comparison.csv').open()))
    # Copy official aggregate rather than recomputing from individual pair scores.
    assert [r['score'] for r in rows] == ['80','75']
    assert all(r['pairs_missing']=='1' for r in rows)
    individual = list(csv.DictReader((output/'per_garment.csv').open()))
    assert len(individual) == 5 and individual[-1]['baseline_score'] == '0'


def test_existing_output_is_refused_without_overwriting(comparison, tmp_path, monkeypatch):
    manifest(tmp_path)
    monkeypatch.setattr(comparison, 'DEV', tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        comparison.main(['--dataset',str(tmp_path),'--out',str(output)])


def test_dry_run_does_not_call_models_or_create_output(comparison, tmp_path, monkeypatch):
    manifest(tmp_path)
    output = tmp_path/'unused'
    monkeypatch.setattr(comparison, 'run_command', lambda *a, **k:pytest.fail('Unexpected subprocess'))
    assert comparison.main(['--dataset',str(tmp_path),'--out',str(output),'--dry-run']) == 0
    assert not output.exists()


def test_studio_worker_passes_only_first_input_and_records_failures(tmp_path, monkeypatch):
    from PIL import Image
    script = Path(__file__).resolve().parents[1] / 'scripts/benchmark_studio.py'
    spec = importlib.util.spec_from_file_location('benchmark_studio_test', script)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    Image.new('RGB',(10,10),'red').save(tmp_path/'first.png')
    Image.new('RGB',(10,10),'blue').save(tmp_path/'second.png')
    rows = [{'pair_id':f't{i:04}','view':'front','inputs':[str(tmp_path/name)],
             'target':'THIS_TARGET_MUST_NOT_BE_READ.jpg'}
            for i,name in enumerate(['first.png','second.png'],start=1)]
    selected = tmp_path/'selected.jsonl'
    selected.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    states = []
    snapshots = []
    class Graph:
        def invoke(self, state):
            states.append(state)
            if len(states) == 2:
                raise RuntimeError('Simulated failure')
            snapshots[-1]('composed', 'Studio composition', Image.new('RGB', (20,30), 'gray'))
            return {'result':Image.new('RGB',(20,30),'white'),
                'mask':Image.new('L', (10,10), 255), 'result_mask':Image.new('L', (20,30), 255),
                **{key:{} for key in ('classification','target_selection','quality','metrics','analysis')},
                'execution_trace':[], 'generation_prompt':'generation','pressing_prompt':'pressing',
                'reference_policy':'source_photographs'}
    def build(*args):
        snapshots.append(args[1])
        return Graph()
    monkeypatch.setattr(worker, 'build_pipeline', build)
    monkeypatch.setattr(worker, 'load_dotenv', lambda *args:None)
    monkeypatch.setattr(worker, 'generation_settings', lambda:None)
    monkeypatch.setattr(worker, 'analysis_settings', lambda:None)
    assert worker.main(['--manifest',str(selected),'--preds',str(tmp_path/'preds'),
                        '--diagnostics',str(tmp_path/'diagnostics')]) == 1
    assert all(set(state) == {'image','additional_images'} for state in states)
    assert all(state['additional_images']==[] for state in states)
    assert states[0]['image'].getpixel((0,0)) == (255,0,0)
    assert (tmp_path/'preds/studio/t0001/front.png').is_file()
    assert Image.open(tmp_path/'diagnostics/t0001/front_before_color.png').getpixel((0,0)) == (128,128,128)
    assert Image.open(tmp_path/'diagnostics/t0001/front_source_mask.png').size == (10,10)
    assert Image.open(tmp_path/'diagnostics/t0001/front_result_mask.png').size == (20,30)
    report = json.loads((tmp_path/'diagnostics/t0001/front.json').read_text())
    assert report['generation_reference_policy'] == 'source_photographs'
    assert not (tmp_path/'preds/studio/t0002/front.png').exists()
    error = json.loads((tmp_path/'diagnostics/t0002/front_error.json').read_text())
    assert error['status'] == 'failed'
