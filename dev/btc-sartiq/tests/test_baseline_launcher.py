import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / 'baseline_nanobanana.py'
    spec = importlib.util.spec_from_file_location('dev_baseline_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'BACKEND', tmp_path)
    for name in ('FAL_KEY', 'FAL_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    yield module
    # dotenv writes directly to os.environ; remove test values before pytest
    # restores any original shell values recorded by monkeypatch.
    for name in ('FAL_KEY', 'FAL_API_KEY'):
        module.os.environ.pop(name, None)


def test_backend_fal_api_key_is_aliased_without_exposing_it(launcher, tmp_path, capsys):
    (tmp_path / '.env').write_text('FAL_API_KEY="fake-backend-key"\n')
    launcher.load_backend_environment()
    assert launcher.os.environ['FAL_KEY'] == 'fake-backend-key'
    assert capsys.readouterr() == ('', '')


def test_explicit_shell_key_takes_precedence(launcher, monkeypatch, tmp_path):
    (tmp_path / '.env').write_text('FAL_KEY=fake-file-key\nFAL_API_KEY=fake-alias\n')
    monkeypatch.setenv('FAL_KEY', 'fake-shell-key')
    launcher.load_backend_environment()
    assert launcher.os.environ['FAL_KEY'] == 'fake-shell-key'


def test_original_baseline_arguments_are_forwarded(launcher, monkeypatch, tmp_path):
    original = tmp_path / 'baseline_nanobanana.py'
    original.touch()
    monkeypatch.setattr(launcher, 'BASELINE', original)
    monkeypatch.setattr(launcher.sys, 'path', list(launcher.sys.path))
    monkeypatch.setattr(launcher.sys, 'argv', ['dev/baseline_nanobanana.py', '--limit', '1', '--dry-run'])
    invoked = []
    monkeypatch.setattr(launcher.runpy, 'run_path', lambda path, **kwargs: invoked.append((path, kwargs)))
    launcher.main()
    assert invoked == [(str(original), {'run_name':'__main__'})]
    assert launcher.sys.argv == [str(original), '--limit', '1', '--dry-run']
    assert launcher.sys.dont_write_bytecode


def test_score_uses_official_evaluator_with_same_subset_after_success(launcher, monkeypatch, tmp_path):
    original = tmp_path / 'baseline_nanobanana.py'
    original.touch()
    monkeypatch.setattr(launcher, 'BASELINE', original)
    monkeypatch.setattr(launcher.sys, 'path', list(launcher.sys.path))
    args = ['--manifest','train/pairs.jsonl','--out','dev/predictions',
            '--team','btc','--limit','3','--score','--score-out','dev/scores/test']
    monkeypatch.setattr(launcher.sys, 'argv', ['dev/baseline_nanobanana.py', *args])
    events = []
    scorer = SimpleNamespace(use_scorer_environment=lambda:None,
        prepare_scoring=lambda *args:events.append(('preflight', args)),
        main=lambda args:events.append(('score', args)) or 0)
    monkeypatch.setitem(launcher.sys.modules, 'eval_val', scorer)
    def generate(*args, **kwargs):
        events.append(('generate', launcher.sys.argv[1:]))
        raise SystemExit(0)
    monkeypatch.setattr(launcher.runpy, 'run_path', generate)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 0
    assert [event[0] for event in events] == ['preflight','generate','score']
    assert events[0][1] == ('train/pairs.jsonl', None, 3)
    assert '--score' not in events[1][1]
    assert events[2][1] == ['--pairs','train/pairs.jsonl','--preds','dev/predictions',
        '--team','btc','--out','dev/scores/test','--jobs','1','--limit','3']


def test_score_cannot_run_on_dry_run(launcher, monkeypatch):
    monkeypatch.setattr(launcher.sys, 'path', list(launcher.sys.path))
    monkeypatch.setattr(launcher.sys, 'argv', ['dev/baseline_nanobanana.py',
        '--manifest','train/pairs.jsonl','--out','dev/predictions','--team','btc','--dry-run','--score'])
    monkeypatch.setitem(launcher.sys.modules, 'eval_val',
                        SimpleNamespace(use_scorer_environment=lambda:None))
    with pytest.raises(SystemExit, match='cannot be combined'):
        launcher.main()


def load_evaluator():
    path = Path(__file__).resolve().parents[2] / 'eval_val.py'
    spec = importlib.util.spec_from_file_location('dev_eval_val', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_scoring_rejects_targetless_split_before_dependencies(tmp_path):
    manifest = tmp_path / 'inputs.jsonl'
    manifest.write_text('{"garment_id":"d001","inputs":["input.jpg"]}\n')
    with pytest.raises(SystemExit, match='needs pair_id/garment_id and target'):
        load_evaluator().prepare_scoring(str(manifest))


def test_missing_scorer_dependencies_fail_before_generation(tmp_path, monkeypatch):
    manifest = tmp_path / 'pairs.jsonl'
    target = tmp_path / 'target.jpg'
    target.touch()
    manifest.write_text('{"pair_id":"t0001","target":"target.jpg"}\n')
    evaluator = load_evaluator()
    monkeypatch.setattr(evaluator.importlib.util, 'find_spec', lambda _:None)
    with pytest.raises(SystemExit, match='No generation was started'):
        evaluator.prepare_scoring(str(manifest))
