"""Keep comparison inputs target-free and reject unscorable runs before API work."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).resolve().parents[2]/'compare_pipelines.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def test_comparison_keeps_targets_separate_and_dry_run_has_no_side_effects(tmp_path,monkeypatch):
    product=tmp_path/'train/t0001';product.mkdir(parents=True)
    for name in ('input_1.jpg','input_2.jpg','target.jpg'):(product/name).touch()
    manifest=tmp_path/'train/pairs.jsonl'
    manifest.write_text(json.dumps({'pair_id':'t0001','inputs':['train/t0001/input_1.jpg','train/t0001/input_2.jpg'],'target':'train/t0001/target.jpg'})+'\n')
    selected=comparison.select_inputs(manifest,tmp_path,1)
    assert selected[0]['inputs']==[str(product/'input_1.jpg')]
    assert selected[0]['target'] not in selected[0]['inputs']
    monkeypatch.setattr(comparison,'run_command',lambda *a,**k:pytest.fail('Dry run launched a model'))
    output=tmp_path/'output'
    assert comparison.main(['--dataset',str(tmp_path),'--count','1','--out',str(output),'--dry-run'])==0
    assert not output.exists()


def test_missing_reference_fails_before_generation(tmp_path):
    product=tmp_path/'train/t0001';product.mkdir(parents=True);(product/'input_1.jpg').touch()
    manifest=tmp_path/'train/pairs.jsonl';manifest.write_text(json.dumps({'pair_id':'t0001','inputs':['train/t0001/input_1.jpg']})+'\n')
    with pytest.raises(ValueError,match='official scoring needs'):
        comparison.select_inputs(manifest,tmp_path,1)
