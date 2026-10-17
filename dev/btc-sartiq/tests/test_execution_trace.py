import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from execution_trace import begin_step, end_step, library_version


def test_trace_prints_provenance_and_does_not_mutate_start_event():
    with patch('execution_trace.logger.info') as log:
        started = begin_step('render', 'Creating photo', 'fal-ai/test/edit', {'Pillow': 'pillow'},
                             'job-123', resolution='2K')
        completed = end_step(started, 1.2345)
        assert log.call_count == 2
        first = log.call_args_list[0]
        assert 'fal-ai/test/edit' in first.args
        assert any('Pillow==' in arg for arg in first.args if isinstance(arg, str))
    assert started['status'] == 'running'
    assert started['seconds'] is None
    assert completed['status'] == 'complete'
    assert completed['libraries']['Pillow'] == library_version('pillow')
    assert completed['seconds'] == 1.234
    assert completed['python']
    assert completed['details'] == {'resolution':'2K'}
