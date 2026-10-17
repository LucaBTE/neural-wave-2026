import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import request_reuse
from request_reuse import ReuseCache


@pytest.mark.parametrize('fail', [False, True])
def test_concurrent_callers_share_work_and_failure_without_retry(monkeypatch, fail):
    started, waiting, release = threading.Event(), threading.Event(), threading.Event()
    class ObservedFuture(Future):
        def result(self, *args, **kwargs):
            waiting.set()
            return super().result(*args, **kwargs)
    monkeypatch.setattr(request_reuse, 'Future', ObservedFuture)
    cache, calls = ReuseCache(), []
    def work():
        calls.append(1)
        started.set()
        assert release.wait(3)
        if fail:
            raise RuntimeError('Provider failed')
        return {'nested': [1]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cache.get_or_compute, 'same', work)
        try:
            assert started.wait(2)
            follower = pool.submit(cache.get_or_compute, 'same', work)
            assert waiting.wait(2)
        finally:
            release.set()
        if fail:
            for future in (owner, follower):
                with pytest.raises(RuntimeError, match='Provider failed'):
                    future.result(timeout=3)
            assert not cache
        else:
            first, second = owner.result(timeout=3), follower.result(timeout=3)
            assert first[1] == 'computed' and second[1] == 'shared'
            first[0]['nested'].append(2)
            assert second[0] == {'nested': [1]}
            assert cache.get_or_compute('same', work) == ({'nested':[1]}, 'cached')
    assert len(calls) == 1
    if fail:
        assert cache.get_or_compute('same', lambda: 'recovered') == ('recovered', 'computed')


def test_cache_enforces_ttl_lru_and_memory_budget():
    clock = [0]
    cache = ReuseCache(ttl=10, max_entries=2, max_bytes=5, clock=lambda: clock[0])
    def get(key, value):
        return cache.get_or_compute(key, lambda: value, weight=len)
    assert get('a', 'aa')[1] == 'computed'
    get('b', 'bb')
    assert get('a', 'aa')[1] == 'cached'
    get('c', 'cc')  # b is least recently used
    assert len(cache) == 2
    assert get('b', 'bb')[1] == 'computed'
    clock[0] = 11
    assert get('b', 'bb')[1] == 'computed'
    assert len(cache) == 1
    get('big', 'oversized')
    assert get('big', 'oversized')[1] == 'computed'
    assert len(cache) == 1


def test_clear_during_running_work_does_not_repopulate_old_cache():
    cache = ReuseCache()
    def work():
        cache.clear()
        return 'old'
    assert cache.get_or_compute('same', work) == ('old', 'computed')
    assert not cache
    assert cache.get_or_compute('same', lambda: 'new') == ('new', 'computed')
