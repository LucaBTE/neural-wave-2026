"""Bounded process-local reuse of validated work, including concurrent callers."""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import Lock
from time import monotonic


class ReuseCache:
    def __init__(self, ttl=3600, max_entries=128, max_bytes=None, clock=monotonic):
        self.ttl, self.max_entries, self.max_bytes = ttl, max_entries, max_bytes
        self.clock = clock
        self._entries = OrderedDict()
        self._pending = {}
        self._bytes = 0
        self._epoch = 0
        self._lock = Lock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._epoch += 1

    @staticmethod
    def _setting(value):
        return value() if callable(value) else value

    def get_or_compute(self, key, work, weight=lambda value: 1):
        with self._lock:
            now = self.clock()
            for expired in [k for k, (deadline, _, _) in self._entries.items() if deadline <= now]:
                self._bytes -= self._entries.pop(expired)[2]
            if key in self._entries:
                self._entries.move_to_end(key)
                return deepcopy(self._entries[key][1]), 'cached'
            pending_key = (self._epoch, key)
            future = self._pending.get(pending_key)
            owner = future is None
            if owner:
                future = self._pending[pending_key] = Future()
        if not owner:
            # Shares only the existing request; failures propagate without retry.
            return deepcopy(future.result()), 'shared'
        try:
            value = work()  # The caller validates its result before returning it.
            size = weight(value)
            stored = deepcopy(value)
        except BaseException as exc:
            with self._lock:
                self._pending.pop(pending_key, None)
            future.set_exception(exc)
            raise
        with self._lock:
            if pending_key[0] == self._epoch and (self.max_bytes is None or size <= self.max_bytes):
                self._entries[key] = (self.clock() + self._setting(self.ttl), stored, size)
                self._bytes += size
                while (len(self._entries) > self._setting(self.max_entries) or
                       self.max_bytes is not None and self._bytes > self.max_bytes):
                    self._bytes -= self._entries.popitem(last=False)[1][2]
            self._pending.pop(pending_key, None)
        future.set_result(value)
        return deepcopy(value), 'computed'
