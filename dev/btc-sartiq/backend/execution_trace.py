"""Runtime provenance for each step, without logging prompts, photos or keys."""
import logging
import platform
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger('sartiq.pipeline')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@lru_cache(maxsize=32)
def library_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return 'unknown'


def begin_step(key, label, model, libraries, run_id=None, **details):
    entry = {'id': key, 'label': label, 'model': model,
             'libraries': {name: library_version(package) for name, package in libraries.items()},
             'python': platform.python_version(), 'status': 'running', 'seconds': None,
             'details': details}
    if run_id:
        entry['run_id'] = run_id
    log_step(entry)
    return entry


def end_step(entry, seconds, status='complete', **details):
    completed = {**entry, 'status': status, 'seconds': round(seconds, 3),
                 'details': {**entry['details'], **details}}
    log_step(completed)
    return completed


def log_step(entry):
    libraries = ', '.join(f'{name}=={value}' for name, value in entry['libraries'].items())
    logger.info('[%s] %s | %s | model=%s | libraries=%s | duration=%s | details=%s',
                entry.get('run_id', 'standalone'), entry['label'], entry['status'],
                entry['model'] or 'local', libraries,
                '-' if entry['seconds'] is None else f"{entry['seconds']:.3f}s", entry['details'])
