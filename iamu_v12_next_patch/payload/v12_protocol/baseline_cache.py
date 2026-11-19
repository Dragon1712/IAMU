"""Immutable, process-safe cache of V1.2 Keep/Greedy/Periodic validation rows.

Keys deliberately EXCLUDE the learned checkpoint, and INCLUDE physical source
bytes, scenario specification, candidate rules and dependency versions. This
module does not change the DDQN evaluator or the physical simulator.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import fcntl                    # The deployed research server is Linux.

SCHEMA = 'iamu-v12-static-baseline-v1'
BASELINES = ('keep', 'greedy', 'periodic')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), default=str)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def baseline_cache_key(case_bytes: bytes, env_config: dict,
                       task: dict, code_hashes: dict, versions: dict | None = None) -> str:
    """Every argument is part of the semantic cache key; no model hash."""
    payload = dict(schema=SCHEMA, case_sha256=hashlib.sha256(case_bytes).hexdigest(),
                   env_config=env_config, task=task, code_hashes=code_hashes,
                   dependency_versions=versions or {})
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def _verify(obj, key):
    if (not isinstance(obj, dict) or obj.get('schema') != SCHEMA or
            obj.get('key') != key):
        raise ValueError('Baseline cache version/key mismatch')
    rows = obj.get('rows')
    if (not isinstance(rows, list) or len(rows) != 3 or
            tuple(x.get('method') for x in rows) != BASELINES):
        raise ValueError('Baseline cache has incomplete or reordered rows')
    checksum = hashlib.sha256(canonical(rows).encode()).hexdigest()
    if obj.get('rows_sha256') != checksum:
        raise ValueError('Baseline cache checksum mismatch')
    return rows


def load_or_compute_baselines(cache_dir, key: str, compute_fn):
    """Return (three rows, hit). A file lock serializes writers/readers per key.

    Corrupt caches fail closed. Failed compute never publishes a partial file.
    Copy rows via JSON decoding so callers cannot mutate the cache's objects.
    """
    if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Cache key must be lowercase SHA-256')
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + '.json')
    lock = directory / (key + '.lock')
    with open(lock, 'a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if path.is_file():
                return _verify(json.loads(path.read_text(encoding='utf-8')), key), True
            rows = compute_fn()
            checksum = hashlib.sha256(canonical(rows).encode()).hexdigest()
            obj = dict(schema=SCHEMA, key=key, rows=rows,
                       rows_sha256=checksum, created_at=time.time())
            _verify(obj, key)
            temp = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                         dir=directory, prefix=key + '.', suffix='.tmp',
                         delete=False) as output:
                    temp = Path(output.name)
                    output.write(canonical(obj) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temp, path)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            return json.loads(canonical(rows)), False
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
