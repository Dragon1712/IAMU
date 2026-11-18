"""Semantic fingerprints for V1.2 experiments, their checkpoints and caches."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

RUNTIME_ARGS = frozenset({'out', 'resume', 'device', 'workers', 'validation_workers',
                          'state_every_rounds', 'dispatch_coordinator'})
# Worker count is an execution parameter *only* under episode-keyed frozen-round
# actor RNG and ordered consumption. Reproducibility contracts still record it.
CODE_PATHS = (
    'iamu_dynamic/core.py', 'iamu_dynamic/rl_env_v11.py',
    'iamu_dynamic/rl_ddqn_v11.py', 'iamu_dynamic/rl_experiment_v11.py',
    'v11_protocol/train_v11.py', 'v11_protocol/indexed_replay.py',
    'v11_protocol/launch_four_gpu.py', 'v11_protocol/margin_scan.py',
    'v11_protocol/audit_candidates.py', 'v11_protocol/replay_reasons.py',
    'v11_protocol/fixed_label_probe.py',
    'v11_protocol/v11_actors.py', 'v11_protocol/rounds.py',
    'v11_protocol/validation_v11.py', 'v11_protocol/research_v12.py',
    'v11_protocol/experiment_fingerprint_v12.py',
    'v11_protocol/recovery_v12.py', 'v11_protocol/dispatch_v12.py',
    'parallel_recovery_v081.py',
    'v11_protocol/build_decision_bank.py', 'v11_protocol/evaluate_qgap_bank.py',
    'v11_protocol/counterfactual_decision_probe.py', 'v11_protocol/margin_scan_v12.py',
    'v11_protocol/replay_tail_audit_v12.py', 'tests_v11/test_research_v12.py',
    'tests_v11/test_replay_snapshot_v12.py', 'tests_v11/test_real_env_v12.py',
    'tests_v11/test_recovery_v12.py', 'tests_v11/test_dispatch_v12.py',
    'tests_v11/test_qgap_v12.py', 'tests_v11/test_fingerprint_v12.py',
)


def digest_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def digest_json(obj):
    return digest_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), default=str).encode('utf-8'))


def _git(root, *args):
    try:
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def fingerprint(args, root):
    """root = iamu_d3dqn directory; never substitute paths for data contents."""
    root = Path(root).resolve()
    argdict = dict(vars(args) if not isinstance(args, dict) else args)
    scientific = {k: v for k, v in argdict.items() if k not in RUNTIME_ARGS}
    runtime = {k: v for k, v in argdict.items() if k in RUNTIME_ARGS}
    hashes = {}
    for rel in CODE_PATHS:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f'Cannot fingerprint missing code: {path}')
        hashes[rel] = digest_bytes(path.read_bytes())
    case_hashes = {}
    for key in ('train_cases', 'val_cases'):
        case_hashes[key] = [(str(p), digest_bytes(Path(p).read_bytes()))
                            for p in argdict.get(key, [])]
    repo_root = root.parent
    status = _git(repo_root, 'status', '--porcelain', '--untracked-files=no')
    revision = _git(repo_root, 'rev-parse', 'HEAD')
    diff = _git(repo_root, 'diff', '--binary', 'HEAD', '--', 'iamu_d3dqn')
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version, cuda_version = None, None
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = None
    return dict(schema='v12-scientific-manifest-v1',
                scientific_config=scientific,
                runtime_config=runtime,
                code_sha256=hashes,
                data_sha256=case_hashes,
                task_list_sha256=None,
                git=dict(commit=revision, dirty=bool(status) if status is not None else None,
                         tracked_diff_sha256=digest_bytes((diff or '').encode('utf8'))),
                software=dict(python=sys.version, cpu_count=__import__('os').cpu_count(),
                              gpu_names=([torch.cuda.get_device_name(i)
                                          for i in range(torch.cuda.device_count())]
                                         if torch_version is not None and torch.cuda.is_available() else []),
                              pytorch=torch_version,
                              cuda=cuda_version, numpy=numpy_version,
                              platform=platform.platform()))


def semantic_digest(manifest):
    return digest_json({k: manifest[k] for k in ('scientific_config', 'code_sha256',
                        'data_sha256', 'task_list_sha256')})


def compare_for_resume(previous, current):
    """Strictly compare scientific semantics; paths, devices and workers are tracked.

    Device/worker changes are allowed only with an explicit comparability warning
    in runtime logs; bitwise deterministic replay is not promised across them.
    """
    if previous.get('schema') != current.get('schema'):
        raise ValueError('Legacy or incompatible manifest schema: strict V1.2 resume refused')
    a, b = semantic_digest(previous), semantic_digest(current)
    if a != b:
        differing = [k for k in ('scientific_config', 'code_sha256',
                                  'data_sha256', 'task_list_sha256')
                     if previous.get(k) != current.get(k)]
        raise ValueError(f'Resume scientific fingerprint mismatch: {differing}')
    return previous.get('runtime_config') != current.get('runtime_config')
