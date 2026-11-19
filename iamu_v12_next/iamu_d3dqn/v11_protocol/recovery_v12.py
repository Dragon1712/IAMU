"""V1.2-only strict round-boundary recovery.

Deliberately separate from legacy parallel_recovery_v081.py, preserving all
historical V0.8/V1.1 loaders. Only load self-generated, locally trusted states:
Python Transition objects require torch.load(weights_only=False).
"""
from __future__ import annotations
import os
from pathlib import Path
import random
import numpy as np
import torch
from v11_protocol.experiment_fingerprint_v12 import semantic_digest


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def save_state(path, *, learner, replay, config, train_rows, val_rows,
               validations, best_val, gradient_credit, protocol_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(replay.actionable) + len(replay.forced) != len(replay):
        raise AssertionError('Replay pools out of sync before recovery snapshot')
    snapshot = dict(version=2, config=config, config_digest=semantic_digest(config),
                    online=cpu_tree(learner.online.state_dict()),
                    target=cpu_tree(learner.target.state_dict()),
                    optimizer=cpu_tree(learner.optimizer.state_dict()),
                    gradient_steps=int(learner.gradient_steps),
                    agent_rng=learner.rng.getstate(),
                    replay_state=replay.state_dict(),
                    replay_items=list(replay.items),
                    replay_rng=replay.rng.getstate(),
                    train_rows=list(train_rows), val_rows=list(val_rows),
                    validations=dict(validations), best_val=float(best_val),
                    gradient_credit=float(gradient_credit),
                    protocol_state=dict(protocol_state or {}),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.random.get_rng_state(),
                    torch_cuda_rng=(torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available() and torch.cuda.is_initialized() else None))
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(snapshot,tmp)
    os.replace(tmp,path)


def load_state(path, *, learner, replay, config):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('version')!=2 or 'replay_state' not in saved:
        raise ValueError('Cannot strictly resume legacy training state; use historical trainer or start new V1.2 run')
    if saved.get('config_digest')!=semantic_digest(config):
        raise ValueError('Resume failed: scientific configuration/code/data/task mismatch')
    learner.online.load_state_dict(saved['online'])
    learner.target.load_state_dict(saved['target'])
    learner.optimizer.load_state_dict(saved['optimizer'])
    learner.gradient_steps=int(saved['gradient_steps'])
    learner.rng.setstate(saved['agent_rng'])
    replay.load_state_dict(saved['replay_state'])
    if len(replay)!=len(saved['replay_items']):
        raise AssertionError('Replay restored with incorrect item count')
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.random.set_rng_state(saved['torch_rng'])
    if saved.get('torch_cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['torch_cuda_rng'])
    return saved
