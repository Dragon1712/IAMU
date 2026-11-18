"""Atomic, restartable learner state for V0.8.1 parallel DQN.

A resumed run preserves completed episode records, replay contents and replay
RNG, optimizer, online/target network, main RNGs and pending frozen validation
snapshots. Worker process scheduling/actor RNG cannot be bitwise resumed after
interruption; interrupted in-flight episodes are safely requeued. Untrusted
PyTorch checkpoint files must never be loaded with weights_only=False.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,tmp)
    os.replace(tmp,path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,ensure_ascii=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def config_digest(config: dict) -> str:
    """Ignore only flags that change logging/recovery, not learning semantics."""
    args=dict(config['training_args'])
    for field in ('out','resume','worker_timeout','max_idle_seconds',
                  'state_every','log_every','validation_progress_every'):
        args.pop(field,None)
    # New V0.9 flag must not invalidate pre-V0.9 DDQN recovery states.
    # D3QN remains in the digest, preventing cross-architecture recovery.
    if args.get('q_network','ddqn')=='ddqn':
        args.pop('q_network',None)
    # Allow CPU/GPU placement & worker counts to change while resuming, but
    # this makes sampling order different; reproducibility is statistical.
    for field in ('device','workers','validation_workers'):
        args.pop(field,None)
    return hashlib.sha256(json.dumps(args,sort_keys=True,default=str).encode()).hexdigest()


def _cpu_tree(value: Any):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def _optimizer(learner):
    op=getattr(learner,'optimizer',None)
    return op if op is not None else learner.opt


def save_state(path: Path, *, learner, replay, config: dict,
               train_rows: list, val_rows: list, validations: dict,
               best_val: float, gradient_credit: float) -> None:
    payload=dict(version=1,config_digest=config_digest(config),
        config=config,online=_cpu_tree(learner.online.state_dict()),
        target=_cpu_tree(learner.target.state_dict()),
        optimizer=_cpu_tree(_optimizer(learner).state_dict()),
        gradient_steps=int(learner.gradient_steps),
        agent_rng=learner.rng.getstate(),
        replay_items=list(replay.items),replay_rng=replay.rng.getstate(),
        train_rows=list(train_rows),val_rows=list(val_rows),
        validations=dict(validations),best_val=float(best_val),
        gradient_credit=float(gradient_credit),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.random.get_rng_state(),
        torch_cuda_rng=(torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() and torch.cuda.is_initialized()
                         else None))
    _atomic_torch_save(payload,path)


def load_state(path: Path, *, learner, replay, config: dict):
    if not path.is_file():
        raise FileNotFoundError(f'Recovery checkpoint missing: {path}')
    # Local-only trusted checkpoint: contains Python Transition objects/NumPy.
    item=torch.load(path,map_location='cpu',weights_only=False)
    if item.get('version')!=1 or item['config_digest']!=config_digest(config):
        raise ValueError('Resume config differs from saved training config')
    learner.online.load_state_dict(item['online'])
    learner.target.load_state_dict(item['target'])
    _optimizer(learner).load_state_dict(item['optimizer'])
    learner.gradient_steps=int(item['gradient_steps'])
    learner.rng.setstate(item['agent_rng'])
    replay.items.clear();replay.items.extend(item['replay_items'])
    replay.rng.setstate(item['replay_rng'])
    random.setstate(item['python_rng'])
    np.random.set_state(item['numpy_rng'])
    torch.random.set_rng_state(item['torch_rng'])
    if item.get('torch_cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(item['torch_cuda_rng'])
    return item
