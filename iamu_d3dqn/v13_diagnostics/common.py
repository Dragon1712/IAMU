"""Dataset integrity, public-state descriptors, balanced jobs, and local trusted IO."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

from . import PROTOCOL


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def scenario_digest(scenario):
    """Hash the *complete pre-generated external tape*, not merely RNG seed."""
    from iamu_dynamic.core import scenario_as_dict
    return stable_digest(scenario_as_dict(scenario))


def atomic_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(tmp,p)


def write_csv(path,rows,columns=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows and not columns: return
    cols=columns or list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,cols,extrasaction='ignore');w.writeheader();w.writerows(rows)


def sha_code():
    base=Path(__file__).resolve().parents[1]
    roots=[base/'v13_diagnostics',base/'v12_protocol',base/'iamu_dynamic/rl_env_v11.py',
           base/'iamu_dynamic/rl_ddqn_v11.py',base/'iamu_dynamic/core.py']
    paths=[]
    for r in roots:
        if r.is_dir():paths.extend(sorted(r.glob('*.py')))
        elif r.is_file():paths.append(r)
    return {str(p.relative_to(base)):digest(p) for p in paths}


def provenance(args,paths=None,inputs=None):
    base=Path(__file__).resolve().parents[1]
    try:
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=base,capture_output=True,text=True,
                           timeout=5,check=True).stdout.strip()
        dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=base,capture_output=True,
                                  text=True,timeout=5,check=True).stdout.strip())
    except (OSError,subprocess.SubprocessError):git=None;dirty=None
    return dict(protocol=PROTOCOL,git_head=git,git_dirty=dirty,
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                code_sha256=sha_code(),
                case_sha256={str(Path(p).resolve()):digest(p) for p in (paths or [])},
                input_sha256={str(Path(p).resolve()):digest(p) for p in (inputs or [])})


def validate_case_split(train,probe):
    a={digest(p) for p in train};b={digest(p) for p in probe}
    if a&b:raise ValueError('Training and probe case files overlap by content')


def balanced_jobs(n, cases, scenarios, budgets, seed, first_episode=1, first_external_seed=600000):
    """Deterministically balance 3x3 conditions; no synthetic duplicated episodes."""
    if n<1 or not cases:raise ValueError('Need cases and positive episode count')
    grid=[(sc,int(b)) for sc in scenarios for b in budgets]
    rng=random.Random(seed)
    pairs=(grid*(n//len(grid)))+rng.sample(grid,n%len(grid))
    rng.shuffle(pairs)
    return [dict(episode_id=first_episode+i,case=str(cases[rng.randrange(len(cases))]),
                 scenario=sc,budget=b,external_seed=first_external_seed+i,
                 uncertainty=8.) for i,(sc,b) in enumerate(pairs)]


def action_descriptor(env,action):
    """Physical meaning, not candidate slot alone; no hidden arrival data."""
    if action==0:return dict(kind='keep',vessel_id=None,old_assignment=None,new_assignment=None)
    vessel=int(env._candidate_ids[action-1])
    before=env.plan[vessel].as_dict()
    after=env._candidates[action-1][vessel].as_dict()
    return dict(kind='edit',vessel_id=vessel,old_assignment=before,
                new_assignment=after,estimated_gain=float(env._candidate_data[action-1]['gain']))


def state_id(episode_id,index):
    return f'{int(episode_id)}:{int(index)}'


def save_trusted(path,obj):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    torch.save(obj,tmp);os.replace(tmp,p)


def load_trusted(path):
    import torch
    # Never load untrusted pickle files. Caller must establish local provenance.
    return torch.load(path,map_location='cpu',weights_only=False)


def load_model(path):
    from v12_protocol.agent import V12Agent
    model,meta=V12Agent.from_checkpoint(str(path),device='cpu')
    model.online.eval();model.target.eval()
    return model


def verify_v12_scientific_model(model,expected_representation=None,gamma=1.):
    """Reject a numerically runnable but scientifically mismatched checkpoint."""
    if expected_representation and model.representation!=expected_representation:
        raise ValueError(f'Expected {expected_representation} model, got '
                         f'{model.representation}')
    if model.q_network!='set_ddqn' or model.hidden_dim!=96:
        raise ValueError('Main V1.2 uses Set-aware DDQN with hidden_dim=96')
    if abs(model.gamma-gamma)>1e-12:
        raise ValueError(f'Checkpoint gamma {model.gamma} != requested {gamma}')
    if abs(model.execution_margin)>1e-12 or abs(model.bootstrap_margin)>1e-12:
        raise ValueError('Main protocol requires zero execution/bootstrap margins')
    if abs(model.reward_scale-100.)>1e-12:
        raise ValueError('Main V1.2 protocol uses reward_scale=100')


def make_env(case,scenario,budget,uncertainty,updates=2):
    from iamu_dynamic import read_case,generate_scenario
    from iamu_dynamic.rl_env_v11 import IAMUEventEnv
    vessels,port=read_case(case)
    sc=generate_scenario(vessels,port,label=scenario['name'],seed=scenario['seed'],
                         uncertainty=uncertainty,updates_per_vessel=updates)
    return IAMUEventEnv(sc,budget=budget,candidate_limit=4,decision_clock='union',
                        period=12.,reward_scale=100.,candidate_screen='positive',
                        min_gain=.1,budget_features='absolute',value_features='global')
