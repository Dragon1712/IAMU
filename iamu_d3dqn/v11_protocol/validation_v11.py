"""Frozen-worker, deterministic-order paired validation for v11 environment."""
from __future__ import annotations
import os
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from iamu_dynamic.rl_experiment_v11 import evaluate_paired,write_csv

_MODEL=None
_CFG=None

def _initializer(checkpoint, config):
    global _MODEL,_CFG
    import torch
    from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    _MODEL,_=DoubleDQNAgent.from_checkpoint(str(checkpoint),device='cpu')
    _MODEL.online.eval();_CFG=config

def _work(group):
    case,sc,seed,budget=group
    c=_CFG
    rr,summary=evaluate_paired(cases=[case],scenarios=[sc],seeds=[seed],budgets=[budget],
        policies=['keep','greedy','ddqn']+(['periodic'] if c['clock']=='union' else []),
        agent=_MODEL,candidate_limit=c['candidate_limit'],decision_clock=c['clock'],
        period=c['period'],reward_scale=c['reward_scale'],uncertainty=c['validation_uncertainty'],
        updates=c['updates'],candidate_screen=c['candidate_screen'],min_gain=c['min_gain'],
        budget_features=c['budget_features'],value_features=c['value_features'])
    return rr

def validate(checkpoint,args,output):
    from itertools import product
    from statistics import mean
    from iamu_dynamic.rl_experiment_v11 import parse_names,parse_ints
    groups=list(product(args.val_cases,parse_names(args.scenarios),
                        parse_ints(args.val_seeds),parse_ints(args.budgets)))
    cfg={k:getattr(args,k) for k in ('clock','candidate_limit','period','reward_scale',
        'validation_uncertainty','updates','candidate_screen','min_gain',
        'budget_features','value_features')}
    # Pool workers are started only at evaluation boundaries. All 16 evaluate
    # independent episodes. Results are flattened in fixed group-index order.
    with ProcessPoolExecutor(max_workers=args.validation_workers,
                             mp_context=mp.get_context("spawn"),initializer=_initializer,initargs=(str(checkpoint),cfg)) as executor:
        chunks=list(executor.map(_work,groups,chunksize=1))
    rows=[r for chunk in chunks for r in chunk]
    if len(rows)!=len(groups)*(4 if args.clock=='union' else 3):
        raise RuntimeError('Incomplete validation grid')
    for r in rows:
        if int(r['spent'])>int(r['budget']):raise AssertionError('Budget overspent')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'runs.csv',rows)
    # Score is mean across all scenarios, budgets and validation ETA seeds.
    selected=[r for r in rows if r['method']=='ddqn']
    score=mean(float(r['wait']) for r in selected)
    summary=[]
    for policy in ('keep','greedy','ddqn','periodic'):
        subset=[r for r in rows if r['method']==policy]
        if subset:
            summary.append(dict(method=policy,mean_wait=mean(float(r['wait']) for r in subset),
                mean_spent=mean(float(r['spent']) for r in subset),n=len(subset)))
    write_csv(out/'summary.csv',summary)
    return dict(score=score,rows=len(rows),groups=len(groups),mean_spent=mean(r['spent'] for r in selected))
