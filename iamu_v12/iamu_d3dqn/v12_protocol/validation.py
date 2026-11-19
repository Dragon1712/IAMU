"""V1.2-specific validation; never silently load checkpoints via V1.1 defaults."""
from __future__ import annotations
import multiprocessing as mp
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from iamu_dynamic.rl_experiment_v11 import evaluate_paired,parse_names,parse_ints,write_csv

_AGENT=None
_CFG=None

def _init_worker(checkpoint,settings):
    global _AGENT,_CFG
    import torch
    from .agent import V12Agent
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    _AGENT,meta=V12Agent.from_checkpoint(checkpoint,device='cpu')
    _AGENT.online.eval();_CFG=settings

def _one(group):
    case,sc,seed,budget=group
    c=_CFG
    rows,_=evaluate_paired(cases=[case],scenarios=[sc],seeds=[seed],budgets=[budget],
        policies=['keep','greedy','ddqn','periodic'],agent=_AGENT,
        candidate_limit=c['candidate_limit'],decision_clock=c['clock'],
        period=c['period'],reward_scale=c['reward_scale'],
        uncertainty=c['validation_uncertainty'],updates=c['updates'],
        candidate_screen=c['candidate_screen'],min_gain=c['min_gain'],
        budget_features=c['budget_features'],value_features=c['value_features'])
    return rows

def validate(checkpoint,args,output):
    from itertools import product
    if args.clock!='union':raise ValueError('V1.2 benchmark requires union clock')
    groups=list(product(args.val_cases,parse_names(args.scenarios),
             parse_ints(args.val_seeds),parse_ints(args.budgets)))
    config={k:getattr(args,k) for k in ('candidate_limit','clock','period','reward_scale',
        'validation_uncertainty','updates','candidate_screen','min_gain',
        'budget_features','value_features')}
    with ProcessPoolExecutor(max_workers=args.validation_workers,
        mp_context=mp.get_context('spawn'),initializer=_init_worker,
        initargs=(str(checkpoint),config)) as executor:
        segments=list(executor.map(_one,groups,chunksize=1))
    rows=[r for grp in segments for r in grp]
    if len(rows)!=4*len(groups):raise AssertionError('Incomplete validation grid')
    if any(int(r['spent'])>int(r['budget']) for r in rows):
        raise AssertionError('Validation overspent budget')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'runs.csv',rows)
    summary=[]
    for mode in ('keep','greedy','periodic','ddqn'):
        subset=[r for r in rows if r['method']==mode]
        summary.append(dict(method=mode,mean_wait=mean(float(r['wait']) for r in subset),
                            mean_spent=mean(int(r['spent']) for r in subset),n=len(subset)))
    write_csv(out/'summary.csv',summary)
    best=next(r for r in summary if r['method']=='ddqn')
    return dict(score=best['mean_wait'],rows=len(rows),groups=len(groups),mean_spent=best['mean_spent'])
