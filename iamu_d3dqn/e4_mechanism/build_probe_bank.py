#!/usr/bin/env python3
"""Build a frozen, PUBLIC-OBSERVATION bank for comparable Q-gap measurements.

Uses only public observation features/action masks. No latent future ETA,
ex-post gains or simulator objects are saved. Keep/point-greedy anchor policies
never depend on a learned checkpoint, eliminating selection by late Q scores.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def choose(n,k):
    if k<1:raise ValueError(k)
    m=min(n,k)
    return [min(n-1,(2*i+1)*n//(2*m)) for i in range(m)]


def collect(instance,budget,anchor,env_args,k):
    env=IAMUEventEnv(instance,budget=budget,**env_args)
    ob=env.reset()
    eligible=[]
    while True:
        if ob.mask[1:].any():
            eligible.append((ob.copy(),int(env.event_index),str(env.event_type)))
        act=0 if anchor=='keep' else int(env.point_greedy_action())
        ob,_,done,_=env.step(act)
        if done:break
    return [eligible[i] for i in choose(len(eligible),k)],len(eligible)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',default='results/v10_e4_b900_seed43/manifest.json')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--seeds',default='400001,400002')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--anchors',default='keep,greedy')
    p.add_argument('--states-per-group',type=int,default=2)
    p.add_argument('--out',default='results/v10_e5_probe/public_bank.npz')
    args=p.parse_args(argv)
    conf=json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
    tr=conf['training_args']
    env_args=dict(candidate_limit=int(tr['candidate_limit']),
        decision_clock=tr['clock'],period=float(tr['period']),
        reward_scale=float(tr['reward_scale']),min_gain=float(tr['min_gain']),
        candidate_screen=tr['candidate_screen'],budget_features=tr['budget_features'],
        value_features=tr['value_features'])
    scenarios=args.scenarios.split(',');seeds=list(map(int,args.seeds.split(',')))
    budgets=list(map(int,args.budgets.split(',')));anchors=args.anchors.split(',')
    if args.states_per_group<1 or len(set(seeds))!=len(seeds):p.error('Invalid count or duplicate seed')
    if not set(anchors)<=set(('keep','greedy')):p.error('anchors must be keep or greedy')
    obs=[];meta=[];coverage=[]
    cases=[Path(f) for f in tr['val_cases']]
    for case in cases:
        vessels,port=read_case(case)
        for scenario in scenarios:
            for seed in seeds:
                instance=generate_scenario(vessels,port,label=scenario,seed=seed,
                    updates_per_vessel=int(tr['updates']),
                    uncertainty=float(tr['validation_uncertainty']))
                for B in budgets:
                    for anchor in anchors:
                        selected,n=collect(instance,B,anchor,env_args,args.states_per_group)
                        coverage.append(dict(case=case.stem,scenario=scenario,seed=seed,
                              budget=B,anchor=anchor,eligible=n,selected=len(selected)))
                        for ob,idx,typ in selected:
                            obs.append(ob)
                            meta.append(dict(case=case.stem,scenario=scenario,seed=seed,
                                  budget=B,anchor=anchor,event_index=idx,event_type=typ,
                                  remaining_budget=ob.remaining_budget,n_legal=int(ob.mask[1:].sum())))
    if not obs:raise RuntimeError('No public states with legal Edit')
    dest=Path(args.out)
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,g=np.stack([o.global_features for o in obs]),
        actions=np.stack([o.action_features for o in obs]),
        mask=np.stack([o.mask for o in obs]),
        event=np.array([str(x['event_index']) for x in meta]))
    with dest.with_suffix('.metadata.json').open('w',encoding='utf-8') as f:
        json.dump(dict(args=vars(args),model_free=True,
              source_manifest_sha256=sha(Path(args.source_manifest)),
              case_sha256={str(c):sha(c) for c in cases},
              bank_sha256=sha(dest),n_states=len(obs),states=meta,coverage=coverage),
              f,ensure_ascii=False,indent=2)
    print(f'[DONE] {len(obs)} public states; bank={dest}, sha256={sha(dest)}')

if __name__=='__main__':main()
