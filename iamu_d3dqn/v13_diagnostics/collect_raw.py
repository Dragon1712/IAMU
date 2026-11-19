#!/usr/bin/env python3
"""Collect 200 COMPLETE V1.1 episodes once, then derive A0/A2/A3 offline.

Behaviors: 100 frozen-A0 with configurable epsilon, 100 50%-Keep/edit-uniform.
No training updates; the engine is executed for every original event.
"""
from __future__ import annotations
import argparse
import math
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .common import (action_descriptor,atomic_json,balanced_jobs,digest,load_model,
                     make_env,provenance,save_trusted,scenario_digest,
                     verify_v12_scientific_model,write_csv)


def collect_one(job,behavior,model=None,epsilon=.2,updates=2):
    from iamu_dynamic.rl_ddqn_v11 import Transition
    from v12_protocol.transitions import RawStep
    env=make_env(job['case'],dict(name=job['scenario'],seed=job['external_seed']),
                 job['budget'],job['uncertainty'],updates)
    generated_scenario_sha256=scenario_digest(env.scenario)
    obs=env.reset();rng=random.Random(job['external_seed']^0xFACED00D)
    if behavior=='a0' and model is None:raise ValueError('a0 requires frozen checkpoint')
    if model is not None:
        model.rng.seed(job['external_seed']*10007+job['episode_id'])
    raw=[];descriptions=[];reward_sum=0.
    while True:
        i=len(raw)
        if behavior=='a0':
            action=model.act(obs,epsilon=epsilon)   # also call on forced-keep state
        elif behavior=='mixed':
            legal=[a for a in range(1,len(obs.mask)) if obs.mask[a]]
            action=(rng.choice(legal) if legal and rng.random()>=.5 else 0)
        else:raise ValueError(behavior)
        desc=action_descriptor(env,action)
        b=int(obs.remaining_budget);t=float(obs.time);ei=int(env.event_index)
        nxt,r,done,info=env.step(action)
        tr=Transition(obs.global_features.copy(),obs.action_features.copy(),obs.mask.copy(),
                      int(action),float(r),nxt.global_features.copy(),nxt.action_features.copy(),
                      nxt.mask.copy(),bool(done),dict(obs.decision_flags or {}),int(job['episode_id']))
        raw.append(RawStep(tr,b,int(nxt.remaining_budget),ei,int(env.event_index),t,
                           float(nxt.time),int(job['episode_id']),0))
        descriptions.append(desc)
        reward_sum+=r;obs=nxt
        if done:break
    wait=float(info['weighted_wait'])
    if not math.isclose(-reward_sum*100.,wait,rel_tol=1e-6,abs_tol=1e-5*max(1,wait)):
        raise AssertionError('Reward/cost mismatch')
    return dict(job=job,behavior=behavior,raw=tuple(raw),descriptors=tuple(descriptions),
                weighted_wait=wait,spent=int(info['spent']),reward_sum=reward_sum,
                scenario_sha256=generated_scenario_sha256)


_WORKER_MODEL=None
_WORKER_EPSILON=None
_WORKER_UPDATES=None


def _initializer(checkpoint,epsilon,updates):
    global _WORKER_MODEL,_WORKER_EPSILON,_WORKER_UPDATES
    import os
    import torch
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    _WORKER_MODEL=load_model(checkpoint)
    verify_v12_scientific_model(_WORKER_MODEL,'primitive')
    _WORKER_EPSILON=epsilon
    _WORKER_UPDATES=updates


def _worker(item):
    behavior,job=item
    return collect_one(job,behavior,_WORKER_MODEL,_WORKER_EPSILON,_WORKER_UPDATES)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--a0-checkpoint',required=True)
    p.add_argument('--episodes-per-behavior',type=int,default=100)
    p.add_argument('--a0-epsilon',type=float,default=.2)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--external-seed-start',type=int,default=600000)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if not 0<=a.a0_epsilon<=1:raise ValueError('Invalid epsilon')
    if not 1<=a.workers<=24:raise ValueError('Expected 1..24 CPU workers')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for i,behavior in enumerate(('a0','mixed')):
        js=balanced_jobs(a.episodes_per_behavior,a.cases,['shock','repeated','small'],
                         [2,5,10],a.seed+i,first_episode=1+i*a.episodes_per_behavior,
                         first_external_seed=a.external_seed_start+i*a.episodes_per_behavior)
        for j in js:j['uncertainty']=a.uncertainty
        jobs.extend((behavior,j) for j in js)
    rows=[];records=[]
    if a.workers==1:
        model=load_model(a.a0_checkpoint)
        verify_v12_scientific_model(model,'primitive')
        stream=(collect_one(job,behavior,model,a.a0_epsilon,a.updates)
                for behavior,job in jobs)
        pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=a.workers,
            mp_context=mp.get_context('spawn'),initializer=_initializer,
            initargs=(a.a0_checkpoint,a.a0_epsilon,a.updates))
        stream=pool.map(_worker,jobs,chunksize=1)
    try:
      for k,record in enumerate(stream,1):
        job=record['job'];behavior=record['behavior']
        records.append(record)
        rows.append(dict(episode_id=job['episode_id'],case=job['case'],scenario=job['scenario'],
                         budget=job['budget'],external_seed=job['external_seed'],behavior=behavior,
                         scenario_sha256=record['scenario_sha256'],
                         raw_steps=len(record['raw']),actionable=sum(x.transition.obs_mask[1:].any()
                                 for x in record['raw']),weighted_wait=record['weighted_wait'],
                         spent=record['spent']))
        if k%10==0:print(f'[V13 collect] {k}/{len(jobs)}',flush=True)
    finally:
      if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    file=out/'raw_episodes.pt'
    save_trusted(file,dict(protocol='v13-P1-raw',records=records))
    write_csv(out/'episodes.csv',rows)
    manifest=provenance(a,a.cases,[a.a0_checkpoint]);manifest.update(
        data_sha256=digest(file),n_episodes=len(records),n_raw_steps=sum(x['raw_steps'] for x in rows),
        behavior_counts={x:sum(row['behavior']==x for row in rows) for x in ('a0','mixed')})
    atomic_json(out/'manifest.json',manifest)
    print(f'[V13 collect] saved={file} SHA256={manifest["data_sha256"]}',flush=True)

if __name__=='__main__':main()
