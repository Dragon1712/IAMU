#!/usr/bin/env python3
"""Independent 120-state probe bank with actual env snapshots and semantic edits.

40 each source remaining budget 1,2,>=3; within each group target
20 shock + 10 repeated + 10 small. No duplicated states to fill shortages.
Probes use independent external-seed range, NOT training trajectories.
"""
from __future__ import annotations
import argparse
import copy
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from .common import (action_descriptor,atomic_json,balanced_jobs,digest,load_model,
                     make_env,provenance,save_trusted,state_id,scenario_digest,
                     verify_v12_scientific_model,write_csv)


def group_for(remaining,scenario):
    return ('1' if remaining==1 else ('2' if remaining==2 else '3+'),scenario)


def replay_snapshot(probe):
    """Exact original-tape replay, including action slots and rewards."""
    env=copy.deepcopy(probe['snapshot'])
    rewards=probe['original_suffix_rewards'];actions=probe['original_suffix_actions']
    if len(actions)!=len(rewards):raise AssertionError('Suffix length differs')
    for i,(action,reward) in enumerate(zip(actions,rewards)):
        ob=env._observation()
        if not ob.mask[action]:raise AssertionError('Original action illegal on replay')
        obs,r,done,_=env.step(int(action))
        if not math.isclose(float(r),float(reward),rel_tol=1e-9,abs_tol=1e-8):
            raise AssertionError(f'Snapshot re-execution differs at event {i}')
        if done!=(i==len(actions)-1):raise AssertionError('Different terminal step')
    return len(rewards)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--a0-checkpoint',required=True)
    p.add_argument('--episodes',type=int,default=240)
    p.add_argument('--start-seed',type=int,default=800000)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--behavior',choices=('mixed','a0'),default='mixed')
    p.add_argument('--epsilon',type=float,default=.25)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    model=load_model(a.a0_checkpoint)
    verify_v12_scientific_model(model,'primitive')
    quotas={(b,sc):(20 if sc=='shock' else 10) for b in ('1','2','3+')
            for sc in ('shock','repeated','small')}
    reservoirs={k:[] for k in quotas};seen=Counter();sampler=random.Random(a.seed+999)
    jobs=balanced_jobs(a.episodes,a.cases,['shock','repeated','small'],[2,5,10],
                       a.seed,first_episode=1000001,first_external_seed=a.start_seed)
    for job in jobs:
        job['uncertainty']=a.uncertainty
        env=make_env(job['case'],dict(name=job['scenario'],seed=job['external_seed']),
                     job['budget'],a.uncertainty,a.updates)
        generated_scenario_sha256=scenario_digest(env.scenario)
        obs=env.reset();rng=random.Random(job['external_seed']^0xABC123)
        model.rng.seed(job['episode_id']*10007+job['external_seed'])
        episode_actions=[];episode_rewards=[];pending=[]
        while True:
            i=len(episode_actions)
            valid=np.flatnonzero(obs.mask[1:])+1
            if len(valid):
                idx=1+max(range(len(env._candidate_data)),key=lambda j:(
                    float(env._candidate_data[j]['gain']),-j))
                bucket=group_for(int(obs.remaining_budget),job['scenario'])
                if bucket in quotas:
                    seen[bucket]+=1
                    selection=seen[bucket]
                    # Reservoir replacement provides unbiased inclusion within
                    # each bucket, subject to the deterministic behavior policy.
                    slot=(len(reservoirs[bucket]) if len(reservoirs[bucket])<quotas[bucket]
                          else sampler.randrange(selection))
                    if slot<quotas[bucket]:
                        chosen=dict(state_id=state_id(job['episode_id'],i),
                            episode_id=job['episode_id'],raw_step_index=i,
                            case=job['case'],scenario=job['scenario'],
                            external_seed=job['external_seed'],
                            original_scenario_sha256=generated_scenario_sha256,
                            source_budget=int(obs.remaining_budget),
                            action_descriptor=action_descriptor(env,idx),
                            selected_edit=int(idx),
                            selected_edit_gain=float(env._candidate_data[idx-1]['gain']),
                            snapshot=copy.deepcopy(env),
                            snapshot_time=float(obs.time),
                            original_suffix_actions=None,original_suffix_rewards=None,
                            _pending_raw_start=i,_reservoir_bucket=bucket)
                        if slot==len(reservoirs[bucket]):reservoirs[bucket].append(chosen)
                        else:reservoirs[bucket][slot]=chosen
                        pending.append(chosen)
            if a.behavior=='mixed':
                action=int(rng.choice(valid.tolist())) if len(valid) and rng.random()>=.5 else 0
            else:
                action=int(model.act(obs,epsilon=a.epsilon))
            obs,r,done,info=env.step(action)
            episode_actions.append(action);episode_rewards.append(float(r))
            if done:break
        # Only selected states from this episode need suffixes. Discarded
        # reservoir entries are intentionally not serialized.
        for candidate in pending:
            start=candidate['_pending_raw_start']
            candidate['original_suffix_actions']=tuple(episode_actions[start:])
            candidate['original_suffix_rewards']=tuple(episode_rewards[start:])
        if job['episode_id']%25==0:
            print(f'[V13 probes] episodes={job["episode_id"]-1000000} counts='+str(
                {str(k):len(v) for k,v in reservoirs.items()}),flush=True)
    probes=[];counts={}
    for key in sorted(quotas):
        counts[str(key)]={'requested':quotas[key],'available':seen[key],
                          'selected':len(reservoirs[key]),
                          'shortfall':max(0,quotas[key]-len(reservoirs[key]))}
        for pr in reservoirs[key]:
            pr.pop('_pending_raw_start',None);pr.pop('_reservoir_bucket',None)
            if pr['original_suffix_actions'] is None:
                raise AssertionError('Unfilled suffix for accepted snapshot')
            replay_snapshot(pr)
            probes.append(pr)
    path=out/'probe_bank.pt'
    save_trusted(path,dict(version=1,probes=probes,quotas=quotas,
                           external_seed_range=[a.start_seed,a.start_seed+a.episodes-1]))
    rows=[{k:v for k,v in p.items() if k not in ('snapshot','original_suffix_actions',
             'original_suffix_rewards')} for p in probes]
    write_csv(out/'probe_states.csv',rows)
    audit=dict(n_probes=len(probes),target=sum(quotas.values()),counts=counts,
               replay_checked=len(probes),data_sha256=digest(path),
               warning=('Probe strata are undersupplied; do not duplicate states'
               if len(probes)<sum(quotas.values()) else None))
    atomic_json(out/'probe_audit.json',audit)
    atomic_json(out/'manifest.json',provenance(a,a.cases,[a.a0_checkpoint]))
    print(f'[V13 probe] states={len(probes)} requested={sum(quotas.values())}',flush=True)

if __name__=='__main__':main()
