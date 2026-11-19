"""Shared actionable-state bank, with full hidden-state reconstruction by replay.

IMPORTANT: observation equality alone is insufficient. Every selected state is
regenerated from the same case, exogenous scenario and behavioral action prefix;
we verify all public features and the full ordered candidate-plan fingerprint.
Only trusted locally generated .pt inputs may be loaded (weights_only=False).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
from .baseline_cache import canonical, file_sha

VERSION = 'v12-shared-probe-v1'
ENV_ARGS = dict(candidate_limit=4, decision_clock='union', period=12.,
                reward_scale=100., candidate_screen='positive', min_gain=.1,
                budget_features='absolute', value_features='global')


def observation_fingerprint(global_features, action_features, mask):
    h = hashlib.sha256()
    for x in (global_features, action_features, mask):
        a = np.ascontiguousarray(x)
        h.update(str(a.dtype).encode());h.update(canonical(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _normalized(x):
    if is_dataclass(x):return _normalized(asdict(x))
    if isinstance(x, dict):return {str(k):_normalized(v) for k,v in sorted(x.items(),key=lambda p:str(p[0]))}
    if isinstance(x,(list,tuple)):return [_normalized(v) for v in x]
    if isinstance(x,np.ndarray):return _normalized(x.tolist())
    if isinstance(x,(np.integer,np.floating)):return x.item()
    return x


def candidate_fingerprint(env):
    """IDs alone do NOT identify moves: each proposed complete plan is hashed."""
    content = dict(ordered_candidate_ids=list(env._candidate_ids),
                   proposal_plans=[_normalized(p) for p in env._candidates],
                   proposal_data=[_normalized(d) for d in env._candidate_data],
                   current_plan=_normalized(env.plan),
                   public_eta=_normalized(env.published_eta))
    return hashlib.sha256(canonical(content).encode()).hexdigest()


def _jobs_for_dataset(ds):
    from v11_protocol.train_v11 import jobs_for
    cfg = ds['config']
    args = SimpleNamespace(seed=cfg['seed'],scenarios=cfg['scenarios'],
        budgets=cfg['budgets'],uncertainties=cfg['uncertainties'],
        train_cases=cfg['cases'],episodes=cfg['episodes'],
        train_seed_start=cfg['train_seed_start'],epsilon_start=1.,
        epsilon_end=1.,epsilon_decay_episodes=1)
    return {j.episode:j for j in jobs_for(args)}


def source_dataset(path):
    """Read only datasets produced locally by fixed_data.collect."""
    ds = torch.load(path,map_location='cpu',weights_only=False)
    if ds.get('version') != 12 or 'episodes' not in ds or 'config' not in ds:
        raise ValueError('Expected trusted V1.2 fixed-data trajectory file')
    return ds


def restore_probe_env(probe: dict, dataset):
    """Restore the *exact same* simulator hidden state, not just its observation."""
    from iamu_dynamic import read_case, generate_scenario
    from iamu_dynamic.rl_env_v11 import IAMUEventEnv
    ds = source_dataset(dataset) if isinstance(dataset,(str,Path)) else dataset
    eps = {int(e['episode_id']):e for e in ds['episodes']}
    if int(probe['episode_id']) not in eps:raise ValueError('Missing source episode')
    ep = eps[int(probe['episode_id'])]
    source_step=int(probe['raw_step_index'])
    if source_step<0 or source_step>=len(ep['raw']):raise ValueError('Bad raw step index')
    if (str(ep['case'])!=str(probe['case']) or ep['scenario']!=probe['scenario'] or
            int(ep['seed'])!=int(probe['scenario_seed']) or int(ep['budget'])!=int(probe['initial_budget'])):
        raise ValueError(f'Probe provenance mismatch: {probe["state_id"]}')
    if file_sha(ep['case'])!=probe['case_sha256']:
        raise ValueError('Case bytes changed')
    cases = _jobs_for_dataset(ds)
    job = cases[int(ep['episode_id'])]
    if (job.case!=ep['case'] or job.scenario!=ep['scenario'] or
            job.seed!=ep['seed'] or job.budget!=ep['budget']):
        raise ValueError('Dataset job table differs from source episode')
    if float(job.uncertainty)!=float(probe['uncertainty']):
        raise ValueError('Scenario uncertainty mismatch')
    vs,port=read_case(ep['case'])
    cfg=ds['config']
    scenario=generate_scenario(vs,port,label=job.scenario,seed=job.seed,
                  updates_per_vessel=int(cfg['updates']),uncertainty=job.uncertainty)
    env=IAMUEventEnv(scenario,budget=int(job.budget),**ENV_ARGS)
    env.reset()
    prefix=list(probe['prefix_actions'])
    expected=[int(x.transition.action) for x in ep['raw'][:source_step]]
    if prefix!=expected:raise ValueError('Prefix actions differ from raw trajectory')
    for idx,action in enumerate(prefix):
        obs=env._observation()
        original=ep['raw'][idx]
        if not np.array_equal(obs.mask, original.transition.obs_mask):
            raise AssertionError(f'Raw prefix mask mismatch at {idx}')
        if not np.array_equal(obs.global_features,original.transition.obs_global):
            raise AssertionError(f'Raw prefix global features mismatch at {idx}')
        if not np.array_equal(obs.action_features,original.transition.obs_actions):
            raise AssertionError(f'Raw prefix candidate features mismatch at {idx}')
        _,_,done,_=env.step(action)
        if done:raise AssertionError('Probe prefix terminated before source state')
    obs=env._observation()
    source=ep['raw'][source_step]
    if (env.done or not np.asarray(obs.mask[1:],dtype=bool).any() or
            int(env.event_index)!=int(probe['source_event_index']) or
            int(obs.remaining_budget)!=int(probe['remaining_budget']) or
            abs(float(obs.time)-float(probe['source_time']))>1e-9):
        raise AssertionError('Restored decision state metadata mismatch: '+probe['state_id'])
    if (not np.array_equal(obs.global_features,source.transition.obs_global) or
            not np.array_equal(obs.action_features,source.transition.obs_actions) or
            not np.array_equal(obs.mask,source.transition.obs_mask)):
        raise AssertionError('Restored source features diverged: '+probe['state_id'])
    if (not np.array_equal(obs.global_features,probe['obs_global']) or
            not np.array_equal(obs.action_features,probe['obs_actions']) or
            not np.array_equal(obs.mask,probe['obs_mask'])):
        raise AssertionError('Probe bank data diverged: '+probe['state_id'])
    fp=observation_fingerprint(obs.global_features,obs.action_features,obs.mask)
    if fp!=probe['observation_sha256']:raise AssertionError('Observation checksum differs')
    ids=tuple(int(x) for x in obs.candidate_ids)
    if 'candidate_ids' in probe and ids!=tuple(probe['candidate_ids']):
        raise AssertionError('Candidate order differs')
    cf=candidate_fingerprint(env)
    if 'candidate_fingerprint' in probe and cf!=probe['candidate_fingerprint']:
        raise AssertionError('Candidate plans/ETA differ')
    return env


def build_probe_bank(raw_dataset_path, out_path, *,seed=1729,
                     per_stratum=8,max_states=144):
    if per_stratum<1 or max_states<1:raise ValueError('Invalid probe limits')
    source_sha=file_sha(raw_dataset_path)
    ds=source_dataset(raw_dataset_path)
    jobs=_jobs_for_dataset(ds)
    pool={}
    for ep in ds['episodes']:
        jid=int(ep['episode_id']);job=jobs[jid]
        for i,raw in enumerate(ep['raw']):
            tr=raw.transition
            if tr.done or not bool(np.asarray(tr.obs_mask)[1:].any()):continue
            if raw.source_budget<1:raise AssertionError('Actionable with zero budget')
            stratum=(str(ep['scenario']),int(ep['budget']),
                     '1' if raw.source_budget==1 else '>1')
            pool.setdefault(stratum,[]).append((ep,i,raw,job))
    rng=random.Random(seed)
    selected=[];coverage=[]
    for stratum in sorted(pool):
        members=pool[stratum]
        take=min(per_stratum,len(members))
        selected.extend(rng.sample(members,take))
        coverage.append(dict(scenario=stratum[0],initial_budget=stratum[1],
            remaining_budget_group=stratum[2],eligible=len(members),selected=take))
    # Explicit zero cells in the full design.
    for sc in ('shock','repeated','small'):
        for b in (2,5,10):
            for rem in ('1','>1'):
                if not any(x['scenario']==sc and x['initial_budget']==b and x['remaining_budget_group']==rem for x in coverage):
                    coverage.append(dict(scenario=sc,initial_budget=b,remaining_budget_group=rem,
                                         eligible=0,selected=0))
    if len(selected)>max_states:
        # Fixed independent RNG; no score-dependent selection or backfill.
        selected=rng.sample(selected,max_states)
        selected_keys={(int(e['episode_id']),i) for e,i,_,_ in selected}
        for c in coverage:
            c['selected']=sum(1 for e,i,r,j in selected if
              e['scenario']==c['scenario'] and int(e['budget'])==c['initial_budget'] and
              ('1' if r.source_budget==1 else '>1')==c['remaining_budget_group'])
    states=[]
    # Replay each *source episode once* in increasing raw-step order. Building
    # 144 states by regenerating 144 separate complete prefixes is needlessly
    # expensive because physical candidate generation dominates wall time.
    grouped=defaultdict(list)
    for ep,i,raw,job in selected:
        grouped[int(ep['episode_id'])].append((ep,i,raw,job))
    from iamu_dynamic import read_case,generate_scenario
    from iamu_dynamic.rl_env_v11 import IAMUEventEnv
    for episode_id in sorted(grouped):
        entries=sorted(grouped[episode_id],key=lambda x:x[1])
        ep=entries[0][0];job=entries[0][3]
        vs,port=read_case(ep['case'])
        scenario=generate_scenario(vs,port,label=job.scenario,seed=job.seed,
                     updates_per_vessel=int(ds['config']['updates']),
                     uncertainty=job.uncertainty)
        env=IAMUEventEnv(scenario,budget=int(job.budget),**ENV_ARGS)
        env.reset()
        target_by_step={i:(item,raw,j) for item,i,raw,j in entries}
        for i in range(max(target_by_step)+1):
            original=ep['raw'][i]
            ob=env._observation()
            if (int(ob.remaining_budget)!=int(original.source_budget) or
                    int(env.event_index)!=int(original.source_event_index) or
                    abs(float(ob.time)-float(original.source_time))>1e-9 or
                    not np.array_equal(ob.global_features,original.transition.obs_global) or
                    not np.array_equal(ob.action_features,original.transition.obs_actions) or
                    not np.array_equal(ob.mask,original.transition.obs_mask)):
                raise AssertionError(f'Dataset prefix cannot be replayed, episode={episode_id}, step={i}')
            if i in target_by_step:
                _,raw,j=target_by_step[i]
                pid=hashlib.sha256(canonical([source_sha,episode_id,i]).encode()).hexdigest()
                tr=raw.transition
                probe=dict(state_id=pid,source_sha256=source_sha,
                  episode_id=episode_id,raw_step_index=i,
                  case=str(ep['case']),case_sha256=file_sha(ep['case']),
                  scenario=str(ep['scenario']),scenario_seed=int(ep['seed']),
                  uncertainty=float(j.uncertainty),initial_budget=int(ep['budget']),
                  remaining_budget=int(raw.source_budget),
                  source_event_index=int(raw.source_event_index),source_time=float(raw.source_time),
                  obs_global=np.array(tr.obs_global,copy=True),
                  obs_actions=np.array(tr.obs_actions,copy=True),
                  obs_mask=np.array(tr.obs_mask,copy=True),
                  observation_sha256=observation_fingerprint(tr.obs_global,tr.obs_actions,tr.obs_mask),
                  prefix_actions=[int(x.transition.action) for x in ep['raw'][:i]],
                  candidate_ids=tuple(int(x) for x in ob.candidate_ids),
                  candidate_fingerprint=candidate_fingerprint(env),
                  generator_config=dict(seed=int(ep['seed']),label=ep['scenario'],
                              uncertainty=float(j.uncertainty),updates=int(ds['config']['updates'])),
                  environment_config=dict(ENV_ARGS,budget=int(ep['budget'])))
                states.append(probe)
            if i<max(target_by_step):
                _,_,done,_=env.step(int(original.transition.action))
                if done:raise AssertionError('Premature termination during shared prefix replay')
    states.sort(key=lambda x:x['state_id'])
    destination=Path(out_path)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    result=dict(version=VERSION,source_sha256=source_sha,
                source_path=str(raw_dataset_path),seed=int(seed),
                per_stratum=per_stratum,max_states=max_states,
                states=states,coverage=sorted(coverage,key=lambda x:(x['scenario'],x['initial_budget'],x['remaining_budget_group'])))
    torch.save(result,destination)
    from iamu_dynamic.rl_experiment_v11 import write_csv
    write_csv(destination.with_suffix('.coverage.csv'),result['coverage'])
    return dict(states=len(states),source_sha256=source_sha,
                bank_sha256=file_sha(destination),coverage=result['coverage'])


def load_probe_bank(path, data=None):
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if bank.get('version')!=VERSION:raise ValueError('Unknown probe version')
    src=data or bank['source_path']
    if file_sha(src)!=bank['source_sha256']:raise ValueError('Probe bank source SHA differs')
    ids=[x['state_id'] for x in bank['states']]
    if len(ids)!=len(set(ids)) or ids!=sorted(ids):raise ValueError('Probe IDs not ordered/unique')
    return bank


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build');b.add_argument('--data',required=True);b.add_argument('--out',required=True)
    b.add_argument('--seed',type=int,default=1729);b.add_argument('--per-stratum',type=int,default=8)
    b.add_argument('--max-states',type=int,default=144)
    t=sub.add_parser('restore-audit');t.add_argument('--bank',required=True);t.add_argument('--data')
    t.add_argument('--limit',type=int,default=8)
    a=p.parse_args(argv)
    if a.mode=='build':print(json.dumps(build_probe_bank(a.data,a.out,seed=a.seed,
                          per_stratum=a.per_stratum,max_states=a.max_states),ensure_ascii=False,indent=2))
    else:
        bank=load_probe_bank(a.bank,a.data)
        ds=source_dataset(a.data or bank['source_path'])
        n=len(bank['states']) if a.limit==0 else min(a.limit,len(bank['states']))
        for state in bank['states'][:n]:restore_probe_env(state,ds)
        print(f'[probe restore] audited {n}/{len(bank["states"])} states')

if __name__=='__main__':main()
