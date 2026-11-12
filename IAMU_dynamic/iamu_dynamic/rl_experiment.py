"""Shared paired evaluation: policy candidates and physical events are identical.

No benchmark/test episode is used in a policy update. Run-seed and scenario
names are never provided as network input. Hard-budget feasibility is checked
on every action. Baselines are evaluated on the very same event clock.
"""
from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from . import generate_scenario, read_case
from .rl_env import IAMUEventEnv
from .risk_selection import RiskConfig,select_from_proposals,upper_cvar


def parse_ints(spec:str):
    out=[]
    for z in spec.split(','):
        z=z.strip()
        if not z:continue
        if ':' in z:
            p=z.split(':')
            if len(p)!=2:raise ValueError('Invalid integer range')
            a,b=map(int,p)
            if b<a or b-a>100000:raise ValueError('Bad integer range')
            out.extend(range(a,b+1))
        else:out.append(int(z))
    if not out or len(set(out))!=len(out):
        raise ValueError('Need nonempty distinct integers')
    return out


def parse_names(spec:str):
    a=[x.strip() for x in spec.split(',') if x.strip()]
    if not a or len(set(a))!=len(a):raise ValueError('Invalid names')
    return a


def verify_cases(train_cases,val_cases,test_cases=()):
    seen=[]
    for group in (train_cases,val_cases,test_cases):
        hashes=set()
        for p in group:
            path=Path(p)
            if not path.is_file():raise FileNotFoundError(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:raise ValueError('Repeated case content within split')
            hashes.add(digest)
        seen.append(hashes)
    if seen[0]&seen[1] or seen[0]&seen[2] or seen[1]&seen[2]:
        raise ValueError('Training/validation/test case contents must be disjoint')


def write_csv(path, rows):
    if not rows:return
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader();w.writerows(rows)


def choose_action(policy,env,agent=None,risk_samples=16,forecast_std=8.):
    if policy=='keep':return 0
    if policy=='periodic' and env.event_type!='clock':return 0
    if policy in ('greedy','periodic'):return env.point_greedy_action()
    if policy=='ddqn':
        if agent is None:raise ValueError('DDQN requires trained agent')
        return agent.act(env._observation(),epsilon=0.)
    if policy in ('expected','risk_lookahead'):
        if not env._candidates:return 0
        proposals=[(x,(i,),'single') for x,i in zip(env._candidates,env._candidate_ids)]
        best,_=select_from_proposals(
            env.runtime,env.planner,env.plan,env.published_eta,proposals,
            now=env.event_time,remaining_global=env.remaining_budget,
            within_event_remaining=1,revision_penalty=0.,min_gain=env.min_gain,
            method=policy,config=RiskConfig(n_samples=risk_samples,
                       forecast_std=forecast_std,lookahead_candidates=1,
                       next_move_candidates=1,risk_aversion=.5),
            run_seed=env.scenario.seed,event_id=env.event_index,
            event_type=env.event_type)
        if best is None:return 0
        i=best[1][0]
        return 1+env._candidate_ids.index(i)
    raise ValueError(f'Unknown policy {policy}')


def run_episode(scenario,budget,policy,agent=None,*,candidate_limit=4,
                decision_clock='union',period=12.,reward_scale=100.,
                risk_samples=16,forecast_std=8.):
    env=IAMUEventEnv(scenario,budget=budget,candidate_limit=candidate_limit,
                     decision_clock=decision_clock,period=period,
                     reward_scale=reward_scale)
    ob=env.reset()
    start=perf_counter()
    total_reward=0.
    while True:
        action=choose_action(policy,env,agent=agent,
                             risk_samples=risk_samples,forecast_std=forecast_std)
        ob,r,done,info=env.step(action)
        total_reward+=r
        if done:break
    assert env.used <= budget
    assert abs(-total_reward*reward_scale-info['weighted_wait']) < 1e-5*max(1.,info['weighted_wait'])
    return dict(wait=info['weighted_wait'],spent=info['spent'],
                runtime_seconds=perf_counter()-start,
                steps=len(env.history),return_unscaled=-info['weighted_wait'],
                edits_by_clock=sum(x[3] is not None and x[0]=='clock' for x in env.history),
                edits_by_eta=sum(x[3] is not None and x[0]=='eta' for x in env.history),
                last_revision_time=max((x[1] for x in env.history if x[3] is not None),default=0.))


def evaluate_paired(*,cases,seeds,scenarios,budgets,policies,agent=None,
                    candidate_limit=4,decision_clock='union',period=12.,
                    reward_scale=100.,uncertainty=8.,updates=2,
                    risk_samples=16,forecast_std=8.,out=None):
    if 'keep' not in policies:raise ValueError('keep baseline is required')
    if 'periodic' in policies and decision_clock!='union':
        raise ValueError('Periodic comparator requires union clock')
    rows=[]
    for case in cases:
        vessels,port=read_case(case)
        for sc in scenarios:
            for seed in seeds:
                instance=generate_scenario(vessels,port,label=sc,seed=seed,
                              updates_per_vessel=updates,uncertainty=uncertainty)
                for budget in budgets:
                    for method in policies:
                        m=run_episode(instance,budget,method,agent=agent,
                            candidate_limit=candidate_limit,decision_clock=decision_clock,
                            period=period,reward_scale=reward_scale,
                            risk_samples=risk_samples,forecast_std=forecast_std)
                        rows.append(dict(case=str(case),scenario=sc,seed=seed,
                                         budget=budget,method=method,**m))
    base={(r['case'],r['scenario'],r['seed'],r['budget']):r['wait']
          for r in rows if r['method']=='keep'}
    per={(r['case'],r['scenario'],r['seed'],r['budget']):r['wait']
         for r in rows if r['method']=='periodic'}
    for row in rows:
        k=(row['case'],row['scenario'],row['seed'],row['budget'])
        row['gain_vs_keep']=base[k]-row['wait']
        row['gain_vs_periodic']=(per[k]-row['wait']) if k in per else ''
        row['regret_vs_keep']=max(0.,row['wait']-base[k])
    groups=defaultdict(list)
    for row in rows:groups[(row['method'],row['budget'])].append(row)
    summary=[]
    for (method,budget),a in sorted(groups.items()):
        gains=[r['gain_vs_keep'] for r in a]
        risks=[r['regret_vs_keep'] for r in a]
        summary.append(dict(method=method,budget=budget,n=len(a),
            mean_wait=mean(r['wait'] for r in a),
            mean_gain=mean(gains),median_gain=median(gains),
            improved=sum(x>1e-6 for x in gains),
            degraded=sum(x< -1e-6 for x in gains),
            worst_gain=min(gains),cvar90_regret=upper_cvar(risks,.90),
            mean_spent=mean(r['spent'] for r in a),
            mean_runtime_seconds=mean(r['runtime_seconds'] for r in a),
            mean_last_revision_time=mean(r['last_revision_time'] for r in a),
            mean_gain_vs_periodic=(mean(r['gain_vs_periodic'] for r in a)
                                   if 'periodic' in policies else '' )))
    if out is not None:
        p=Path(out);p.mkdir(parents=True,exist_ok=True)
        write_csv(p/'runs.csv',rows)
        write_csv(p/'summary.csv',summary)
    return rows,summary
