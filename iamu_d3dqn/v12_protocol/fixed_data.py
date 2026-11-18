#!/usr/bin/env python3
"""V1.2 E1: paired fixed-trajectory A0-A3 diagnostic (NOT offline policy proof).

Collect once with a predeclared public-policy mixture, then train each
representation from scratch using exactly the same episode IDs and gradient
budget. No V1.1 optimizer/checkpoint is reused. The generated trajectories
are local trusted training data; load only datasets you personally collected.
"""
from __future__ import annotations
import argparse,math,random,time,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from statistics import mean
import numpy as np
import torch
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.rl_env_v11 import IAMUEventEnv
from iamu_dynamic.rl_ddqn_v11 import Transition
from iamu_dynamic.rl_experiment_v11 import write_csv,parse_ints,parse_names
from .transitions import RawStep
from .compress import compress_episode
from .replay import EpisodeReplay
from .agent import V12Agent
from .validation import validate


def collect(a):
    from v11_protocol.v11_actors import _actor_env_kwargs
    from v11_protocol.train_v11 import jobs_for
    rng_jobs=SimpleNamespace(seed=a.seed,scenarios=a.scenarios,budgets=a.budgets,
        uncertainties=a.uncertainties,train_cases=a.cases,episodes=a.episodes,
        train_seed_start=a.train_seed_start,epsilon_start=1.,epsilon_end=1.,epsilon_decay_episodes=1)
    jobs=jobs_for(rng_jobs)
    episodes=[]
    cache={}
    for job in jobs:
        if job.case not in cache:cache[job.case]=read_case(job.case)
        vs,port=cache[job.case]
        scenario=generate_scenario(vs,port,label=job.scenario,seed=job.seed,
            updates_per_vessel=a.updates,uncertainty=job.uncertainty)
        env=IAMUEventEnv(scenario,**_actor_env_kwargs(dict(budget=job.budget,
            candidate_limit=4,clock='union',period=12.,reward_scale=100.,
            candidate_screen='positive',min_gain=.1,budget_features='absolute',value_features='global')))
        obs=env.reset();rng=random.Random(a.seed*1000003+job.episode*10007+job.seed)
        trajectory=[];total=0.;counts={'keep':0,'greedy':0,'random_legal':0}
        while True:
            z=rng.random()
            legal=np.flatnonzero(obs.mask)
            if z<1./3:act=0;counts['keep']+=1
            elif z<2./3:act=int(env.point_greedy_action());counts['greedy']+=1
            else:act=int(rng.choice(legal.tolist()));counts['random_legal']+=1
            source_budget=int(obs.remaining_budget)
            source_event=int(env.event_index)
            nxt,reward,done,info=env.step(act)
            tr=Transition(obs.global_features.copy(),obs.action_features.copy(),
                obs.mask.copy(),act,float(reward),nxt.global_features.copy(),
                nxt.action_features.copy(),nxt.mask.copy(),bool(done),
                dict(obs.decision_flags or {}),job.episode)
            trajectory.append(RawStep(tr,source_budget,int(nxt.remaining_budget),
                source_event,int(env.event_index),float(obs.time),float(nxt.time),
                job.episode,0))
            total+=reward;obs=nxt
            if done:break
        if not math.isclose(-total*100.,info['weighted_wait'],rel_tol=1e-6,abs_tol=.001):
            raise AssertionError('Fixed-data reward audit failed')
        episodes.append(dict(episode_id=job.episode,case=job.case,scenario=job.scenario,
                             seed=job.seed,budget=job.budget,wait=info['weighted_wait'],
                             raw=trajectory,policy_mix=counts))
        print(f'[fixed collect] {job.episode}/{len(jobs)} steps={len(trajectory)}',flush=True)
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(version=12,provenance='mixed Keep/Greedy/Uniform legal; train cases only',
        episodes=episodes,config=dict(vars(a)))
    torch.save(payload,out)
    print(f'[fixed collect] wrote {len(episodes)} episodes => {out}',flush=True)


def _qgap(agent,macro):
    vals=[]
    agent.online.eval()
    with torch.no_grad():
        for x in macro:
            if not x.obs_mask[1:].any():continue
            g=torch.as_tensor(x.obs_global,device=agent.device).unsqueeze(0)
            aa=torch.as_tensor(x.obs_actions,device=agent.device).unsqueeze(0)
            mask=torch.as_tensor(x.obs_mask,device=agent.device).unsqueeze(0)
            qs=agent.online(g,aa,mask).squeeze(0).cpu().numpy()
            vals.append(float(np.max(qs[1:][x.obs_mask[1:]])-qs[0]))
    return vals


def train(a):
    source=torch.load(a.data,map_location='cpu',weights_only=False)  # TRUSTED local dataset only
    if source.get('version')!=12:raise ValueError('Not a V1.2 raw-trajectory dataset')
    mode=dict(A0=('primitive','uniform'),A1=('primitive','stratified'),
              A2=('budget_tail','uniform'),A3=('decision_epoch','uniform'))[a.group]
    representation,sampling=mode
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    model=V12Agent(hidden_dim=96,learning_rate=3e-4,gamma=a.gamma,
         target_update_steps=250,seed=a.seed,device=a.device,
         q_network='set_ddqn',execution_margin=0.,bootstrap_margin=0.,
         representation=representation,reward_scale=100.)
    replay=EpisodeReplay(a.replay_episodes,a.seed+100,representation)
    audits=[];all_macro=[]
    for ep in source['episodes']:
        result=compress_episode(ep['raw'],representation,a.gamma)
        result.audit['source_episode_wait']=ep['wait']
        replay.append_episode(ep['episode_id'],result.transitions,len(ep['raw']))
        all_macro.extend(result.transitions)
        audits.append(dict(episode=ep['episode_id'],weighted_wait=ep['wait'],**result.audit))
    write_csv(out/'conversion_audit.csv',audits)
    if len(replay)<a.batch_size:
        raise ValueError(f'Only {len(replay)} replay transitions; need batch_size={a.batch_size}')
    curve=[];fixed=all_macro[:min(200,len(all_macro))]
    for step in range(a.gradient_updates):
        batch,meta=replay.sample(a.batch_size,sampling,.25)
        loss=model.update(batch)
        if (step+1)%a.diagnostic_every==0 or step+1==a.gradient_updates:
            qg=_qgap(model,fixed)
            curve.append(dict(gradient_steps=model.gradient_steps,loss=loss,
                mean_q_gap=mean(qg) if qg else '',n_qgap=len(qg),
                replay_size=len(replay),sample_reuse=replay.stats()['sample_reuse']))
            print(f'[fixed {a.group}] grad={model.gradient_steps} loss={loss:.4f}',flush=True)
    write_csv(out/'train_curve.csv',curve)
    torch.save(model.checkpoint(dict(fixed_dataset=str(a.data),gradient_steps=model.gradient_steps)),out/'final.pt')
    manifest=dict(version='v12-fixed-trajectory',group=a.group,
        representation=representation,sampling=sampling,source_file=str(a.data),
        source_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        gamma=a.gamma,gradient_updates=a.gradient_updates,
        actor_data_episodes=len(source['episodes']),replay_stats=replay.stats(),
        seed=a.seed,original_data_config=source['config'])
    if not a.skip_validation:
        if not a.val_cases:raise ValueError('Need --val-cases unless --skip-validation')
        cfg=SimpleNamespace(val_cases=a.val_cases,scenarios=a.scenarios,
          val_seeds=a.val_seeds,budgets=a.budgets,validation_workers=a.validation_workers,
          clock='union',candidate_limit=4,period=12.,reward_scale=100.,
          validation_uncertainty=a.validation_uncertainty,updates=a.updates,
          candidate_screen='positive',min_gain=.1,budget_features='absolute',value_features='global')
        manifest['validation']=validate(out/'final.pt',cfg,out/'validation')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('collect')
    c.add_argument('--cases',nargs='+',required=True)
    c.add_argument('--episodes',type=int,default=64)
    c.add_argument('--scenarios',default='shock,repeated,small')
    c.add_argument('--budgets',default='2,5,10')
    c.add_argument('--uncertainties',default='4,8,12')
    c.add_argument('--updates',type=int,default=2)
    c.add_argument('--train-seed-start',type=int,default=200000)
    c.add_argument('--seed',type=int,default=42)
    c.add_argument('--out',required=True)
    t=sub.add_parser('train')
    t.add_argument('--data',required=True)
    t.add_argument('--group',choices=('A0','A1','A2','A3'),required=True)
    t.add_argument('--gradient-updates',type=int,default=500)
    t.add_argument('--diagnostic-every',type=int,default=25)
    t.add_argument('--replay-episodes',type=int,default=200)
    t.add_argument('--batch-size',type=int,default=64)
    t.add_argument('--gamma',type=float,default=1.)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--device',default='cpu')
    t.add_argument('--val-cases',nargs='+')
    t.add_argument('--scenarios',default='shock,repeated,small')
    t.add_argument('--val-seeds',default='400001:400003')
    t.add_argument('--budgets',default='2,5,10')
    t.add_argument('--updates',type=int,default=2)
    t.add_argument('--validation-uncertainty',type=float,default=8.)
    t.add_argument('--validation-workers',type=int,default=4)
    t.add_argument('--skip-validation',action='store_true')
    t.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.cmd=='collect':collect(a)
    else:train(a)

if __name__=='__main__':main()
