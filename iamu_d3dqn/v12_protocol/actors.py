"""V1.2 CPU actor: same V1.1 physical rollout, explicit per-event budget metadata.

Do not remove calls to agent.act at forced Keep events: RNG parity matters.
"""
from __future__ import annotations
from dataclasses import dataclass
import os,queue,time,math,traceback
import torch
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent,Transition
from iamu_dynamic.rl_env_v11 import IAMUEventEnv
from v11_protocol.v11_actors import RolloutJob,_actor_env_kwargs
from .transitions import RawStep


def _actor_loop(worker_id,inbox,outbox,args):
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        torch.set_num_threads(1)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        agent=DoubleDQNAgent(hidden_dim=args['hidden_dim'],seed=args['seed']+worker_id+10000,
                              device='cpu',q_network=args['q_network'],
                              execution_margin=args['execution_margin'],
                              bootstrap_margin=args['bootstrap_margin'])
        agent.online.eval()
        outbox.put(('actor-online',worker_id,os.getpid()))
        cache={};version=-1
        while True:
            msg=inbox.get()
            if msg[0]=='stop':break
            if msg[0]=='sync':
                _,weights,version=msg
                agent.online.load_state_dict(weights,strict=True)
                agent.online.eval()
                continue
            if msg[0]!='rollout':raise ValueError('Invalid worker command')
            job=msg[1]
            if version<0:raise RuntimeError('Rollout before sync')
            agent.rng.seed(args['seed']*1000003+job.episode*10007+job.seed)
            started=time.perf_counter()
            outbox.put(('actor-start',worker_id,os.getpid(),job.episode,version))
            if job.case not in cache:cache[job.case]=read_case(job.case)
            vessels,port=cache[job.case]
            instance=generate_scenario(vessels,port,label=job.scenario,seed=job.seed,
                        updates_per_vessel=args['updates'],uncertainty=job.uncertainty)
            env=IAMUEventEnv(instance,**_actor_env_kwargs(dict(args,budget=job.budget)))
            obs=env.reset()
            trajectory=[];total=0.;teacher_steps=0;heartbeat=started
            while True:
                teacher=None
                if args['greedy_exploration_prob']>0:
                    teacher=env.point_greedy_action()
                # Call act() on every event, including no-choice observations.
                kw=dict(epsilon=job.epsilon)
                if teacher is not None:
                    kw.update(teacher_action=teacher,teacher_prob=args['greedy_exploration_prob'])
                action=agent.act(obs,**kw)
                teacher_steps+=int(teacher is not None and action==teacher)
                source_budget=int(obs.remaining_budget)
                source_event=int(env.event_index)
                source_time=float(obs.time)
                nxt,reward,done,info=env.step(action)
                t=Transition(obs.global_features.copy(),obs.action_features.copy(),obs.mask.copy(),
                    int(action),float(reward),nxt.global_features.copy(),nxt.action_features.copy(),
                    nxt.mask.copy(),bool(done),dict(obs.decision_flags or {}),int(job.episode))
                trajectory.append(RawStep(t,source_budget,int(nxt.remaining_budget),
                    source_event,int(env.event_index),source_time,float(nxt.time),
                    int(job.episode),int(version)))
                total+=reward;obs=nxt
                if time.perf_counter()-heartbeat>=15:
                    try:outbox.put_nowait(('actor-heartbeat',worker_id,os.getpid(),job.episode,
                                           len(trajectory),round(time.perf_counter()-started,2)))
                    except queue.Full:pass
                    heartbeat=time.perf_counter()
                if done:break
            wait=float(info['weighted_wait'])
            if not math.isclose(-total*args['reward_scale'],wait,rel_tol=1e-6,
                                abs_tol=1e-5*max(1.,wait)):
                raise AssertionError('Reward total differs from weighted waiting')
            if info['spent']>job.budget:raise AssertionError('Overspent budget')
            flags={}
            for raw in trajectory:
                for k,v in (raw.transition.reason_flags or {}).items():
                    if isinstance(v,bool):flags[k]=flags.get(k,0)+int(v)
            report=dict(worker_id=worker_id,worker_pid=os.getpid(),
                policy_version=int(version),episode=int(job.episode),case=job.case,
                scenario=job.scenario,seed=job.seed,uncertainty=job.uncertainty,
                budget=job.budget,epsilon=job.epsilon,spent=int(info['spent']),
                weighted_wait=wait,reward_sum_scaled=total,steps=len(trajectory),
                forced_flags=flags,actor_seconds=time.perf_counter()-started,
                teacher_action_matches=teacher_steps,
                screened_proposals=getattr(env,'screened_proposals',0))
            outbox.put(('ok',report,trajectory))
    except BaseException:
        try:outbox.put(('error',worker_id,traceback.format_exc()),timeout=2)
        except Exception:pass
        raise
