#!/usr/bin/env python3
"""Parallel CPU rollout actors + one central (optional CUDA) Double-DQN learner.

Additive IAMU V0.6/V0.7-compatible training entry point. Does not change
SlotPlanner, EventEnv, reward semantics, model architecture, or original trainer.

Workers use spawn + CPU-only policy copies; only the main process owns the
optimizer, replay buffer, validation and GPU. Actors are *off-policy* and may
use a snapshot a few episodes old; model staleness is explicit in output.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import time
import traceback
from typing import Any

# Limit BLAS/torch thread fanout *before* importing numpy and torch.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

from iamu_dynamic import generate_scenario, read_case
from iamu_dynamic.rl_ddqn import DoubleDQNAgent, ReplayBuffer, Transition
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_experiment import (parse_ints, parse_names, verify_cases,
                                        evaluate_paired, write_csv)


@dataclass(frozen=True)
class RolloutJob:
    episode: int
    case: str
    scenario: str
    seed: int
    uncertainty: float
    budget: int
    epsilon: float


def _supports(function, parameter: str) -> bool:
    return parameter in inspect.signature(function).parameters


def _actor_env_kwargs(settings: dict) -> dict:
    kwargs = dict(budget=settings['budget'], candidate_limit=settings['candidate_limit'],
                  decision_clock=settings['clock'], period=settings['period'],
                  reward_scale=settings['reward_scale'])
    optional = {
        'candidate_screen': settings['candidate_screen'],
        'min_gain': settings['min_gain'],
        'budget_features': settings['budget_features'],
        'value_features': settings['value_features'],
    }
    for name, value in optional.items():
        if _supports(IAMUEventEnv.__init__, name):
            kwargs[name] = value
        else:
            defaults = {'candidate_screen': 'none', 'min_gain': .1,
                        'budget_features': 'legacy', 'value_features': 'legacy'}
            if value != defaults[name]:
                raise RuntimeError(f'Installed environment does not support --{name.replace("_", "-")}')
    return kwargs


def _actor_loop(worker_id: int, inbox, outbox, args: dict):
    """Top-level spawn entry: intentionally never imports/initializes CUDA."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        # Distinct seeds; no global actor random generator used for test cases.
        agent_kwargs = dict(hidden_dim=args['hidden_dim'], seed=args['seed'] + worker_id + 10000,
                            device='cpu')
        if _supports(DoubleDQNAgent.__init__, 'keep_margin'):
            agent_kwargs['keep_margin'] = args['keep_margin']
        elif args['keep_margin'] != 0.:
            raise RuntimeError('Installed DDQN does not support --keep-margin')
        agent = DoubleDQNAgent(**agent_kwargs)
        agent.online.eval()
        case_cache: dict[str, Any] = {}
        model_version = -1
        while True:
            message = inbox.get()
            if message[0] == 'stop':
                break
            if message[0] == 'sync':
                _, weights, model_version = message
                agent.online.load_state_dict(weights, strict=True)
                agent.online.eval()
                continue
            if message[0] != 'rollout':
                raise ValueError(f'Unknown worker message: {message[0]}')
            _, job = message
            if model_version < 0:
                raise RuntimeError('Actor attempted rollout before weight synchronization')
            started = time.perf_counter()
            if job.case not in case_cache:
                case_cache[job.case] = read_case(job.case)
            vessels, port = case_cache[job.case]
            scenario = generate_scenario(vessels, port, label=job.scenario,
                seed=job.seed, updates_per_vessel=args['updates'],
                uncertainty=job.uncertainty)
            env_settings = dict(args, budget=job.budget)
            env = IAMUEventEnv(scenario, **_actor_env_kwargs(env_settings))
            obs = env.reset()
            transitions = []
            total_reward = 0.
            teacher_steps = 0
            while True:
                teacher = None
                if args['greedy_exploration_prob'] > 0:
                    if not _supports(agent.act, 'teacher_action'):
                        raise RuntimeError('Installed DDQN does not support Greedy exploration')
                    teacher = env.point_greedy_action()
                act_kw = dict(epsilon=job.epsilon)
                if teacher is not None:
                    act_kw.update(teacher_action=teacher,
                                  teacher_prob=args['greedy_exploration_prob'])
                action = agent.act(obs, **act_kw)
                teacher_steps += int(teacher is not None and action == teacher)
                nxt, reward, done, info = env.step(action)
                transitions.append(Transition(
                    obs.global_features.copy(), obs.action_features.copy(),
                    obs.mask.copy(), int(action), float(reward),
                    nxt.global_features.copy(), nxt.action_features.copy(),
                    nxt.mask.copy(), bool(done)))
                total_reward += reward
                obs = nxt
                if done:
                    break
            raw_wait = float(info['weighted_wait'])
            if not math.isclose(-total_reward*args['reward_scale'], raw_wait,
                                abs_tol=1e-5*max(1.,raw_wait), rel_tol=1e-6):
                raise AssertionError('Episode rewards do not sum to -weighted_wait')
            if info['spent'] > job.budget:
                raise AssertionError('Actor violated edit budget')
            audit = getattr(env, 'action_audit', [])
            edits = [a for a in audit if a.get('action', 0) != 0]
            report = {
                'worker_id': worker_id, 'worker_pid': os.getpid(),
                'policy_version': model_version, 'episode': job.episode,
                'case': job.case, 'scenario': job.scenario, 'seed': job.seed,
                'uncertainty': job.uncertainty, 'budget': job.budget,
                'epsilon': job.epsilon, 'spent': int(info['spent']),
                'weighted_wait': raw_wait, 'reward_sum_scaled': total_reward,
                'steps': len(transitions),
                'actor_seconds': time.perf_counter()-started,
                'teacher_action_matches': teacher_steps,
                'screened_proposals': getattr(env, 'screened_proposals', 0),
                'last_revision_time': float(getattr(env, 'last_revision_time', 0.) or 0.),
                'mean_estimated_edit_gain': (
                    sum(x['estimated_gain'] for x in edits)/len(edits) if edits else 0.),
            }
            outbox.put(('ok', report, transitions))
    except BaseException:
        # Child failures cannot be allowed to leave the learner hanging forever.
        try:
            outbox.put(('error', worker_id, traceback.format_exc()), timeout=2)
        except Exception:
            pass
        raise


def _cpu_weights(agent: DoubleDQNAgent) -> dict:
    """Detach from learner: torch multiprocessing must not share GPU storage."""
    return {key: val.detach().cpu().clone() for key,val in agent.online.state_dict().items()}


def _save_checkpoint(path: Path, agent: DoubleDQNAgent, meta: dict):
    tmp = path.with_name(path.name+'.tmp')
    torch.save(agent.checkpoint(meta), tmp)
    tmp.replace(path)


def _make_jobs(args, scenarios, budgets, uncertainties) -> list[RolloutJob]:
    rng = random.Random(args.seed)
    jobs=[]
    for ep in range(1,args.episodes+1):
        fraction = min(1., (ep-1)/max(1,args.epsilon_decay_episodes))
        epsilon = args.epsilon_start + (args.epsilon_end-args.epsilon_start)*fraction
        jobs.append(RolloutJob(ep, rng.choice(args.train_cases),
                               rng.choice(scenarios), args.train_seed_start+ep-1,
                               rng.choice(uncertainties), rng.choice(budgets),epsilon))
    return jobs


def _validation_kwargs(args, config, scenarios, budgets, agent):
    kw = dict(cases=args.val_cases, seeds=config['val_seeds'],
              scenarios=scenarios, budgets=budgets,
              policies=['keep','greedy','ddqn'] + (['periodic'] if args.clock=='union' else []),
              agent=agent,candidate_limit=args.candidate_limit,
              decision_clock=args.clock, period=args.period,
              reward_scale=args.reward_scale, uncertainty=args.validation_uncertainty,
              updates=args.updates)
    for name, value in [('candidate_screen',args.candidate_screen),('min_gain',args.min_gain),
                        ('budget_features',args.budget_features),('value_features',args.value_features)]:
        if _supports(evaluate_paired,name):
            kw[name] = value
        elif value != {'candidate_screen':'none','min_gain':.1,
                       'budget_features':'legacy','value_features':'legacy'}[name]:
            raise RuntimeError(f'evaluate_paired lacks {name}; cannot evaluate this model consistently')
    return kw


def _arguments():
    p=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--train-cases',nargs='+',required=True)
    p.add_argument('--val-cases',nargs='+',required=True)
    p.add_argument('--episodes',type=int,default=300)
    p.add_argument('--workers',type=int,default=4,
                   help='CPU rollout processes; 4–8 recommended for 12 physical cores')
    p.add_argument('--prefetch-per-worker',type=int,default=2)
    p.add_argument('--sync-every',type=int,default=4,
                   help='Assigned episodes per actor between snapshot updates')
    p.add_argument('--worker-timeout',type=float,default=60.,
                   help='Failure-check interval; does not kill a long but living rollout')
    p.add_argument('--max-idle-seconds',type=float,default=0.,
                   help='Abort if no worker completes an episode for this many seconds; 0 disables')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--train-seed-start',type=int,default=200000)
    p.add_argument('--val-seeds',default='400001:400005')
    p.add_argument('--uncertainties',default='4,8,12')
    p.add_argument('--validation-uncertainty',type=float,default=8.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--clock',choices=['eta','union','clock'],default='union')
    p.add_argument('--period',type=float,default=12.)
    p.add_argument('--candidate-limit',type=int,default=4)
    p.add_argument('--reward-scale',type=float,default=100.)
    p.add_argument('--candidate-screen',choices=['none','positive'],default='none')
    p.add_argument('--budget-features',choices=['legacy','absolute'],default='absolute')
    p.add_argument('--value-features',choices=['legacy','global'],default='global')
    p.add_argument('--min-gain',type=float,default=.1)
    p.add_argument('--keep-margin',type=float,default=0.)
    p.add_argument('--greedy-exploration-prob',type=float,default=0.)
    p.add_argument('--hidden-dim',type=int,default=96)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--replay-capacity',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--warmup',type=int,default=512)
    p.add_argument('--updates-per-step',type=float,default=1.,
                   help='Gradient updates per observed transition (off-policy learner)')
    p.add_argument('--max-updates-per-episode',type=int,default=0,
                   help='0 = no cap; nonzero caps learner work per returned rollout')
    p.add_argument('--target-update',type=int,default=250)
    p.add_argument('--epsilon-start',type=float,default=1.)
    p.add_argument('--epsilon-end',type=float,default=.05)
    p.add_argument('--epsilon-decay-episodes',type=int,default=300)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu',help='Learner only: cpu or cuda:0, cuda:1 ...')
    p.add_argument('--out',default='results/v08_ddqn_parallel')
    return p.parse_args()


def main(args=None):
    if args is None: args=_arguments()
    if args.episodes<1 or args.workers<1 or args.workers>32:
        raise ValueError('episodes > 0, 1 <= workers <= 32 required')
    if not 1<=args.prefetch_per_worker<=16 or args.sync_every<1:
        raise ValueError('prefetch must be 1..16 and sync_every >=1')
    if args.worker_timeout<=0 or args.max_idle_seconds<0 or args.batch_size<1 or args.warmup<args.batch_size:
        raise ValueError('Invalid timeout/batch/warmup')
    if args.eval_every<1 or args.epsilon_decay_episodes<1 or args.target_update<1:
        raise ValueError('Invalid evaluation/target/epsilon settings')
    if args.updates_per_step<0 or not math.isfinite(args.updates_per_step):
        raise ValueError('updates-per-step must be finite and nonnegative')
    if args.max_updates_per_episode<0 or not 0<=args.greedy_exploration_prob<=1:
        raise ValueError('Invalid cap or teacher probability')
    if args.min_gain<0 or args.keep_margin<0:
        raise ValueError('Negative min gain/keep margin')
    if not 0<=args.epsilon_start<=1 or not 0<=args.epsilon_end<=1:
        raise ValueError('Epsilon must lie in [0,1]')
    if args.replay_capacity<args.warmup:
        raise ValueError('Replay capacity cannot be smaller than warmup')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested ({args.device}) but PyTorch cannot access CUDA')
    torch.set_num_threads(1)
    verify_cases(args.train_cases,args.val_cases)
    scenarios=parse_names(args.scenarios)
    if not set(scenarios).issubset({'shock','repeated','small'}):
        raise ValueError('Unknown scenario')
    budgets=parse_ints(args.budgets)
    if min(budgets)<0: raise ValueError('Negative budget')
    uncertainties=[float(x) for x in parse_names(args.uncertainties)]
    if any(not math.isfinite(x) or x<0 for x in uncertainties):
        raise ValueError('Invalid uncertainty')
    val_seeds=parse_ints(args.val_seeds)
    train_seeds=set(range(args.train_seed_start,args.train_seed_start+args.episodes))
    if train_seeds&set(val_seeds):
        raise ValueError('Train and validation seeds overlap')
    # Fail fast on unsupported optional V0.7 features before starting actors.
    _actor_env_kwargs(dict(vars(args),budget=budgets[0]))
    if args.greedy_exploration_prob and not _supports(DoubleDQNAgent.act,'teacher_action'):
        raise ValueError('Greedy-guided exploration requires the V0.7 DDQN agent')
    if args.keep_margin and not _supports(DoubleDQNAgent.__init__,'keep_margin'):
        raise ValueError('Keep margin requires the V0.7 DDQN agent')
    learner_kw = dict(hidden_dim=args.hidden_dim, learning_rate=args.lr,
                      gamma=args.gamma, target_update_steps=args.target_update,
                      seed=args.seed, device=args.device)
    if _supports(DoubleDQNAgent.__init__, 'keep_margin'):
        learner_kw['keep_margin']=args.keep_margin
    learner=DoubleDQNAgent(**learner_kw)
    print(f'[learner] device={next(learner.online.parameters()).device}, '
          f'workers={args.workers}, torch_threads={torch.get_num_threads()}',flush=True)
    replay=ReplayBuffer(args.replay_capacity,args.seed+100)
    jobs=_make_jobs(args,scenarios,budgets,uncertainties)
    settings=dict(vars(args))
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    config=dict(train_cases=args.train_cases,val_cases=args.val_cases,
                candidate_limit=args.candidate_limit,decision_clock=args.clock,
                period=args.period,reward_scale=args.reward_scale,
                candidate_screen=args.candidate_screen,min_gain=args.min_gain,
                budget_features=args.budget_features,value_features=args.value_features,
                scenarios=scenarios,budgets=budgets,uncertainties=uncertainties,
                updates=args.updates,train_seed_start=args.train_seed_start,
                val_seeds=val_seeds,training_args=vars(args),
                parallel_workers=args.workers,prefetch_per_worker=args.prefetch_per_worker,
                actor_sync_every=args.sync_every,
                parallel_off_policy=True,
                warning='Actors use stale CPU snapshots; not numerically identical to serial V0.7')
    (output/'manifest.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    ctx=mp.get_context('spawn')
    result_q=ctx.Queue(maxsize=max(4,args.workers*3))
    cmd_queues=[ctx.Queue(maxsize=args.prefetch_per_worker*3+6) for _ in range(args.workers)]
    children=[ctx.Process(target=_actor_loop,args=(i,cmd_queues[i],result_q,settings),
                          name=f'iamu-rollout-{i}',daemon=False) for i in range(args.workers)]
    train_rows=[];val_rows=[];best_val=float('inf')
    assigned=[0]*args.workers
    current_snapshot=[-1]*args.workers
    next_job=0;completed=0;queued=0;rem_grad_credit=0.
    start=time.perf_counter();last_report=start;last_result=start
    workers_used=set()
    try:
        for child in children:child.start()
        def assign(wid:int):
            nonlocal next_job,queued
            if next_job>=len(jobs):return False
            if assigned[wid]%args.sync_every==0:
                v=learner.gradient_steps
                cmd_queues[wid].put(('sync',_cpu_weights(learner),v))
                current_snapshot[wid]=v
            job=jobs[next_job]
            cmd_queues[wid].put(('rollout',job))
            assigned[wid]+=1;next_job+=1;queued+=1
            return True
        for i in range(args.workers):
            for _ in range(args.prefetch_per_worker):
                assign(i)
        while completed<len(jobs):
            try:
                msg=result_q.get(timeout=args.worker_timeout)
            except queue.Empty:
                if args.max_idle_seconds and time.perf_counter()-last_result > args.max_idle_seconds:
                    raise TimeoutError(f'No completed rollout for {args.max_idle_seconds:.1f}s; inspect worker stack before retry')
                failed=[(p.name,p.exitcode) for p in children if not p.is_alive()]
                if failed:raise RuntimeError(f'Workers exited unexpectedly: {failed}')
                print(f'[monitor] {completed}/{len(jobs)} episodes complete; '
                      f'{queued-completed} outstanding; workers alive; '
                      f'elapsed={time.perf_counter()-start:.1f}s',flush=True)
                continue
            if msg[0]=='error':
                raise RuntimeError(f'Actor {msg[1]} failed:\n{msg[2]}')
            if msg[0]!='ok':raise RuntimeError(f'Unexpected actor result {msg[0]}')
            report,transitions=msg[1],msg[2]
            last_result=time.perf_counter()
            workers_used.add(report['worker_pid'])
            completed+=1
            # Prefetch BEFORE learner updates so the actor runs in parallel
            # with CUDA backprop instead of waiting idle.
            assign(report['worker_id'])
            ready=0
            for trans in transitions:
                replay.items.append(trans)
                if len(replay)>=args.warmup:ready+=1
            rem_grad_credit+=ready*args.updates_per_step
            updates=int(rem_grad_credit)
            if args.max_updates_per_episode:
                updates=min(updates,args.max_updates_per_episode)
            rem_grad_credit-=updates
            # When capped, unserved gradient credit is discarded, not allowed
            # to accumulate unbounded and cause a later training stall.
            if args.max_updates_per_episode:
                rem_grad_credit=min(rem_grad_credit,1.)
            losses=[]
            learning_start=time.perf_counter()
            for _ in range(updates):
                losses.append(learner.update(replay.sample(args.batch_size)))
            learn_seconds=time.perf_counter()-learning_start
            wall=time.perf_counter()-start
            row=dict(report,completed=completed,
                     completion_order=completed,
                     policy_staleness_steps=learner.gradient_steps-report['policy_version'],
                     learner_updates=updates,
                     loss=(sum(losses)/len(losses) if losses else ''),
                     gradient_steps=learner.gradient_steps,replay_size=len(replay),
                     learner_seconds=learn_seconds,wall_seconds=wall,
                     episodes_per_hour=3600*completed/max(1e-9,wall))
            train_rows.append(row)
            if completed%args.eval_every==0 or completed==len(jobs):
                validate_start=time.perf_counter()
                _,summary=evaluate_paired(**_validation_kwargs(args,config,
                                          scenarios,budgets,learner))
                eval_seconds=time.perf_counter()-validate_start
                rl=[r for r in summary if r['method']=='ddqn']
                score=sum(r['mean_wait'] for r in rl)/len(rl)
                item=dict(completed=completed,mean_wait=score,
                          mean_gain=sum(r['mean_gain'] for r in rl)/len(rl),
                          worst_gain=min(r['worst_gain'] for r in rl),
                          mean_spent=sum(r['mean_spent'] for r in rl)/len(rl),
                          gradient_steps=learner.gradient_steps,
                          validation_seconds=eval_seconds,
                          wall_seconds=time.perf_counter()-start)
                val_rows.append(item)
                cp_meta={**config,'selected_episode':completed,
                         'selection_score':score,'n_gradient_updates':learner.gradient_steps}
                _save_checkpoint(output/'latest.pt',learner,cp_meta)
                if score<best_val:
                    best_val=score
                    _save_checkpoint(output/'best.pt',learner,cp_meta)
                write_csv(output/'train_curve.csv',train_rows)
                write_csv(output/'validation_curve.csv',val_rows)
                print(f'[validation] completed={completed}/{len(jobs)} '
                      f'wait={score:.3f} gain={item["mean_gain"]:+.3f} '
                      f'best={best_val:.3f} grads={learner.gradient_steps} '
                      f'workers={len(workers_used)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            elif time.perf_counter()-last_report>30:
                print(f'[train] completed={completed}/{len(jobs)}, '
                      f'grads={learner.gradient_steps}, workers={len(workers_used)}',flush=True)
                last_report=time.perf_counter()
        write_csv(output/'train_curve.csv',train_rows)
        write_csv(output/'validation_curve.csv',val_rows)
        (output/'performance.json').write_text(json.dumps({
           'completed_episodes':completed,'workers_requested':args.workers,
           'workers_used':len(workers_used),'actor_pids':sorted(workers_used),
           'gradient_steps':learner.gradient_steps,
           'total_wall_seconds':time.perf_counter()-start,
           'avg_episodes_per_hour':3600*completed/max(1e-9,time.perf_counter()-start),
           'avg_actor_seconds':sum(r['actor_seconds'] for r in train_rows)/len(train_rows),
           'avg_learner_seconds':sum(r['learner_seconds'] for r in train_rows)/len(train_rows),
           'note':'End-to-end CUDA utilization depends on environment cost and network size.'
        },indent=2),encoding='utf-8')
        print(f'[complete] {completed} episodes, {learner.gradient_steps} updates, '
              f'{len(workers_used)} CPU actor PIDs, best={output/"best.pt"}',flush=True)
    finally:
        for q in cmd_queues:
            try:q.put_nowait(('stop',))
            except queue.Full:pass
        for child in children:
            child.join(timeout=5)
            if child.is_alive():child.terminate()
            child.join(timeout=3)
            if child.is_alive():child.kill()
        for q in cmd_queues:
            q.close()
        result_q.close()


if __name__ == '__main__':
    mp.freeze_support()
    main()
