#!/usr/bin/env python3
"""Phase-synchronous 4-CPU rollout / 4-CPU validation with a central DDQN learner.

Additive IAMU V0.6/V0.7-compatible training entry point. Does not change
SlotPlanner, EventEnv, reward semantics, model architecture, or original trainer.

Workers use spawn + CPU-only policy copies; only the main process owns the
optimizer, replay buffer and GPU. Actors are *off-policy* within a training
phase; all rollouts in that phase finish before checkpointed CPU validation.
Training does NOT run concurrently with validation.
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
import shutil
import time
import traceback
from typing import Any
from datetime import datetime, timezone
from collections import deque

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
from parallel_validation_v081 import validation_coordinator, file_sha256
from parallel_recovery_v081 import atomic_json, save_state, load_state


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
        outbox.put(('actor-online', worker_id, os.getpid()))
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
            outbox.put(('actor-start', worker_id, os.getpid(), job.episode, model_version))
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
            last_heartbeat = started
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
                # Nonblocking heartbeat: slow candidate projections can take
                # substantial CPU time and must be observable in live logs.
                if time.perf_counter()-last_heartbeat >= 15.:
                    try:
                        outbox.put_nowait(('actor-heartbeat', worker_id,
                                           os.getpid(),job.episode,len(transitions),
                                           round(time.perf_counter()-started,2)))
                    except queue.Full:
                        pass
                    last_heartbeat=time.perf_counter()
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
    p.add_argument('--validation-workers',type=int,default=4,
                   help='Four CPU validation processes; training rollouts pause during validation')
    p.add_argument('--validation-progress-every',type=int,default=6,
                   help='Emit progress after every N completed validation groups')
    p.add_argument('--state-every',type=int,default=5,
                   help='Save complete learner/replay state every N completed episodes')
    p.add_argument('--resume',action='store_true',
                   help='Resume from trusted OUT/training_state.pt; actor scheduling is not bitwise reproducible')
    p.add_argument('--log-every',type=int,default=1,help='Emit training status every N episodes')
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


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp=path.with_suffix(path.suffix+'.tmp')
    write_csv(tmp,rows)
    tmp.replace(path)


def _record_event(output: Path, event: str, **fields):
    item=dict(timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds'),
              event=event, **fields)
    with (output/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(item,ensure_ascii=False,default=str)+'\n')
        f.flush()
    return item


def _validation_settings(args, config, scenarios, budgets) -> dict:
    # Exactly the V0.8 validation methods and Cartesian-product cardinality.
    return dict(val_cases=list(args.val_cases),val_seeds=list(config['val_seeds']),
                scenarios=list(scenarios),budgets=list(budgets),
                policies=['keep','greedy','ddqn']+(['periodic'] if args.clock=='union' else []),
                candidate_limit=args.candidate_limit,clock=args.clock,
                period=args.period,reward_scale=args.reward_scale,
                validation_uncertainty=args.validation_uncertainty,updates=args.updates,
                candidate_screen=args.candidate_screen,min_gain=args.min_gain,
                budget_features=args.budget_features,value_features=args.value_features)


def _select_best_snapshot(output: Path, snapshot: Path, *, score: float,
                          validation_id: str, completed: int,
                          gradient_steps: int, snapshot_sha256: str) -> None:
    """Save the EXACT evaluated network, never learner's current parameters."""
    item=torch.load(snapshot,map_location='cpu',weights_only=True)
    item['meta'].update(selection_score=score,validation_id=validation_id,
                        selected_episode=completed,n_gradient_updates=gradient_steps,
                        validated_snapshot_sha256=snapshot_sha256)
    tmp=output/'best.pt.tmp'
    torch.save(item,tmp)
    os.replace(tmp,output/'best.pt')


def main(args=None):
    if args is None: args=_arguments()
    if args.episodes<1 or args.workers<1 or args.workers>32:
        raise ValueError('episodes > 0, 1 <= workers <= 32 required')
    if not 1<=args.validation_workers<=16 or args.validation_progress_every<1:
        raise ValueError('validation-workers 1..16, progress-every >= 1 required')
    if args.state_every<1 or args.log_every<1:
        raise ValueError('state-every and log-every must be >= 1')
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
    if min(budgets)<0:raise ValueError('Negative budget')
    uncertainties=[float(x) for x in parse_names(args.uncertainties)]
    if any(not math.isfinite(x) or x<0 for x in uncertainties):
        raise ValueError('Invalid uncertainty')
    val_seeds=parse_ints(args.val_seeds)
    if set(range(args.train_seed_start,args.train_seed_start+args.episodes))&set(val_seeds):
        raise ValueError('Train and validation seeds overlap')
    _actor_env_kwargs(dict(vars(args),budget=budgets[0]))
    if args.greedy_exploration_prob and not _supports(DoubleDQNAgent.act,'teacher_action'):
        raise ValueError('Greedy exploration requires V0.7 DDQN')
    if args.keep_margin and not _supports(DoubleDQNAgent.__init__,'keep_margin'):
        raise ValueError('Keep margin requires V0.7 DDQN')

    learner_kw=dict(hidden_dim=args.hidden_dim,learning_rate=args.lr,
                    gamma=args.gamma,target_update_steps=args.target_update,
                    seed=args.seed,device=args.device)
    if _supports(DoubleDQNAgent.__init__,'keep_margin'):
        learner_kw['keep_margin']=args.keep_margin
    learner=DoubleDQNAgent(**learner_kw)
    replay=ReplayBuffer(args.replay_capacity,args.seed+100)
    jobs=_make_jobs(args,scenarios,budgets,uncertainties)
    config=dict(train_cases=args.train_cases,val_cases=args.val_cases,
                candidate_limit=args.candidate_limit,decision_clock=args.clock,
                period=args.period,reward_scale=args.reward_scale,
                candidate_screen=args.candidate_screen,min_gain=args.min_gain,
                budget_features=args.budget_features,value_features=args.value_features,
                scenarios=scenarios,budgets=budgets,uncertainties=uncertainties,
                updates=args.updates,train_seed_start=args.train_seed_start,
                val_seeds=val_seeds,training_args=vars(args),
                parallel_workers=args.workers,prefetch_per_worker=args.prefetch_per_worker,
                actor_sync_every=args.sync_every,parallel_off_policy=True,
                validation_parallel=True,validation_sync_phases=True,
                warning='Phase-synchronous validation; rollout completion order remains nondeterministic')
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=True)
    state_file=output/'training_state.pt'
    if not args.resume and any((output/name).exists() for name in
                               ('training_state.pt','manifest.json','train_curve.csv')):
        raise FileExistsError(f'{output} already contains training artifacts; '
                              'use --resume (for V0.8.1 state) or a NEW --out')
    settings=dict(vars(args))
    val_settings=_validation_settings(args,config,scenarios,budgets)
    expected_groups=(len(val_settings['val_cases'])*len(val_settings['val_seeds'])*
                     len(scenarios)*len(budgets))
    expected_runs=expected_groups*len(val_settings['policies'])
    # Original union-clock 2x3x3x3x4 = 216 every validation, unchanged.
    train_rows=[];val_rows=[];validations={};best_val=float('inf');grad_credit=0.
    if args.resume:
        item=load_state(state_file,learner=learner,replay=replay,config=config)
        if not item['config'].get('validation_sync_phases'):
            raise ValueError('This is an asynchronous V0.8.1 state. For phase-synchronous V0.8.2, '
                             'start a NEW --out; old mid-validation trajectories cannot be rewound.')
        train_rows=item['train_rows'];val_rows=item['val_rows']
        validations=item['validations'];best_val=float(item['best_val'])
        grad_credit=float(item['gradient_credit'])
        print(f'[resume] restored completed={len(train_rows)} gradients={learner.gradient_steps} '
              f'pending_validations={sum(v["status"]=="pending" for v in validations.values())}; '
              'in-flight actor episodes will be requeued',flush=True)
    else:
        atomic_json(config,output/'manifest.json')
    if args.resume:
        # Validation callbacks that arrived after the LAST full state may have
        # written CSVs. Canonical recovery rolls back to last atomic state.
        _atomic_csv(output/'train_curve.csv',train_rows)
        _atomic_csv(output/'validation_curve.csv',val_rows)
    else:
        _record_event(output,'startup',episodes=len(jobs),workers=args.workers,
                      validation_workers=args.validation_workers,
                      validation_runs_per_check=expected_runs)
    completed_ids={int(r['episode']) for r in train_rows}
    if len(completed_ids)!=len(train_rows):
        raise ValueError('Duplicate completed episodes in recovery state')
    pending=deque(job for job in jobs if job.episode not in completed_ids)
    completed=len(completed_ids)
    # Do not send a single episode beyond the next validation boundary.
    # Prefetch cannot cross this barrier even if earlier rollouts finish out of order.
    stage_limits=list(range(args.eval_every,len(jobs),args.eval_every))+[len(jobs)]
    stage_index=0
    while (stage_index<len(stage_limits) and completed>=stage_limits[stage_index]
           and validations.get(f'val_{stage_limits[stage_index]:06d}',{}).get('status')=='done'):
        stage_index+=1
    stage_limit=(stage_limits[stage_index] if stage_index<len(stage_limits) else None)
    if stage_limit is not None and completed>stage_limit:
        raise ValueError('Invalid resume state: completed episodes exceed active phase boundary')
    # Existing best checkpoint may have been written AFTER the last state save.
    # Restore its identity from the state, never silently use a future model.
    complete_validations=[v for v in validations.values() if v['status']=='done']
    if complete_validations:
        winner=min(complete_validations,key=lambda v:(v['score'],v['completed']))
        best_val=winner['score']
        _select_best_snapshot(output,Path(winner['snapshot']),
            score=winner['score'],validation_id=winner['id'],
            completed=winner['completed'],gradient_steps=winner['gradient_steps'],
            snapshot_sha256=winner['snapshot_sha256'])
    elif args.resume and (output/'best.pt').exists():
        (output/'best.pt').unlink()  # never expose stale unaccounted best model

    def persist(*,save_full: bool,reason: str):
        _atomic_csv(output/'train_curve.csv',train_rows)
        _atomic_csv(output/'validation_curve.csv',val_rows)
        atomic_json(dict(validation_runs_per_check=expected_runs,
                         validation_groups_per_check=expected_groups,
                         records=validations),output/'validation_ledger.json')
        if save_full:
            # Also save an inference-ready unvalidated 'latest' network.
            _save_checkpoint(output/'latest.pt',learner,
                             {**config,'selected_episode':completed,
                              'n_gradient_updates':learner.gradient_steps,
                              'validation_status':'not necessarily validated'})
            save_state(state_file,learner=learner,replay=replay,config=config,
                       train_rows=train_rows,val_rows=val_rows,
                       validations=validations,best_val=best_val,
                       gradient_credit=grad_credit)
            _record_event(output,'training_state_saved',reason=reason,
                          completed=completed,gradient_steps=learner.gradient_steps,
                          replay_size=len(replay))
    if args.resume:
        persist(save_full=False,reason='resume-log-reconciliation')
        _record_event(output,'resumed',completed=completed,gradient_steps=learner.gradient_steps)

    ctx=mp.get_context('spawn')
    result_q=ctx.Queue(maxsize=max(8,args.workers*4))
    cmd_queues=[ctx.Queue(maxsize=args.prefetch_per_worker*3+6)
                for _ in range(args.workers)]
    children=[ctx.Process(target=_actor_loop,
                          args=(i,cmd_queues[i],result_q,settings),
                          name=f'iamu-rollout-{i}',daemon=False)
              for i in range(args.workers)]
    validation_req=ctx.Queue()  # all required validation snapshots must be evaluated
    validation_report=ctx.Queue()
    validator=ctx.Process(target=validation_coordinator,
        args=(validation_req,validation_report,args.validation_workers,
              str(output),args.validation_progress_every),
        name='iamu-validation-coordinator',daemon=False)
    assigned=[0]*args.workers;queued=completed
    start=time.perf_counter();last_actor_result=start;last_monitor=start
    worker_pids=set();validation_pids=set();actor_status={}
    validation_count=0

    def schedule_validation(completed_number: int):
        validation_id=f'val_{completed_number:06d}'
        if validation_id in validations:
            return
        grad=learner.gradient_steps
        snapshot=output/'validation_snapshots'/f'{validation_id}_grad_{grad:09d}.pt'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        _save_checkpoint(snapshot,learner,
                         {**config,'selected_episode':completed_number,
                          'n_gradient_updates':grad,'validation_id':validation_id,
                          'validation_status':'pending'})
        record=dict(id=validation_id,completed=completed_number,gradient_steps=grad,
                    snapshot=str(snapshot),snapshot_sha256=file_sha256(snapshot),
                    status='pending',expected_runs=expected_runs,
                    expected_groups=expected_groups)
        validations[validation_id]=record
        # All recovery state must be durable BEFORE request enters the queue.
        persist(save_full=True,reason='before-validation')
        validation_req.put(('validate',validation_id,str(snapshot),val_settings))
        _record_event(output,'validation_queued',validation_id=validation_id,
                      completed=completed_number,gradient_steps=grad,
                      snapshot=str(snapshot),expected_runs=expected_runs,
                      queue_backlog=sum(v['status']=='pending' for v in validations.values()))
        print(f'[validation-queued] {validation_id} frozen_grad={grad} '
              f'runs={expected_runs} groups={expected_groups}; training paused',flush=True)

    def drain_validation():
        nonlocal best_val,validation_count
        while True:
            try:msg=validation_report.get_nowait()
            except queue.Empty:break
            tag=msg[0]
            if tag=='validation-worker-online':
                _,vid,pid=msg
                validation_pids.add(pid)
                print(f'[validation-worker-online] {vid} pid={pid}',flush=True)
                _record_event(output,'validation_worker_online',validation_id=vid,pid=pid)
                continue
            if tag=='validation-task-start':
                _,vid,idx,pid,case,scenario,seed,budget=msg
                _record_event(output,'validation_task_start',validation_id=vid,
                              group_index=idx,worker_pid=pid,case=case,
                              scenario=scenario,seed=seed,budget=budget)
                print(f'[validation-task-start] {vid} worker={pid} group={idx+1} '
                      f'case={Path(case).name} scenario={scenario} seed={seed} B={budget}',
                      flush=True)
                continue
            if tag=='validation-error':
                raise RuntimeError(f'Validation coordinator failed:\n{msg[1]}')
            if tag=='validation-start':
                _,vid,groups,runs,snapshot,pid=msg
                _record_event(output,'validation_started',validation_id=vid,
                              groups=groups,runs=runs,coordinator_pid=pid)
                print(f'[validation-start] {vid} {groups} groups / {runs} simulations '
                      f'coordinator={pid}',flush=True)
            elif tag=='validation-progress':
                _,vid,groups,total,done_runs,total_runs,elapsed,pids=msg
                validation_pids.update(pids)
                _record_event(output,'validation_progress',validation_id=vid,
                    completed_groups=groups,total_groups=total,completed_runs=done_runs,
                    total_runs=total_runs,elapsed_seconds=elapsed,worker_pids=pids)
                print(f'[validation-progress] {vid} {done_runs}/{total_runs} '
                      f'elapsed={elapsed:.1f}s pids={pids}',flush=True)
            elif tag=='validation-finished':
                (_,vid,score,summary,runs,groups,seconds,snapshot,checksum,pids)=msg
                record=validations[vid]
                if record['status']=='done':continue
                if queued != completed:
                    raise RuntimeError('Validation completed while training rollouts were in flight')
                if learner.gradient_steps != record['gradient_steps']:
                    raise RuntimeError('Learner parameters changed during phase-synchronous validation')
                if (runs!=record['expected_runs'] or groups!=record['expected_groups']
                        or snapshot!=record['snapshot'] or checksum!=record['snapshot_sha256']):
                    raise RuntimeError(f'{vid}: validation metadata/snapshot mismatch')
                rld=[r for r in summary if r['method']=='ddqn']
                if len(rld)!=len(budgets):raise RuntimeError('Missing DDQN in validation')
                row=dict(validation_id=vid,completed=record['completed'],
                         mean_wait=score,mean_gain=sum(r['mean_gain'] for r in rld)/len(rld),
                         worst_gain=min(r['worst_gain'] for r in rld),
                         mean_spent=sum(r['mean_spent'] for r in rld)/len(rld),
                         gradient_steps=record['gradient_steps'],
                         learner_gradient_steps_at_receive=learner.gradient_steps,
                         validation_seconds=seconds,expected_runs=runs,
                         evaluated_runs=runs,snapshot_sha256=checksum,
                         worker_pids=';'.join(str(x) for x in pids),
                         wall_seconds=time.perf_counter()-start)
                val_rows.append(row)
                record.update(status='done',score=score,validation_seconds=seconds,
                              evaluated_runs=runs,worker_pids=pids,
                              snapshot_sha256=checksum)
                if score<best_val:
                    best_val=score
                    _select_best_snapshot(output,Path(snapshot),score=score,
                         validation_id=vid,completed=record['completed'],
                         gradient_steps=record['gradient_steps'],
                         snapshot_sha256=checksum)
                    _record_event(output,'best_updated',validation_id=vid,
                                  mean_wait=score,snapshot_sha256=checksum)
                validation_count+=1
                persist(save_full=True,reason='validation-finished')
                _record_event(output,'validation_finished',validation_id=vid,
                              mean_wait=score,elapsed_seconds=round(seconds,3),
                              frozen_gradient_steps=record['gradient_steps'],
                              current_gradient_steps=learner.gradient_steps,
                              worker_pids=pids)
                print(f'[validation-finished] {vid} wait={score:.3f} best={best_val:.3f} '
                      f'validated_grad={record["gradient_steps"]} '
                      f'current_grad={learner.gradient_steps} '
                      f'{runs} runs in {seconds:.1f}s',flush=True)
            else:raise RuntimeError(f'Unexpected validation message {tag}')

    def assign(wid:int):
        nonlocal queued
        if not pending or stage_limit is None or pending[0].episode>stage_limit:
            return False
        if assigned[wid]%args.sync_every==0:
            v=learner.gradient_steps
            cmd_queues[wid].put(('sync',_cpu_weights(learner),v))
        job=pending.popleft()
        cmd_queues[wid].put(('rollout',job))
        assigned[wid]+=1;queued+=1
        return True

    def wait_for_validation(validation_id: str):
        """Strict phase barrier. Learner and rollout actors are idle here."""
        started=time.perf_counter()
        last_heartbeat=started
        _record_event(output,'phase_validation_start',validation_id=validation_id,
                      completed=completed,gradient_steps=learner.gradient_steps,
                      outstanding_rollouts=queued-completed)
        if queued != completed:
            raise RuntimeError('Validation barrier reached with unfinished actor rollouts')
        while validations[validation_id]['status']!='done':
            drain_validation()
            if validations[validation_id]['status']=='done':break
            check_processes()
            time.sleep(.2)
            if time.perf_counter()-last_heartbeat>=15.:
                print(f'[phase-validation] {validation_id} completed={completed} '
                      f'gradients={learner.gradient_steps} '
                      f'elapsed={time.perf_counter()-started:.1f}s',flush=True)
                last_heartbeat=time.perf_counter()
        _record_event(output,'phase_validation_end',validation_id=validation_id,
                      completed=completed,gradient_steps=learner.gradient_steps,
                      elapsed_seconds=time.perf_counter()-started)

    def check_processes():
        failed=[(p.name,p.exitcode) for p in children if not p.is_alive()]
        if failed:raise RuntimeError(f'Actor processes exited unexpectedly: {failed}')
        if not validator.is_alive():
            raise RuntimeError(f'Validation coordinator exited unexpectedly: {validator.exitcode}')

    try:
        validator.start()
        for child in children:child.start()
        _record_event(output,'processes_started',validator_pid=validator.pid,
                      actor_pids=[p.pid for p in children],
                      validation_workers=args.validation_workers)
        print(f'[learner] device={next(learner.online.parameters()).device} '
              f'actors={args.workers}, validation_workers={args.validation_workers}, '
              f'episodes={len(jobs)} completed={completed}, '
              f'validation_runs_per_check={expected_runs} mode=synchronous-phases',flush=True)
        # Resume must complete its frozen validation before ANY new rollout.
        for r in sorted(validations.values(),key=lambda x:x['completed']):
            if r['status']=='pending':
                snapshot=Path(r['snapshot'])
                if not snapshot.exists() or file_sha256(snapshot)!=r['snapshot_sha256']:
                    raise RuntimeError(f'Resumed validation snapshot missing/corrupt: {snapshot}')
                validation_req.put(('validate',r['id'],str(snapshot),val_settings))
                print(f'[validation-resumed] {r["id"]} snapshot={snapshot.name}',flush=True)
        for r in sorted(validations.values(),key=lambda x:x['completed']):
            if r['status']=='pending':
                wait_for_validation(r['id'])
        # All pending validations were evaluated using their original frozen snapshots.
        while (stage_index<len(stage_limits) and completed>=stage_limits[stage_index]
               and validations.get(f'val_{stage_limits[stage_index]:06d}',{}).get('status')=='done'):
            stage_index+=1
        stage_limit=(stage_limits[stage_index] if stage_index<len(stage_limits) else None)
        if stage_limit is not None and completed==stage_limit:
            # Defensive recovery if an episode boundary had been saved but its
            # validation request was not yet persisted.
            schedule_validation(completed)
            wait_for_validation(f'val_{completed:06d}')
            stage_index+=1
            stage_limit=(stage_limits[stage_index] if stage_index<len(stage_limits) else None)
        for wid in range(args.workers):
            for _ in range(args.prefetch_per_worker):
                assign(wid)
        last_progress=0.
        if stage_limit is not None:
            _record_event(output,'phase_training_start',stage_limit=stage_limit,
                          completed=completed,gradient_steps=learner.gradient_steps)
            print(f'[phase-training] completed={completed} next_validation={stage_limit}',flush=True)
        while completed<len(jobs):
            drain_validation()
            try:
                msg=result_q.get(timeout=min(args.worker_timeout,2.0))
            except queue.Empty:
                now=time.perf_counter()
                if args.max_idle_seconds and now-last_actor_result>args.max_idle_seconds:
                    raise TimeoutError('No completed actor rollout; inspect worker stacks')
                check_processes()
                if now-last_monitor>15.:
                    print(f'[monitor] completed={completed}/{len(jobs)} '
                          f'outstanding={queued-completed} gradients={learner.gradient_steps} '
                          f'validation_pending={sum(v["status"]=="pending" for v in validations.values())} '
                          f'actor_status={actor_status}',flush=True)
                    _record_event(output,'monitor',completed=completed,
                                  outstanding=queued-completed,gradient_steps=learner.gradient_steps,
                                  actor_status=actor_status)
                    last_monitor=now
                continue
            if msg[0]=='error':raise RuntimeError(f'Actor {msg[1]} failed:\n{msg[2]}')
            if msg[0]=='actor-online':
                _,wid,pid=msg;actor_status[wid]=dict(pid=pid,state='online')
                print(f'[actor-online] worker={wid} pid={pid}',flush=True)
                _record_event(output,'actor_online',worker_id=wid,pid=pid)
                continue
            if msg[0]=='actor-start':
                _,wid,pid,ep,version=msg
                actor_status[wid]=dict(pid=pid,state='running',episode=ep,policy_version=version)
                print(f'[actor-start] worker={wid} pid={pid} episode={ep} '
                      f'policy_version={version}',flush=True)
                _record_event(output,'actor_start',worker_id=wid,pid=pid,
                              episode=ep,policy_version=version)
                continue
            if msg[0]=='actor-heartbeat':
                _,wid,pid,ep,steps,seconds=msg
                actor_status[wid]=dict(pid=pid,state='running',episode=ep,
                                       observed_steps=steps,seconds=seconds)
                print(f'[actor-heartbeat] worker={wid} ep={ep} '
                      f'steps={steps} elapsed={seconds}s',flush=True)
                continue
            if msg[0]!='ok':raise RuntimeError(f'Unexpected actor result: {msg[0]}')
            report,transitions=msg[1],msg[2]
            last_actor_result=time.perf_counter()
            wid=report['worker_id'];worker_pids.add(report['worker_pid'])
            actor_status[wid]=dict(pid=report['worker_pid'],state='completed',
                                   episode=report['episode'])
            if report['episode'] in completed_ids:
                raise RuntimeError(f'Duplicate completed episode {report["episode"]}')
            completed_ids.add(report['episode']);completed+=1
            # Prefetch ONLY inside the current phase. No future-phase work
            # may execute while the frozen validation is pending.
            assign(wid)
            staleness_at_receive=learner.gradient_steps-report['policy_version']
            ready=0
            for trans in transitions:
                replay.items.append(trans)
                if len(replay)>=args.warmup:ready+=1
            grad_credit+=ready*args.updates_per_step
            updates=int(grad_credit)
            if args.max_updates_per_episode:
                updates=min(updates,args.max_updates_per_episode)
            grad_credit-=updates
            if args.max_updates_per_episode:
                grad_credit=min(grad_credit,1.)
            losses=[];learning_start=time.perf_counter()
            for _ in range(updates):
                losses.append(learner.update(replay.sample(args.batch_size)))
            learn_seconds=time.perf_counter()-learning_start
            wall=time.perf_counter()-start
            train_rows.append(dict(report,completed=completed,
                completion_order=completed,
                policy_staleness_at_receive=staleness_at_receive,
                policy_staleness_steps=learner.gradient_steps-report['policy_version'],
                learner_updates=updates,
                loss=(sum(losses)/len(losses) if losses else ''),
                gradient_steps=learner.gradient_steps,replay_size=len(replay),
                learner_seconds=learn_seconds,wall_seconds=wall,
                episodes_per_hour=3600*completed/max(1e-9,wall)))
            # Realtime CSV: no longer wait for hundreds of CPU validation runs.
            _atomic_csv(output/'train_curve.csv',train_rows)
            if completed%args.log_every==0:
                print(f'[train] completed={completed}/{len(jobs)} '
                      f'ep={report["episode"]} actor={wid} '
                      f'grads={learner.gradient_steps} wait={report["weighted_wait"]:.3f} '
                      f'replay={len(replay)} validation_pending='
                      f'{sum(v["status"]=="pending" for v in validations.values())}',flush=True)
                _record_event(output,'train_completed',completed=completed,
                              episode=report['episode'],gradient_steps=learner.gradient_steps,
                              worker_id=wid,worker_pid=report['worker_pid'])
            if completed==stage_limit:
                # Every assigned episode in this phase has returned; prefetch
                # was bounded by episode number, so there are no in-flight jobs.
                if queued!=completed:
                    raise RuntimeError('Reached stage boundary with pending rollouts')
                _record_event(output,'phase_training_end',stage_limit=stage_limit,
                              completed=completed,gradient_steps=learner.gradient_steps)
                schedule_validation(completed)
                wait_for_validation(f'val_{completed:06d}')
                stage_index+=1
                stage_limit=(stage_limits[stage_index] if stage_index<len(stage_limits) else None)
                # Validation does not contribute gradient steps or replay items.
                last_actor_result=time.perf_counter()
                last_monitor=last_actor_result
                if stage_limit is not None:
                    _record_event(output,'phase_training_start',stage_limit=stage_limit,
                                  completed=completed,gradient_steps=learner.gradient_steps)
                    print(f'[phase-training] completed={completed} next_validation={stage_limit}',flush=True)
                    for worker_id in range(args.workers):
                        for _ in range(args.prefetch_per_worker):
                            assign(worker_id)
            elif completed%args.state_every==0:
                persist(save_full=True,reason='periodic')
        # This should be redundant: every validation is awaited at its barrier.
        if any(v['status']=='pending' for v in validations.values()):
            raise RuntimeError('Finished training with pending validation: barrier violation')
        print(f'[training-complete] {completed} episodes; all validation phases finished',
              flush=True)
        persist(save_full=True,reason='all-validations-complete')
        elapsed=time.perf_counter()-start
        # On a no-op resume, retain the original performance measurements.
        if not (args.resume and not worker_pids and (output/'performance.json').exists()):
            atomic_json(dict(completed_episodes=completed,workers_requested=args.workers,
            workers_used=len(worker_pids),actor_pids=sorted(worker_pids),
            validation_workers_requested=args.validation_workers,
            validation_worker_pids=sorted(validation_pids),
            completed_validations=len([v for v in validations.values() if v['status']=='done']),
            runs_per_validation=expected_runs,
            total_validated_runs=sum(v.get('evaluated_runs',0) for v in validations.values()),
            gradient_steps=learner.gradient_steps,total_wall_seconds=elapsed,
            avg_episodes_per_hour=3600*completed/max(1e-9,elapsed),
            avg_actor_seconds=(sum(r['actor_seconds'] for r in train_rows)/len(train_rows)),
            avg_learner_seconds=(sum(r['learner_seconds'] for r in train_rows)/len(train_rows)),
            note='Phase-synchronous: 4 rollout actors -> frozen snapshot -> 4 validation workers; '
                 'validation never overlaps rollout training; actor completion order may differ'),output/'performance.json')
        _record_event(output,'completed',episodes=completed,
                      gradients=learner.gradient_steps,best_score=best_val,
                      total_validated_runs=sum(v.get('evaluated_runs',0) for v in validations.values()))
        print(f'[complete] completed={completed} gradients={learner.gradient_steps} '
              f'validations={len(validations)} validated_runs='
              f'{sum(v.get("evaluated_runs",0) for v in validations.values())} '
              f'best={output/"best.pt"}',flush=True)
    finally:
        for q in cmd_queues:
            try:q.put_nowait(('stop',))
            except queue.Full:pass
        try:validation_req.put_nowait(('stop',))
        except queue.Full:pass
        for child in children:
            if child.pid is None:continue
            child.join(timeout=3)
            if child.is_alive():child.terminate()
            child.join(timeout=2)
            if child.is_alive():child.kill()
        if validator.pid is not None:
            validator.join(timeout=4)
            if validator.is_alive():validator.terminate()
            validator.join(timeout=2)
            if validator.is_alive():validator.kill()
        for q in cmd_queues:q.close()
        result_q.close();validation_req.close();validation_report.close()


if __name__ == '__main__':
    mp.freeze_support()
    main()
