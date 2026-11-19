#!/usr/bin/env python3
"""IAMU V1.1 fixed-behavior-round multi-actor DDQN learner.

1. Freeze actor network at start of each numbered collection round;
2. dynamically dispatch episode jobs to CPU workers inside the round;
3. receive out-of-order, consume strictly ascending episode_id;
4. update learner immediately on contiguous arrivals, overlapping CPU rollout;
5. validate frozen checkpoints after each complete validation boundary.

This is a NEW training protocol. Matching old async results bit-for-bit is
neither promised nor scientifically valid. Resume saves only complete rounds.
"""
from __future__ import annotations
import argparse,hashlib,json,math,multiprocessing as mp,os,queue,random,sys,time,shutil
from collections import deque,Counter
from pathlib import Path
from types import SimpleNamespace
from statistics import mean
from datetime import datetime,timezone

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import torch
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
from iamu_dynamic.rl_experiment_v11 import parse_names,parse_ints,verify_cases,write_csv
from parallel_recovery_v081 import atomic_json
from v11_protocol.recovery_v12 import save_state,load_state
from v11_protocol.indexed_replay import IndexedReplay
from v11_protocol.rounds import protocol_rounds
from v11_protocol.validation_v11 import validate
from v11_protocol.v11_actors import RolloutJob,_actor_loop
from v11_protocol.dispatch_v12 import RoundDispatcher
from v11_protocol.research_v12 import collapse_exhaustion_tail,verify_collapse,class_counts,classify_transition
from v11_protocol.experiment_fingerprint_v12 import fingerprint,compare_for_resume,digest_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def atomic_checkpoint(path,learner,meta):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(learner.checkpoint(meta),tmp);tmp.replace(path)

def jobs_for(args):
    rng=random.Random(args.seed)
    scenes=parse_names(args.scenarios);budgets=parse_ints(args.budgets)
    uncertainties=[float(x) for x in parse_names(args.uncertainties)]
    return [RolloutJob(ep,rng.choice(args.train_cases),rng.choice(scenes),
                args.train_seed_start+ep-1,rng.choice(uncertainties),rng.choice(budgets),
                args.epsilon_start+(args.epsilon_end-args.epsilon_start)*min(1.,(ep-1)/max(1,args.epsilon_decay_episodes)),
                args.seed*1000003+ep*10007+(args.train_seed_start+ep-1))
            for ep in range(1,args.episodes+1)]


def arg_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--train-cases',nargs='+',required=True)
    p.add_argument('--val-cases',nargs='+',required=True)
    p.add_argument('--episodes',type=int,default=300)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validation-workers',type=int,default=4)
    p.add_argument('--round-size',type=int,default=16)
    p.add_argument('--state-every-rounds',type=int,default=5)
    p.add_argument('--fork-episode',type=int,default=200)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--exhaustion-tail-mode',choices=('raw','collapse'),default='raw')
    p.add_argument('--dispatch-coordinator',action='store_true')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--train-seed-start',type=int,default=200000)
    p.add_argument('--val-seeds',default='400001:400003')
    p.add_argument('--uncertainties',default='4,8,12')
    p.add_argument('--validation-uncertainty',type=float,default=8.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--clock',choices=('eta','union','clock'),default='union')
    p.add_argument('--period',type=float,default=12.)
    p.add_argument('--candidate-limit',type=int,default=4)
    p.add_argument('--reward-scale',type=float,default=100.)
    p.add_argument('--candidate-screen',choices=('positive','none','mixed'),default='positive')
    p.add_argument('--budget-features',choices=('absolute','legacy'),default='absolute')
    p.add_argument('--value-features',choices=('global','legacy'),default='global')
    p.add_argument('--min-gain',type=float,default=.1)
    p.add_argument('--execution-margin',type=float,default=.1)
    p.add_argument('--bootstrap-margin',type=float,default=0.)
    p.add_argument('--greedy-exploration-prob',type=float,default=0.)
    p.add_argument('--hidden-dim',type=int,default=96)
    p.add_argument('--q-network',choices=('ddqn','d3qn','set_ddqn','set_d3qn'),default='set_ddqn')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--replay-capacity',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--warmup',type=int,default=512)
    p.add_argument('--updates-per-step',type=float,default=1.)
    p.add_argument('--max-updates-per-episode',type=int,default=0)
    p.add_argument('--target-update',type=int,default=250)
    p.add_argument('--replay-rho',type=float,default=.25)
    p.add_argument('--replay-weighting',choices=('unweighted','importance','uniform'),default='unweighted')
    p.add_argument('--epsilon-start',type=float,default=1.)
    p.add_argument('--epsilon-end',type=float,default=.05)
    p.add_argument('--epsilon-decay-episodes',type=int,default=300)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',required=True)
    return p

def validate_args(a):
    if not (1<=a.workers<=32 and 1<=a.validation_workers<=16 and a.round_size>=1):
        raise ValueError('Invalid parallelism')
    if not (1<=a.state_every_rounds and a.episodes>=1 and a.eval_every>=1 and a.fork_episode>=0):
        raise ValueError('Invalid round/episode grid')
    if not(0<=a.replay_rho<=1) or (a.replay_weighting=='importance' and a.replay_rho in (0.,1.)):
        raise ValueError('Importance correction requires 0 < rho < 1')
    if any(x<0 or not math.isfinite(x) for x in (a.execution_margin,a.bootstrap_margin,a.min_gain)):
        raise ValueError('Invalid margins/min_gain')
    if a.warmup<a.batch_size or a.replay_capacity<a.warmup:
        raise ValueError('Invalid Replay warmup/capacity')
    if a.max_updates_per_episode<0 or a.updates_per_step<0 or a.target_update<=0:
        raise ValueError('Invalid update settings')
    if not 0<=a.epsilon_start<=1 or not 0<=a.epsilon_end<=1:raise ValueError('Invalid epsilon')
    if a.fork_episode and a.fork_episode<=a.episodes and a.fork_episode%a.eval_every:
        raise ValueError('Fork checkpoint must coincide with validation boundary')
    verify_cases(a.train_cases,a.val_cases)

def snapshot_weights(learner):
    return {k:v.detach().cpu().clone() for k,v in learner.online.state_dict().items()}

def main(argv=None):
    a=arg_parser().parse_args(argv);validate_args(a)
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable')
    torch.set_num_threads(1)
    all_jobs=jobs_for(a)
    rounds=protocol_rounds(all_jobs,a.round_size,a.eval_every)
    learner=DoubleDQNAgent(hidden_dim=a.hidden_dim,learning_rate=a.lr,gamma=a.gamma,
        target_update_steps=a.target_update,seed=a.seed,device=a.device,
        q_network=a.q_network,execution_margin=a.execution_margin,
        bootstrap_margin=a.bootstrap_margin)
    replay=IndexedReplay(a.replay_capacity,a.seed+100)
    cfg=fingerprint(a,Path(__file__).resolve().parents[1])
    cfg['task_list_sha256']=digest_json([vars(j) for j in all_jobs])
    cfg['protocol_version']='V1.2-frozen-round-ordered-consumption'
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    state=out/'training_state.pt';rows=[];val_rows=[];validations={};best=float('inf');credit=0.
    observed_raw_steps=0;cumulative_sample_counts=Counter()
    if a.resume:
        if not (out/'manifest.json').exists():raise RuntimeError('Missing original run manifest')
        changed_runtime=compare_for_resume(json.loads((out/'manifest.json').read_text()),cfg)
        if changed_runtime:
            print('[v12] Runtime device/workers/paths differ; bitwise cross-platform replay not guaranteed',flush=True)
        saved=load_state(state,learner=learner,replay=replay,config=cfg)
        rows=saved['train_rows'];val_rows=saved['val_rows'];validations=saved['validations']
        best=float(saved['best_val']);credit=float(saved['gradient_credit'])
        observed_raw_steps=int(saved.get('protocol_state',{}).get('observed_raw_steps',0))
        cumulative_sample_counts=Counter(saved.get('protocol_state',{}).get('cumulative_sample_counts',{}))
    else:
        if any(out.iterdir()):
            raise FileExistsError('Output directory already contains training artifacts')
        atomic_json(cfg,out/'manifest.json')
        atomic_json([vars(j) for j in all_jobs],out/'task_list.json')
    done=len(rows)
    if [int(r['episode']) for r in rows]!=list(range(1,done+1)):
        raise RuntimeError('Recovery log is not in deterministic episode-ID order')
    start_round=0
    for i,rr in enumerate(rounds):
        if rr[-1].episode<=done:start_round=i+1
        elif rr[0].episode<=done:
            raise RuntimeError('Recovery state is not on complete round boundary')
    def persist(reason):
        write_csv(out/'train_curve.csv',rows)
        write_csv(out/'validation_curve.csv',val_rows)
        atomic_json(dict(records=validations),out/'validation_ledger.json')
        atomic_checkpoint(out/'latest.pt',learner,dict(selected_episode=done,
                             gradient_steps=learner.gradient_steps,reason=reason))
        save_state(state,learner=learner,replay=replay,config=cfg,train_rows=rows,
            val_rows=val_rows,validations=validations,best_val=best,gradient_credit=credit,
            protocol_state=dict(observed_raw_steps=observed_raw_steps,
                cumulative_sample_counts=dict(cumulative_sample_counts)))
    def eval_boundary():
        nonlocal best
        key=f'val_{done:06d}'
        snap=out/'validation_snapshots'/f'{key}_grad_{learner.gradient_steps:09d}.pt'
        if snap.exists():
            raise FileExistsError(f'Old uncommitted validation snapshot exists: {snap}; use a clean output')
        atomic_checkpoint(snap,learner,dict(selected_episode=done,gradient_steps=learner.gradient_steps,
            validation_id=key,training_protocol='v11'))
        validation_started=time.perf_counter()
        summary=validate(snap,a,out/'validation_details'/key)
        validation_seconds=time.perf_counter()-validation_started
        result=dict(id=key,completed=done,gradient_steps=learner.gradient_steps,
            snapshot=str(snap),snapshot_sha256=sha(snap),status='done',
            expected_runs=summary['rows'],evaluated_runs=summary['rows'],score=summary['score'])
        validations[key]=result
        val_rows.append(dict(completed=done,validation_id=key,mean_wait=summary['score'],
            mean_spent=summary['mean_spent'],validation_seconds=validation_seconds,
            gradient_steps=learner.gradient_steps,
            snapshot_sha256=result['snapshot_sha256'],evaluated_runs=summary['rows']))
        if summary['score']<best:
            best=summary['score'];atomic_checkpoint(out/'best.pt',learner,
                dict(selected_episode=done,validation_id=key,validation_score=best,
                     gradient_steps=learner.gradient_steps))
        print(f'[v11 validation] episode={done} wait={summary["score"]:.3f} '
              f'grad={learner.gradient_steps} best={best:.3f}',flush=True)
    # Set up CPU actors only for unfinished rounds. No actor is allowed to
    # acquire updated learner weights inside its current frozen round.
    if start_round==len(rounds):
        print('[v11] already complete',flush=True);return
    ctx=mp.get_context('spawn')
    result_q=ctx.Queue(maxsize=max(16,a.workers*4))
    commands=[ctx.Queue(maxsize=8) for _ in range(a.workers)]
    children=[]
    settings=dict(vars(a),keep_margin=a.execution_margin)
    for wid in range(a.workers):
        child=ctx.Process(target=_actor_loop,args=(wid,commands[wid],result_q,settings),
                          name=f'v11-actor-{wid}',daemon=False)
        child.start();children.append(child)
    try:
        for round_id in range(start_round,len(rounds)):
            batch=rounds[round_id]
            round_started=time.perf_counter()
            wait_seconds=0.;learning_seconds=0.;max_out_of_order=0
            frozen_grad=learner.gradient_steps
            weights=snapshot_weights(learner)
            for cmd in commands:cmd.put(('sync',weights,frozen_grad))
            waiting=deque(batch)
            order=[job.episode for job in batch]
            ready={};inflight=set();next_index=0;arrival=0
            def assign(wid):
                if not waiting:return
                job=waiting.popleft()
                if job.episode in inflight:raise AssertionError('Duplicate job')
                commands[wid].put(('rollout',job));inflight.add(job.episode)
            dispatcher=None
            if a.dispatch_coordinator:
                waiting.clear()
                inflight=set(order)
                dispatcher=RoundDispatcher(result_q,commands,batch)
                dispatcher.start()
            else:
                for wid in range(a.workers):assign(wid)
            while next_index<len(order):
                wait_started=time.perf_counter()
                try:msg=(dispatcher.get(timeout=30) if dispatcher is not None else result_q.get(timeout=30))
                except queue.Empty:
                    if any(not c.is_alive() for c in children):raise RuntimeError('Actor process exited')
                    print(f'[v11 waiting] round={round_id} done={next_index}/{len(order)}',flush=True)
                    continue
                wait_seconds+=time.perf_counter()-wait_started
                tag=msg[0]
                if tag=='error':raise RuntimeError('Actor failed:\n'+msg[2])
                if tag!='ok':continue
                report,trajectory=msg[1],msg[2];ep=int(report['episode'])
                arrival+=1
                if ep not in inflight or ep in ready:raise RuntimeError('Unexpected or duplicate rollout')
                ready[ep]=(report,trajectory,arrival)
                max_out_of_order=max(max_out_of_order,len(ready))
                # Decouple actor throughput from learner's ordered consumption.
                if dispatcher is None: assign(int(report['worker_id']))
                while next_index<len(order) and order[next_index] in ready:
                    ep=order[next_index];report,trajectory,arrival_order=ready.pop(ep)
                    if int(report['episode'])!=ep:
                        raise AssertionError('Ordered report episode ID mismatch')
                    if int(report['policy_version'])!=frozen_grad:
                        raise AssertionError('Actor policy changed within frozen round')
                    raw_trajectory=trajectory
                    if len(raw_trajectory)!=int(report['steps']):
                        raise AssertionError('Actor raw step count mismatch')
                    if a.exhaustion_tail_mode=='collapse':
                        trajectory=collapse_exhaustion_tail(raw_trajectory,gamma=a.gamma)
                        verify_collapse(raw_trajectory,trajectory,gamma=a.gamma)
                    stored_steps=len(trajectory)
                    old_replay_size=len(replay)
                    for tr in trajectory: replay.items.append(tr)
                    raw_before=observed_raw_steps
                    observed_raw_steps+=len(raw_trajectory)
                    n_ready=max(0,observed_raw_steps-(a.warmup-1))-max(0,raw_before-(a.warmup-1))
                    # Original schedule: each observed post-warmup transition
                    # contributes one fractional learner gradient credit.
                    credit+=n_ready*a.updates_per_step
                    updates=int(credit)
                    if a.max_updates_per_episode:updates=min(updates,a.max_updates_per_episode)
                    if len(replay)<a.batch_size: updates=0
                    credit-=updates
                    if a.max_updates_per_episode:credit=min(credit,1.)
                    staleness_at_consume=learner.gradient_steps-frozen_grad
                    learn_started=time.perf_counter()
                    losses=[];sample_metas=[];sampled_classes=Counter()
                    for _ in range(updates):
                        if a.replay_weighting=='uniform':
                            sampled=replay.sample(a.batch_size);weights=None;meta=dict(actual_rho=None,shortfall=0)
                        else:
                            sampled,weights,meta=replay.sample_stratified(a.batch_size,a.replay_rho,
                                importance=a.replay_weighting=='importance')
                        sampled_classes.update(classify_transition(tr) for tr in sampled)
                        losses.append(learner.update(sampled,weights=weights));sample_metas.append(meta)
                    this_learning=time.perf_counter()-learn_started
                    learning_seconds+=this_learning
                    done+=1;next_index+=1;inflight.remove(ep)
                    replay_counts=class_counts(replay.items)
                    cumulative_sample_counts.update(sampled_classes)
                    rows.append(dict(episode=ep,completed=done,round_id=round_id,
                        policy_version=frozen_grad,arrival_order_in_round=arrival_order,
                        policy_staleness_at_consume=staleness_at_consume,
                        learner_seconds=this_learning,
                        gradient_steps=learner.gradient_steps,learner_updates=updates,
                        mean_loss=mean(losses) if losses else '',replay_size=len(replay),
                        replay_actionable=replay.stats()['actionable'],
                        mean_sampled_actionable_rho=(mean(x['actual_rho'] for x in sample_metas)
                             if sample_metas and sample_metas[0]['actual_rho'] is not None else ''),
                        sample_shortfall_updates=sum(x['shortfall'] for x in sample_metas),
                        weighted_wait=report['weighted_wait'],spent=report['spent'],
                        scenario=report['scenario'],budget=report['budget'],
                        epsilon=report['epsilon'],actor_seconds=report['actor_seconds'],
                        forced_flags=json.dumps(report.get('forced_flags',{}),sort_keys=True),
                        raw_events_observed=observed_raw_steps,
                        episode_raw_events=len(raw_trajectory),episode_stored_transitions=stored_steps,
                        exhaustion_tail_mode=a.exhaustion_tail_mode,
                        replay_budget_exhausted=replay_counts['budget_exhausted'],
                        replay_forced_with_budget=replay_counts['forced_with_budget'],
                        replay_actionable_keep=replay_counts['actionable_keep'],
                        replay_actionable_edit=replay_counts['actionable_edit'],
                        sampled_actionable=sampled_classes['actionable'],
                        sampled_budget_exhausted=sampled_classes['budget_exhausted'],
                        sampled_forced_with_budget=sampled_classes['forced_with_budget'],
                        batch_actionable_fraction=(sampled_classes['actionable']/max(1,updates*a.batch_size)),
                        cumulative_sampled_actionable=cumulative_sample_counts['actionable'],
                        cumulative_sampled_budget_exhausted=cumulative_sample_counts['budget_exhausted'],
                        cumulative_sampled_forced_with_budget=cumulative_sample_counts['forced_with_budget'],
                        first_exhaustion_event_index=next((tr.event_index for tr in raw_trajectory
                            if tr.budget_before>0 and tr.budget_after==0),''),
                        remaining_budget_trace=json.dumps([tr.budget_after for tr in raw_trajectory]),
                        candidate_generation_seconds=report.get('candidate_generation_seconds',0.),
                        forecast_projection_seconds=report.get('forecast_projection_seconds',0.),
                        worker_idle_seconds=report.get('worker_idle_seconds',0.)))
            if dispatcher is not None: dispatcher.finish()
            if inflight or waiting or ready:raise AssertionError('Incomplete collection round')
            round_wall=time.perf_counter()-round_started
            for record in rows[-len(batch):]:record['round_wall_seconds']=round_wall
            print(f'[v11 throughput] round={round_id} wall={round_wall:.3f}s '
                f'queue_wait={wait_seconds:.3f}s learner={learning_seconds:.3f}s '
                f'max_buffered={max_out_of_order} ep_per_hour={3600*len(batch)/max(round_wall,1e-6):.2f}',flush=True)
            if done%a.eval_every==0 or done==a.episodes:eval_boundary()
            if ((round_id+1)%a.state_every_rounds==0 or
                done%a.eval_every==0 or done==a.episodes):
                persist('round-boundary')
                if done==a.fork_episode:
                    fork=out/f'full_state_ep{done:04d}.pt'
                    if not fork.exists():
                        temp=fork.with_suffix('.pt.tmp');shutil.copyfile(state,temp);temp.replace(fork)
                        atomic_json(dict(episode=done,full_state_sha256=sha(fork),
                            source_gradient_steps=learner.gradient_steps,
                            validation_id=f'val_{done:06d}' if done%a.eval_every==0 else None,
                            protocol='v11 frozen behavior, ordered replay'),
                            out/f'full_state_ep{done:04d}.metadata.json')
            print(f'[v11 round] {round_id+1}/{len(rounds)} ep={done} '
                f'grad={learner.gradient_steps} replay={len(replay)}',flush=True)
        if done!=a.episodes or len(rows)!=a.episodes or len({int(r['episode']) for r in rows})!=a.episodes:
            raise RuntimeError('Incomplete or duplicate training episodes: cannot write complete.json')
        atomic_json(dict(completed_episodes=done,gradient_steps=learner.gradient_steps,
            best_val=best,protocol='fixed-behavior-round, ordered-consumption',
            workers=a.workers,validation_workers=a.validation_workers),out/'complete.json')
    finally:
        if 'dispatcher' in locals() and dispatcher is not None:
            dispatcher.cancel()
        for c in commands:
            try:c.put_nowait(('stop',))
            except queue.Full:pass
        for proc in children:
            proc.join(timeout=3)
            if proc.is_alive():proc.terminate();proc.join(timeout=2)
        for c in commands:c.close()
        result_q.close()

if __name__=='__main__':
    mp.freeze_support();main()
