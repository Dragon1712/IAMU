#!/usr/bin/env python3
"""V1.2 A0-A3, frozen-round, job-keyed actor, ordered learner protocol.

Only trajectory representation and primitive sampling mode change. No V1.1
physics, reward, network, or candidate-selection code is patched. Resume is
allowed from a COMPLETE saved round only; outputs are never overwritten.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,multiprocessing as mp,os,queue,random,sys,time,subprocess,shutil
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from iamu_dynamic.rl_experiment_v11 import verify_cases,write_csv
from v11_protocol.train_v11 import jobs_for,snapshot_weights
from v11_protocol.rounds import protocol_rounds
from .agent import V12Agent
from .actors import _actor_loop
from .compress import compress_episode
from .replay import EpisodeReplay
from .validation import validate

CODE_ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=CODE_ROOT.parent


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(1024*1024),b''):h.update(piece)
    return h.hexdigest()

def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding='utf-8')
    os.replace(tmp,p)

def atomic_torch(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--representation',choices=('primitive','budget_tail','decision_epoch'),required=True)
    p.add_argument('--sampling',choices=('uniform','stratified'),default='uniform')
    p.add_argument('--replay-rho',type=float,default=.25)
    p.add_argument('--replay-episodes',type=int,default=200)
    p.add_argument('--max-replay-numpy-mib',type=float,default=1024.)
    p.add_argument('--train-cases',nargs='+',required=True)
    p.add_argument('--val-cases',nargs='+',required=True)
    p.add_argument('--episodes',type=int,default=400)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validation-workers',type=int,default=4)
    p.add_argument('--round-size',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--state-every-rounds',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--train-seed-start',type=int,default=200000)
    p.add_argument('--val-seeds',default='400001:400003')
    p.add_argument('--uncertainties',default='4,8,12')
    p.add_argument('--validation-uncertainty',type=float,default=8.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--clock',choices=('union',),default='union')
    p.add_argument('--period',type=float,default=12.)
    p.add_argument('--candidate-limit',type=int,default=4)
    p.add_argument('--reward-scale',type=float,default=100.)
    p.add_argument('--candidate-screen',choices=('positive',),default='positive')
    p.add_argument('--budget-features',choices=('absolute',),default='absolute')
    p.add_argument('--value-features',choices=('global',),default='global')
    p.add_argument('--min-gain',type=float,default=.1)
    p.add_argument('--execution-margin',type=float,default=0.)
    p.add_argument('--bootstrap-margin',type=float,default=0.)
    p.add_argument('--greedy-exploration-prob',type=float,default=0.)
    p.add_argument('--hidden-dim',type=int,default=96)
    p.add_argument('--q-network',choices=('set_ddqn',),default='set_ddqn')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--warmup-raw-steps',type=int,default=512)
    p.add_argument('--warmup-actionable',type=int,default=64)
    p.add_argument('--updates-per-raw-step',type=float,default=.25)
    p.add_argument('--max-updates-per-episode',type=int,default=0)
    p.add_argument('--target-update',type=int,default=250)
    p.add_argument('--epsilon-start',type=float,default=1.)
    p.add_argument('--epsilon-end',type=float,default=.05)
    p.add_argument('--epsilon-decay-episodes',type=int,default=400)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.sampling=='stratified' and a.representation!='primitive':
        p.error('A1 stratification only allowed for primitive representation')
    if (a.episodes<1 or a.eval_every<1 or a.round_size<1 or a.workers<1 or
        a.validation_workers<1 or a.state_every_rounds<1 or
        a.replay_episodes<1 or a.max_replay_numpy_mib<=0 or a.batch_size<1 or a.target_update<1 or
        a.max_updates_per_episode<0 or a.warmup_raw_steps<0 or a.warmup_actionable<0 or
        a.updates_per_raw_step<0 or not math.isfinite(a.updates_per_raw_step) or
        not 0<a.gamma<=1 or a.reward_scale<=0 or not 0<=a.replay_rho<=1):
        p.error('Invalid train protocol numeric configuration')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    verify_cases(a.train_cases,a.val_cases)
    return a


def code_fingerprint():
    paths=[*sorted((CODE_ROOT/'v12_protocol').glob('*.py')),
        CODE_ROOT/'iamu_dynamic/rl_env_v11.py',
        CODE_ROOT/'iamu_dynamic/rl_ddqn_v11.py',
        CODE_ROOT/'iamu_dynamic/core.py',
        CODE_ROOT/'iamu_dynamic/rl_experiment_v11.py',
        CODE_ROOT/'v11_protocol/v11_actors.py',
        CODE_ROOT/'v11_protocol/train_v11.py',
        CODE_ROOT/'v11_protocol/rounds.py']
    return {str(p.relative_to(CODE_ROOT)):digest(p) for p in paths}


def manifest_for(a,jobs):
    args=vars(a).copy()
    for name in ('out','resume','workers','validation_workers','state_every_rounds'):
        args.pop(name,None)
    git_sha=None;dirty=None
    try:
        cp=subprocess.run(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,capture_output=True,text=True,timeout=3)
        if cp.returncode==0:
            git_sha=cp.stdout.strip()
            x=subprocess.run(['git','status','--porcelain'],cwd=PROJECT_ROOT,capture_output=True,text=True,timeout=5)
            dirty=bool(x.stdout.strip()) if x.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):pass
    jobs_payload=[dict(episode=j.episode,case=j.case,scenario=j.scenario,seed=j.seed,
                       budget=j.budget,epsilon=j.epsilon,uncertainty=j.uncertainty)
                  for j in jobs]
    jobs_hash=hashlib.sha256(json.dumps(jobs_payload,sort_keys=True).encode()).hexdigest()
    stable=dict(version='v12.0-episode-window',scientific_args=args,
        case_hash={str(Path(p).resolve()):digest(p) for p in [*a.train_cases,*a.val_cases]},
        code_sha256=code_fingerprint(),task_table_sha256=jobs_hash,
        sampling_protocol='uniform or primitive-25pct-stratified',
        physical_env='V1.1, unmodified',
        pytorch=torch.__version__,numpy=np.__version__)
    # Git dirty/head are recorded, but not used for strict resume comparisons:
    # untracked output files must not cause false resume mismatches.
    stable_hash=hashlib.sha256(json.dumps(stable,sort_keys=True,default=str).encode()).hexdigest()
    return dict(scientific=stable,science_hash=stable_hash,
                git_head=git_sha,git_dirty=dirty,
                run_controls=dict(workers=a.workers,validation_workers=a.validation_workers,
                                  state_every_rounds=a.state_every_rounds,device=a.device))


def save_full_state(path,agent,replay,manifest,rows,validation_rows,ledger,best,credit,raw_count,action_count):
    # Full training-state is LOCAL TRUSTED pickle; do not deserialize third-party checkpoints.
    payload=dict(protocol=12,science_hash=manifest['science_hash'],
        online=agent.online.state_dict(),target=agent.target.state_dict(),
        optimizer=agent.optimizer.state_dict(),gradient_steps=agent.gradient_steps,
        agent_rng=agent.rng.getstate(),
        replay=replay.state_dict(),rows=rows,validation_rows=validation_rows,
        ledger=ledger,best=best,credit=credit,raw_count=raw_count,action_count=action_count,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.random.get_rng_state(),
        cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None))
    atomic_torch(path,payload)


def restore_full_state(path,agent,replay,manifest):
    saved=torch.load(path,map_location='cpu',weights_only=False) # Local trusted state only.
    if saved['protocol']!=12 or saved['science_hash']!=manifest['science_hash']:
        raise ValueError('V1.2 resume manifest/scientific config mismatch')
    agent.online.load_state_dict(saved['online'])
    agent.target.load_state_dict(saved['target'])
    agent.optimizer.load_state_dict(saved['optimizer'])
    agent.gradient_steps=saved['gradient_steps']
    agent.rng.setstate(saved['agent_rng'])
    replay.load_state_dict(saved['replay'])
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    torch.random.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved


def quarantine_uncommitted_validation(out:Path, committed_episodes:int):
    """Preserve validation artifacts created after the last saved complete round."""
    snap_root=out/'validation_snapshots'
    if not snap_root.exists():return
    for path in sorted(snap_root.glob('val_*_grad_*.pt')):
        ep=int(path.name.split('_')[1])
        if ep<=committed_episodes:continue
        dest=out/'orphaned_on_resume'/f'val_{ep:06d}'
        dest.mkdir(parents=True,exist_ok=True)
        target=dest/path.name
        if target.exists():raise FileExistsError('Orphan quarantine collision: '+str(target))
        shutil.move(str(path),str(target))
        old=out/'validation_details'/f'val_{ep:06d}'
        if old.exists():
            target=dest/'validation_details'
            if target.exists():raise FileExistsError('Orphan details quarantine collision')
            shutil.move(str(old),str(target))
        print(f'[V12 recovery] quarantined uncommitted validation EP{ep}',flush=True)


def main(argv=None):
    a=parse_args(argv)
    torch.set_num_threads(1)
    jobs=jobs_for(a)
    rounds=protocol_rounds(jobs,a.round_size,a.eval_every)
    meta=manifest_for(a,jobs)
    out=Path(a.out).resolve()
    if a.resume:
        if not (out/'manifest.json').is_file():raise FileNotFoundError('Missing manifest.json')
        original=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if original['science_hash']!=meta['science_hash']:
            raise ValueError('Refusing resume: code/config/task table differs; fork must use a new run')
        meta=original
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Refusing to overwrite nonempty output: {out}')
        out.mkdir(parents=True,exist_ok=True);atomic_json(out/'manifest.json',meta)
    agent=V12Agent(hidden_dim=a.hidden_dim,learning_rate=a.lr,gamma=a.gamma,
           target_update_steps=a.target_update,seed=a.seed,device=a.device,
           q_network=a.q_network,execution_margin=a.execution_margin,
           bootstrap_margin=a.bootstrap_margin,representation=a.representation,
           reward_scale=a.reward_scale)
    replay=EpisodeReplay(a.replay_episodes,a.seed+100,a.representation)
    rows=[];val_rows=[];ledger={};best=float('inf');credit=0.;raw_count=0;action_count=0
    if a.resume:
        saved=restore_full_state(out/'training_state.pt',agent,replay,meta)
        rows=saved['rows'];val_rows=saved['validation_rows'];ledger=saved['ledger']
        best=float(saved['best']);credit=float(saved['credit'])
        raw_count=int(saved['raw_count']);action_count=int(saved['action_count'])
    done=len(rows)
    if [int(x['episode']) for x in rows]!=list(range(1,done+1)):
        raise AssertionError('Noncontiguous recovered episode ledger')
    start_round=0
    for i,rr in enumerate(rounds):
        if rr[-1].episode<=done:start_round=i+1
        elif rr[0].episode<=done:raise ValueError('Recovery did not end on complete round boundary')
    if a.resume:
        quarantine_uncommitted_validation(out,done)
    if done==a.episodes:
        print('[V12] already complete',flush=True);return

    def persist():
        write_csv(out/'train_curve.csv',rows)
        write_csv(out/'validation_curve.csv',val_rows)
        atomic_json(out/'validation_ledger.json',ledger)
        atomic_torch(out/'latest.pt',agent.checkpoint(dict(selected_episode=done,
                   gradient_steps=agent.gradient_steps,representation=a.representation)))
        save_full_state(out/'training_state.pt',agent,replay,meta,rows,val_rows,
                        ledger,best,credit,raw_count,action_count)

    def evaluate():
        nonlocal best
        stamp=f'val_{done:06d}'
        ck=out/'validation_snapshots'/f'{stamp}_grad_{agent.gradient_steps:09d}.pt'
        if ck.exists():raise FileExistsError(f'Validation snapshot already exists {ck}')
        atomic_torch(ck,agent.checkpoint(dict(selected_episode=done,
                      gradient_steps=agent.gradient_steps,validation_id=stamp)))
        started=time.perf_counter()
        summary=validate(ck,a,out/'validation_details'/stamp)
        validation_seconds=time.perf_counter()-started
        ledger[stamp]=dict(snapshot=str(ck),sha256=digest(ck),
                            status='done',score=summary['score'],rows=summary['rows'],
                            validation_seconds=validation_seconds)
        val_rows.append(dict(completed=done,validation_id=stamp,
            mean_wait=summary['score'],mean_spent=summary['mean_spent'],
            gradient_steps=agent.gradient_steps,evaluated_runs=summary['rows'],
            validation_seconds=validation_seconds))
        if summary['score']<best:
            best=summary['score']
            atomic_torch(out/'best.pt',agent.checkpoint(dict(selected_episode=done,
                  validation_id=stamp,validation_score=best,gradient_steps=agent.gradient_steps)))
        print(f'[V12 validation] EP={done} mean_wait={summary["score"]:.5f} best={best:.5f}',flush=True)

    ctx=mp.get_context('spawn')
    result_q=ctx.Queue(maxsize=max(16,a.workers*4))
    commands=[ctx.Queue(maxsize=8) for _ in range(a.workers)]
    children=[]
    for wid in range(a.workers):
        proc=ctx.Process(target=_actor_loop,
            args=(wid,commands[wid],result_q,dict(vars(a),keep_margin=a.execution_margin)),
            name=f'v12-actor-{wid}',daemon=False)
        proc.start();children.append(proc)
    start_wall=time.perf_counter()
    try:
        for round_id in range(start_round,len(rounds)):
            round_start=time.perf_counter();batch=rounds[round_id]
            frozen_grad=agent.gradient_steps
            weights=snapshot_weights(agent)
            for q in commands:q.put(('sync',weights,frozen_grad))
            waiting=deque(batch);order=[j.episode for j in batch];ready={}
            inflight=set();next_index=0;arrival=0
            def assign(wid):
                if not waiting:return
                job=waiting.popleft()
                if job.episode in inflight:raise AssertionError('Duplicate rollout')
                inflight.add(job.episode);commands[wid].put(('rollout',job))
            for wid in range(a.workers):assign(wid)
            while next_index<len(order):
                try:msg=result_q.get(timeout=30)
                except queue.Empty:
                    if any(not x.is_alive() for x in children):
                        raise RuntimeError('V1.2 actor terminated')
                    print(f'[V12 waiting] round={round_id} {next_index}/{len(order)}',flush=True)
                    continue
                if msg[0]=='error':raise RuntimeError('V1.2 actor error:\n'+msg[2])
                if msg[0]!='ok':continue
                report,trajectory=msg[1],msg[2]
                ep=int(report['episode']);arrival+=1
                if ep not in inflight or ep in ready:raise AssertionError('Duplicate/unknown job report')
                ready[ep]=(report,trajectory,arrival)
                assign(int(report['worker_id']))
                while next_index<len(order) and order[next_index] in ready:
                    ep=order[next_index];report,trajectory,arrival_order=ready.pop(ep)
                    if int(report['policy_version'])!=frozen_grad:
                        raise AssertionError('Behavior network changed within round')
                    result=compress_episode(trajectory,a.representation,a.gamma)
                    result.audit['source_episode_wait']=report['weighted_wait']
                    if not math.isclose(-result.audit['source_total_reward']*a.reward_scale,
                                         report['weighted_wait'],rel_tol=1e-6,
                                         abs_tol=1e-5*max(1.,report['weighted_wait'])):
                        raise AssertionError('Compression changed physical reward')
                    replay.append_episode(ep,result.transitions,result.audit['raw_steps'])
                    raw_count+=result.audit['raw_steps']
                    action_count+=result.audit['actionable_steps']
                    # New common warmup in raw interactions AND actionable transitions.
                    # Never accrue retroactive gradient credit while warming up.
                    credit_before=credit
                    if raw_count>=a.warmup_raw_steps and action_count>=a.warmup_actionable and len(replay)>=a.batch_size:
                        credit+=a.updates_per_raw_step*result.audit['raw_steps']
                    updates=int(credit)
                    if a.max_updates_per_episode:updates=min(updates,a.max_updates_per_episode)
                    credit-=updates
                    if a.max_updates_per_episode:credit=min(credit,1.)
                    started=time.perf_counter();losses=[];sample_metas=[]
                    for _ in range(updates):
                        sample,sm=replay.sample(a.batch_size,a.sampling,a.replay_rho)
                        losses.append(agent.update(sample))
                        sample_metas.append(sm)
                    learner_seconds=time.perf_counter()-started
                    rstats=replay.stats()
                    if rstats['replay_numpy_mib']>a.max_replay_numpy_mib:
                        raise MemoryError('Replay NumPy buffers exceed --max-replay-numpy-mib')
                    done+=1;next_index+=1;inflight.remove(ep)
                    rows.append(dict(episode=ep,completed=done,round_id=round_id,
                        policy_version=frozen_grad,arrival_order_in_round=arrival_order,
                        gradient_steps=agent.gradient_steps,learner_updates=updates,
                        mean_loss=mean(losses) if losses else '',
                        raw_env_steps=raw_count,macro_transitions=len(result.transitions),
                        total_macro_seen=sum(int(x['macro_transitions']) for x in rows)+len(result.transitions),
                        unique_decision_transitions=action_count,
                        prefix_steps=result.audit['prefix_steps'],
                        tail_raw_steps=result.audit['tail_raw_steps'],
                        prefix_reward=result.prefix.reward_undiscounted,
                        reconstructed_discounted_return=result.audit['reconstructed_discounted_return'],
                        replay_size=len(replay),replay_episode_count=rstats['replay_episode_count'],
                        replay_raw_coverage=rstats['replay_raw_coverage'],
                        replay_numpy_mib=rstats['replay_numpy_mib'],
                        mean_macro_length=rstats['mean_macro_length'],
                        max_macro_length=rstats['max_macro_length'],
                        sample_reuse=rstats['sample_reuse'],
                        sampled_actionable_ratio=(mean(sm['sample_actionable']/sm['sample_size'] for sm in sample_metas)
                                                  if sample_metas else ''),
                        sampling_shortfall_updates=sum(int(sm['shortfall']) for sm in sample_metas),
                        updates_per_raw_env_step=agent.gradient_steps/max(1,raw_count),
                        updates_per_unique_decision_transition=agent.gradient_steps/max(1,action_count),
                        gradient_credit=credit,actor_seconds=report['actor_seconds'],
                        learner_seconds=learner_seconds,weighted_wait=report['weighted_wait'],
                        spent=report['spent'],scenario=report['scenario'],budget=report['budget'],
                        epsilon=report['epsilon']))
            if inflight or waiting or ready:raise AssertionError('Round not fully consumed')
            if done%a.eval_every==0 or done==a.episodes:evaluate()
            if ((round_id+1)%a.state_every_rounds==0 or
                done%a.eval_every==0 or done==a.episodes):persist()
            print(f'[V12] round={round_id+1}/{len(rounds)} EP={done} '
                  f'grad={agent.gradient_steps} replay={len(replay)} '
                  f'wall={time.perf_counter()-round_start:.2f}s',flush=True)
        atomic_json(out/'complete.json',dict(completed_episodes=done,gradient_steps=agent.gradient_steps,
                    best_validation_wait=best,raw_env_steps=raw_count,
                    unique_decision_transitions=action_count,
                    wall_seconds=time.perf_counter()-start_wall,
                    actor_seconds_sum=sum(float(r['actor_seconds']) for r in rows),
                    learner_seconds=sum(float(r['learner_seconds']) for r in rows),
                    validation_seconds=sum(float(r['validation_seconds']) for r in val_rows),
                    representation=a.representation,sampling=a.sampling))
    finally:
        for q in commands:
            try:q.put_nowait(('stop',))
            except queue.Full:pass
        for proc in children:
            proc.join(timeout=3)
            if proc.is_alive():proc.terminate();proc.join(timeout=2)
        for q in commands:q.close()
        result_q.close()

if __name__=='__main__':
    mp.freeze_support();main()
