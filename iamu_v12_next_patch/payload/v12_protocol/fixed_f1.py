"""F1: identical raw trajectories + common probe, four V1.2 representations.

This is a NEW opt-in path for fixed_data.train. Legacy CLI behavior is
unchanged when --probe-bank is omitted. F2 is deliberately NOT auto-launched.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from statistics import mean
import numpy as np
import torch
from .baseline_cache import file_sha,canonical
from .probe_bank import load_probe_bank,source_dataset
from .probe_eval import evaluate_agent_on_probes
from .compress import compress_episode
from .replay import EpisodeReplay
from .agent import V12Agent
from .diagnostics import TDRecorder,FIELDS

MODES={'A0':('primitive','uniform'),'A1':('primitive','stratified'),
       'A2':('budget_tail','uniform'),'A3':('decision_epoch','uniform')}


def online_sha(net):
    """Stable tensor-content digest, not a random zip serialisation hash."""
    h=hashlib.sha256()
    for name,v in sorted(net.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode())
        h.update(canonical(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def _code_snapshot():
    root=Path(__file__).resolve().parents[1]
    source=[root/'iamu_dynamic/core.py',root/'iamu_dynamic/rl_env_v11.py',
            root/'iamu_dynamic/rl_ddqn_v11.py',root/'iamu_dynamic/rl_experiment_v11.py',
            *sorted((root/'v12_protocol').glob('*.py'))]
    digest={str(p.relative_to(root)):file_sha(p) for p in source if p.exists()}
    fingerprint=hashlib.sha256(canonical(digest).encode()).hexdigest()
    git_head=None;git_dirty=None
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=root.parent,
                    capture_output=True,text=True,timeout=2)
        if result.returncode==0:
            git_head=result.stdout.strip()
            dirty=subprocess.run(['git','status','--porcelain'],cwd=root.parent,
                     capture_output=True,text=True,timeout=3)
            if dirty.returncode==0:git_dirty=bool(dirty.stdout.strip())
    except (OSError,subprocess.TimeoutExpired):pass
    return dict(git_head=git_head,git_dirty=git_dirty,
                git_unavailable_reason=('offline_copy_or_no_git' if git_head is None else ''),
                source_sha256=digest,source_snapshot_sha256=fingerprint)


def _write_rows(path,rows,columns):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def _summarize_probe(rows, step):
    # One row per STATE, not per Edit, to avoid giving high-candidate states
    # extra weight. All models share these same common physical states.
    by={}
    for r in rows:
        by.setdefault(r['state_id'],[]).append(r)
    from collections import defaultdict
    strata=defaultdict(list)
    for rs in by.values():
        first=rs[0]
        best=max(float(r['delta_q']) for r in rs)
        chosen=int(first['selected_action'])
        strata[('ALL','ALL','ALL')].append((best,chosen))
        strata[(first['scenario'],str(first['initial_budget']),
                str(first['remaining_budget']))].append((best,chosen))
    stats=[]
    for (sc,b,r),val in sorted(strata.items()):
        stats.append(dict(gradient_steps=step,scenario=sc,initial_budget=b,
             remaining_budget=r,n_states=len(val),
             mean_best_qgap=mean(x[0] for x in val),
             fraction_qgap_positive=sum(x[0]>0 for x in val)/len(val),
             fraction_choose_keep=sum(x[1]==0 for x in val)/len(val)))
    return stats


def train_f1(a):
    if not a.probe_bank:raise ValueError('F1 requires --probe-bank')
    if a.gradient_updates<1 or a.batch_size<1 or a.checkpoint_every<0 or a.probe_every<1:
        raise ValueError('Invalid F1 gradient/batch/probe grid')
    if a.group not in MODES:raise ValueError('Invalid A0–A3 group')
    started=time.perf_counter()
    source=source_dataset(a.data)
    probe=load_probe_bank(a.probe_bank)
    if file_sha(a.data)==probe['source_sha256']:
        raise ValueError('Probe raw data must be independent of train raw data')
    probe_source=source_dataset(probe['source_path'])
    if int(source['config']['train_seed_start'])==int(probe_source['config']['train_seed_start']):
        raise ValueError('Train/probe scenario seed ranges must be distinct')
    train_seeds={int(x['seed']) for x in source['episodes']}
    probe_seeds={int(x['seed']) for x in probe_source['episodes']}
    if train_seeds & probe_seeds:raise ValueError('Train/probe episode seeds overlap')
    for ep in source['episodes']:
        if Path(ep['case']).stem not in {str(i) for i in range(1,8)}:
            raise ValueError('F1 training data includes non-training Case 8+')
    for ep in probe_source['episodes']:
        if Path(ep['case']).stem not in {str(i) for i in range(1,8)}:
            raise ValueError('Probe data includes non-training Case 8+')
    out=Path(a.out)
    if out.exists():raise FileExistsError(f'F1 output already exists: {out}')
    out.mkdir(parents=True)
    representation,sampling=MODES[a.group]
    model=V12Agent(hidden_dim=96,learning_rate=3e-4,gamma=a.gamma,
         target_update_steps=250,seed=a.seed,device=a.device,
         q_network='set_ddqn',execution_margin=0.,bootstrap_margin=0.,
         representation=representation,reward_scale=100.)
    initial_sha=online_sha(model.online)
    replay=EpisodeReplay(a.replay_episodes,a.seed+100,representation)
    audits=[];raw_steps=0;macro_steps=0
    for ep in source['episodes']:
        result=compress_episode(ep['raw'],representation,a.gamma)
        if not math.isclose(result.audit['source_total_reward']*-100.,
                            float(ep['wait']),rel_tol=1e-6,abs_tol=1e-3):
            raise AssertionError('F1 source weighted_wait audit mismatch')
        replay.append_episode(ep['episode_id'],result.transitions,len(ep['raw']))
        raw_steps+=len(ep['raw']);macro_steps+=len(result.transitions)
        audits.append(dict(episode=ep['episode_id'],weighted_wait=ep['wait'],**result.audit))
    _write_rows(out/'conversion_audit.csv',audits,list(dict.fromkeys(k for x in audits for k in x)))
    if len(replay)<a.batch_size:
        raise ValueError(f'F1 {a.group} replay={len(replay)} < batch={a.batch_size}; '
                         'extend the SAME public dataset by 32 episodes')
    snapshot=_code_snapshot()
    now=dict(protocol='v12-fixed-data-F1-v1',group=a.group,representation=representation,
             sampling=sampling,source_file=str(a.data),source_sha256=file_sha(a.data),
             probe_file=str(a.probe_bank),probe_sha256=file_sha(a.probe_bank),
             probe_source_sha256=probe['source_sha256'],
             train_seed_range_start=int(source['config']['train_seed_start']),
             probe_seed_range_start=int(probe_source['config']['train_seed_start']),
             train_case_sha256={str(e['case']):file_sha(e['case']) for e in source['episodes']},
             probe_case_sha256={str(e['case']):file_sha(e['case']) for e in probe_source['episodes']},
             validation_case_sha256=({str(p):file_sha(p) for p in a.val_cases}
                                     if a.val_cases else {}),
             executed_command=list(sys.argv),
             initial_online_state_sha256=initial_sha,
             gradient_updates=a.gradient_updates,batch_size=a.batch_size,
             replay_episodes=a.replay_episodes,replay_size=len(replay),
             replay_stats_before_training=replay.stats(),
             raw_steps_in_dataset=raw_steps,macro_steps_before_window=macro_steps,
             gamma=a.gamma,seed=a.seed,device=a.device,
             update_rule='exactly 250 gradient steps per target update',
             probe_every=a.probe_every,diagnostic_every=a.diagnostic_every,
             checkpoint_every=a.checkpoint_every,
             td_diagnostics_enabled=bool(a.diagnostics),validation_requested=not a.skip_validation,
             source_snapshot=snapshot,
             python=sys.version,torch_version=torch.__version__,numpy_version=np.__version__)
    (out/'manifest.json').write_text(json.dumps(now,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    rec=TDRecorder(sample_every=a.diagnostic_every,window=250) if a.diagnostics else None
    if rec is not None:model.diagnostic_callback=rec
    curve=[];qgap=[];metrics=[]
    if a.checkpoint_every:
        cp=out/'checkpoints';cp.mkdir(exist_ok=True)
        torch.save(model.checkpoint(dict(f1_model_only=True,gradient_steps=0)),
                   cp/'grad_000000.pt')
    def probe_now(step):
        rows=evaluate_agent_on_probes(model,probe['states'],batch_size=32)
        for row in rows:row.update(gradient_steps=step,group=a.group,training_seed=a.seed)
        qgap.extend(rows);metrics.extend(_summarize_probe(rows,step))
    probe_now(0)
    for step in range(1,a.gradient_updates+1):
        batch,meta=replay.sample(a.batch_size,sampling,.25)
        loss=model.update(batch)
        if step%a.diagnostic_every==0 or step==a.gradient_updates:
            curve.append(dict(gradient_steps=step,loss=loss,
                   replay_size=len(replay),sample_reuse=replay.stats()['sample_reuse'],
                   sample_draws=step*a.batch_size,
                   normalized_reuse=step*a.batch_size/len(replay),
                   sampled_actionable=meta['sample_actionable'],
                   sampling_shortfall=int(bool(meta['shortfall']))))
        if step%a.probe_every==0 or step==a.gradient_updates:probe_now(step)
        if a.checkpoint_every and (step%a.checkpoint_every==0 or step==a.gradient_updates):
            torch.save(model.checkpoint(dict(f1_model_only=True,gradient_steps=step)),
                    out/'checkpoints'/f'grad_{step:06d}.pt')
    _write_rows(out/'train_curve.csv',curve,list(curve[0]))
    _write_rows(out/'probe_qgap.csv',qgap,list(qgap[0]) if qgap else ['state_id','action_index','delta_q'])
    _write_rows(out/'probe_metrics.csv',metrics,list(metrics[0]) if metrics else ['gradient_steps','n_states'])
    if rec is None:_write_rows(out/'td_diagnostics.csv',[],list(FIELDS))
    else:rec.write(out/'td_diagnostics.csv',model.gradient_steps)
    torch.save(model.checkpoint(dict(fixed_dataset=str(a.data),gradient_steps=model.gradient_steps,
                                     f1_model_only=True)),out/'final.pt')
    if not a.skip_validation:
        from types import SimpleNamespace
        from .validation import validate
        if not a.val_cases:raise ValueError('Need --val-cases unless --skip-validation')
        cfg=SimpleNamespace(val_cases=a.val_cases,scenarios=a.scenarios,
           val_seeds=a.val_seeds,budgets=a.budgets,
           validation_workers=a.validation_workers,
           clock='union',candidate_limit=4,period=12.,reward_scale=100.,
           validation_uncertainty=a.validation_uncertainty,updates=a.updates,
           candidate_screen='positive',min_gain=.1,
           budget_features='absolute',value_features='global',
           baseline_cache_dir=getattr(a,'baseline_cache_dir',None))
        now['validation']=validate(out/'final.pt',cfg,out/'validation')
        now['validation_executed']=True
    else:
        now['validation_executed']=False
        now['validation_skip_reason']='explicit --skip-validation; no closed-loop results'
    now['gradient_updates_actual']=model.gradient_steps
    now['replay_stats_after_training']=replay.stats()
    now['target_sync_count']=model.gradient_steps//250
    now['wall_seconds']=time.perf_counter()-started
    now['final_online_state_sha256']=online_sha(model.online)
    (out/'manifest.json').write_text(json.dumps(now,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    if model.gradient_steps!=a.gradient_updates:raise AssertionError('F1 incomplete gradient run')
    (out/'complete.json').write_text(json.dumps(dict(status='complete',
       group=a.group,gradient_steps=model.gradient_steps,validation_executed=not a.skip_validation,
       probe_states=len(probe['states']),f1_source_sha256=now['source_sha256'],
       replay_size=len(replay),wall_seconds=now['wall_seconds']),indent=2),encoding='utf-8')
    print(f'[F1] {a.group} seed={a.seed} steps={model.gradient_steps} replay={len(replay)} '
          f'probe={len(probe["states"])} complete={out}',flush=True)
    return now
