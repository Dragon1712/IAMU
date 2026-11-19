#!/usr/bin/env python3
"""E4-E controlled Ep200->400 offline learning from the SAME full-state fork.

Branches:
  actionable_uniform: Frozen Replay, uniform among legal Edit states only;
                      natural actionable Keep/Edit ratio.
  uniform: Frozen Replay, original uniform without-replacement sampler,
           normal online/target bootstrap and target-network synchronization.
  fixed_target: Frozen Replay, uniform sampler, target frozen at fork snapshot.
  balanced: Frozen Replay, oversample ACTIONABLE Keep/Edit 1:1,
            normal target updates. Diagnostic distribution intervention.

Gradient count per completed episode is taken from the matching observed
ONLINE training trace (200<completed<=400). A new record is added only after
completed replay-independent updates; no interaction / no new transitions.
No simulator future labels or candidate modifications enter training.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,sys
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from e4_mechanism.mechanism_core import replay_stats,deterministic_probe,balanced_batch,actionable_batch,replay_fingerprint
from e4_mechanism.telemetry import sha,eval_qgap,append_csv
from e4_mechanism.td_probe import td_rows,td_summary


def load_fork(path,*,must_trust):
    if not must_trust:
        raise PermissionError('Full recovery state uses pickle. Only load a locally produced trusted state with --trust-local-training-state.')
    import torch
    p=Path(path)
    meta=p.with_suffix('.metadata.json')
    if not p.is_file() or not meta.is_file():raise FileNotFoundError(p)
    attrs=json.loads(meta.read_text(encoding='utf-8'))
    if attrs['full_state_sha256']!=sha(p):raise ValueError('Fork state SHA256 mismatch')
    # Required: Python Transition and NumPy legacy classes in original
    # optimizer/replay state; NEVER load untrusted downloaded checkpoints.
    item=torch.load(p,map_location='cpu',weights_only=False)
    if item.get('version')!=1:raise ValueError('Expected E4 full-state version=1')
    if len(item['train_rows'])!=int(attrs['completed']):
        raise ValueError('Unmatched completed episode count at fork')
    if item['validations'].get(f'val_{attrs["completed"]:06d}',{}).get('status')!='done':
        raise ValueError('Fork validation incomplete')
    if len(item['replay_items'])!=int(attrs['replay_size']):
        raise ValueError('Fork Replay truncated')
    if int(item['gradient_steps'])!=int(attrs['gradient_steps']):
        raise ValueError('Fork gradient count mismatch')
    # The full-state fork must contain the exact online weights evaluated at
    # EP200, not an unvalidated policy saved from a future episode.
    record=item['validations'][f'val_{attrs["completed"]:06d}']
    validated=Path(record['snapshot'])
    if not validated.exists():
        validated=Path(attrs['source_out'])/'validation_snapshots'/validated.name
    if not validated.is_file() or sha(validated)!=attrs['validation_sha256']:
        raise ValueError('Corresponding EP200 validation snapshot missing/bad')
    fixed=torch.load(validated,map_location='cpu',weights_only=True)['online']
    if set(fixed)!=set(item['online']) or any(not torch.equal(fixed[k],item['online'][k]) for k in fixed):
        raise ValueError('Full-state online network differs from validated EP200 snapshot')
    return item,attrs


def read_schedule(path,start,end,initial_steps):
    """Future update counts are known *ex post* for matched-gradient ablation.

    Using the ONLINE completion-order credit schedule is a comparison design
    choice, not online deployment information. No extra experience is exposed.
    """
    with Path(path).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    lookup={int(x['completed']):x for x in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate completed in training trace')
    if start not in lookup or end not in lookup:raise ValueError('Reference online trace incomplete')
    if int(lookup[start]['gradient_steps'])!=initial_steps:
        raise ValueError('Reference online trace is NOT the same Episode200 fork')
    out=[]
    prev=initial_steps
    for ep in range(start+1,end+1):
        item=lookup.get(ep)
        if item is None:raise ValueError(f'Missing online episode {ep}')
        n=int(item['learner_updates'])
        grad=int(item['gradient_steps'])
        if n<0 or grad!=prev+n:raise ValueError(f'Invalid learner-update accounting at ep{ep}')
        out.append((ep,n))
        prev=grad
    return out


def make_learner(item,device):
    import torch
    from iamu_dynamic.rl_ddqn import DoubleDQNAgent,ReplayBuffer
    cfg=item['config']['training_args']
    learner=DoubleDQNAgent(hidden_dim=int(cfg['hidden_dim']),
        learning_rate=float(cfg['lr']),gamma=float(cfg['gamma']),
        target_update_steps=int(cfg['target_update']),seed=int(cfg['seed']),
        device=device,keep_margin=float(cfg['keep_margin']),q_network=cfg['q_network'])
    learner.online.load_state_dict(item['online'])
    learner.target.load_state_dict(item['target'])
    learner.optimizer.load_state_dict(item['optimizer'])
    learner.gradient_steps=int(item['gradient_steps'])
    learner.rng.setstate(item['agent_rng'])
    replay=ReplayBuffer(int(cfg['replay_capacity']),int(cfg['seed'])+100)
    replay.items.extend(item['replay_items'])
    replay.rng.setstate(item['replay_rng'])
    return learner,replay,cfg


def snapshot(out,learner,episode,fork_attrs,mode):
    import torch
    sub=Path(out)/'checkpoints';sub.mkdir(parents=True,exist_ok=True)
    path=sub/f'ep_{episode:06d}_grad_{learner.gradient_steps:09d}.pt'
    if path.exists():raise FileExistsError(path)
    info=dict(source='E4-E frozen replay offline experiment',selected_episode=episode,
         n_gradient_updates=learner.gradient_steps,comparison_mode=mode,
         full_fork_sha256=fork_attrs['full_state_sha256'])
    temp=path.with_suffix('.pt.tmp')
    torch.save(learner.checkpoint(info),temp);temp.replace(path)
    return str(path),sha(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork',required=True,help='E5 monitored source: e5_mechanism/full_state_ep0200.pt')
    p.add_argument('--online-trace',required=True,help='E5 ONLINE run train_curve.csv from SAME episode200 fork')
    p.add_argument('--probe-bank',required=True)
    p.add_argument('--mode',choices=('uniform','fixed_target','actionable_uniform','balanced'),required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--until',type=int,default=400)
    p.add_argument('--every',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=0,help='0=original training batch size')
    p.add_argument('--trust-local-training-state',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.every<1 or a.until<1:p.error('Invalid --until/--every')
    dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):p.error('Choose a NEW empty --out')
    import numpy as np
    item,attrs=load_fork(a.fork,must_trust=a.trust_local_training_state)
    learner,replay,cfg=make_learner(item,a.device)
    fork_ep=int(attrs['completed'])
    if a.until<=fork_ep:p.error('--until must exceed fork episode')
    schedule=read_schedule(a.online_trace,fork_ep,a.until,learner.gradient_steps)
    bs=int(cfg['batch_size']) if a.batch_size==0 else a.batch_size
    if bs<2:p.error('batch size must be >=2')
    if a.batch_size and bs!=int(cfg['batch_size']):
        p.error('Matched-gradient ablation requires SAME original batch size')
    with np.load(a.probe_bank,allow_pickle=False) as data:
        bank={x:data[x].copy() for x in ('g','actions','mask')}
    if attrs['probe_sha256']!=sha(a.probe_bank):p.error('Probe bank mismatch')
    td_probe_path=Path(a.fork).parent/f'td_probe_ep{fork_ep:04d}.pt'
    if attrs['td_probe_sha256']!=sha(td_probe_path):p.error('Frozen TD probe changed')
    # trusted local archive, fixed sampled transitions, no future labels
    import torch
    td_probe=torch.load(td_probe_path,map_location='cpu',weights_only=False)
    baseline_replay=list(replay.items)
    baseline_replay_sha=replay_fingerprint(baseline_replay)
    baseline_stat=replay_stats(baseline_replay)
    if a.mode in ('balanced','actionable_uniform') and baseline_stat['n_legal_choice']<bs:
        p.error('Actionable branch requires >=batch_size actionable transitions')
    if a.mode=='balanced':
        if baseline_stat['n_edit']==0 or baseline_stat['n_keep_with_legal_edit']==0:
            p.error('Balanced branch needs both replay Edit and actionable Keep')
    if a.mode=='fixed_target':
        learner.target_update_steps=10**15
    brng=random.Random(17371+int(cfg['seed']))
    dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(a),source_run=attrs['source_out'],source_fork_sha256=attrs['full_state_sha256'],
        source_training_args=cfg,online_trace_sha256=sha(a.online_trace),
        probe_sha256=attrs['probe_sha256'],td_probe_sha256=attrs['td_probe_sha256'],
        source_learner_grad=attrs['gradient_steps'],source_replay_stats=baseline_stat,
        frozen_replay_sha256=baseline_replay_sha,
        logic='fixed dataset, identical online phase update counts; no external data; balanced is a distribution intervention',
        causal_limits='fixed snapshot is off-policy: offline performance changes do not prove original online replay drift causes collapse')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    completed_updates=0
    samples_repeated=0
    def record(ep):
        q=eval_qgap(learner,bank)
        append_csv(dest/'qgap_probe.csv',dict(mode=a.mode,episode=ep,
            gradient_steps=learner.gradient_steps,**q))
        rep=replay_stats(replay.items)
        append_csv(dest/'replay_check.csv',dict(mode=a.mode,episode=ep,
            gradient_steps=learner.gradient_steps,**rep))
        for row in td_summary(td_rows(learner,td_probe),name='ep200_fixed'):
            append_csv(dest/'td_probe.csv',dict(mode=a.mode,episode=ep,
                gradient_steps=learner.gradient_steps,**row))
        name,checksum=snapshot(dest,learner,ep,attrs,a.mode)
        append_csv(dest/'checkpoints.csv',dict(mode=a.mode,episode=ep,
            gradient_steps=learner.gradient_steps,checkpoint=name,sha256=checksum))
        print(f'[E5-OFFLINE] mode={a.mode} ep={ep} grad={learner.gradient_steps} '
              f'qgap={q["mean_best_q_gap"]:.4f} keep={q["fraction_keep_with_legal"]:.3f}',flush=True)
    record(fork_ep)
    last_ep=fork_ep
    for ep,n_updates in schedule:
        losses=[];unique=[]
        for _ in range(n_updates):
            if a.mode=='balanced':
                batch,nb=balanced_batch(replay.items,bs,brng)
                unique.append(nb)
            elif a.mode=='actionable_uniform':
                batch,nb=actionable_batch(replay.items,bs,brng)
                unique.append(nb)
            else:
                batch=replay.sample(bs)
            losses.append(learner.update(batch))
        completed_updates+=n_updates
        append_csv(dest/'offline_updates.csv',dict(mode=a.mode,episode=ep,
            learner_updates=n_updates,cumulative_updates=completed_updates,
            gradient_steps=learner.gradient_steps,mean_loss=mean(losses) if losses else None,
            mean_unique_per_batch=mean(unique) if unique else None,
            replay_size=len(replay)))
        last_ep=ep
        if (ep-fork_ep)%a.every==0 or ep==a.until:record(ep)
    # Exact frozen Replay still stored in original deque, no appends occurred.
    from e4_mechanism.mechanism_core import assert_immutable_replay
    assert_immutable_replay(baseline_replay,replay.items)
    if replay_fingerprint(replay.items)!=baseline_replay_sha:
        raise AssertionError('Frozen Replay CONTENT changed during offline training')
    if learner.gradient_steps != int(attrs['gradient_steps'])+completed_updates:
        raise AssertionError('Offline gradient count mismatch')
    (dest/'complete.json').write_text(json.dumps(dict(mode=a.mode,completed_ep=last_ep,
       total_gradient_updates=completed_updates,final_gradient_steps=learner.gradient_steps,
       replay_unchanged=True),indent=2))
    print(f'[DONE] {a.mode} {len(schedule)} additional episode-schedule slices '
          f'{completed_updates} updates => {dest}')

if __name__=='__main__':main()
