"""Train two A0/A2 offline curves from the EXACT same V13 raw trajectories.

At Gshort=round(G*N2/N0) and G, store checkpoints and fixed-probe Q gaps.
"""
from __future__ import annotations
import argparse,hashlib,time,csv
from pathlib import Path
from statistics import mean
from collections import defaultdict
import numpy as np
import torch
from v13_diagnostics.common import load_trusted,write_csv,atomic_json
from .utils import boundary_provenance


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def check_data(p0,p2):
    if p0.get('representation')!='primitive' or p2.get('representation')!='budget_tail':
        raise ValueError('Require primitive/budget_tail derived datasets')
    if p0.get('source_sha256')!=p2.get('source_sha256') or p0.get('gamma')!=p2.get('gamma'):
        raise ValueError('Datasets must share raw source hash/gamma')
    if p0.get('gamma')!=1.:raise ValueError('Expected gamma=1 main protocol')
    e0,e2=p0['episodes'],p2['episodes']
    if not e0 or len(e0)!=len(e2):raise ValueError('Episode population mismatch')
    for x,y in zip(e0,e2):
        if x['episode_id']!=y['episode_id'] or x['job']!=y['job']:
            raise ValueError('Episode order/physical job mismatch')
        if len(x['transitions'])<len(y['transitions']):raise AssertionError('Compressed set expanded')
    n0=sum(len(x['transitions']) for x in e0)
    n2=sum(len(x['transitions']) for x in e2)
    if not 0<n2<=n0:raise ValueError('Invalid transition counts')
    return n0,n2


def read_reference(path):
    data={}
    with open(path,encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            sid=r['state_id'];val=float(r['reference_gap_mean'])
            if sid in data and abs(data[sid]-val)>1e-7:
                raise ValueError('Inconsistent same-state reference')
            data[sid]=val
    return data


def shared_probe_gap(model,bank,reference=None):
    from v13_diagnostics.counterfactual import predict_pair
    was_training=model.online.training
    model.online.eval()
    out=[]
    try:
        for p in bank['probes']:
            sid=p['state_id'];qk,qe=predict_pair(model,p['snapshot']._observation(),p['selected_edit'])
            row=dict(state_id=sid,episode_id=p['episode_id'],source_budget=p['source_budget'],
                     scenario=p['scenario'],q_keep=qk,q_edit=qe,predicted_gap=qe-qk)
            if reference is not None:
                if sid not in reference:raise ValueError('Missing independent reference for '+sid)
                row.update(reference_gap_mean=reference[sid],gap_error=qe-qk-reference[sid],
                    abs_gap_error=abs(qe-qk-reference[sid]),
                    missed_positive=int(reference[sid]>0 and qe<=qk))
            out.append(row)
    finally:
        if was_training:model.online.train()
    return out


def run_fixed(a):
    from v12_protocol.agent import V12Agent
    from v12_protocol.replay import EpisodeReplay
    p0,p2=load_trusted(a.primitive),load_trusted(a.budget_tail)
    n0,n2=check_data(p0,p2)
    short=max(1,round(a.gradient_updates*n2/n0))
    if short>=a.gradient_updates:raise ValueError('No exposure contrast')
    bank=load_trusted(a.probes) if a.probes else None
    reference=read_reference(a.reference_csv) if a.reference_csv else None
    if reference is not None and bank is None:raise ValueError('Need independent probe bank')
    if bank and ({p['episode_id'] for p in bank['probes']} &
                 {p['episode_id'] for p in p0['episodes']}):raise ValueError('Probe/training overlap')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    curves=[];gaprows=[];initial={}
    for group,source in (('A0',p0),('A2',p2)):
        mode=source['representation']
        replay=EpisodeReplay(max_episodes=len(source['episodes']),seed=a.seed+100,representation=mode)
        for ep in source['episodes']:
            replay.append_episode(ep['episode_id'],tuple(ep['transitions']),
                                  int(ep['compression_audit']['raw_steps']))
        model=V12Agent(hidden_dim=96,learning_rate=a.learning_rate,gamma=1.,
            target_update_steps=a.target_update,seed=a.seed,device=a.device,
            q_network='set_ddqn',execution_margin=0.,bootstrap_margin=0.,
            representation=mode,reward_scale=100.)
        init=tensor_digest(model.online.state_dict());initial[group]=init
        probe_points={0,short,a.gradient_updates}
        if a.diagnostic_every>0:probe_points.update(range(a.diagnostic_every,a.gradient_updates+1,a.diagnostic_every))
        begin=time.perf_counter();exposure_actionable=0
        def report(step,loss=None):
            total=sum(replay.sample_counts.values())
            curves.append(dict(group=group,gradient_steps=step,
                sampled_total=total,sample_exposure=total/len(replay),
                actionable_sample_exposure=exposure_actionable,
                target_syncs=step//a.target_update,raw_episodes=len(source['episodes']),
                n_transitions=len(replay),loss=loss if loss is not None else '',
                elapsed_seconds=time.perf_counter()-begin))
            if bank:
                gaprows.extend(dict(group=group,gradient_steps=step,**p)
                               for p in shared_probe_gap(model,bank,reference))
            if step in (0,short,a.gradient_updates):
                name='initial' if step==0 else 'short' if step==short else 'full'
                torch.save(model.checkpoint(dict(protocol='v13-boundary-fixed',
                   group=group,gradient_steps=step,initial_network_sha256=init,
                   source_sha256=source['source_sha256'],short_updates=short,
                   full_updates=a.gradient_updates)),out/f'{group}_{name}.pt')
        report(0)
        for step in range(1,a.gradient_updates+1):
            batch,meta=replay.sample(a.batch_size,'uniform')
            exposure_actionable+=int(meta['sample_actionable'])
            loss=model.update(batch)
            if step in probe_points:report(step,loss)
    if initial['A0']!=initial['A2']:raise AssertionError('Networks not equally initialized')
    write_csv(out/'fixed_data_curve.csv',curves)
    write_csv(out/'gap_metrics.csv',gaprows,columns=['group','gradient_steps','state_id',
       'episode_id','source_budget','scenario','q_keep','q_edit','predicted_gap',
       'reference_gap_mean','gap_error','abs_gap_error','missed_positive'])
    buckets=defaultdict(list)
    for row in gaprows:
        for budget in ('ALL',str(row['source_budget']) if int(row['source_budget'])<3 else '3+'):
            for sc in ('ALL',row['scenario']):
                buckets[(row['group'],row['gradient_steps'],budget,sc)].append(row)
    summary=[]
    for (group,step,budget,sc),rows in sorted(buckets.items()):
        errors=[r['gap_error'] for r in rows if 'gap_error' in r]
        summary.append(dict(group=group,gradient_steps=step,budget=budget,scenario=sc,
            n_states=len(rows),mean_gap=mean(r['predicted_gap'] for r in rows),
            gap_bias=mean(errors) if errors else '',gap_mae=mean(abs(x) for x in errors) if errors else '',
            missed_positive=sum(r.get('missed_positive',0) for r in rows)))
    write_csv(out/'fixed_gap_summary.csv',summary)
    atomic_json(out/'manifest.json',boundary_provenance(a,inputs=[x for x in
       (a.primitive,a.budget_tail,a.probes,a.reference_csv) if x])|
       dict(N0=n0,N2=n2,Gshort=short,G=a.gradient_updates,
            initial_online_sha256=initial['A0']))
    print('[fixed] N0=',n0,'N2=',n2,'Gshort=',short,'G=',a.gradient_updates)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primitive',required=True);p.add_argument('--budget-tail',required=True)
    p.add_argument('--probes');p.add_argument('--reference-csv')
    p.add_argument('--gradient-updates',type=int,default=4000)
    p.add_argument('--diagnostic-every',type=int,default=250)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--target-update',type=int,default=250)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=43);p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.gradient_updates<2 or a.batch_size<1 or a.target_update<1 or a.diagnostic_every<0:
        p.error('Invalid hyperparameters')
    run_fixed(a)
if __name__=='__main__':main()
