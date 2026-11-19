#!/usr/bin/env python3
"""Read-only decomposition of interval DDQN target and Keep/Edit predictions.

A model-only checkpoint resets target := online and is NOT the historical
training target. Pass --full-state for the exact saved target net at its round
boundary. Both modes are diagnostic; neither performs gradient updates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from .common import (atomic_json,digest,load_model,load_trusted,provenance,
                     verify_v12_scientific_model,write_csv)


def _arrays(batch,name,dtype):
    return torch.as_tensor(np.stack([getattr(x,name) for x in batch]),dtype=dtype)


def inspect_batch(agent,trs):
    if not trs:return []
    og=_arrays(trs,'obs_global',torch.float32)
    oa=_arrays(trs,'obs_actions',torch.float32)
    om=_arrays(trs,'obs_mask',torch.bool)
    ng=_arrays(trs,'next_global',torch.float32)
    na=_arrays(trs,'next_actions',torch.float32)
    nm=_arrays(trs,'next_mask',torch.bool)
    a=torch.as_tensor([x.action for x in trs],dtype=torch.long)
    r=torch.as_tensor([x.reward for x in trs],dtype=torch.float32)
    d=torch.as_tensor([x.bootstrap_discount for x in trs],dtype=torch.float32)
    with torch.no_grad():
        q=agent.online(og,oa,om)
        taken=q.gather(1,a[:,None]).squeeze(1)
        nq=agent.online(ng,na,nm).masked_fill(~nm,-torch.inf).clone()
        nq[:,0]+=agent.bootstrap_margin
        nxt=nq.argmax(1)
        tgt=agent.target(ng,na,nm).gather(1,nxt[:,None]).squeeze(1)
        boot=d*tgt
        y=r+boot
        q_keep=q[:,0]
    out=[]
    for i,t in enumerate(trs):
        legal=np.flatnonzero(np.asarray(t.obs_mask,dtype=bool)[1:])+1
        chosen=int(t.action)
        if chosen>0:gap=float(q[i,chosen]-q_keep[i])
        elif len(legal):gap=float(torch.max(q[i,legal])-q_keep[i])
        else:gap=float('nan')
        out.append(dict(state_id=f'{t.episode_id}:{t.raw_start_index}',
             episode_id=t.episode_id,raw_start_index=t.raw_start_index,
             raw_end_index=t.raw_end_index,source_budget=t.source_budget,
             next_budget=t.next_budget,action=chosen,action_is_edit=int(chosen!=0),
             actionable=int(bool(len(legal))),done=int(t.done),
             n_raw_steps=t.n_raw_steps,elapsed_hours=t.elapsed_hours,
             interval_reward=float(r[i]),bootstrap_discount=float(d[i]),
             next_selected_action=int(nxt[i]),next_target_value=float(tgt[i]),
             bootstrap=float(boot[i]),td_target=float(y[i]),
             q_taken=float(taken[i]),q_keep=float(q_keep[i]),
             current_max_edit_gap=gap,td_error=float(y[i]-taken[i]),
             macro_length_bin=('1' if t.n_raw_steps==1 else '2-4' if t.n_raw_steps<5
                               else '5-19' if t.n_raw_steps<20 else '20+')))
    return out


def audit(data_path,representation,model_path,full_state=None,max_rows=0,batch_size=128):
    from v12_protocol.compress import compress_episode
    agent=load_model(model_path)
    verify_v12_scientific_model(agent,representation,agent.gamma)
    if representation!=agent.representation:
        raise ValueError(f'Model representation {agent.representation} != {representation}')
    source='model_only_target_equal_online'
    if full_state:
        state=load_trusted(full_state)
        if state.get('protocol')!=12 or 'target' not in state:
            raise ValueError('Expected trusted full V1.2 training state')
        if any(not torch.equal(v,state['online'][key])
               for key,v in agent.online.state_dict().items()):
            raise ValueError('Model checkpoint and full training-state online weights differ. '
                             'Use matching round/gradient snapshots; never mix best.pt with '
                             'a final-episode training_state.pt')
        agent.online.load_state_dict(state['online'],strict=True)
        agent.target.load_state_dict(state['target'],strict=True)
        source='full_training_state_round_boundary'
    agent.online.eval();agent.target.eval()
    data=load_trusted(data_path)
    trs=[]
    episode_jobs={r['job']['episode_id']:r['job'] for r in data['records']}
    episode_behaviors={r['job']['episode_id']:r['behavior'] for r in data['records']}
    for record in data['records']:
        trs.extend(compress_episode(list(record['raw']),representation,agent.gamma).transitions)
    n_all=len(trs)
    if max_rows>0 and len(trs)>max_rows:
        # Deterministic uniform index grid; not a random dataset resampling.
        idx=np.linspace(0,len(trs)-1,num=max_rows,dtype=np.int64)
        trs=[trs[i] for i in idx]
    rows=[]
    for j in range(0,len(trs),batch_size):rows.extend(inspect_batch(agent,trs[j:j+batch_size]))
    for row in rows:
        job=episode_jobs[row['episode_id']]
        row.update(case=job['case'],scenario=job['scenario'],
                   original_budget=job['budget'],behavior=episode_behaviors[row['episode_id']])
    return rows,dict(representation=representation,model_sha256=digest(model_path),
                     full_state_sha256=digest(full_state) if full_state else None,
                     target_source=source,full_dataset_macro_count=n_all,
                     n_logged=len(rows),gamma=agent.gamma)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--representation',choices=('primitive','budget_tail','decision_epoch'),required=True)
    p.add_argument('--full-state',help='LOCAL trusted training_state.pt, not GitHub model weights')
    p.add_argument('--max-rows',type=int,default=10000)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    rows,meta=audit(a.data,a.representation,a.checkpoint,a.full_state,a.max_rows)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'target_audit.csv',rows)
    atomic_json(out/'audit_meta.json',meta)
    atomic_json(out/'manifest.json',provenance(a,inputs=[x for x in
                       [a.data,a.checkpoint,a.full_state] if x]))
    print('[V13 target audit]',meta,flush=True)

if __name__=='__main__':main()
