#!/usr/bin/env python3
"""H1: SAME-STATE paired SMDP TD targets under SAME pre-generated future tape.

Construct exactly one decision_epoch macro per branch: source action then forced
Keep steps until the next public legal-Edit decision or terminal. Report the
interval reward and model bootstrap separately. No gradients and no oracle
future information is ever passed to the network.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from .common import (atomic_json,digest,load_model,load_trusted,provenance,
                     verify_v12_scientific_model,write_csv)
from .future_tapes import apply_tape,draw_tape,tape_hash
from .counterfactual import predict_pair


def first_macro(env,initial_action,gamma=1.):
    """Construct A3 interval online from a restored frozen state.

    Only Keep may be executed AFTER the source action until the next decision
    anchor. A state with legal Edit is retained even when the policy chose Keep.
    """
    if env.done:raise ValueError('Already terminal snapshot')
    obs=env._observation()
    if not bool(obs.mask[initial_action]):raise ValueError('Illegal initial action')
    reward=0.;discount=1.;steps=0;first=int(initial_action)
    while True:
        obs2,r,done,_=env.step(first if steps==0 else 0)
        reward+=discount*float(r)
        steps+=1;discount*=gamma
        if done or bool(np.asarray(obs2.mask,dtype=bool)[1:].any()):break
    return dict(interval_reward=reward,bootstrap_discount=(0. if done else discount),
                n_raw_steps=steps,done=bool(done),next_observation=obs2,
                remaining_budget=int(obs2.remaining_budget))


def _values(agent,obs,action):
    with torch.no_grad():
        g=torch.as_tensor(obs.global_features,dtype=torch.float32).unsqueeze(0)
        x=torch.as_tensor(obs.action_features,dtype=torch.float32).unsqueeze(0)
        m=torch.as_tensor(obs.mask,dtype=torch.bool).unsqueeze(0)
        q=agent.online(g,x,m)[0,action].item()
    return float(q)


def _bootstrap(agent,macro):
    if macro['done']:return 0.,None,0.
    obs=macro['next_observation']
    with torch.no_grad():
        g=torch.as_tensor(obs.global_features,dtype=torch.float32).unsqueeze(0)
        x=torch.as_tensor(obs.action_features,dtype=torch.float32).unsqueeze(0)
        m=torch.as_tensor(obs.mask,dtype=torch.bool).unsqueeze(0)
        scores=agent.online(g,x,m).masked_fill(~m,-torch.inf).clone()
        scores[:,0]+=agent.bootstrap_margin
        next_action=int(scores.argmax(dim=1)[0])
        v=float(agent.target(g,x,m)[0,next_action])
    return macro['bootstrap_discount']*v,next_action,v


def audit_pairs(probes,model,tapes=8,seed=910000,gamma=1.,uncertainty=8.,updates=2):
    rows=[];comparisons=[]
    for p in probes:
        snap=p['snapshot'];sid=p['state_id'];edit=p['selected_edit']
        if not snap._observation().mask[edit]:raise AssertionError('Stale saved Edit slot')
        q_keep,q_edit=predict_pair(model,snap._observation(),edit)
        for j in range(tapes):
            tape=draw_tape(snap,sid,j,base_seed=seed,
                           uncertainty=uncertainty,updates_per_vessel=updates)
            th=tape_hash(tape)
            vals={}
            for label,action in [('Keep',0),('Edit',edit)]:
                world=apply_tape(snap,tape)
                interval=first_macro(world,action,gamma)
                bootstrap,nxt_action,nxt_value=_bootstrap(model,interval)
                q_taken=q_keep if label=='Keep' else q_edit
                td=interval['interval_reward']+bootstrap
                vals[label]=dict(state_id=sid,tape_index=j,tape_sha256=th,
                    scenario=p['scenario'],source_budget=p['source_budget'],
                    action_descriptor=json.dumps({'kind':'keep'} if label=='Keep'
                      else p['action_descriptor'],sort_keys=True),
                    branch=label,action_slot=action,done=int(interval['done']),
                    n_raw_steps=interval['n_raw_steps'],
                    next_budget=interval['remaining_budget'],
                    interval_reward=interval['interval_reward'],
                    bootstrap_discount=interval['bootstrap_discount'],
                    next_target_action=nxt_action,next_target_q=nxt_value,
                    bootstrap=bootstrap,q_taken=q_taken,td_target=td,
                    td_error=td-q_taken)
                rows.append(vals[label])
            a,b=vals['Keep'],vals['Edit']
            comparisons.append(dict(state_id=sid,tape_index=j,
                    scenario=p['scenario'],source_budget=p['source_budget'],
                    tape_sha256=th,
                    predicted_q_gap=q_edit-q_keep,
                    paired_target_gap=b['td_target']-a['td_target'],
                    paired_target_gap_error=(q_edit-q_keep)-(b['td_target']-a['td_target']),
                    interval_reward_gap=b['interval_reward']-a['interval_reward'],
                    bootstrap_gap=b['bootstrap']-a['bootstrap'],
                    keep_bootstrap=a['bootstrap'],edit_bootstrap=b['bootstrap'],
                    keep_done=a['done'],edit_done=b['done'],
                    keep_n=a['n_raw_steps'],edit_n=b['n_raw_steps'],
                    keep_td_error=a['td_error'],edit_td_error=b['td_error'],
                    action_descriptor=json.dumps(p['action_descriptor'],sort_keys=True)))
    return rows,comparisons


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--full-state',help='Only same-gradient, locally trusted V1.2 training state')
    p.add_argument('--tapes',type=int,default=8)
    p.add_argument('--max-states',type=int,default=0)
    p.add_argument('--start-index',type=int,default=0,
                   help='Inclusive index into probe_bank for parallel shards')
    p.add_argument('--stop-index',type=int,default=0,
                   help='Exclusive index; 0 means end of bank')
    p.add_argument('--seed',type=int,default=910000)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.tapes<1 or not 0<a.gamma<=1:raise ValueError('Invalid tapes/gamma')
    model=load_model(a.checkpoint)
    verify_v12_scientific_model(model,'decision_epoch',a.gamma)
    if model.representation!='decision_epoch':
        raise ValueError('Paired target macro definition requires A3 decision_epoch checkpoint')
    if not math.isclose(model.gamma,a.gamma):
        raise ValueError('Gamma differs from saved V1.2 model')
    if a.full_state:
        state=load_trusted(a.full_state)
        if state.get('protocol')!=12:raise ValueError('Not V1.2 full-state')
        if any(not torch.equal(v,state['online'][k])
               for k,v in model.online.state_dict().items()):
            raise ValueError('Full-state and checkpoint gradient versions disagree')
        model.target.load_state_dict(state['target'],strict=True)
    bank=load_trusted(a.probes)
    stop=a.stop_index or len(bank['probes'])
    if not 0<=a.start_index<stop<=len(bank['probes']):
        raise ValueError('Invalid probe index range')
    probes=bank['probes'][a.start_index:stop]
    if a.max_states>0 and a.max_states<len(probes):
        # Bank order is stratum-sorted. Taking its first N would silently
        # omit later budgets/scenarios, so spread the pilot across all strata.
        indices=np.linspace(0,len(probes)-1,a.max_states,dtype=int)
        probes=[probes[int(i)] for i in indices]
    rows,pairs=audit_pairs(probes,model,a.tapes,a.seed,a.gamma,a.uncertainty,a.updates)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'paired_target_components.csv',rows)
    write_csv(out/'paired_target_gap.csv',pairs)
    info=dict(n_states=len(probes),n_tapes=a.tapes,n_rows=len(rows),
        n_pairs=len(pairs),target_kind=('matching_full_training_state'
                    if a.full_state else 'model_only_target_equal_online'),
        meaning='Paired one-macro TD targets, NOT true counterfactual action values',
        probes_sha256=digest(a.probes),checkpoint_sha256=digest(a.checkpoint))
    atomic_json(out/'summary.json',info)
    atomic_json(out/'manifest.json',provenance(a,inputs=[x for x in
                      (a.probes,a.checkpoint,a.full_state) if x]))
    print('[V13 paired targets]',info,flush=True)

if __name__=='__main__':main()
