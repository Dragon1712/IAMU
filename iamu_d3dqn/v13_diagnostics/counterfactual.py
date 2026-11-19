#!/usr/bin/env python3
"""Paired Keep/Edit counterfactual with explicit common exogenous future tapes.

The target is Q^continuation for specified fixed Greedy or frozen A0 policy,
NOT optimal Q*. A0/A2/A3 networks see identical public probe observations.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from statistics import mean,stdev
from dataclasses import asdict

import numpy as np
import torch
from .common import (atomic_json,digest,load_model,load_trusted,provenance,
                     verify_v12_scientific_model,write_csv)
from .future_tapes import apply_tape,draw_tape,tape_hash


def predict_pair(model,obs,edit_index):
    with torch.no_grad():
        g=torch.as_tensor(obs.global_features,dtype=torch.float32).unsqueeze(0)
        ac=torch.as_tensor(obs.action_features,dtype=torch.float32).unsqueeze(0)
        mask=torch.as_tensor(obs.mask,dtype=torch.bool).unsqueeze(0)
        q=model.online(g,ac,mask).flatten()
        if not bool(mask[0,edit_index]):raise ValueError('Probe Edit is illegal')
        return float(q[0]),float(q[edit_index])


def rollout_branch(env,first_action,continuation='greedy',frozen=None,gamma=1.):
    if continuation=='a0' and frozen is None:raise ValueError('Need frozen A0 continuation')
    rewards=[];actions=[]
    action=int(first_action)
    while True:
        ob=env._observation()
        if not bool(ob.mask[action]):raise AssertionError('Illegal counterfactual action')
        nxt,reward,done,_=env.step(action)
        rewards.append(float(reward));actions.append(int(action))
        if done:break
        if continuation=='greedy':action=int(env.point_greedy_action())
        elif continuation=='a0':action=int(frozen.act(nxt,epsilon=0.))
        else:raise ValueError(continuation)
    return sum((gamma**i)*v for i,v in enumerate(rewards)),tuple(actions)


def approximate_t_critical(n):
    """Two-sided Student-t 95% cutoff. Exact table for n <= 31."""
    table=(None, None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447,
           2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
           2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
           2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
           2.048, 2.045, 2.042)
    if n<2:raise ValueError('Need at least two tapes')
    if n<len(table):return table[n]
    return 1.96+2.4/(n-1)


def run_counterfactual(probes,models,tapes=8,seed=910000,continuation='greedy',
                       frozen=None,gamma=1.,uncertainty=8.,updates=2):
    rows=[];case_audit=[];tape_records=[]
    for i,p in enumerate(probes,1):
        snapshot=p['snapshot'];sid=p['state_id'];edit=p['selected_edit']
        public=snapshot._observation()
        if not public.mask[edit]:raise AssertionError('Saved Edit has become illegal')
        if int(public.remaining_budget)!=p['source_budget']:
            raise AssertionError('Snapshot source budget differs')
        predictions={name:predict_pair(model,public,edit) for name,model in models.items()}
        outcomes=[];hashes=[]
        for j in range(tapes):
            tape=draw_tape(snapshot,sid,j,base_seed=seed,
                           uncertainty=uncertainty,updates_per_vessel=updates)
            th=tape_hash(tape);hashes.append(th)
            tape_records.append(dict(state_id=sid,tape_index=j,tape_sha256=th,
                  exogenous_seed=tape.seed,origin_time=tape.origin_time,
                  true_etas=json.dumps(tape.true_etas,sort_keys=True),
                  public_future_updates=json.dumps([asdict(v) for v in tape.future_updates],
                                                    sort_keys=True)))
            # The tape is constructed BEFORE either action branch starts.
            a=apply_tape(snapshot,tape);b=apply_tape(snapshot,tape)
            if a.events!=b.events or a.scenario.true_etas!=b.scenario.true_etas:
                raise AssertionError('Not a common exogenous tape')
            keep,keep_actions=rollout_branch(a,0,continuation,frozen,gamma)
            change,change_actions=rollout_branch(b,edit,continuation,frozen,gamma)
            outcomes.append(change-keep)
            case_audit.append(dict(state_id=sid,tape_index=j,tape_sha256=th,
                    initial_budget=p['source_budget'],scenario=p['scenario'],
                    continuation=continuation,return_keep=keep,return_edit=change,
                    ref_gap=change-keep,keep_steps=len(keep_actions),
                    edit_steps=len(change_actions),
                    changed_physical_trajectory=(keep_actions!=change_actions)))
        ref=mean(outcomes)
        se=(stdev(outcomes)/(len(outcomes)**.5) if len(outcomes)>1 else float('nan'))
        # Small-M Student-t is approximate (2.365 for 8 samples). This is
        # *sampling* uncertainty under this synthetic tape family only.
        tcrit=approximate_t_critical(tapes)
        low=ref-tcrit*se if math.isfinite(se) else float('nan')
        high=ref+tcrit*se if math.isfinite(se) else float('nan')
        label=('beneficial' if low>0 else 'harmful' if high<0 else 'uncertain')
        for name,(qkeep,qedit) in predictions.items():
            gap=qedit-qkeep
            rows.append(dict(state_id=sid,episode_id=p['episode_id'],
                raw_step_index=p['raw_step_index'],case=p['case'],
                scenario=p['scenario'],external_seed=p['external_seed'],
                source_budget=p['source_budget'],
                action_slot=edit,action_descriptor=json.dumps(p['action_descriptor'],sort_keys=True),
                evaluated_model=name,continuation=continuation,
                q_keep=qkeep,q_edit=qedit,predicted_gap=gap,
                reference_gap_mean=ref,reference_gap_se=se,
                reference_gap_ci_low=low,reference_gap_ci_high=high,
                reference_sign=label,gap_error=gap-ref,
                missed_beneficial=int(label=='beneficial' and gap<=0),
                false_positive_edit=int(label=='harmful' and gap>0),
                tapes=tapes,tape_family='synthetic_conditional_future',
                tape_sha256_list=json.dumps(hashes)))
        if i%10==0:print(f'[V13 counterfactual] {i}/{len(probes)}',flush=True)
    return rows,case_audit,tape_records


def original_future_pairs(probes,models,continuation='greedy',frozen=None,gamma=1.):
    """Single immutable ORIGINAL simulator future, without synthetic retaping.

    This is a realized-world sensitivity check, NOT eight independent samples.
    No confidence interval is attached to its one observed value.
    """
    rows=[]
    for probe in probes:
        snapshot=probe['snapshot'];edit=probe['selected_edit']
        left=copy.deepcopy(snapshot);right=copy.deepcopy(snapshot)
        keep,_=rollout_branch(left,0,continuation,frozen,gamma)
        changed,_=rollout_branch(right,edit,continuation,frozen,gamma)
        ref=changed-keep
        for name,model in models.items():
            obs=snapshot._observation()
            qkeep,qedit=predict_pair(model,obs,edit)
            rows.append(dict(state_id=probe['state_id'],source_budget=probe['source_budget'],
                scenario=probe['scenario'],evaluated_model=name,continuation=continuation,
                q_keep=qkeep,q_edit=qedit,predicted_gap=qedit-qkeep,
                realized_scenario_gap=ref,gap_error=qedit-qkeep-ref,
                tape_family='original_frozen_scenario',
                note='One realized synthetic simulator world, not optimal Q or an expectation'))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes',required=True)
    p.add_argument('--checkpoint',nargs='+',required=True,
                   help='Repeat: A0=path A2=path A3=path (V1.2 model checkpoints)')
    p.add_argument('--continuation',choices=('greedy','a0'),default='greedy')
    p.add_argument('--a0-continuation',help='required with --continuation a0')
    p.add_argument('--tapes',type=int,default=8)
    p.add_argument('--start-index',type=int,default=0,
                   help='Inclusive index into immutable probe_bank, for parallel shards')
    p.add_argument('--stop-index',type=int,default=0,
                   help='Exclusive index; 0 means end of bank')
    p.add_argument('--seed',type=int,default=910000)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.tapes<2 or a.gamma<=0 or a.gamma>1:raise ValueError('Need tapes>=2 and 0<gamma<=1')
    if a.continuation=='a0' and not a.a0_continuation:
        p.error('--a0-continuation required with --continuation a0')
    models={};paths=[]
    for item in a.checkpoint:
        name,sep,path=item.partition('=')
        if not sep or name in models or not name:raise ValueError('Duplicate/malformed checkpoint')
        models[name]=load_model(path)
        verify_v12_scientific_model(models[name],
            {'A0':'primitive','A2':'budget_tail','A3':'decision_epoch'}.get(name),a.gamma)
        paths.append(path)
    frozen=load_model(a.a0_continuation) if a.a0_continuation else None
    if frozen is not None:verify_v12_scientific_model(frozen,'primitive',a.gamma)
    if a.a0_continuation:paths.append(a.a0_continuation)
    bank=load_trusted(a.probes)
    stop=a.stop_index or len(bank['probes'])
    if not 0<=a.start_index<stop<=len(bank['probes']):
        raise ValueError('Invalid probe index range')
    selected=bank['probes'][a.start_index:stop]
    rows,audit,tape_records=run_counterfactual(selected,models,a.tapes,a.seed,
                                  a.continuation,frozen,a.gamma,a.uncertainty,a.updates)
    original=original_future_pairs(selected,models,a.continuation,frozen,a.gamma)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'counterfactual_pairs.csv',rows)
    write_csv(out/'tape_pairs.csv',audit)
    write_csv(out/'future_tapes.csv',tape_records)
    write_csv(out/'original_future_pairs.csv',original)
    summary=dict(n_states=len(selected),range_start=a.start_index,range_stop=stop,
                 n_predictions=len(rows),
                 n_independent_tapes=a.tapes,continuation=a.continuation,
                 original_future_rows=len(original),
                 note='Synthetic conditional continuation Q; not optimal Q*, no future data used by policy',
                 probe_sha256=digest(a.probes),model_sha256={str(x):digest(x) for x in paths})
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'manifest.json',provenance(a,inputs=[a.probes,*paths]))
    print('[V13 counterfactual]',summary,flush=True)

if __name__=='__main__':main()
