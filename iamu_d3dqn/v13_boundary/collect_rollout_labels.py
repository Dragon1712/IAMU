"""Collect training-only, one-step rollout-improvement labels for all legal edits.

Freeze the A0 continuation. Keep is simulated once per state/tape and cached
for comparison with every legal Edit. All future ETA tapes are identical
within the matched pair. These are not Q* labels.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev
import numpy as np
from v13_diagnostics.common import (load_trusted,load_model,verify_v12_scientific_model,
    action_descriptor,write_csv,atomic_json,digest)
from v13_diagnostics.future_tapes import draw_tape,apply_tape,tape_hash
from v13_diagnostics.counterfactual import rollout_branch,approximate_t_critical
from .utils import boundary_provenance


def generate(probes,frozen,tapes=8,seed=940000,updates=2,uncertainty=8.):
    labels=[];details=[];costs=[]
    for p in probes:
        s=p['snapshot'];sid=p['state_id'];obs=s._observation()
        edits=(np.flatnonzero(np.asarray(obs.mask,dtype=bool)[1:])+1).tolist()
        if not edits:raise ValueError('Non-actionable label-bank state')
        gaps=defaultdict(list);edit_steps=defaultdict(int);keep_steps=0
        for j in range(tapes):
            tape=draw_tape(s,sid,j,base_seed=seed,uncertainty=uncertainty,
                           updates_per_vessel=updates)
            env=apply_tape(s,tape)
            keep,keep_actions=rollout_branch(env,0,'a0',frozen,1.)
            keep_steps+=len(keep_actions)
            for edit in edits:
                e=apply_tape(s,tape)
                result,edit_actions=rollout_branch(e,int(edit),'a0',frozen,1.)
                gap=result-keep;gaps[edit].append(gap);edit_steps[edit]+=len(edit_actions)
                details.append(dict(state_id=sid,tape_index=j,action_slot=edit,
                    tape_sha256=tape_hash(tape),keep_return=keep,edit_return=result,
                    paired_gap=gap,keep_raw_steps=len(keep_actions),edit_raw_steps=len(edit_actions)))
        for edit in edits:
            vals=gaps[edit];mu=mean(vals)
            se=stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else float('nan')
            t=approximate_t_critical(len(vals)) if len(vals)>1 else float('nan')
            low=mu-t*se;high=mu+t*se
            sign=('beneficial' if low>0 else 'harmful' if high<0 else
                  'equivalent' if all(abs(x)<1e-9 for x in vals) else 'uncertain')
            labels.append(dict(state_id=sid,episode_id=p['episode_id'],
               external_seed=p['external_seed'],case=p['case'],scenario=p['scenario'],
               source_budget=p['source_budget'],action_slot=edit,
               action_descriptor=json.dumps(action_descriptor(s,edit),sort_keys=True),
               paired_return_gap=mu,paired_gap_se=se,ci_lo=low,ci_hi=high,
               label_kind=sign,tapes=len(vals),
               allocated_sim_steps=keep_steps/len(edits)+edit_steps[edit],
               keep_steps_share=keep_steps/len(edits),edit_steps=edit_steps[edit]))
        costs.append(dict(state_id=sid,episode_id=p['episode_id'],scenario=p['scenario'],
           budget=p['source_budget'],n_edits=len(edits),tapes=tapes,
           keep_raw_steps=keep_steps,edit_raw_steps=sum(edit_steps.values()),
           total_raw_steps=keep_steps+sum(edit_steps.values())))
    return labels,details,costs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-probes',required=True)
    p.add_argument('--diagnostic-probes',required=True)
    p.add_argument('--frozen-a0',required=True)
    p.add_argument('--tapes',type=int,default=8)
    p.add_argument('--seed',type=int,default=940000)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.tapes<2:p.error('Need at least two tapes')
    train=load_trusted(a.training_probes);diag=load_trusted(a.diagnostic_probes)
    if not train.get('probes') or not diag.get('probes'):raise ValueError('Missing probes')
    train_sources={(x['external_seed'],x['case']) for x in train['probes']}
    diag_sources={(x['external_seed'],x['case']) for x in diag['probes']}
    if train_sources&diag_sources or ({x['state_id'] for x in train['probes']} &
                                     {x['state_id'] for x in diag['probes']}):
        raise ValueError('Training label/probe overlap')
    frozen=load_model(a.frozen_a0);verify_v12_scientific_model(frozen,'primitive')
    labels,details,costs=generate(train['probes'],frozen,a.tapes,a.seed,a.updates,a.uncertainty)
    dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'rollout_labels.csv',labels)
    write_csv(dest/'rollout_tape_pairs.csv',details)
    write_csv(dest/'label_cost.csv',costs)
    atomic_json(dest/'summary.json',dict(n_states=len(costs),n_action_labels=len(labels),
       physical_env_steps=sum(x['total_raw_steps'] for x in costs),
       frozen_checkpoint_sha256=digest(a.frozen_a0),
       label='Return(Edit)-Return(Keep) under frozen A0 continuation; not Q*'))
    atomic_json(dest/'manifest.json',boundary_provenance(a,inputs=[a.training_probes,
                      a.diagnostic_probes,a.frozen_a0]))
    print('[labels]',len(labels),'action comparisons on',len(costs),'states')
if __name__=='__main__':main()
