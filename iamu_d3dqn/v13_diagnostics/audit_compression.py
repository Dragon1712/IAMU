#!/usr/bin/env python3
"""Check event ledger, legal masks, semantic anchor mapping, and toy SMDP DP."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
from .common import atomic_json,digest,load_trusted,provenance,save_trusted,state_id,write_csv


def exactly_solved_toy(gamma):
    """Small deterministic budget-one MDP; only decision-state anchors retained.

    States: D0 (decision, budget=1), F1 (forced Keep), D2 (decision), terminal.
    At D0: Keep incurs -1 then forced -2, reaches D2; Edit incurs -5,
    consumes budget, then forced costs -1,-4 before terminal.
    At D2 with budget one: Keep -8, Edit -3. Episode finite and acyclic.
    """
    v_d2=max(-8.,-3.)
    q_keep=-1.+gamma*(-2.+gamma*v_d2)
    q_edit=-5.+gamma*(-1.+gamma*(-4.))
    raw_gap=q_edit-q_keep
    compressed_keep=(-1.-2.*gamma)+(gamma**2)*v_d2
    compressed_edit=(-5.-1.*gamma-4.*gamma**2)
    compressed_gap=compressed_edit-compressed_keep
    assert math.isclose(raw_gap,compressed_gap,rel_tol=1e-13,abs_tol=1e-13)
    return dict(gamma=gamma,q_keep=q_keep,q_edit=q_edit,q_gap=raw_gap,
                compressed_gap=compressed_gap)


def audit(data_path,out,gamma_values=(1.,.9)):
    from v12_protocol.compress import compress_episode,anchor_indices,validate_raw_episode
    data=load_trusted(data_path);records=data['records']
    audit_rows=[]; anchor_rows=[]
    derived={m:[] for m in ('primitive','budget_tail','decision_epoch')}
    for r in records:
        raw=list(r['raw']);validate_raw_episode(raw)
        if len(r['descriptors'])!=len(raw):raise ValueError('Missing action semantic descriptor')
        eid=r['job']['episode_id']
        for gamma in gamma_values:
            representations={mode:compress_episode(raw,mode,gamma)
                             for mode in ('primitive','budget_tail','decision_epoch')}
            if gamma==1.:
                for mode,result in representations.items():
                    derived[mode].append(dict(episode_id=eid,
                        job=r['job'],behavior=r['behavior'],
                        transitions=result.transitions,prefix=result.prefix,
                        compression_audit=result.audit))
            trueG=sum((gamma**i)*s.transition.reward for i,s in enumerate(raw))
            for mode,res in representations.items():
                recon=res.audit['reconstructed_discounted_return']
                if not math.isclose(trueG,recon,rel_tol=1e-9,abs_tol=1e-8):
                    raise AssertionError('Reconstructed discounted return differs')
                if any(t.done and t.bootstrap_discount!=0 for t in res.transitions):
                    raise AssertionError('Terminal macro bootstraps')
                if any(not t.done and not math.isclose(t.bootstrap_discount,gamma**t.n_raw_steps,
                                                    rel_tol=1e-10,abs_tol=1e-10)
                       for t in res.transitions):
                    raise AssertionError('Non-terminal discount mismatch')
                if res.prefix.raw_steps+sum(t.n_raw_steps for t in res.transitions)!=len(raw):
                    raise AssertionError('Event coverage error')
                audit_rows.append(dict(episode_id=eid,behavior=r['behavior'],
                    case=r['job']['case'],scenario=r['job']['scenario'],budget=r['job']['budget'],
                    representation=mode,gamma=gamma,raw_steps=len(raw),
                    macro_steps=len(res.transitions),prefix_steps=res.prefix.raw_steps,
                    raw_discounted_return=trueG,reconstructed_discounted_return=recon,
                    absolute_error=abs(trueG-recon),mean_macro_length=(sum(t.n_raw_steps
                    for t in res.transitions)/len(res.transitions) if res.transitions else 0.),
                    terminal_segments=sum(t.done for t in res.transitions)))
            if gamma==gamma_values[0]:
                # Physical action descriptors must agree at every shared decision anchor.
                anchors={m: {t.raw_start_index:t for t in z.transitions}
                         for m,z in representations.items()}
                for idx,tr in anchors['decision_epoch'].items():
                    original=raw[idx].transition
                    if not np.array_equal(tr.obs_mask,original.obs_mask):
                        raise AssertionError('Action mask differs for shared anchor')
                    if not np.array_equal(tr.obs_global,original.obs_global) or not np.array_equal(tr.obs_actions,original.obs_actions):
                        raise AssertionError('Source observation differs')
                    if tr.action!=original.action:raise AssertionError('Source action differs')
                    semantic=r['descriptors'][idx]
                    if (semantic['kind']=='keep')!=(tr.action==0):
                        raise AssertionError('Action descriptor kind mismatches')
                    if idx in anchors['budget_tail']:
                        tail=anchors['budget_tail'][idx]
                        if tail.action!=tr.action:raise AssertionError('Cross-representation action differs')
                    anchor_rows.append(dict(state_id=state_id(eid,idx),episode_id=eid,
                        raw_step_index=idx,scenario=r['job']['scenario'],
                        source_budget=tr.source_budget,action=tr.action,
                        vessel_id=semantic.get('vessel_id'),
                        new_assignment=semantic.get('new_assignment'),
                        decision_epoch_n_raw_steps=tr.n_raw_steps))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'compression_episodes.csv',audit_rows)
    write_csv(out/'shared_anchors.csv',anchor_rows)
    derived_sha={}
    for mode,episodes in derived.items():
        f=out/f'{mode}.pt'
        save_trusted(f,dict(protocol='v13-P0-derived',representation=mode,
                           gamma=1.,source_sha256=digest(data_path),episodes=episodes))
        derived_sha[mode]=digest(f)
    dps=[exactly_solved_toy(g) for g in gamma_values]
    result=dict(status='passed',episodes=len(records),checked_ledger_rows=len(audit_rows),
                checked_anchor_rows=len(anchor_rows),max_ledger_error=max((r['absolute_error']
                    for r in audit_rows),default=0),exact_dp=dps,
                derived_data_sha256=derived_sha,
                per_mode={m:dict(mean_transition_count=sum(row['macro_steps'] for row in audit_rows
                              if row['representation']==m and row['gamma']==gamma_values[0])/len(records),
                              episode_count=len(records)) for m in ('primitive','budget_tail','decision_epoch')},
                input_sha256=digest(data_path))
    atomic_json(out/'compression_audit.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    res=audit(a.data,a.out)
    atomic_json(Path(a.out)/'manifest.json',provenance(a,inputs=[a.data]))
    print('[V13 audit]',res['status'],res['episodes'],'episodes',flush=True)

if __name__=='__main__':main()
