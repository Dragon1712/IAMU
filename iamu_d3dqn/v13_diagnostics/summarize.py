#!/usr/bin/env python3
"""Group calibrated Q-gap errors and read-only TD target components.

No inference of causal mechanisms from aggregate averages. Synthetic futures
support P2 diagnosis, not claims about an optimal Q or a calibrated posterior.
"""
from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from .common import atomic_json,provenance,write_csv


def read_rows(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def group_metrics(rows,group_cols):
    buckets=defaultdict(list)
    for r in rows:
        key=tuple(r[k] for k in group_cols)
        buckets[key].append(r)
    out=[]
    for key,group in sorted(buckets.items()):
        def nums(k):return [float(x[k]) for x in group if x.get(k) not in ('','nan',None)
                             and math.isfinite(float(x[k]))]
        errors=nums('gap_error');signs=[x['reference_sign'] for x in group]
        nontrivial=[x for x in group if x['reference_sign']!='uncertain']
        metrics=dict(zip(group_cols,key))
        metrics.update(n=len(group),mean_signed_gap_error=mean(errors) if errors else float('nan'),
            gap_mae=mean(abs(x) for x in errors) if errors else float('nan'),
            n_beneficial=signs.count('beneficial'),n_harmful=signs.count('harmful'),
            n_uncertain=signs.count('uncertain'),
            missed_beneficial=sum(int(x['missed_beneficial']) for x in nontrivial),
            false_positive_edit=sum(int(x['false_positive_edit']) for x in nontrivial),
            beneficial_recall=(1-sum(int(x['missed_beneficial']) for x in group
                              if x['reference_sign']=='beneficial')/signs.count('beneficial'))
                               if signs.count('beneficial') else float('nan'))
        out.append(metrics)
    return out


def summarize_targets(rows):
    group=defaultdict(list)
    for r in rows:
        b=int(r['source_budget']);key=(str(r.get('target_source_file','unknown')),
             ('1' if b==1 else '2+' if b>1 else '0'),
             'Edit' if int(r['action_is_edit']) else 'Keep',
             str(int(r.get('actionable',0))),str(r.get('scenario','unknown')),
             r['macro_length_bin'],str(int(r['done'])))
        group[key].append(r)
    out=[]
    for (src,budget,action,actionable,scenario,length,terminal),items in sorted(group.items()):
        def avg(field):return mean(float(x[field]) for x in items)
        out.append(dict(target_source_file=src,budget_group=budget,
                        action=action,macro_length_bin=length,
                        actionable=actionable,scenario=scenario,
                        done=terminal,n=len(items),mean_reward=avg('interval_reward'),
                        mean_bootstrap=avg('bootstrap'),mean_target=avg('td_target'),
                        mean_q_taken=avg('q_taken'),mean_td_error=avg('td_error'),
                        mean_abs_td_error=mean(abs(float(x['td_error'])) for x in items)))
    return out


def summarize_paired_targets(rows):
    groups=defaultdict(list)
    for r in rows:
        b=int(r['source_budget']);bg='1' if b==1 else '2' if b==2 else '3+'
        groups[(bg,r['scenario'],r['edit_done'])].append(r)
    out=[]
    for (budget,scenario,edit_done),xs in sorted(groups.items()):
        def av(k):return mean(float(x[k]) for x in xs)
        out.append(dict(budget=budget,scenario=scenario,edit_done=edit_done,n=len(xs),
            mean_predicted_gap=av('predicted_q_gap'),
            mean_td_target_gap=av('paired_target_gap'),
            mean_target_gap_error=av('paired_target_gap_error'),
            mean_interval_reward_gap=av('interval_reward_gap'),
            mean_bootstrap_gap=av('bootstrap_gap'),
            mean_keep_bootstrap=av('keep_bootstrap'),
            mean_edit_bootstrap=av('edit_bootstrap'),
            mean_keep_n_raw_steps=av('keep_n'),
            mean_edit_n_raw_steps=av('edit_n')))
    return out


def join_h1_pairs(pairs,paired_targets):
    by=defaultdict(list)
    for r in paired_targets:by[r['state_id']].append(r)
    out=[]
    for r in pairs:
        if r['evaluated_model']!='A3':continue
        matched=by.get(r['state_id'])
        if not matched:continue
        def av(k):return mean(float(x[k]) for x in matched)
        out.append(dict(state_id=r['state_id'],continuation=r['continuation'],
             scenario=r['scenario'],source_budget=r['source_budget'],
             predicted_gap=r['predicted_gap'],
             reference_gap_mean=r['reference_gap_mean'],
             reference_gap_se=r['reference_gap_se'],
             reference_sign=r['reference_sign'],
             q_gap_reference_error=r['gap_error'],
             paired_td_gap=av('paired_target_gap'),
             paired_td_gap_error=av('paired_target_gap_error'),
             mean_keep_bootstrap=av('keep_bootstrap'),
             mean_edit_bootstrap=av('edit_bootstrap'),
             mean_bootstrap_gap=av('bootstrap_gap'),
             mean_interval_reward_gap=av('interval_reward_gap'),
             edit_terminal_fraction=av('edit_done'),
             missed_beneficial=r['missed_beneficial']))
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',nargs='+',required=True,help='One or more counterfactual_pairs.csv')
    p.add_argument('--targets',nargs='*',default=[],help='One or more target_audit.csv')
    p.add_argument('--paired-targets',nargs='*',default=[],
                   help='One or more paired_target_gap.csv on the SAME probe bank')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    pairs=[]
    for f in a.pairs:pairs.extend(read_rows(f))
    targets=[]
    for f in a.targets:
        for row in read_rows(f):
            row['target_source_file']=str(f);targets.append(row)
    paired_targets=[]
    for f in a.paired_targets:paired_targets.extend(read_rows(f))
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    metrics=group_metrics(pairs,['evaluated_model','continuation','source_budget','scenario'])
    write_csv(out/'gap_metrics.csv',metrics)
    write_csv(out/'target_groups.csv',summarize_targets(targets),
              columns=['target_source_file','budget_group','action','actionable','scenario',
                       'macro_length_bin','done','n','mean_reward',
                       'mean_bootstrap','mean_target','mean_q_taken','mean_td_error',
                       'mean_abs_td_error'])
    write_csv(out/'paired_target_groups.csv',summarize_paired_targets(paired_targets))
    h1=join_h1_pairs(pairs,paired_targets)
    write_csv(out/'h1_joined_states.csv',h1)
    inconclusive=sum(x['n_uncertain'] for x in metrics)
    findings=dict(n_prediction_rows=len(pairs),n_target_rows=len(targets),
        n_paired_target_rows=len(paired_targets),n_h1_joined_states=len(h1),
        warning='Cannot causally attribute error to bootstrap/overuse from these tables alone',
        interpretations=dict(
            H1='Compare same-state budget-one Keep/Edit Q gaps to frozen-policy reference;'
               ' use target_groups only to locate components, not to infer causality',
            H2='Unresolved until fixed-data replay-reuse ablation P4',
            H3='Deferred; do not introduce additional state features in P0-P3'),
        n_uncertain_group_counts=inconclusive)
    atomic_json(out/'mechanism_report.json',findings)
    atomic_json(out/'manifest.json',provenance(a,inputs=a.pairs+a.targets+a.paired_targets))
    print('[V13 summary]',findings,flush=True)

if __name__=='__main__':main()
