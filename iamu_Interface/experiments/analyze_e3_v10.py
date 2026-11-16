#!/usr/bin/env python3
"""V1.0-E3: transparent paired summaries, risk tails and exploratory CIs.

Expected input: 1-3 E3B result folders, plus optional E3A output folder.
Hierarchical bootstrap samples training seeds and Case×ETA blocks, retaining
all budgets and disturbance types inside a block. With only 3 training seeds,
intervals are exploratory, not a substitute for held-out Cases 10-14.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_csv(path:Path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path:Path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def upper_cvar(values,alpha=.9):
    if not values:return None
    tail=len(values)*(1-alpha)
    w=tail;s=0.
    for x in sorted(values,reverse=True):
        z=min(1.,w);s+=z*x;w-=z
        if w<=1e-10:break
    return s/tail


def summarize_pairs(values):
    if not values:return {}
    v=list(values)
    return dict(n=len(v),mean_gain=mean(v),
        win=sum(x>1e-8 for x in v),loss=sum(x< -1e-8 for x in v),
        tie=sum(abs(x)<=1e-8 for x in v),
        worst_gain=min(v),best_gain=max(v),
        cvar90_positive_regret=upper_cvar([max(0.,-x) for x in v]))


def hierarchical_ci(paired,*,rounds=3000,seed=2026):
    """Paired data: rows with train_seed, case, eta_seed, gain.

    Resample seeds and (case, eta seed) blocks separately. For each selected
    combination, preserve all budgets×scenarios then average them. If the data
    form a sparse grid, skip unmatched draws rather than imputing false data.
    """
    seeds=sorted(set(int(r['train_seed']) for r in paired))
    blocks=sorted(set((str(r['case']),int(r['eta_seed'])) for r in paired))
    data=defaultdict(list)
    for r in paired:
        data[(int(r['train_seed']),str(r['case']),int(r['eta_seed']))].append(float(r['gain']))
    if not paired:return None
    rng=random.Random(seed)
    bootstrap=[]
    for _ in range(rounds):
        ss=rng.choices(seeds,k=len(seeds))
        bb=rng.choices(blocks,k=len(blocks))
        means=[]
        for s in ss:
            for c,eta in bb:
                sample=data.get((s,c,eta))
                if sample:means.append(mean(sample))
        if means:bootstrap.append(mean(means))
    bootstrap.sort()
    if not bootstrap:return None
    return dict(ci025=bootstrap[int((len(bootstrap)-1)*.025)],
                ci975=bootstrap[int((len(bootstrap)-1)*.975)],
                n_training_seeds=len(seeds),n_case_eta_blocks=len(blocks),
                warning='Exploratory bootstrap; selected validation checkpoint and few training seeds')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fusion',nargs='+',required=True,help='E3B output folders containing runs.csv/manifest.json')
    p.add_argument('--value-calibration',default=None,help='Optional E3A folder')
    p.add_argument('--bootstrap-rounds',type=int,default=3000)
    p.add_argument('--out',default='results/v10_e3_analysis')
    args=p.parse_args()
    all_runs=[]
    for source in args.fusion:
        root=Path(source)
        meta=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        train=int(meta['training_seed'])
        for row in read_csv(root/'runs.csv'):
            item=dict(row)
            for col in ('budget','eta_seed','spent','weighted_wait','elapsed_sec'):
                item[col]=float(item[col])
            item['train_seed']=train
            all_runs.append(item)
    keys=('train_seed','case','scenario','eta_seed','budget')
    groups=defaultdict(dict)
    for r in all_runs:
        k=tuple(r[t] for t in keys)
        if r['policy'] in groups[k]:
            raise ValueError('Duplicate episode/mode: '+str((k,r['policy'])))
        groups[k][r['policy']]=r
    modes=sorted(set(r['policy'] for r in all_runs))
    pairings=[]
    for k,by in groups.items():
        if 'q_only' not in by:
            raise ValueError('Missing V0.9 q_only comparator at '+str(k))
        for mode in modes:
            if mode not in by:continue
            for ref in ('q_only','q_fixed'):
                if ref not in by:continue
                x=by[mode];b=by[ref]
                pairings.append(dict(zip(keys,k),policy=mode,comparator=ref,
                    gain=float(b['weighted_wait'])-float(x['weighted_wait']),
                    spent_delta=float(x['spent'])-float(b['spent']),
                    runtime_delta=float(x['elapsed_sec'])-float(b['elapsed_sec'])))
    summary=[]
    for mode in modes:
        for comparator in ('q_only','q_fixed'):
            for dim in ('ALL','train_seed','budget','scenario'):
                vals=['ALL'] if dim=='ALL' else sorted(set(r[dim] for r in pairings))
                for val in vals:
                    group=[r for r in pairings if r['policy']==mode and r['comparator']==comparator
                           and (dim=='ALL' or r[dim]==val)]
                    if not group:continue
                    result=summarize_pairs(r['gain'] for r in group)
                    bootstrap=(hierarchical_ci(group,rounds=args.bootstrap_rounds)
                               if dim=='ALL' and len(group)>1 else None)
                    summary.append(dict(policy=mode,comparator=comparator,
                        breakdown=dim,value=val,**result,
                        mean_spent_delta=mean(r['spent_delta'] for r in group),
                        mean_runtime_delta=mean(r['runtime_delta'] for r in group),
                        ci_lower=bootstrap['ci025'] if bootstrap else None,
                        ci_upper=bootstrap['ci975'] if bootstrap else None))
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):p.error('Output not empty')
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'fusion_pairwise.csv',pairings)
    write_csv(out/'fusion_summary.csv',summary)
    if args.value_calibration:
        src=Path(args.value_calibration)
        records=read_csv(src/'details.csv')
        if records:
            diag=[]
            for sc in ['ALL',*sorted(set(r['scenario'] for r in records))]:
                rr=[r for r in records if sc=='ALL' or r['scenario']==sc]
                diag.append(dict(scenario=sc,n=len(rr),
                    posterior_mpc_mae=mean(abs(float(x['mc_reference_error'])) for x in rr),
                    model_future_mae=mean(abs(float(x['synthetic_frozen_error'])) for x in rr),
                    h0_future_mae=mean(abs(float(x['h0_synthetic_frozen_error'])) for x in rr),
                    real_future_mae=mean(abs(float(x['realized_frozen_error'])) for x in rr),
                    mean_confidence=mean(float(x['mpc_effective_strength']) for x in rr)))
            write_csv(out/'calibration_summary.csv',diag)
        policies=src/'fixed_state_policy.csv'
        if policies.exists():
            r=read_csv(policies)
            ss=[]
            for mode in sorted(set(x['mode'] for x in r)):
                grp=[x for x in r if x['mode']==mode]
                ss.append(dict(mode=mode,n_states=len(grp),
                    mean_synthetic_oracle_regret=mean(float(x['oracle_regret']) for x in grp),
                    mean_realized_oracle_regret=mean(float(x['realized_regret']) for x in grp),
                    synthetic_oracle_match=mean(float(x['matches_oracle']) for x in grp)))
            write_csv(out/'fixed_state_policy_summary.csv',ss)
    meta=dict(fusion_input=args.fusion,value_calibration=args.value_calibration,
              n_training_seeds=len(set(r['train_seed'] for r in all_runs)),
              n_pair_groups=len(groups),bootstrap_rounds=args.bootstrap_rounds,
              note='Same validation instances. CI is exploratory; do NOT claim independent generalization.')
    (out/'analysis_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[DONE] {len(groups)} paired instances => {out}')


if __name__=='__main__':main()
