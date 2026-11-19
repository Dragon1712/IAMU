#!/usr/bin/env python3
"""Analyze matched V1.2 validation: mean waiting, edits, paired strong baselines.

Usage: python -m v12_protocol.analyze --runs A0_42=.../runs.csv A1_42=... \
    A2_42=... A3_42=... --out results/v12_summary
This is descriptive; correlated case/seed/budget groups are not independent.
"""
from __future__ import annotations
import argparse,csv,math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from iamu_dynamic.risk_selection import upper_cvar
from iamu_dynamic.rl_experiment_v11 import write_csv


def read(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='MODEL_SEED=.../runs.csv')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    data={}
    for s in a.runs:
        if '=' not in s: p.error('Each --runs item must have NAME=PATH')
        label,path=s.split('=',1)
        if label in data: p.error('Duplicate name '+label)
        rows=read(path)
        if not rows:p.error('No rows in '+path)
        data[label]=rows
    pooled=[]
    for label,rows in data.items():
        groups=defaultdict(dict)
        for r in rows:
            k=(Path(r['case']).stem,r['scenario'],r['seed'],r['budget'])
            if r['method'] in groups[k]:raise ValueError('Duplicate method in same eval cell')
            groups[k][r['method']]=r
        for key,rr in groups.items():
            if set(rr)!= {'keep','greedy','periodic','ddqn'}:
                raise ValueError(f'Incomplete matched group in {label}: {key}')
            for mode,r in rr.items():
                pooled.append(dict(run=label,case=key[0],scenario=key[1],eta_seed=key[2],
                    budget=key[3],method=mode,wait=float(r['wait']),spent=int(r['spent']),
                    gain_vs_periodic=float(rr['periodic']['wait'])-float(r['wait']),
                    gain_vs_greedy=float(rr['greedy']['wait'])-float(r['wait']),
                    positive_regret_vs_periodic=max(0.,float(r['wait'])-float(rr['periodic']['wait'])),
                    positive_regret_vs_greedy=max(0.,float(r['wait'])-float(rr['greedy']['wait']))))
    summary=[]
    for label in data:
        for b in ('ALL','2','5','10'):
            for sc in ('ALL','shock','repeated','small'):
                xx=[r for r in pooled if r['run']==label and r['method']=='ddqn' and
                    (b=='ALL' or r['budget']==b) and (sc=='ALL' or r['scenario']==sc)]
                if not xx:continue
                summary.append(dict(run=label,budget=b,scenario=sc,n=len(xx),
                    mean_wait=mean(r['wait'] for r in xx),mean_spent=mean(r['spent'] for r in xx),
                    mean_gain_vs_periodic=mean(r['gain_vs_periodic'] for r in xx),
                    mean_gain_vs_greedy=mean(r['gain_vs_greedy'] for r in xx),
                    cvar90_regret_vs_periodic=upper_cvar([r['positive_regret_vs_periodic'] for r in xx],.9),
                    cvar90_regret_vs_greedy=upper_cvar([r['positive_regret_vs_greedy'] for r in xx],.9),
                    max_wait=max(r['wait'] for r in xx)))
    # Pair learned methods directly with A0 under matching training-seed +
    # case/scenario/ETA seed/budget, rather than comparing aggregate means.
    ddqn=[r for r in pooled if r['method']=='ddqn']
    refs={}
    for r in ddqn:
        name=r['run']
        if name.startswith('A0_'):
            refs[(name.split('_',1)[1],r['case'],r['scenario'],r['eta_seed'],r['budget'])]=r
    pair_rows=[]
    for r in ddqn:
        if not any(r['run'].startswith(x+'_') for x in ('A1','A2','A3')):continue
        seed=r['run'].split('_',1)[1]
        key=(seed,r['case'],r['scenario'],r['eta_seed'],r['budget'])
        if key not in refs:continue
        base=refs[key]
        pair_rows.append(dict(run=r['run'],training_seed=seed,
            case=r['case'],scenario=r['scenario'],eta_seed=r['eta_seed'],budget=r['budget'],
            wait_improvement_vs_A0=base['wait']-r['wait'],
            spent_delta_vs_A0=r['spent']-base['spent'],
            positive_harm_vs_A0=max(0.,r['wait']-base['wait'])))
    pair_summary=[]
    for label in sorted({r['run'] for r in pair_rows}):
        for b in ('ALL','2','5','10'):
            z=[r for r in pair_rows if r['run']==label and (b=='ALL' or r['budget']==b)]
            if not z:continue
            pair_summary.append(dict(run=label,budget=b,n=len(z),
                mean_gain_vs_A0=mean(x['wait_improvement_vs_A0'] for x in z),
                mean_spent_delta=mean(x['spent_delta_vs_A0'] for x in z),
                wins=sum(x['wait_improvement_vs_A0']>1e-8 for x in z),
                harms=sum(x['wait_improvement_vs_A0']<-1e-8 for x in z),
                cvar90_harm_vs_A0=upper_cvar([x['positive_harm_vs_A0'] for x in z],.9)))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'matched_runs.csv',pooled);write_csv(out/'summary.csv',summary)
    write_csv(out/'a0_pairwise.csv',pair_rows)
    write_csv(out/'a0_pairwise_summary.csv',pair_summary)
    print(f'[V12 analyze] {len(summary)} baseline summary groups; {len(pair_rows)} paired A1-A3/A0 cells => {out}')

if __name__=='__main__':main()
