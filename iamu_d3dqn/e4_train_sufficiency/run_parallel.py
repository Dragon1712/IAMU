#!/usr/bin/env python3
"""CPU-parallel E4 Keep/Q-gap diagnostic; one independent shard per scenario.

The 4 GPUs were needed for E4 training. This is frozen Q inference + physical
CPU replay; 8 CPU workers are a practical starting point. Never merge failed
or scientifically mismatched shards. Snapshots MUST be present on the server.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import product
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from diagnose_keep_qgap import ints, write_csv, group_summaries


def read_csv(p):
    if not p.exists():return []
    with p.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',required=True,
                   help='Repeat run-folder@episode')
    p.add_argument('--anchor-model',default=None)
    p.add_argument('--cases',nargs='+',default=['../2024LiERLDDE/case_code/8.txt',
                                                 '../2024LiERLDDE/case_code/9.txt'])
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--eta-seeds',default='400001:400003')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--anchors',default='early,keep')
    p.add_argument('--continuations',default='anchor,greedy')
    p.add_argument('--states-per-group',type=int,default=2)
    p.add_argument('--max-edit-actions',type=int,default=4)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if args.workers<1 or args.states_per_group<1 or args.max_edit_actions<1:
        p.error('Workers/states/action cap must be positive')
    cases=[(ROOT/path).resolve() for path in args.cases]
    for case in cases:
        if not case.exists():p.error('Missing case file: '+str(case))
    scenarios=args.scenarios.split(',')
    seeds=ints(args.eta_seeds)
    budgets=ints(args.budgets)
    jobs=[]
    out=(ROOT/args.out).resolve()
    for case,sc,seed,b in product(cases,scenarios,seeds,budgets):
        label=f'case{case.stem}__{sc}__seed{seed}__B{b}'
        if any(label==x[0] for x in jobs):p.error('Duplicate shard: '+label)
        destination=out/'shards'/label
        cmd=[sys.executable,'-u',str(HERE/'diagnose_keep_qgap.py')]
        for model in args.model:
            cmd.extend(['--model',model])
        if args.anchor_model:
            cmd.extend(['--anchor-model',args.anchor_model])
        cmd.extend(['--cases',str(case),'--scenarios',sc,
                    '--eta-seeds',str(seed),'--budgets',str(b),
                    '--anchors',args.anchors,
                    '--continuations',args.continuations,
                    '--states-per-group',str(args.states_per_group),
                    '--max-edit-actions',str(args.max_edit_actions),
                    '--device',args.device,'--out',str(destination)])
        jobs.append((label,destination,cmd))
    if args.dry_run:
        print(json.dumps({'n_shards':len(jobs),'workers':args.workers,
              'examples':[j[2] for j in jobs[:2]]},ensure_ascii=False,indent=2))
        return
    plan=out/'plan.json'
    meta=dict(vars(args))
    ignore={'resume','workers','dry_run'}
    if out.exists() and any(out.iterdir()):
        if not args.resume or not plan.exists():
            p.error('Nonempty output directory without compatible --resume plan')
        earlier=json.loads(plan.read_text())['arguments']
        if {k:v for k,v in earlier.items() if k not in ignore} != \
           {k:v for k,v in meta.items() if k not in ignore}:
            p.error('--resume parameters differ; choose a NEW output path')
    out.mkdir(parents=True,exist_ok=True)
    plan.write_text(json.dumps({'arguments':meta,'n_shards':len(jobs),
          'shards':[{'label':x[0],'out':str(x[1])} for x in jobs]},
          ensure_ascii=False,indent=2))

    def worker(job):
        label,dest,cmd=job
        if args.resume and (dest/'summary.json').is_file():
            result=json.loads((dest/'summary.json').read_text())
            if result.get('all_parity_pass'):
                return label,0,'previously completed'
        if dest.exists() and any(dest.iterdir()):
            return label,99,'incomplete existing shard, inspect before retry'
        log=out/'logs'/f'{label}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with log.open('w',encoding='utf-8') as stream:
            stream.write(' '.join(cmd)+'\n');stream.flush()
            result=subprocess.run(cmd,cwd=ROOT,env=env,
                                  stdout=stream,stderr=subprocess.STDOUT)
        return label,result.returncode,str(log)

    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(worker,j) for j in jobs]
        for i,future in enumerate(as_completed(futures),1):
            label,code,log=future.result()
            print(f'[{i}/{len(jobs)}] {"OK" if code==0 else "FAIL"} {label} ({log})',flush=True)
            if code:failures.append(dict(label=label,exit_code=code,log=log))
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} shards failed: NO merged result was generated')
    source_files=('trajectory_episodes.csv','trajectory_events.csv','state_bank.csv',
                  'q_values.csv','counterfactual.csv','calibration_pairs.csv',
                  'decision_regret.csv','coverage.csv','validation_parity.csv',
                  'matched_q_shift.csv')
    combined={}
    for filename in source_files:
        collected=[]
        for label,dest,_ in jobs:
            collected.extend(read_csv(dest/filename))
        combined[filename]=collected
        write_csv(out/'merged'/filename,collected)
    ep=combined['trajectory_episodes.csv']
    pairs=combined['calibration_pairs.csv']
    decisions=combined['decision_regret.csv']
    # CSV fields read from disk are strings. Convert only numeric data used
    # downstream, leaving scenario/label/anchor as stable strings.
    for row in ep:
        for name in ['weighted_wait','spent','budget','actionable_events',
                     'keep_with_legal_edit','keep_due_to_positive_margin',
                     'final_screened_proposals']:
            row[name]=float(row[name])
    for row in pairs:
        for name in ['q_gap','true_gain_scaled','keep_margin']:
            row[name]=float(row[name])
    filtered=[]
    for row in decisions:
        if row['action_labeled']!='True':continue
        for name in ['first_action_regret','remaining_budget','n_legal_edits',
                     'selected_action']:
            row[name]=float(row[name])
        for name in ['missed_beneficial_keep','harmful_edit']:
            row[name]=(row[name]=='True')
        filtered.append(row)
    reports=group_summaries(ep,combined['trajectory_events.csv'],pairs,filtered)
    for filename,report in zip(('trajectory_summary.csv','calibration_summary.csv',
                                'decision_summary.csv'),reports):
        write_csv(out/'merged'/filename,report)
    metrics=dict(n_shards=len(jobs),n_episodes=len(ep),n_state_rows=len(combined['state_bank.csv']),
                 n_labels=len(pairs),n_matched_q_rows=len(combined['matched_q_shift.csv']),
                 n_parity=len(combined['validation_parity.csv']))
    (out/'merge_report.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
    print('[DONE] '+json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
