#!/usr/bin/env python3
"""Isolated, resumable CPU parallel runner for E3A and E3B.

Each shard receives ONE case/scenario/ETA seed/budget and ONE training seed;
no shard can overwrite old E2 results. Empty E3A strata are recorded, not
considered crashes. Merge only after ALL shards exit successfully.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def ints(s):
    z=[]
    for v in s.split(','):
        if ':' in v:
            a,b=map(int,v.split(':'));z.extend(range(a,b+1))
        else:z.append(int(v))
    if len(z)!=len(set(z)):raise ValueError('Duplicate integer in grid')
    return z


def read_csv(p):
    if not Path(p).exists():return []
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


@dataclass(frozen=True)
class Job:
    workflow:str
    seed:int
    case:Path
    scenario:str
    eta_seed:int
    budget:int
    out:Path
    cmd:tuple[str,...]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workflow',choices=('fusion','calibration','both'),default='fusion')
    p.add_argument('--training-seeds',default='42,43,44')
    p.add_argument('--cases',nargs='+',default=['assets/cases/8.txt','assets/cases/9.txt'])
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--eta-seeds',default='400001:400003')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--mpc-samples',type=int,default=8)
    p.add_argument('--risk-samples',type=int,default=16)
    p.add_argument('--reference-samples',type=int,default=64)
    p.add_argument('--future-worlds',type=int,default=4)
    p.add_argument('--states-per-group',type=int,default=1)
    p.add_argument('--min-candidates',type=int,default=1)
    p.add_argument('--stages',type=int,default=2)
    p.add_argument('--beta-budget',type=float,default=.25)
    p.add_argument('--beta-risk',type=float,default=.25)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Only first case, scenario, ETA seed and budget, train seed42')
    p.add_argument('--out',default='results/v10_e3_parallel')
    args=p.parse_args()
    if args.workers<1 or args.mpc_samples<2 or args.mpc_samples>args.risk_samples:
        p.error('Invalid workers or MPC/risk scenario count')
    if args.future_worlds<1 or args.states_per_group<1 or args.reference_samples<2:
        p.error('Invalid calibration sample settings')
    case_paths=[(ROOT/x).resolve() for x in args.cases]
    for case in case_paths:
        if not case.is_file():p.error('Missing case: '+str(case))
    train_seeds=ints(args.training_seeds)
    scenarios=args.scenarios.split(',');etas=ints(args.eta_seeds);budgets=ints(args.budgets)
    if args.smoke:
        train_seeds=[42];case_paths=case_paths[:1]
        scenarios=scenarios[:1];etas=etas[:1];budgets=budgets[:1]
    out=(ROOT/args.out).resolve()
    workflows=['fusion','calibration'] if args.workflow=='both' else [args.workflow]
    jobs=[]
    for train_seed in train_seeds:
        source=ROOT.parent/'iamu_d3dqn/results'/f'v09b_set_ddqn_seed{train_seed}_300'
        bundled=ROOT/'assets/checkpoints'/f'set_ddqn_seed{train_seed}'
        cp=(bundled if (bundled/'best.pt').is_file() else source)
        if not (cp/'best.pt').is_file() or not (cp/'manifest.json').is_file():
            p.error('Missing checkpoint or manifest for train seed '+str(train_seed))
        for case in case_paths:
            for sc in scenarios:
                for eta in etas:
                    for b in budgets:
                        for workflow in workflows:
                            tag=f'{case.stem}__{sc}__eta{eta}__B{b}'
                            jobdir=out/'shards'/workflow/f'train{train_seed}'/tag
                            script='eval_e3_fusion_v10.py' if workflow=='fusion' else 'eval_e3_value_calibration_v10.py'
                            cmd=[sys.executable,'-u',str(ROOT/'experiments'/script),
                                 '--checkpoint',str(cp/'best.pt'),
                                 '--manifest',str(cp/'manifest.json'),
                                 '--cases',str(case),'--scenarios',sc,
                                 '--seeds',str(eta),'--budgets',str(b),
                                 '--risk-samples',str(args.risk_samples),
                                 '--mpc-samples',str(args.mpc_samples),
                                 '--beta-risk',str(args.beta_risk),
                                 '--out',str(jobdir)]
                            if workflow=='fusion':
                                cmd+=['--lookahead-stages',str(args.stages),
                                      '--beta-budget',str(args.beta_budget),
                                      '--verify-baseline']
                                if args.audit:cmd.append('--audit')
                            else:
                                cmd+=['--stages',str(args.stages),
                                      '--reference-samples',str(args.reference_samples),
                                      '--future-worlds',str(args.future_worlds),
                                      '--states-per-group',str(args.states_per_group),
                                      '--min-candidates',str(args.min_candidates),
                                      '--selection','near_margin']
                            jobs.append(Job(workflow,train_seed,case,sc,eta,b,jobdir,tuple(cmd)))
    if out.exists() and any(out.iterdir()) and not args.resume:
        p.error('Output nonempty: choose a fresh --out, or --resume')
    plan_file=out/'plan.json'
    if args.resume and plan_file.is_file():
        previous=json.loads(plan_file.read_text(encoding='utf-8'))['args']
        # Changing the science while reusing shards silently invalidates a
        # merged result. Worker count and resume are purely operational.
        ignored={'resume','workers'}
        old={k:v for k,v in previous.items() if k not in ignored}
        new={k:v for k,v in vars(args).items() if k not in ignored}
        if old!=new:
            p.error('--resume plan mismatch: use a NEW --out for modified settings')
    out.mkdir(parents=True,exist_ok=True)
    (out/'plan.json').write_text(json.dumps({'args':vars(args),'jobs':[
       {'workflow':j.workflow,'seed':j.seed,'case':str(j.case),'scenario':j.scenario,
        'eta_seed':j.eta_seed,'budget':j.budget,'output':str(j.out)} for j in jobs]},
       indent=2,ensure_ascii=False))
    def completed(j):
        return ((j.out/'runs.csv').is_file() if j.workflow=='fusion'
                else (j.out/'details.csv').is_file() or (j.out/'empty_group.json').is_file())
    def run(j):
        if args.resume and completed(j):return (j,'cached',0)
        if j.out.exists() and any(j.out.iterdir()):
            return (j,'incomplete shard; choose a new --out or manually inspect it',1)
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1')
        logfile=out/'logs'/j.workflow/f'train{j.seed}'/f'{j.out.name}.log'
        logfile.parent.mkdir(parents=True,exist_ok=True)
        with logfile.open('w',encoding='utf-8') as f:
            f.write(' '.join(j.cmd)+'\n');f.flush()
            x=subprocess.run(j.cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return (j,str(logfile),x.returncode)
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            j,log,rc=f.result()
            print(f'[{i}/{len(jobs)}] {j.workflow} train{j.seed} {j.out.name}: '
                  f'{"OK" if rc==0 else "FAILED"} ({log})',flush=True)
            if rc:failures.append(dict(workflow=j.workflow,train_seed=j.seed,
                         group=j.out.name,log=log,exit_code=rc))
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures:
        raise SystemExit(f'FAIL {len(failures)} shards; DO NOT interpret merged partial results')
    # Merge ONLY successfully completed shards. Empty E3A strata remain
    # documented by coverage and empty_group.json; no false pseudo-samples.
    merges=[]
    for workflow in workflows:
        for seed in train_seeds:
            shards=[j for j in jobs if j.workflow==workflow and j.seed==seed]
            merged=out/'merged'/workflow/f'train{seed}'
            targets=(('runs.csv',) if workflow=='fusion'
                     else ('details.csv','fixed_state_policy.csv','coverage.csv'))
            for filename in targets:
                all_rows=[]
                for j in shards:all_rows.extend(read_csv(j.out/filename))
                write_csv(merged/filename,all_rows)
                merges.append(dict(workflow=workflow,seed=seed,file=filename,rows=len(all_rows)))
            if workflow=='fusion':
                valid=next((j for j in shards if (j.out/'manifest.json').is_file()),None)
                if valid:
                    meta=json.loads((valid.out/'manifest.json').read_text(encoding='utf-8'))
                    meta['parallel_merged_from']=[str(j.out) for j in shards]
                    merged.mkdir(parents=True,exist_ok=True)
                    (merged/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    (out/'merge_report.json').write_text(json.dumps(merges,indent=2))
    print(f'[DONE] {len(jobs)} jobs; merged outputs => {out}/merged')


if __name__=='__main__':main()
