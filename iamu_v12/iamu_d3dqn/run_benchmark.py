#!/usr/bin/env python3
"""Paired, multiple-seed evaluation for the no-retraining IAMU prototype.

Identical scenarios and hidden actual arrivals are shared across all policies,
repair modes and budgets.  Results are not paper-level evidence until tested
on independent held-out cases with a validated shore-power model.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from iamu_dynamic import demo_case, generate_scenario, read_case, simulate


def parse_list(s: str):
    return [t.strip() for t in s.split(',') if t.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='*',default=[],help='Space-separated original ERL-DDE TSV paths')
    p.add_argument('--demo',action='store_true')
    p.add_argument('--seeds',default='11,22,33')
    p.add_argument('--scenarios',default='small,shock,repeated')
    p.add_argument('--policies',default='no_replan,budget_aware,threshold,periodic')
    p.add_argument('--repair-methods',default='single,alns2')
    p.add_argument('--budgets',default='0,2,5')
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--uncertainty',type=float,default=8)
    p.add_argument('--threshold',type=float,default=3)
    p.add_argument('--period',type=float,default=12)
    p.add_argument('--max-edits-per-event',type=int,default=2)
    p.add_argument('--revision-penalty',type=float,default=0)
    p.add_argument('--out',default='results/benchmark_v02')
    args=p.parse_args()
    if not args.cases and not args.demo:p.error('Provide --cases or --demo')
    setups=[('demo',*demo_case())] if args.demo else [(x,*read_case(x)) for x in args.cases]
    seeds=[int(x) for x in parse_list(args.seeds)]
    policies=parse_list(args.policies)
    repairs=parse_list(args.repair_methods)
    budgets=[None if x.lower()=='unlimited' else int(x) for x in parse_list(args.budgets)]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,vessels,port in setups:
        for scenario_name in parse_list(args.scenarios):
            for seed in seeds:
                scenario=generate_scenario(vessels,port,scenario_name,seed,args.updates,args.uncertainty)
                for policy in policies:
                    for budget in budgets:
                        for repair in repairs:
                            start=perf_counter()
                            result=simulate(scenario,policy,budget,args.threshold,args.period,
                                            args.max_edits_per_event,repair_method=repair,
                                            revision_penalty=args.revision_penalty)
                            m=result['metrics']
                            rows.append(dict(case=name,scenario=scenario_name,seed=seed,
                                             policy=policy,budget=('unlimited' if budget is None else budget),
                                             repair_method=repair,spent=m['spent'],
                                             weighted_wait=m['weighted_wait'],
                                             makespan=m['makespan'],
                                             arrival_lateness_slip=m['arrival_lateness_slip'],
                                             resource_conflict_slip=m['resource_conflict_slip'],
                                             candidate_evaluations=m['candidate_evaluations'],
                                             runtime_seconds=round(perf_counter()-start,5)))
                print(f'case={name} scenario={scenario_name} seed={seed}: {len(rows)} evaluations done',flush=True)
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    base={(r['case'],r['scenario'],r['seed']):r['weighted_wait']
          for r in rows if r['policy']=='no_replan'}
    paired=[]
    for r in rows:
        key=(r['case'],r['scenario'],r['seed'])
        b=base.get(key)
        if b is not None:
            paired.append({**r,'wait_reduction_vs_no_replan':round(b-r['weighted_wait'],6)})
    if paired:
        with (out/'paired_runs.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=paired[0].keys());writer.writeheader();writer.writerows(paired)
        grouped={}
        for row in paired:
            key=(row['policy'],str(row['budget']),row['repair_method'])
            grouped.setdefault(key,[]).append(row)
        summary=[]
        for (policy,budget,method),recs in sorted(grouped.items()):
            summary.append(dict(policy=policy,budget=budget,repair_method=method,
                                n=len(recs),mean_wait=round(mean(r['weighted_wait'] for r in recs),5),
                                mean_wait_reduction=round(mean(r['wait_reduction_vs_no_replan'] for r in recs),5),
                                mean_spent=round(mean(r['spent'] for r in recs),3),
                                mean_runtime_seconds=round(mean(r['runtime_seconds'] for r in recs),5)))
        with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    (out/'manifest.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Saved {len(rows)} runs to {out}/runs.csv; independent seed-paired comparison to summary.csv')


if __name__=='__main__':main()
