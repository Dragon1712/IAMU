#!/usr/bin/env python3
"""Paired and non-oracle experiment: point ETA vs MC mean vs CVaR + budget recourse.

All selector variants use the *same* public ETA events, candidate generator,
resource feasibility checks, and hidden realization sequence. No train.py run.
This pilot does not claim calibrated future arrival distributions.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from iamu_dynamic import read_case, demo_case, generate_scenario, simulate, save_result
from iamu_dynamic.risk_selection import upper_cvar


def split(x):return [v.strip() for v in x.split(',') if v.strip()]


def output_csv(path, rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    setups=[('demo',*demo_case())] if args.demo else [(p,*read_case(p)) for p in args.cases]
    budgets=[int(v) for v in split(args.budgets)]
    seeds=[int(v) for v in split(args.seeds)]
    methods=split(args.methods)
    if not set(methods).issubset({'deterministic','expected','risk_lookahead'}):
        raise ValueError('Unsupported methods list')
    risk_aversions=[float(v) for v in split(args.risk_aversions)]
    rows=[]
    for case,vessels,port in setups:
        for scenario_name in split(args.scenarios):
            for seed in seeds:
                scenario=generate_scenario(vessels,port,scenario_name,seed,args.updates,args.uncertainty)
                for budget in budgets:
                    jobs=[('no_replan','deterministic',0.0)]
                    if args.periodic_reference:jobs.append(('periodic','deterministic',0.0))
                    for method in methods:
                        if method=='risk_lookahead':
                            jobs.extend([('budget_aware',method,l) for l in risk_aversions])
                        else:jobs.append(('budget_aware',method,0.0))
                    for policy,method,lam in jobs:
                        start=perf_counter()
                        result=simulate(scenario,policy,budget,
                            threshold=args.threshold,period=args.period,
                            max_edits_per_event=args.max_edits_per_event,
                            candidate_limit=args.candidate_limit,
                            repair_method=args.repair_method,
                            revision_penalty=args.revision_penalty,
                            selection_method=method,forecast_std=args.forecast_std,
                            risk_samples=args.risk_samples,cvar_alpha=args.cvar_alpha,
                            risk_aversion=lam,lookahead_horizon=args.lookahead_horizon,
                            lookahead_candidates=args.lookahead_candidates,
                            next_move_candidates=args.next_move_candidates,
                            future_signal_strength=args.future_signal_strength)
                        m=result['metrics']
                        label=(policy if policy!='budget_aware' else
                               method+(('_lambda'+str(lam)) if method=='risk_lookahead' else ''))
                        rows.append(dict(case=case,scenario=scenario_name,seed=seed,
                            budget=budget,method=label,policy=policy,selector=method,
                            risk_aversion=lam,
                            spent=m['spent'],weighted_wait=m['weighted_wait'],
                            arrival_lateness_slip=m['arrival_lateness_slip'],
                            resource_conflict_slip=m['resource_conflict_slip'],
                            makespan=m['makespan'],candidate_evaluations=m['candidate_evaluations'],
                            sampled_evaluations=m['sampled_evaluations'],
                            two_stage_candidates=m['two_stage_candidates'],
                            kept_due_to_risk=m['kept_due_to_risk'],
                            runtime_seconds=round(perf_counter()-start,5)))
                        if args.save_details:
                            prefix=f'{Path(case).stem}_{scenario_name}_{seed}_{label}_B{budget}'
                            save_result(out/'details'/prefix,scenario,result)
                print(f'{case} {scenario_name} seed={seed}: {len(rows)} runs',flush=True)
    output_csv(out/'runs.csv',rows)
    baselines={(x['case'],x['scenario'],x['seed'],x['budget']):x
               for x in rows if x['method']=='no_replan'}
    periodic={(x['case'],x['scenario'],x['seed'],x['budget']):x
              for x in rows if x['method']=='periodic'}
    paired=[]
    for row in rows:
        key=(row['case'],row['scenario'],row['seed'],row['budget'])
        base=baselines[key]
        p=periodic.get(key)
        paired.append({**row,'gain_vs_no_replan':round(base['weighted_wait']-row['weighted_wait'],6),
            'gain_pct_vs_no_replan':round(100*(base['weighted_wait']-row['weighted_wait'])/base['weighted_wait'],6),
            'gain_vs_periodic':(None if p is None else round(p['weighted_wait']-row['weighted_wait'],6))})
    output_csv(out/'paired_runs.csv',paired)
    groups=defaultdict(list)
    for row in paired:groups[row['method'],row['budget']].append(row)
    summary=[]
    for (method,budget),recs in sorted(groups.items()):
        gains=[x['gain_vs_no_replan'] for x in recs]
        summary.append(dict(method=method,budget=budget,n=len(recs),
            mean_wait=round(mean(x['weighted_wait'] for x in recs),5),
            mean_gain=round(mean(gains),5),median_gain=round(median(gains),5),
            mean_gain_pct=round(mean(x['gain_pct_vs_no_replan'] for x in recs),5),
            improved=sum(g>1e-6 for g in gains),unchanged=sum(abs(g)<=1e-6 for g in gains),
            degraded=sum(g<-1e-6 for g in gains),worst_gain=round(min(gains),5),
            mean_spent=round(mean(x['spent'] for x in recs),3),
            mean_runtime_seconds=round(mean(x['runtime_seconds'] for x in recs),3),
            max_regret=round(max(max(0.,-g) for g in gains),5),
            cvar90_regret=round(upper_cvar([max(0.,-g) for g in gains],.9),5),
            mean_arrival_lateness_slip=round(mean(x['arrival_lateness_slip'] for x in recs),5),
            mean_resource_conflict_slip=round(mean(x['resource_conflict_slip'] for x in recs),5),
            mean_sampled_evaluations=round(mean(x['sampled_evaluations'] for x in recs),2),
            mean_two_stage_candidates=round(mean(x['two_stage_candidates'] for x in recs),2),
            mean_kept_due_to_risk=round(mean(x['kept_due_to_risk'] for x in recs),2),
            mean_gain_vs_periodic=(None if not args.periodic_reference else round(
                mean(x['gain_vs_periodic'] for x in recs),5))))
    output_csv(out/'summary.csv',summary)
    (out/'manifest.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Completed {len(rows)} runs; saved {out}/runs.csv and summary.csv',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--cases',nargs='+')
    group.add_argument('--demo',action='store_true')
    p.add_argument('--scenarios',default='shock,repeated')
    p.add_argument('--seeds',default='11,22,33')
    p.add_argument('--budgets',default='0,2,5')
    p.add_argument('--methods',default='deterministic,expected,risk_lookahead')
    p.add_argument('--risk-aversions',default='0,0.5')
    p.add_argument('--risk-samples',type=int,default=4)
    p.add_argument('--forecast-std',type=float,default=8.0)
    p.add_argument('--cvar-alpha',type=float,default=.75)
    p.add_argument('--lookahead-horizon',type=float,default=12.0)
    p.add_argument('--lookahead-candidates',type=int,default=1)
    p.add_argument('--next-move-candidates',type=int,default=1)
    p.add_argument('--future-signal-strength',type=float,default=.5)
    p.add_argument('--candidate-limit',type=int,default=4)
    p.add_argument('--max-edits-per-event',type=int,default=2)
    p.add_argument('--repair-method',choices=['single','alns2'],default='single')
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--uncertainty',type=float,default=8)
    p.add_argument('--threshold',type=float,default=3)
    p.add_argument('--period',type=float,default=12)
    p.add_argument('--revision-penalty',type=float,default=0)
    p.add_argument('--periodic-reference',action='store_true',default=True)
    p.add_argument('--no-periodic-reference',action='store_false',dest='periodic_reference')
    p.add_argument('--save-details',action='store_true')
    p.add_argument('--out',default='results/v04_risk_pilot')
    args=p.parse_args()
    if not math.isfinite(args.uncertainty) or args.uncertainty<0:p.error('invalid uncertainty')
    if args.risk_samples<1 or args.candidate_limit<1 or args.next_move_candidates<1 or args.lookahead_candidates<1:
        p.error('sampling and candidate counts must be >= 1')
    run(args)
if __name__=='__main__':main()
