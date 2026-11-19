#!/usr/bin/env python3
"""Run independent dynamic commitment experiments; original ERL-DDE stays untouched."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from iamu_dynamic import (read_case,demo_case,generate_scenario,simulate,save_result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--case',help='Original case_code/0.txt (ERL-DDE TSV)')
    g.add_argument('--demo',action='store_true',help='Run built-in self-contained demonstration')
    p.add_argument('--scenario',choices=['small','shock','repeated'],default='repeated')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--uncertainty',type=float,default=8.0)
    p.add_argument('--updates',type=int,default=3)
    p.add_argument('--budgets',default='0,2,5,unlimited')
    p.add_argument('--policies',default='no_replan,threshold,periodic,always,budget_aware')
    p.add_argument('--threshold',type=float,default=3.0)
    p.add_argument('--period',type=float,default=12.0)
    p.add_argument('--max-edits-per-event',type=int,default=2)
    p.add_argument('--repair-method',choices=['single','alns2'],default='single',
                   help='single-ship repair or bounded two-ship destroy/repair neighborhoods')
    p.add_argument('--selection-method',choices=['deterministic','expected','risk_lookahead'],default='deterministic',
                   help='Causal scenario-based selector; deterministic preserves V0.3')
    p.add_argument('--forecast-std',type=float,default=8.0)
    p.add_argument('--risk-samples',type=int,default=8)
    p.add_argument('--cvar-alpha',type=float,default=0.75)
    p.add_argument('--risk-aversion',type=float,default=0.5)
    p.add_argument('--lookahead-horizon',type=float,default=12.0)
    p.add_argument('--lookahead-candidates',type=int,default=3)
    p.add_argument('--next-move-candidates',type=int,default=3)
    p.add_argument('--future-signal-strength',type=float,default=0.5)
    p.add_argument('--revision-penalty',type=float,default=0.0,
                   help='Forecast wait-time penalty per charged published change')
    p.add_argument('--max-repair-pairs',type=int,default=4)
    p.add_argument('--periodic-horizon',type=float,
                   help='End of independent periodic decision clock; default initial published makespan + period')
    p.add_argument('--out',default='results/dynamic_demo')
    p.add_argument('--backend',choices=['heuristic','erldde'],default='heuristic')
    p.add_argument('--erldde-root',help='Path to original ERL-DDE source tree')
    p.add_argument('--checkpoint',help='One saved original synced_model_*.pth')
    p.add_argument('--probe-so',help='Path to separately compiled libBACAP_env_probe.so')
    p.add_argument('--build-probe',action='store_true',help='Build probe without editing original C++')
    args=p.parse_args()
    vessels,port=demo_case() if args.demo else read_case(args.case)
    scenario=generate_scenario(vessels,port,args.scenario,args.seed,args.updates,args.uncertainty)
    hint_provider=None
    if args.backend=='erldde':
        if not args.erldde_root or not args.checkpoint:
            p.error('--backend erldde requires --erldde-root and --checkpoint')
        from iamu_dynamic.legacy_erldde import FrozenERLDDEHints,build_probe
        probe_path=Path(args.probe_so) if args.probe_so else Path(args.out)/'probe_build/libBACAP_env_probe.so'
        if args.build_probe:probe_path=build_probe(args.erldde_root,probe_path.parent)
        hint_provider=FrozenERLDDEHints(args.erldde_root,args.checkpoint,probe_path)
        print(f'Frozen ES policy loaded from {args.checkpoint} (inference only)',flush=True)
    policies=[x.strip() for x in args.policies.split(',') if x.strip()]
    budgets=[None if x.strip().lower()=='unlimited' else int(x) for x in args.budgets.split(',')]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for policy in policies:
        for budget in budgets:
            # Consistent settings: even 'no_replan' gets a budget label.
            result=simulate(scenario,policy,budget,args.threshold,args.period,
                            args.max_edits_per_event,hint_provider=hint_provider,
                            repair_method=args.repair_method,
                            revision_penalty=args.revision_penalty,
                            max_repair_pairs=args.max_repair_pairs,
                            periodic_horizon=args.periodic_horizon,
                            selection_method=args.selection_method,
                            forecast_std=args.forecast_std,risk_samples=args.risk_samples,
                            cvar_alpha=args.cvar_alpha,risk_aversion=args.risk_aversion,
                            lookahead_horizon=args.lookahead_horizon,
                            lookahead_candidates=args.lookahead_candidates,
                            next_move_candidates=args.next_move_candidates,
                            future_signal_strength=args.future_signal_strength)
            label='unlimited' if budget is None else str(budget)
            target=out/f'{policy}_B{label}'
            save_result(target,scenario,result)
            metrics=dict(result['metrics'],backend=args.backend,case=(args.case or 'demo'),
                         scenario=args.scenario,seed=args.seed)
            entries.append(metrics)
            print(f'{policy:>14s} B={label:>9s} spent={metrics["spent"]:3d} '
                  f'wait={metrics["weighted_wait"]:11.3f} '
                  f'slip={metrics["total_operational_slip"]:10.3f}',flush=True)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=entries[0]);w.writeheader();w.writerows(entries)
    (out/'manifest.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print(f'Comparison saved to: {out.resolve() / "comparison.csv"}')

if __name__=='__main__':main()
