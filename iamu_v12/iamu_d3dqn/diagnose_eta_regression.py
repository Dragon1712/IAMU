#!/usr/bin/env python3
"""Offline comparison of legacy oracle-timed ETA schedules and exogenous ETA schedules.

CRITICAL: hindsight_audit uses hidden true arrivals strictly AFTER actions are
selected, only to explain failures. Never feed this information into a policy.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from iamu_dynamic import (Port, Scenario, Update, Vessel, generate_scenario,
                          read_case, simulate)


def load_legacy(path:Path)->Scenario:
    raw=json.loads(path.read_text(encoding='utf-8'))
    return Scenario(
        [Vessel(**x) for x in raw['vessels']], Port(**raw['port']),
        {int(k):float(v) for k,v in raw['true_etas'].items()},
        [Update(**x) for x in raw['updates']], raw['label'],raw['seed'],
    )


def reclock_same_truth(sc:Scenario)->Scenario:
    """Controlled offline A/B: hold actual arrivals and ETA values fixed.

    The *only intended intervention* is replacing the legacy timestamp formula
    with an initially-public schedule; late notifications are suppressed once
    the ship has actually arrived.  This is a diagnostic fixture construction,
    not an online feature supplied to the decision policy.
    """
    by_ship={v.id:[] for v in sc.vessels}
    for e in sc.updates:
        by_ship[e.vessel_id].append(e)
    updates=[];k=0
    for v in sc.vessels:
        series=sorted(by_ship[v.id],key=lambda e:e.update_id)
        n=len(series)
        for u,e in enumerate(series):
            t=v.eta0*(u+1)/(n+1)
            if t>=sc.true_etas[v.id]-1e-8:continue
            updates.append(Update(t,v.id,max(t,e.new_eta),k))
            k+=1
    updates.sort(key=lambda e:(e.time,e.update_id))
    return Scenario(sc.vessels,sc.port,sc.true_etas.copy(),updates,
                    'reclocked_'+sc.label,sc.seed)


def diagnose(sc:Scenario,label:str,target:Path):
    target.mkdir(parents=True,exist_ok=True)
    base=simulate(sc,'no_replan',budget=0)
    rows=[];edits=[];per_vessel=[]
    for policy,b in [('no_replan',0),('budget_aware',2),('budget_aware',5),('periodic',5)]:
        res=base if policy=='no_replan' else simulate(
            sc,policy,b,diagnostic_hindsight=True)
        m=res['metrics'];name=f'{policy}_B{b}'
        rows.append(dict(scenario_format=label,policy=policy,budget=b,
                         spent=m['spent'],weighted_wait=m['weighted_wait'],
                         change_vs_no_replan=round(m['weighted_wait']-base['metrics']['weighted_wait'],6),
                         makespan=m['makespan'],arrival_lateness_slip=m['arrival_lateness_slip'],
                         resource_conflict_slip=m['resource_conflict_slip']))
        for step,log in enumerate(m.get('hindsight_audit',[]),1):
            edits.append({k:v for k,v in dict(run=name,step=step,**log).items()
                          if k!='per_ship_start_deltas'})
            per_vessel.extend(dict(run=name,step=step,vessel_id=ship,
                                   start_delta=value)
                              for ship,value in log['per_ship_start_deltas'].items())
        (target/f'{name}_revisions.json').write_text(
            json.dumps(res['revisions'],ensure_ascii=False,indent=2),encoding='utf-8')
    for path, records in [('comparison.csv',rows),('offline_hindsight_actions.csv',edits),
                          ('offline_hindsight_ship_effects.csv',per_vessel)]:
        if records:
            with (target/path).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    print(f'=== {label} ===')
    for r in rows:
        print(f"  {r['policy']:>13} B={r['budget']} wait={r['weighted_wait']:.6f} "
              f"delta={r['change_vs_no_replan']:+.6f} edits={r['spent']}")
    for e in edits:
        if e['run']=='budget_aware_B5':
            print('  action',e['step'],'at',e['time'],'vessels',e['changed'],
                  'estimated_gain',e['estimated_gain'],
                  'offline_hindsight_gain',e['hindsight_gain'],
                  'worst_ship',e['worst_ship'],
                  'worst_ship_start_delta',e['worst_start_delta'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='../2024LiERLDDE/case_code/1.txt')
    parser.add_argument('--legacy-scenario',default='fixtures/legacy_case1_shock22.json')
    parser.add_argument('--mode',choices=['legacy','reclocked','new','all'],default='all')
    parser.add_argument('--out',default='results/eta_fix_diagnosis')
    parser.add_argument('--seed',type=int,default=22)
    parser.add_argument('--scenario',default='shock',choices=['shock','small','repeated'])
    parser.add_argument('--updates',type=int,default=2)
    parser.add_argument('--uncertainty',type=float,default=8)
    args=parser.parse_args()
    out=Path(args.out)
    legacy=None
    if args.mode in ('legacy','reclocked','all'):
        legacy=load_legacy(Path(args.legacy_scenario))
    if args.mode in ('legacy','all'):
        diagnose(legacy, 'historical_oracle_clock', out/'legacy')
    if args.mode in ('reclocked','all'):
        diagnose(reclock_same_truth(legacy), 'reclocked_same_hidden_arrivals',
                 out/'reclocked')
    if args.mode in ('new','all'):
        v,p=read_case(args.case)
        s=generate_scenario(v,p,args.scenario,args.seed,args.updates,args.uncertainty)
        diagnose(s,'exogenous_public_clock',out/'fixed')
    (out/'manifest.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print(f'Offline diagnostic CSVs saved under {out.resolve()}')


if __name__=='__main__':main()
