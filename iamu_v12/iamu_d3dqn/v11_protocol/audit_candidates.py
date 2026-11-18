#!/usr/bin/env python3
"""Public-state candidate coverage + ex-post diagnostic of screened proposals.

Keep/Greedy anchors build states without any RL retraining. Every audit record
separates physical proposals, point-positive proposals and returned K actions.
Optional paired ex-post single-future replay only explains observed choices;
it is NOT a deployable oracle or a conditional expected Q value.
"""
from __future__ import annotations
import argparse,csv,copy,json,sys
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.rl_env_v11 import IAMUEventEnv
from iamu_dynamic.rl_experiment_v11 import parse_ints,parse_names,write_csv

def proposals(env):
    entries=[];base=None
    for vessel_id in env._eligible():
        plan,reason=env._proposal_with_reason(vessel_id)
        if plan is None:continue
        if base is None:base=env._forecast_projection(env.plan)
        original_starts,original_costs,original_score=base
        starts,costs,score=env._forecast_projection(plan)
        gain=original_score-score
        affected=[j for j in starts if j!=vessel_id]
        worsened=[j for j in affected if starts[j]>original_starts[j]+1e-9]
        data=dict(gain=gain,base_score=original_score,new_score=score,
            external_worse_count=len(worsened),
            external_max_delay=max((starts[j]-original_starts[j] for j in worsened),default=0.),
            external_worse_cost=sum(max(0.,costs[j]-original_costs[j]) for j in affected),
            external_relief_cost=sum(max(0.,original_costs[j]-costs[j]) for j in affected))
        entries.append((vessel_id,plan,gain,data))
    return entries

def realized(env,first=None):
    e=copy.deepcopy(env)
    if first is None:action=0
    else:
        vid,plan,gain,data=first
        e._candidate_ids=(vid,)
        e._candidates=[plan]
        # Only step() needs gain and the legal physical mask; this is an
        # audit-only action in a cloned physical environment.
        e._candidate_data=[data]
        e._physical_mask[:]=False;e._physical_mask[0]=True;e._physical_mask[1]=True
        e._heuristic_mask[:]=False;e._heuristic_mask[0]=True
        e.decision_flags=dict(e.decision_flags,has_retained_edit=True,returned_candidates=1)
        action=1
    _,_,done,info=e.step(action)
    while not done:
        _,_,done,info=e.step(0)
    return float(info['weighted_wait'])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--seeds',default='400001:400002')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--anchors',default='keep,greedy')
    p.add_argument('--states-per-group',type=int,default=3)
    p.add_argument('--counterfactual-per-state',type=int,default=2)
    p.add_argument('--candidate-limit',type=int,default=4)
    p.add_argument('--clock',default='union')
    p.add_argument('--period',type=float,default=12.)
    p.add_argument('--min-gain',type=float,default=.1)
    p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):p.error('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    rows=[];cf=[];flags=[]
    for path in args.cases:
        vessels,port=read_case(path)
        for sc in parse_names(args.scenarios):
            for seed in parse_ints(args.seeds):
                future=generate_scenario(vessels,port,label=sc,seed=seed,updates_per_vessel=2,uncertainty=8.)
                for budget in parse_ints(args.budgets):
                    for anchor in parse_names(args.anchors):
                        env=IAMUEventEnv(future,budget=budget,candidate_limit=args.candidate_limit,
                            decision_clock=args.clock,period=args.period,min_gain=args.min_gain,candidate_screen='positive')
                        obs=env.reset();states=[]
                        while True:
                            flags.append(dict(case=Path(path).stem,scenario=sc,seed=seed,budget=budget,
                                anchor=anchor,event=env.event_index,**obs.decision_flags))
                            if obs.decision_flags.get('decision_allowed') and env.remaining_budget>0 and not obs.decision_flags.get('all_started'):
                                states.append(copy.deepcopy(env))
                            action=0 if anchor=='keep' else env.point_greedy_action()
                            obs,_,done,_=env.step(action)
                            if done:break
                        stride=min(args.states_per_group,len(states))
                        idx=[(2*i+1)*len(states)//(2*stride) for i in range(stride)] if stride else []
                        for j in idx:
                            state=states[j];all_raw=proposals(state)
                            point_pos=[x for x in all_raw if x[2]>args.min_gain+1e-12]
                            chosen=set(state._candidate_ids)
                            negative=[x for x in all_raw if x[2]<=args.min_gain+1e-12]
                            rank_first_k=set(state._eligible()[:args.candidate_limit])
                            old_count=sum(i in rank_first_k and g>args.min_gain+1e-12 for i,_,g,_ in all_raw)
                            rows.append(dict(case=Path(path).stem,scenario=sc,seed=seed,budget=budget,
                                anchor=anchor,event=state.event_index,full_raw=len(all_raw),
                                point_positive=len(point_pos),returned=len(chosen),
                                missed_point_positive=sum(x[0] not in chosen for x in point_pos),
                                old_first_k_count=old_count,new_actionable=int(bool(chosen)),
                                old_actionable=int(old_count>0),
                                gained_actionability=int(old_count==0 and bool(chosen)),
                                point_nonpositive=len(negative),
                                **{k:state.decision_flags[k] for k in ('pending_ships','ships_examined',
                                    'unchanged_proposals','screened_proposals','search_exhausted')}))
                            if args.counterfactual_per_state:
                                baseline=realized(state)
                                for cand in negative[:args.counterfactual_per_state]:
                                    cost=realized(state,cand)
                                    cf.append(dict(case=Path(path).stem,scenario=sc,seed=seed,budget=budget,
                                        anchor=anchor,event=state.event_index,vessel=cand[0],point_gain=cand[2],
                                        ex_post_gain=baseline-cost,keep_wait=baseline,edit_wait=cost))
    write_csv(out/'reason_flags.csv',flags)
    write_csv(out/'candidate_coverage.csv',rows)
    write_csv(out/'screened_counterfactual.csv',cf)
    (out/'summary.json').write_text(json.dumps(dict(n_states=len(rows),n_screened_pairs=len(cf),
        mean_returned=mean(r['returned'] for r in rows) if rows else None,
        fraction_screened_positive_expost=sum(r['ex_post_gain']>0 for r in cf)/len(cf) if cf else None,
        interpretation='ex-post single-realization, fixed Keep continuation; not an oracle'),indent=2))
    print('[DONE] candidate states=',len(rows),'diagnostic ex-post pairs=',len(cf))
if __name__=='__main__':main()
