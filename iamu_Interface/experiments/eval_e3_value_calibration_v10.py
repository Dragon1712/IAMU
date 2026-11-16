#!/usr/bin/env python3
"""V1.0-E3-A: separate MPC estimation error from Q/MPC fusion error.

For each FROZEN public decision state, evaluate Keep and each legal Edit using:
  (1) online MPC estimate from M shared ETA samples (no future leaked),
  (2) independent larger-M MPC reference (Monte Carlo accuracy only),
  (3) multiple independently sampled synthetic future worlds whose ETA updates
      are released only at predeclared future times, then BOTH arms continue
      with the SAME frozen Set-aware DDQN policy (model-based future benchmark),
  (4) original held-out realized future replayed from the same public state
      with both arms using the frozen continuation (single-world diagnostic).

No simulated or actual future outcome is used to CHOOSE an action in the policy.
Do not claim the synthetic-world reference is an empirical port arrival truth.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(ROOT/'experiments') not in sys.path:sys.path.insert(0,str(ROOT/'experiments'))
if not (ROOT/'iamu_dynamic/core.py').is_file():
    raise SystemExit('Run setup_interface.py to install the V0.9 engine snapshot first')
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.core import Update
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.v10_e3_policy import E3Specification,estimates_at_state
from iamu_dynamic.v10_e3_confidence import paired_confidence,fused_scores,MODES
from iamu_dynamic.v10_policy import PolicySpecification,public_arrival_flags,frozen_q_values
from iamu_dynamic.v10_budget_lookahead import LookaheadSpecification,evaluate_multistage
from iamu_dynamic.v10_risk_features import RiskSpecification,public_eta_scenarios,event_seed
from eval_risk_budget_v10 import ints,write_csv,digest


def make_synthetic_world(env,eta_sample,*,step_hours:float,signal_strength:float,
                         future_signals:int):
    """Create a counterfactual world conditional on an unchanged PUBLIC prefix.

    Old future notifications are removed; new notifications are scheduled at
    fixed future times, irrespective of actual sampled arrivals. Notifications
    contain only a partial forecast (not exact hidden arrivals), released at
    each event time. Started / already arrived ships retain physical history.
    Use exclusively for EX-POST diagnostics, never on an online policy path.
    """
    if not 0. <= signal_strength <= 1. or step_hours<=0 or future_signals<0:
        raise ValueError('Invalid hypothetical future public signal model')
    c=copy.deepcopy(env)
    now=float(env.event_time)
    arrived=public_arrival_flags(env)
    future_ids=[i for i in c.ids if i not in arrived]
    arrivals=dict(c.scenario.true_etas)
    for i in future_ids:
        if eta_sample[i] < now-1e-8:
            raise ValueError('Sample contradicts public not-arrived status')
        arrivals[i]=float(eta_sample[i])
    c.scenario.true_etas=arrivals
    old_prefix=list(c.events[:c.event_index])
    public_clocks=[e for e in c.events[c.event_index:]
                   if e[3]=='clock' and e[0]>now+1e-8]
    updates=[]
    count=0
    for stage in range(1,future_signals+1):
        time=now+stage*step_hours
        for i in future_ids:
            mu=float(env.published_eta[i])
            new_eta=max(time,mu+signal_strength*(float(eta_sample[i])-mu))
            count+=1
            # Stable synthetic IDs unique beyond all existing updates.
            update=Update(time=float(time),vessel_id=int(i),
                          new_eta=float(new_eta),update_id=1000000+count)
            updates.append((float(time),0,1000000+count,'eta',update))
    future=sorted([*public_clocks,*updates],key=lambda t:(t[0],t[1],t[2]))
    c.events=old_prefix+future
    c.scenario.updates=[ev[4] for ev in c.events if ev[3]=='eta']
    if c.event_index!=len(old_prefix):
        raise AssertionError('Public state history changed')
    if tuple(c.published_eta.items())!=tuple(env.published_eta.items()):
        raise AssertionError('Current public ETA changed')
    return c


def continue_to_end(env,first_action:int,agent,*,max_steps=20000):
    """Continuation sees ONLY signals emitted by env.step, never future data."""
    obs=env._observation()
    if first_action<0 or first_action>=len(obs.mask) or not bool(obs.mask[first_action]):
        raise ValueError('Illegal counterfactual first action')
    obs,_,done,info=env.step(first_action)
    count=1
    while not done:
        if count>=max_steps:
            raise RuntimeError('Counterfactual environment exceeded event bound')
        action=int(agent.act(obs,epsilon=0.))
        obs,_,done,info=env.step(action)
        count+=1
    return float(info['weighted_wait']),int(info['spent'])


def realized_counterfactual(env,actions,agent):
    """Diagnostic only: actual future information is never supplied to scorer."""
    return {a:continue_to_end(copy.deepcopy(env),a,agent) for a in actions}


def evaluate_snapshot(env,obs,agent,spec,*,reference_samples:int,
                      future_worlds:int,signal_strength:float,future_signals:int,
                      max_edit_actions:int):
    q,mpc,stats,risk,t_risk,t_mpc=estimates_at_state(env,obs,agent,spec,
                      want_risk=spec.base.beta_risk>0)
    if mpc is None:return [],[]
    candidates={a:env._candidates[a-1] for a in range(1,len(obs.mask)) if obs.mask[a]}
    # All actions if possible. When capped, select using only public Q, never
    # hindsight nor the reference evaluations that will be computed below.
    choices=sorted(candidates,key=lambda a:(-float(q[a]-q[0]),a))[:max_edit_actions]
    subset={a:candidates[a] for a in choices}
    actions=[0,*choices]
    seed=event_seed(env.scenario.seed,env.event_index,env.event_type)
    arrivals=public_arrival_flags(env)
    ref_spec=RiskSpecification(spec.base.risk.sigma,reference_samples,
                               spec.base.risk.alpha,spec.base.risk.belief)
    ref_samples=public_eta_scenarios(env.published_eta,env.event_time,arrivals,
                            ref_spec,seed=seed ^ 0xE3A20D2E5F171A)
    # Independent high-sample reference: SAME greedy MPC continuation.
    ref=evaluate_multistage(env,subset,ref_samples,spec.base.lookahead)
    # H=0 ablation on the SAME samples, so an observed value-of-lookahead
    # difference is not a change in posterior or random-number sequence.
    online_draws=public_eta_scenarios(env.published_eta,env.event_time,arrivals,
                    spec.base.risk,seed=seed)
    h0spec=LookaheadSpecification(
        stages=0,step_hours=spec.base.lookahead.step_hours,
        signal_strength=spec.base.lookahead.signal_strength,
        min_gain=spec.base.lookahead.min_gain,
        candidates_per_stage=spec.base.lookahead.candidates_per_stage,
        alpha=spec.base.lookahead.alpha)
    mpc_h0=evaluate_multistage(env,subset,
                     online_draws[:spec.base.mpc_samples],h0spec)
    reference_h0=evaluate_multistage(env,subset,ref_samples,h0spec)
    # Truly independent, paired, synthetic FUTURE WORLDS with a FROZEN learned
    # continuation, not the MPC greedy continuation used by the predictor.
    truth_spec=RiskSpecification(spec.base.risk.sigma,future_worlds,
                                 spec.base.risk.alpha,spec.base.risk.belief)
    truth_samples=public_eta_scenarios(env.published_eta,env.event_time,arrivals,
                                 truth_spec,seed=seed ^ 0xE3A46B59852C43)
    truth_costs={a:[] for a in actions}
    truth_spent={a:[] for a in actions}
    start=perf_counter()
    for sample in truth_samples:
        synthetic=make_synthetic_world(env,sample,
            step_hours=spec.base.lookahead.step_hours,
            signal_strength=signal_strength,future_signals=future_signals)
        for a in actions:
            cost,spent=continue_to_end(copy.deepcopy(synthetic),a,agent)
            truth_costs[a].append(cost)
            truth_spent[a].append(spent)
    synthetic_seconds=perf_counter()-start
    # Original realized future is a SINGLE paired observation, not a label for
    # true conditional expected value. This evaluation may use hidden actual
    # future, but nothing here feeds the policy or the Q/MPC estimators.
    realized=realized_counterfactual(env,actions,agent)
    rows=[]
    for a in choices:
        paired_truth=[k-v for k,v in zip(truth_costs[0],truth_costs[a])]
        truth_mean=mean(paired_truth)
        prediction=stats[a].mean*env.reward_scale
        half=stats[a].half_width*env.reward_scale
        reference=ref.advantage(a)
        realized_gain=realized[0][0]-realized[a][0]
        rows.append(dict(
            case=None,scenario=None,eta_seed=None,budget=None,
            event_index=env.event_index,event_time=env.event_time,
            event_type=env.event_type,remaining_budget=env.remaining_budget,
            n_candidates=len(candidates),action=a,vessel_id=env._candidate_ids[a-1],
            q_gap=float(q[a]-q[0]),mpc_mean=prediction,
            mpc_std=stats[a].sd*env.reward_scale,
            mpc_sem=stats[a].se*env.reward_scale,
            mpc_ci_low=prediction-half,mpc_ci_high=prediction+half,
            mpc_effective_strength=stats[a].signal_strength,
            mpc_reference_advantage=reference,
            mc_reference_error=prediction-reference,
            mc_reference_in_ci=int(prediction-half<=reference<=prediction+half),
            mc_reference_sign_agree=int((prediction>0)==(reference>0)),
            q_mpc_direction_agree=int(float(q[a]-q[0])*prediction>=0),
            mpc_h0_advantage=mpc_h0.advantage(a),
            mpc_h0_reference_advantage=reference_h0.advantage(a),
            mc_h0_reference_error=mpc_h0.advantage(a)-reference_h0.advantage(a),
            h2_minus_h0_predicted=prediction-mpc_h0.advantage(a),
            synthetic_frozen_advantage=truth_mean,
            synthetic_frozen_error=prediction-truth_mean,
            h0_synthetic_frozen_error=mpc_h0.advantage(a)-truth_mean,
            value_of_extra_lookahead_abs_error=(abs(mpc_h0.advantage(a)-truth_mean)
                                                  -abs(prediction-truth_mean)),
            mean_synthetic_budget_spent_keep=mean(truth_spent[0]),
            mean_synthetic_budget_spent_edit=mean(truth_spent[a]),
            synthetic_frozen_sign_agree=int((prediction>0)==(truth_mean>0)),
            realized_frozen_advantage=realized_gain,
            realized_frozen_error=prediction-realized_gain,
            future_worlds=future_worlds,
            mpc_samples=mpc.scenario_count,
            mpc_reference_samples=reference_samples,
            risk_tail_delta=risk[a].tail_cost_delta if a in risk else None,
            evaluation_seconds=t_risk+t_mpc+synthetic_seconds,
        ))
    # Crucial separation: evaluate EVERY fusion rule on the SAME fixed state,
    # identical predicted MPC statistics, and identical hypothetical futures.
    # Thus prediction error is held fixed while only the fusion rule changes.
    restricted_mask=[bool(i==0 or i in choices) for i in range(len(obs.mask))]
    realized_cost={a:realized[a][0] for a in actions}
    synthetic_cost={a:mean(truth_costs[a]) for a in actions}
    oracle_action=min(actions,key=lambda a:(synthetic_cost[a],a!=0,a))
    oracle_realized=min(actions,key=lambda a:(realized_cost[a],a!=0,a))
    policy_rows=[]
    for mode in MODES:
        keep_margin=(env.min_gain/env.reward_scale if mode in ('pure_mpc','pure_mpc_risk')
                     else float(agent.keep_margin))
        selected,scores=fused_scores(q,restricted_mask,stats,mode=mode,
             beta_budget=spec.base.beta_budget,beta_risk=spec.base.beta_risk,
             risk_metrics=risk,reward_scale=env.reward_scale,
             keep_margin=keep_margin,neutral=spec.neutral_q)
        policy_rows.append(dict(
            case=None,scenario=None,eta_seed=None,budget=None,
            event_index=env.event_index,event_time=env.event_time,
            n_available_actions=len(actions),
            n_candidate_actions=len(choices),
            mode=mode,selected_action=selected,
            predicted_qgap=float(q[selected]-q[0]),
            selected_effective_weight=(scores[selected]['effective_weight']
                                       if selected else 0.),
            oracle_action=oracle_action,
            oracle_realized_action=oracle_realized,
            oracle_regret=synthetic_cost[selected]-synthetic_cost[oracle_action],
            realized_regret=realized_cost[selected]-realized_cost[oracle_realized],
            matches_oracle=int(selected==oracle_action),
            matches_realized_oracle=int(selected==oracle_realized),
        ))
    return rows,policy_rows


def select_snapshots(instance,b,agent,env_args,*,state_count:int,stride:int,
                     selection:str,min_candidates:int):
    env=IAMUEventEnv(instance,budget=b,**env_args)
    obs=env.reset()
    candidates=[]
    eligible=0
    while True:
        n=int(obs.mask[1:].sum())
        if n>=min_candidates and n>0:
            eligible+=1
            if selection=='near_margin':
                q=frozen_q_values(agent,obs)
                gap=max(float(q[a]-q[0]) for a in range(1,len(q)) if obs.mask[a])
                priority=abs(gap-float(agent.keep_margin))
                candidates.append((priority,env.event_index,copy.deepcopy(env)))
            elif (eligible-1)%stride==0 and len(candidates)<state_count:
                candidates.append((float(eligible),env.event_index,copy.deepcopy(env)))
        act=int(agent.act(obs,epsilon=0.))
        obs,_,done,_=env.step(act)
        if done:break
    if selection=='near_margin':
        candidates.sort(key=lambda x:(x[0],x[1]))
        chosen=[]
        for item in candidates:
            if all(abs(item[1]-k[1])>=3 for k in chosen):
                chosen.append(item)
            if len(chosen)>=state_count:break
        candidates=chosen
    return [(env0,env0._observation()) for _,_,env0 in candidates],eligible


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    bundled=ROOT/'assets/checkpoints/set_ddqn_seed42'
    fallback=ROOT.parent/'iamu_d3dqn/results/v09b_set_ddqn_seed42_300'
    checkpoint=bundled if (bundled/'best.pt').is_file() else fallback
    cdir=ROOT/'assets/cases'
    if not (cdir/'8.txt').is_file():cdir=ROOT.parent/'2024LiERLDDE/case_code'
    p.add_argument('--checkpoint',default=str(checkpoint/'best.pt'))
    p.add_argument('--manifest',default=str(checkpoint/'manifest.json'))
    p.add_argument('--cases',nargs='+',default=[str(cdir/'8.txt'),str(cdir/'9.txt')])
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--seeds',default='400001:400003')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--states-per-group',type=int,default=1)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--selection',choices=('first','near_margin'),default='near_margin')
    p.add_argument('--min-candidates',type=int,default=1)
    p.add_argument('--max-edit-actions',type=int,default=4)
    p.add_argument('--risk-samples',type=int,default=16)
    p.add_argument('--mpc-samples',type=int,default=8)
    p.add_argument('--reference-samples',type=int,default=64)
    p.add_argument('--future-worlds',type=int,default=4)
    p.add_argument('--future-signals',type=int,default=2)
    p.add_argument('--signal-strength',type=float,default=.5)
    p.add_argument('--stages',type=int,default=2)
    p.add_argument('--step-hours',type=float,default=12.)
    p.add_argument('--beta-risk',type=float,default=.25)
    p.add_argument('--sigma',type=float,default=7.12463033710398)
    p.add_argument('--alpha',type=float,default=.90)
    p.add_argument('--belief',choices=('censored','truncated'),default='censored')
    p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-groups',type=int,default=0)
    p.add_argument('--candidate-screen',choices=('positive','none'),default=None)
    p.add_argument('--out',default='results/v10_e3_value_calibration_seed42')
    a=p.parse_args(argv)
    if min(a.future_worlds,a.reference_samples,a.mpc_samples,a.risk_samples,a.stride,a.states_per_group,a.max_edit_actions,a.min_candidates)<1:
        p.error('Sample, state and candidate counts must be >=1')
    if a.mpc_samples<2 or a.mpc_samples>a.risk_samples:
        p.error('2<=MPC sample size<=risk sample size required')
    for f in [a.checkpoint,a.manifest,*a.cases]:
        if not Path(f).is_file():p.error('Missing: '+str(f))
    train=json.loads(Path(a.manifest).read_text(encoding='utf-8'))['training_args']
    agent,_=DoubleDQNAgent.from_checkpoint(a.checkpoint,device=a.device)
    if agent.q_network!='set_ddqn':p.error('Must use a Set-aware DDQN model')
    env_args=dict(candidate_limit=int(train['candidate_limit']),
        decision_clock=train['clock'],period=float(train['period']),
        reward_scale=float(train['reward_scale']),min_gain=float(train['min_gain']),
        candidate_screen=a.candidate_screen or train['candidate_screen'],
        budget_features=train['budget_features'],value_features=train['value_features'])
    base=PolicySpecification(
       risk=RiskSpecification(a.sigma,a.risk_samples,a.alpha,a.belief),
       lookahead=LookaheadSpecification(stages=a.stages,
            step_hours=a.step_hours,min_gain=env_args['min_gain'],
            candidates_per_stage=env_args['candidate_limit'],alpha=a.alpha),
       beta_risk=a.beta_risk,beta_budget=.25,mpc_samples=a.mpc_samples)
    spec=E3Specification(base)
    paths=[Path(x) for x in a.cases]
    scenarios=a.scenarios.split(',');seeds=ints(a.seeds);budgets=ints(a.budgets)
    if a.smoke:
        paths,scenarios,seeds,budgets=paths[:1],scenarios[:1],seeds[:1],budgets[:1]
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Choose empty --out')
    out.mkdir(parents=True)
    files=['iamu_dynamic/v10_e3_confidence.py',
           'iamu_dynamic/v10_e3_policy.py',
           'experiments/eval_e3_value_calibration_v10.py',
           'iamu_dynamic/v10_budget_lookahead.py']
    meta=dict(arguments=vars(a),checkpoint_sha256=digest(Path(a.checkpoint)),
              model_manifest_sha256=digest(Path(a.manifest)),
              case_sha256={str(f):digest(f) for f in paths},
              code_sha256={f:digest(ROOT/f) for f in files},
              training_seed=train['seed'],environment=env_args,
              design='Synthetic posterior worlds and realized future are EVALUATION ONLY; not independent real-world truth',
              continuation='frozen Set-aware DDQN; MPC future continuation uses greedy repairs')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    records=[];policy_records=[];coverage=[];groups=0
    for path in paths:
        vessels,port=read_case(path)
        for scen in scenarios:
            for seed in seeds:
                instance=generate_scenario(vessels,port,label=scen,seed=seed,
                   updates_per_vessel=int(train['updates']),
                   uncertainty=float(train['validation_uncertainty']))
                for budget in budgets:
                    if a.max_groups and groups>=a.max_groups:break
                    snapshots,eligible=select_snapshots(instance,budget,agent,env_args,
                         state_count=a.states_per_group,stride=a.stride,
                         selection=a.selection,min_candidates=a.min_candidates)
                    coverage.append(dict(case=path.stem,scenario=scen,
                             eta_seed=seed,budget=budget,
                             n_eligible_states=eligible,n_selected_states=len(snapshots)))
                    for env,obs in snapshots:
                        rr,pr=evaluate_snapshot(env,obs,agent,spec,
                              reference_samples=a.reference_samples,
                              future_worlds=a.future_worlds,
                              signal_strength=a.signal_strength,
                              future_signals=a.future_signals,
                              max_edit_actions=a.max_edit_actions)
                        records.extend([dict(r,case=path.stem,scenario=scen,
                                            eta_seed=seed,budget=budget) for r in rr])
                        policy_records.extend([dict(r,case=path.stem,scenario=scen,
                                                 eta_seed=seed,budget=budget) for r in pr])
                    groups+=1
                    write_csv(out/'details.partial.csv',records)
                    write_csv(out/'fixed_state_policy.partial.csv',policy_records)
                    write_csv(out/'coverage.partial.csv',coverage)
                    print(f'[E3A] group={groups} case={path.stem} {scen} seed={seed} B={budget} selected={len(snapshots)} eligible={eligible}',flush=True)
                if a.max_groups and groups>=a.max_groups:break
            if a.max_groups and groups>=a.max_groups:break
        if a.max_groups and groups>=a.max_groups:break
    write_csv(out/'coverage.csv',coverage)
    if not records:
        # Zero eligible states is a VALID stratum outcome, not a crashed job.
        # This matters when the runner shards by case/scenario/seed/budget.
        (out/'empty_group.json').write_text(json.dumps({
            'status':'no_actionable_state','groups':groups,
            'coverage':coverage},ensure_ascii=False,indent=2))
        print(f'[EMPTY] groups={groups}; no legal candidate state; see coverage.csv',flush=True)
        return
    write_csv(out/'details.csv',records)
    write_csv(out/'fixed_state_policy.csv',policy_records)
    summary=[]
    for sc in ['ALL','shock','repeated','small']:
        group=[r for r in records if sc=='ALL' or r['scenario']==sc]
        if not group:continue
        summary.append(dict(scenario=sc,n_action_pairs=len(group),
            n_states=len(set((r['case'],r['scenario'],r['eta_seed'],r['budget'],r['event_index']) for r in group)),
            mc_reference_mae=mean(abs(r['mc_reference_error']) for r in group),
            mc_reference_ci_coverage=mean(r['mc_reference_in_ci'] for r in group),
            mc_reference_sign_agreement=mean(r['mc_reference_sign_agree'] for r in group),
            synthetic_future_mae=mean(abs(r['synthetic_frozen_error']) for r in group),
            h0_synthetic_future_mae=mean(abs(r['h0_synthetic_frozen_error']) for r in group),
            incremental_lookahead_abs_error_improvement=mean(
                r['value_of_extra_lookahead_abs_error'] for r in group),
            realized_future_mae=mean(abs(r['realized_frozen_error']) for r in group),
            synthetic_sign_agreement=mean(r['synthetic_frozen_sign_agree'] for r in group),
            average_predicted_ci_width=mean(r['mpc_ci_high']-r['mpc_ci_low'] for r in group),
            mean_mpc_strength=mean(r['mpc_effective_strength'] for r in group)))
    write_csv(out/'summary.csv',summary)
    if policy_records:
        summaries=[]
        for mode in MODES:
            rr=[x for x in policy_records if x['mode']==mode]
            if not rr:continue
            summaries.append(dict(mode=mode,n_states=len(rr),
              mean_oracle_regret=mean(x['oracle_regret'] for x in rr),
              mean_realized_regret=mean(x['realized_regret'] for x in rr),
              oracle_action_accuracy=mean(x['matches_oracle'] for x in rr),
              realized_action_accuracy=mean(x['matches_realized_oracle'] for x in rr)))
        write_csv(out/'fixed_state_summary.csv',summaries)
    print(f'[DONE] groups={groups} action pairs={len(records)} => {out}',flush=True)


if __name__=='__main__':main()
