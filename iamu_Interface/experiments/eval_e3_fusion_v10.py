#!/usr/bin/env python3
"""V1.0-E3-B: paired online comparison of SAME-estimator fusion policies.

All modes use the SAME frozen Set-aware checkpoint, public ETA belief, CRN
sampling rule, current candidates and 2-stage MPC at each visited state.
Policies may visit different states after decisions diverge; do not interpret
an entire trajectory's MPC samples as the same posterior after divergence.

Never tune on held-out Cases 10-14. This script intentionally defaults to 8/9.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
if str(ROOT/'experiments') not in sys.path:
    sys.path.insert(0,str(ROOT/'experiments'))
if not (ROOT/'iamu_dynamic/core.py').is_file():
    raise SystemExit('First run: python setup_interface.py --source ../iamu_d3dqn --copy-assets')
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import run_episode as original_episode
from iamu_dynamic.v10_policy import PolicySpecification
from iamu_dynamic.v10_budget_lookahead import LookaheadSpecification
from iamu_dynamic.v10_risk_features import RiskSpecification
from iamu_dynamic.v10_e3_policy import E3Policy,E3Specification
from iamu_dynamic.v10_e3_confidence import MODES
from eval_risk_budget_v10 import ints,write_csv,digest,summarize


def run_one(scenario,budget,agent,mode,env_args,spec,*,audit=False):
    env=IAMUEventEnv(scenario,budget=budget,**env_args)
    obs=env.reset()
    policy=E3Policy(mode,agent,spec)
    logs=[]
    candidates_log=[]
    diagnostics=dict(eligible_states=0,eligible_edit_actions=0,
        agreement_count=0,conflict_count=0,confidence_gate_off_count=0,
        effective_weight_sum=0., action_diff_vs_q=0,
        total_mpc_seconds=0.,total_risk_seconds=0.)
    begin=perf_counter()
    while True:
        now=env.event_time
        event=env.event_index
        budget_before=env.remaining_budget
        original_action=(int(agent.act(obs,epsilon=0.))
                         if bool(obs.mask[1:].any()) and mode!='q_only' else None)
        action,extra=policy.choose(env,obs)
        if not isinstance(action,int) or action<0 or action>=len(obs.mask) or not bool(obs.mask[action]):
            raise AssertionError(f'Illegal action {action} at {event}')
        chosen_vessel=env._candidate_ids[action-1] if action else None
        all_details=extra.pop('_all_actions',{})
        # action 已单独记录，避免 **extra 重复传入
        extra.pop('action', None)
        if all_details:
            diagnostics['eligible_states']+=1
            diagnostics['eligible_edit_actions']+=len(all_details)
            diagnostics['agreement_count']+=sum(int(v['agree']) for v in all_details.values())
            diagnostics['conflict_count']+=sum(not v['agree'] for v in all_details.values())
            diagnostics['confidence_gate_off_count']+=sum(v['mpc_strength']<=1e-12 for v in all_details.values())
            diagnostics['effective_weight_sum']+=sum(v['effective_weight'] for v in all_details.values())
        if original_action is not None and action!=original_action:
            diagnostics['action_diff_vs_q']+=1
        diagnostics['total_mpc_seconds']+=float(extra.get('mpc_seconds',0.))
        diagnostics['total_risk_seconds']+=float(extra.get('risk_seconds',0.))
        if audit:
            logs.append(dict(event_index=event,event_time=now,
                             event_type=env.event_type,budget_before=budget_before,
                             action=action,changed_vessel=chosen_vessel,**extra))
            for a,d in all_details.items():
                candidates_log.append(dict(event_index=event,event_time=now,
                                           budget_before=budget_before,action_id=a,
                                           chosen=int(a==action),**d))
        obs,reward,done,info=env.step(action)
        if done:
            break
    if env.used>budget:
        raise AssertionError('Hard budget violation')
    nc=diagnostics['eligible_edit_actions']
    return dict(weighted_wait=float(info['weighted_wait']),spent=int(env.used),
                mean_effective_weight=diagnostics['effective_weight_sum']/max(1,nc),
                conflicting_fraction=diagnostics['conflict_count']/max(1,nc),
                unconfident_fraction=diagnostics['confidence_gate_off_count']/max(1,nc),
                eligible_states=diagnostics['eligible_states'],
                eligible_edit_actions=nc,
                action_changes_vs_q=diagnostics['action_diff_vs_q'],
                mpc_seconds=diagnostics['total_mpc_seconds'],
                risk_seconds=diagnostics['total_risk_seconds'],
                elapsed_sec=perf_counter()-begin,decisions=len(env.history),
                edits_eta=sum(x[3] is not None and x[0]=='eta' for x in env.history),
                edits_clock=sum(x[3] is not None and x[0]=='clock' for x in env.history)),logs,candidates_log


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    bundled=ROOT/'assets/checkpoints/set_ddqn_seed42'
    fallback=ROOT.parent/'iamu_d3dqn/results/v09b_set_ddqn_seed42_300'
    model=bundled if (bundled/'best.pt').is_file() else fallback
    cases=ROOT/'assets/cases'
    case_dir=cases if (cases/'8.txt').is_file() else ROOT.parent/'2024LiERLDDE/case_code'
    p.add_argument('--checkpoint',default=str(model/'best.pt'))
    p.add_argument('--manifest',default=str(model/'manifest.json'))
    p.add_argument('--cases',nargs='+',default=[str(case_dir/'8.txt'),str(case_dir/'9.txt')])
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--seeds',default='400001:400003')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--modes',default='q_only,q_risk_only,pure_mpc,pure_mpc_risk,q_fixed,q_confidence,q_agreement,q_adaptive')
    p.add_argument('--beta-budget',type=float,default=.25)
    p.add_argument('--beta-risk',type=float,default=.25)
    p.add_argument('--neutral-q',type=float,default=0.,help='Q/MPC conflict neutrality tolerance in Q units')
    p.add_argument('--risk-samples',type=int,default=16)
    p.add_argument('--mpc-samples',type=int,default=8)
    p.add_argument('--lookahead-stages',type=int,default=2)
    p.add_argument('--lookahead-step-hours',type=float,default=12.)
    p.add_argument('--sigma',type=float,default=7.12463033710398)
    p.add_argument('--belief',choices=('censored','truncated'),default='censored')
    p.add_argument('--risk-alpha',type=float,default=.90)
    p.add_argument('--device',default='cpu')
    p.add_argument('--verify-baseline',action='store_true')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--out',default='results/v10_e3_fusion_seed42')
    a=p.parse_args(argv)
    if a.mpc_samples<2 or a.mpc_samples>a.risk_samples:
        p.error('Requires 2<=mpc-samples<=risk-samples')
    modes=a.modes.split(',')
    if len(set(modes))!=len(modes) or any(x not in MODES for x in modes):
        p.error('Invalid or repeated --modes')
    modes=list(dict.fromkeys(['q_only',*modes]))
    for x in [a.checkpoint,a.manifest,*a.cases]:
        if not Path(x).is_file():
            p.error(f'Missing input: {x}')
    train=json.loads(Path(a.manifest).read_text(encoding='utf-8'))['training_args']
    agent,metadata=DoubleDQNAgent.from_checkpoint(a.checkpoint,device=a.device)
    if agent.q_network!='set_ddqn':
        p.error('Expected frozen Set-aware DDQN checkpoint')
    env_args=dict(candidate_limit=int(train['candidate_limit']),
                  decision_clock=train['clock'],period=float(train['period']),
                  reward_scale=float(train['reward_scale']),
                  min_gain=float(train['min_gain']),candidate_screen=train['candidate_screen'],
                  budget_features=train['budget_features'],value_features=train['value_features'])
    base=PolicySpecification(
        risk=RiskSpecification(a.sigma,a.risk_samples,a.risk_alpha,a.belief),
        lookahead=LookaheadSpecification(stages=a.lookahead_stages,
            step_hours=a.lookahead_step_hours,alpha=a.risk_alpha,
            min_gain=env_args['min_gain'],candidates_per_stage=env_args['candidate_limit']),
        beta_risk=a.beta_risk,beta_budget=a.beta_budget,mpc_samples=a.mpc_samples)
    spec=E3Specification(base,neutral_q=a.neutral_q)
    scenarios=a.scenarios.split(',');seeds=ints(a.seeds);budgets=ints(a.budgets);paths=[Path(x) for x in a.cases]
    if a.smoke:
        scenarios=scenarios[:1];seeds=seeds[:1];budgets=budgets[:1];paths=paths[:1]
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error('Output nonempty; choose distinct --out')
    files=['iamu_dynamic/v10_e3_confidence.py','iamu_dynamic/v10_e3_policy.py',
           'iamu_dynamic/v10_budget_lookahead.py','iamu_dynamic/v10_policy.py',
           'experiments/eval_e3_fusion_v10.py']
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,
                          stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):head=None
    meta=dict(args=vars(a),modes=modes,git_head=head,
              implementation_sha256={f:digest(ROOT/f) for f in files},
              checkpoint_sha256=digest(Path(a.checkpoint)),
              manifest_sha256=digest(Path(a.manifest)),
              case_sha256={str(f):digest(f) for f in paths},
              training_seed=train['seed'],env_args=env_args,
              model_spec=asdict(spec),
              interpretation='Matched state-estimators; divergent policy paths have different future states. Validation-only.')
    if a.dry_run:
        print(json.dumps(meta,indent=2,ensure_ascii=False));return
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    rows=[]
    for case in paths:
        vessels,port=read_case(case)
        for label in scenarios:
            for seed in seeds:
                episode=generate_scenario(vessels,port,label=label,seed=seed,
                    updates_per_vessel=int(train['updates']),
                    uncertainty=float(train['validation_uncertainty']))
                for b in budgets:
                    group=[]
                    for mode in modes:
                        r,decision_audit,details=run_one(episode,b,agent,mode,env_args,spec,
                                                         audit=a.audit)
                        group.append(dict(case=case.stem,scenario=label,eta_seed=seed,
                                          budget=b,policy=mode,**r))
                        if a.audit:
                            # Separate files by policy and group to bound memory usage.
                            prefix=f'{case.stem}_{label}_{seed}_B{b}_{mode}'
                            write_csv(out/'audit'/f'{prefix}.csv',decision_audit)
                            write_csv(out/'action_audit'/f'{prefix}.csv',details)
                    ref={r['policy']:r for r in group}
                    if a.verify_baseline:
                        base_val=original_episode(episode,b,'ddqn',agent=agent,
                            candidate_limit=env_args['candidate_limit'],
                            decision_clock=env_args['decision_clock'],period=env_args['period'],
                            reward_scale=env_args['reward_scale'],
                            candidate_screen=env_args['candidate_screen'],min_gain=env_args['min_gain'],
                            budget_features=env_args['budget_features'],value_features=env_args['value_features'])
                        if abs(base_val['wait']-ref['q_only']['weighted_wait'])>1e-6 or base_val['spent']!=ref['q_only']['spent']:
                            raise AssertionError('Frozen V0.9 baseline mismatch')
                    for r in group:
                        r['gain_vs_q_only']=ref['q_only']['weighted_wait']-r['weighted_wait']
                        r['gain_vs_q_fixed']=(ref['q_fixed']['weighted_wait']-r['weighted_wait']
                                            if 'q_fixed' in ref else None)
                        rows.append(r)
                    write_csv(out/'runs.partial.csv',rows)
                    print(f'[E3B] case={case.stem} {label} ETA={seed} B={b} policies={len(modes)}',flush=True)
    write_csv(out/'runs.csv',rows)
    # Native E3 grouped mean and tail relative to q_only, to avoid accidentally
    # using E2's special Periodic-policy summaries.
    from collections import defaultdict
    from statistics import mean
    from iamu_dynamic.risk_selection import upper_cvar
    groups=defaultdict(list)
    for r in rows:
        for sc,B in [('ALL','ALL'),(r['scenario'],'ALL'),('ALL',r['budget']),
                     (r['scenario'],r['budget'])]:
            groups[(r['policy'],sc,B)].append(r)
    summary=[]
    for (mode,sc,B),rr in sorted(groups.items(),key=lambda x:tuple(map(str,x[0]))):
        summary.append(dict(policy=mode,scenario=sc,budget=B,n=len(rr),
            mean_wait=mean(t['weighted_wait'] for t in rr),
            mean_spent=mean(t['spent'] for t in rr),
            mean_gain_vs_q_only=mean(t['gain_vs_q_only'] for t in rr),
            mean_runtime_sec=mean(t['elapsed_sec'] for t in rr),
            mean_mpc_seconds=mean(t['mpc_seconds'] for t in rr),
            mean_effective_weight=mean(t['mean_effective_weight'] for t in rr),
            mean_conflicting_fraction=mean(t['conflicting_fraction'] for t in rr),
            mean_unconfident_fraction=mean(t['unconfident_fraction'] for t in rr),
            mean_action_changes_vs_q=mean(t['action_changes_vs_q'] for t in rr),
            cvar90_negative_regret_vs_q_only=upper_cvar(
                [max(0.,-t['gain_vs_q_only']) for t in rr],.9)))
    write_csv(out/'paired_summary.csv',summary)
    print(f'[DONE] {len(rows)} trajectories => {out}',flush=True)


if __name__=='__main__':main()
