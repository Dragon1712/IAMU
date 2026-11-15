#!/usr/bin/env python3
"""V1.0 E2: stability of MC ETA risk ranking and Q+Risk/Joint decisions.

Freeze observable decision states from the original Set-aware DDQN trajectory.
For each state: score candidates with independent CRN batches at N=16/64/128;
compare to an independent high-sample reference (default N=256).

NO test-case future ETA values or future notifications are used by the scorer.
The reference is a Monte Carlo approximation, NOT the true arrival distribution.
For --joint, MPC scores are held FIXED across sampling sizes, so the experiment
isolates risk sampling noise rather than conflating it with MPC sampling noise.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not (ROOT / 'iamu_dynamic' / 'core.py').is_file():
    raise SystemExit('Run python setup_interface.py --copy-assets first')

from iamu_dynamic import read_case, generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.v10_risk_features import (
    RiskSpecification, event_seed, public_eta_scenarios, evaluate_proposals,
)
from iamu_dynamic.v10_budget_lookahead import LookaheadSpecification, evaluate_multistage
from iamu_dynamic.v10_policy import (public_arrival_flags, frozen_q_values,
                                     choose_adjusted_action)
from eval_risk_budget_v10 import ints, write_csv, digest


def evaluate_state(env, obs, agent, *, sizes, repeats, reference_samples,
                   sigma, alpha, belief, beta_risk, joint,
                   beta_budget, mpc_samples, mpc_stages):
    candidates = {i: env._candidates[i-1] for i in range(1, len(obs.mask))
                  if bool(obs.mask[i])}
    if not candidates:
        return []
    q = frozen_q_values(agent, obs)
    base_seed = event_seed(env.scenario.seed, env.event_index, env.event_type)
    arrivals = public_arrival_flags(env)
    reference_spec = RiskSpecification(sigma=sigma, samples=reference_samples,
                                       alpha=alpha, belief=belief)
    # Independent reference batch from every tested replicate.
    ref_seed = base_seed ^ 0x65A41EA029E70567
    ref, _, _ = evaluate_proposals(env.runtime, env.plan, candidates,
                                    env.published_eta, env.event_time,
                                    arrivals, reference_spec, seed=ref_seed)
    mpc = None
    if joint:
        # Fixed independent MPC batch: joint action variation across N then
        # reflects ONLY changing MC risk estimates, not MPC sample changes.
        mpc_spec = RiskSpecification(sigma=sigma, samples=mpc_samples,
                                     alpha=alpha, belief=belief)
        mpc_draws = public_eta_scenarios(
            env.published_eta, env.event_time, arrivals, mpc_spec,
            seed=base_seed ^ 0xA0B37B5E19D45B61)
        mpc = evaluate_multistage(
            env, candidates, mpc_draws,
            LookaheadSpecification(stages=mpc_stages, alpha=alpha,
                                      min_gain=env.min_gain,
                                      candidates_per_stage=env.candidate_limit))
    def action_from(metrics):
        chosen, _, _ = choose_adjusted_action(
            q, obs.mask, margin=float(agent.keep_margin),
            reward_scale=env.reward_scale, risk_metrics=metrics,
            mpc_estimate=mpc, beta_risk=beta_risk,
            beta_budget=beta_budget if joint else 0.)
        return chosen
    ref_action = action_from(ref)
    ref_top_tail = min(ref, key=lambda a: (ref[a].tail_cost_delta, a))
    ref_top_mean = max(ref, key=lambda a: (ref[a].mean_gain, -a))
    rows = []
    for rep in range(repeats):
        batch_seed = base_seed + 1000003 * (rep + 1)
        # M is a nested prefix within each replicate; independent across reps.
        for n in sizes:
            specification = RiskSpecification(sigma=sigma, samples=n,
                                              alpha=alpha, belief=belief)
            start = perf_counter()
            estimate, _, _ = evaluate_proposals(
                env.runtime, env.plan, candidates,
                env.published_eta, env.event_time, arrivals,
                specification, seed=batch_seed)
            elapsed = perf_counter() - start
            selected = action_from(estimate)
            tail_top = min(estimate, key=lambda a: (estimate[a].tail_cost_delta, a))
            mean_top = max(estimate, key=lambda a: (estimate[a].mean_gain, -a))
            tail_err = [abs(estimate[a].tail_cost_delta - ref[a].tail_cost_delta)
                        for a in candidates]
            gain_err = [abs(estimate[a].mean_gain - ref[a].mean_gain)
                        for a in candidates]
            harm_err = [abs(estimate[a].harm_probability - ref[a].harm_probability)
                        for a in candidates]
            rows.append({
                'event_index': env.event_index, 'time': env.event_time,
                'event_type': env.event_type, 'remaining_budget': env.remaining_budget,
                'n_candidates': len(candidates), 'sample_count': n,
                'replicate': rep, 'reference_samples': reference_samples,
                'reference_action': ref_action, 'selected_action': selected,
                'action_agreement': int(selected == ref_action),
                'best_tail_agreement': int(tail_top == ref_top_tail),
                'best_mean_agreement': int(mean_top == ref_top_mean),
                'tail_mae': float(mean(tail_err)),
                'tail_max_error': float(max(tail_err)),
                'gain_mae': float(mean(gain_err)),
                'harm_prob_mae': float(mean(harm_err)),
                'risk_eval_seconds': elapsed,
                'decision_policy': 'q_joint_fixed_mpc' if joint else 'q_risk',
            })
    return rows


def summarize(rows):
    """Within-state repeatability and independent-reference agreement by N."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['decision_policy'], r['sample_count'], 'ALL')].append(r)
        groups[(r['decision_policy'], r['sample_count'], r['scenario'])].append(r)
    summary = []
    for (policy, n, scenario), group in sorted(groups.items()):
        per_state = defaultdict(list)
        for r in group:
            state_id = (r['case'],r['scenario'],r['eta_seed'],
                        r['budget'],r['event_index'])
            per_state[state_id].append(r['selected_action'])
        # 0: every replicate selects identical action in this state.
        # 2/3: all three replicates differ (maximum for 3 samples).
        flip_rates = []
        for decisions in per_state.values():
            counts = {v: decisions.count(v) for v in set(decisions)}
            flip_rates.append(1. - max(counts.values())/len(decisions))
        out = dict(policy=policy, sample_count=n, scenario=scenario,
                   n_state_replicates=len(group),
                   n_unique_states=len(per_state),
                   mean_within_state_action_flip_rate=mean(flip_rates),
                   n_states_with_action_disagreement=sum(x>0 for x in flip_rates))
        for k in ('action_agreement', 'best_tail_agreement', 'best_mean_agreement',
                  'tail_mae', 'tail_max_error', 'gain_mae', 'harm_prob_mae',
                  'risk_eval_seconds'):
            out[f'mean_{k}'] = mean(float(r[k]) for r in group)
        summary.append(out)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    bundled = ROOT / 'assets/checkpoints/set_ddqn_seed42'
    fallback = ROOT.parent / 'iamu_d3dqn/results/v09b_set_ddqn_seed42_300'
    base = bundled if (bundled / 'best.pt').is_file() else fallback
    cd = ROOT / 'assets/cases'
    cd = cd if (cd / '8.txt').exists() else ROOT.parent / '2024LiERLDDE/case_code'
    p.add_argument('--checkpoint', default=str(base / 'best.pt'))
    p.add_argument('--manifest', default=str(base / 'manifest.json'))
    p.add_argument('--cases', nargs='+', default=[str(cd / '8.txt'), str(cd / '9.txt')])
    p.add_argument('--scenarios', default='shock,repeated,small')
    p.add_argument('--seeds', default='400001:400003')
    p.add_argument('--budgets', default='2,5,10')
    p.add_argument('--sample-sizes', default='16,64,128')
    p.add_argument('--reference-samples', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--states-per-group', type=int, default=2)
    p.add_argument('--stride', type=int, default=10,
                   help='Retain every Nth eligible state on the DDQN trajectory')
    p.add_argument('--max-groups', type=int, default=0, help='0=all groups')
    p.add_argument('--sigma', type=float, default=7.12463033710398)
    p.add_argument('--alpha', type=float, default=.90)
    p.add_argument('--belief', choices=('censored','truncated'), default='censored')
    p.add_argument('--beta-risk', type=float, default=.25)
    p.add_argument('--joint', action='store_true',
                   help='Hold the multistage MPC term fixed and vary risk sampling')
    p.add_argument('--beta-budget', type=float, default=.25)
    p.add_argument('--mpc-stages', type=int, default=2)
    p.add_argument('--mpc-samples', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='results/v10_e2_risk_stability')
    args = p.parse_args(argv)
    sizes = ints(args.sample_sizes)
    if min(sizes) < 2 or sizes != sorted(sizes) or args.reference_samples < max(sizes):
        p.error('Use increasing sizes >=2 with independent reference >= maximum')
    if args.repeats < 2 or args.stride < 1 or args.states_per_group < 1 or args.max_groups < 0:
        p.error('Invalid repeats, stride, states-per-group or max-groups')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error('Output exists: choose a fresh --out')
    if not Path(args.checkpoint).is_file() or not Path(args.manifest).is_file():
        p.error('Missing checkpoint or manifest')
    train = json.loads(Path(args.manifest).read_text())['training_args']
    env_args = dict(candidate_limit=int(train['candidate_limit']),
                    decision_clock=train['clock'], period=float(train['period']),
                    reward_scale=float(train['reward_scale']),
                    min_gain=float(train['min_gain']), candidate_screen=train['candidate_screen'],
                    budget_features=train['budget_features'],value_features=train['value_features'])
    agent, _ = DoubleDQNAgent.from_checkpoint(args.checkpoint,device=args.device)
    if agent.q_network != 'set_ddqn':
        p.error('Must load Set-aware DDQN')
    cases = [Path(x) for x in args.cases]
    for f in cases:
        if not f.is_file():
            p.error('Missing case: '+str(f))
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps({
        'args': vars(args), 'checkpoint_sha256': digest(Path(args.checkpoint)),
        'manifest_sha256': digest(Path(args.manifest)),
        'source_sha256': {'risk':digest(ROOT/'iamu_dynamic/v10_risk_features.py'),
                          'policy':digest(ROOT/'iamu_dynamic/v10_policy.py'),
                          'mpc':digest(ROOT/'iamu_dynamic/v10_budget_lookahead.py'),
                          'stability':digest(Path(__file__))},
        'case_sha256':{str(f):digest(f) for f in cases},
        'warning': 'Reference MC is an independent finite-sample approximation, not truth.'
    }, indent=2,ensure_ascii=False))
    details = []
    groups_done = 0
    for case in cases:
        vessels,port=read_case(case)
        for scenario_name in args.scenarios.split(','):
            for eta_seed in ints(args.seeds):
                instance=generate_scenario(vessels,port,label=scenario_name,seed=eta_seed,
                    updates_per_vessel=int(train['updates']),
                    uncertainty=float(train['validation_uncertainty']))
                for budget in ints(args.budgets):
                    if args.max_groups and groups_done >= args.max_groups:
                        break
                    env=IAMUEventEnv(instance,budget=budget,**env_args)
                    obs=env.reset()
                    eligible_seen=0
                    saved=0
                    while True:
                        if bool(obs.mask[1:].any()):
                            eligible_seen += 1
                            if eligible_seen % args.stride == 0 and saved < args.states_per_group:
                                rows=evaluate_state(
                                    env, obs, agent, sizes=sizes, repeats=args.repeats,
                                    reference_samples=args.reference_samples,
                                    sigma=args.sigma,alpha=args.alpha,belief=args.belief,
                                    beta_risk=args.beta_risk,joint=args.joint,
                                    beta_budget=args.beta_budget,
                                    mpc_samples=args.mpc_samples,mpc_stages=args.mpc_stages)
                                for r in rows:
                                    details.append(dict(case=case.stem,scenario=scenario_name,
                                                        eta_seed=eta_seed,budget=budget,**r))
                                saved += 1
                        # Follow frozen original model, not the experimental policy.
                        action=int(agent.act(obs,epsilon=0.))
                        obs,_,done,_=env.step(action)
                        if done:
                            break
                    groups_done += 1
                    write_csv(out/'stability_details.partial.csv',details)
                    print(f'[stability] groups={groups_done} case={case.stem}'
                          f' {scenario_name} ETA={eta_seed} B={budget} states={saved}',flush=True)
                if args.max_groups and groups_done >= args.max_groups:
                    break
            if args.max_groups and groups_done >= args.max_groups:
                break
        if args.max_groups and groups_done >= args.max_groups:
            break
    if not details:
        raise RuntimeError('No actionable states selected; reduce --stride')
    write_csv(out/'stability_details.csv',details)
    write_csv(out/'stability_summary.csv',summarize(details))
    print(f'[DONE] states={len(details)//(len(sizes)*args.repeats)},'
          f' rows={len(details)} => {out}',flush=True)

if __name__ == '__main__':
    main()
