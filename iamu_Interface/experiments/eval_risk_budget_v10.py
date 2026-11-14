#!/usr/bin/env python3
"""Paired frozen Set-aware DDQN risk/budget evaluation on V0.9 event environment.

No training, no future ETA leakage in policy inputs, no change to core or RL.
The script intentionally defaults to the *validation* split Cases 8,9.
Do not tune on held-out Cases 10-14.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict
from statistics import mean
from time import perf_counter

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if not (HERE / "iamu_dynamic" / "core.py").is_file():
    raise SystemExit(
        "The V0.9 engine snapshot is not installed. "
        "Run: python setup_interface.py [--copy-assets] from iamu_Interface/"
    )
from iamu_dynamic import read_case, generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import run_episode as original_episode
from iamu_dynamic.risk_selection import upper_cvar
from iamu_dynamic.v10_risk_features import RiskSpecification
from iamu_dynamic.v10_budget_lookahead import LookaheadSpecification
from iamu_dynamic.v10_policy import V10Policy, PolicySpecification, POLICIES
from iamu_dynamic.v10_candidate_api import SingleMoveProvider, FrozenERLDDEHintProvider
from iamu_dynamic.v10_candidate_integration import install_candidate_source


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def ints(text: str) -> list[int]:
    out = []
    for entry in text.split(','):
        if ':' in entry:
            a, b = map(int, entry.split(':'))
            if b < a:
                raise ValueError('Backwards range')
            out.extend(range(a, b+1))
        else:
            out.append(int(entry))
    if len(out) != len(set(out)):
        raise ValueError('Duplicate values')
    return out


def evaluate_once(scenario, budget, agent, name, spec, env_args,
                  *, collect_audit, provider=None):
    env = IAMUEventEnv(scenario, budget=budget, **env_args)
    observation = env.reset()
    policy = V10Policy(name, agent, spec)
    audit = []
    start = perf_counter()
    while True:
        was_budget = env.remaining_budget
        t, typ, event_id = env.event_time, env.event_type, env.event_index
        install_candidate_source(env, provider)
        observation = env._observation()
        action, extra = policy.choose(env, observation)
        if not isinstance(action, int) or not 0 <= action < len(observation.mask) or not observation.mask[action]:
            raise AssertionError(f'Invalid V1.0 action {name}: {action}')
        selected_vessel = int(env._candidate_ids[action-1]) if action else None
        observation, reward, done, info = env.step(action)
        if collect_audit:
            audit.append({"policy": name, "time": t, "event_type": typ,
                          "event_index": event_id, "budget_before": was_budget,
                          "budget_after": env.remaining_budget, "action": action,
                          "vessel_id": selected_vessel, **extra})
        if done:
            break
    if env.used > budget or info['spent'] != budget - env.remaining_budget:
        raise AssertionError('Hard budget violation')
    return {"weighted_wait": info['weighted_wait'], "spent": info['spent'],
            "elapsed_sec": perf_counter()-start, "decisions": len(env.history),
            "edits_eta": sum(x[3] is not None and x[0] == 'eta' for x in env.history),
            "edits_clock": sum(x[3] is not None and x[0] == 'clock' for x in env.history)}, audit


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        dims = [("ALL", "ALL"), (r['scenario'], "ALL"),
                ("ALL", str(r['budget'])), (r['scenario'], str(r['budget']))]
        for scenario, budget in dims:
            by[(r['policy'], scenario, budget)].append(r)
    result = []
    for (policy, scenario, budget), group in sorted(by.items()):
        reg = [max(0., -x['gain_vs_periodic']) for x in group]
        result.append({"policy": policy, "scenario": scenario, "budget": budget,
                       "n": len(group), "mean_wait": mean(x['weighted_wait'] for x in group),
                       "mean_spent": mean(x['spent'] for x in group),
                       "mean_gain_vs_periodic": mean(x['gain_vs_periodic'] for x in group),
                       "mean_gain_vs_set_ddqn": mean(x['gain_vs_set_ddqn'] for x in group),
                       "cvar90_regret_vs_periodic": upper_cvar(reg, .9),
                       "mean_runtime_seconds": mean(x['elapsed_sec'] for x in group)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    bundled_model = HERE / 'assets' / 'checkpoints' / 'set_ddqn_seed42'
    original_model = HERE.parent / 'iamu_d3dqn' / 'results' / 'v09b_set_ddqn_seed42_300'
    default_model = bundled_model if (bundled_model / 'best.pt').is_file() else original_model
    bundled_cases = HERE / 'assets' / 'cases'
    original_cases = HERE.parent / '2024LiERLDDE' / 'case_code'
    case_dir = bundled_cases if (bundled_cases / '8.txt').is_file() and (bundled_cases / '9.txt').is_file() else original_cases
    p.add_argument('--checkpoint', default=str(default_model / 'best.pt'))
    p.add_argument('--manifest', default=str(default_model / 'manifest.json'))
    p.add_argument('--cases', nargs='+', default=[str(case_dir/'8.txt'),str(case_dir/'9.txt')])
    p.add_argument('--scenarios', default='shock,repeated,small')
    p.add_argument('--seeds', default='400001:400003')
    p.add_argument('--budgets', default='2,5,10')
    p.add_argument('--policies', default='keep,periodic,greedy,set_ddqn,q_risk,q_budget,q_joint,expected,lookahead')
    p.add_argument('--sigma', type=float, default=7.12463033710398)
    p.add_argument('--belief', choices=['censored','truncated'], default='censored')
    p.add_argument('--risk-samples', type=int, default=16)
    p.add_argument('--risk-alpha', type=float, default=.9)
    p.add_argument('--beta-risk', type=float, default=.25)
    p.add_argument('--beta-budget', type=float, default=.10)
    p.add_argument('--lookahead-stages', type=int, default=1)
    p.add_argument('--lookahead-step-hours', type=float, default=12.)
    p.add_argument('--lookahead-risk-weight', type=float, default=0.)
    p.add_argument('--device', default='cpu')
    p.add_argument('--candidate-provider', choices=['single','erldde_hint'], default='single')
    p.add_argument('--erldde-repo', default=None, help='Local original ERL-DDE directory with model.py')
    p.add_argument('--erldde-checkpoint', default=None, help='Trusted frozen model state_dict')
    p.add_argument('--erldde-probe', default=None, help='Compiled read-only C++ probe .so')
    p.add_argument('--smoke', action='store_true', help='One case/scenario/seed/budget')
    p.add_argument('--verify-baseline', action='store_true')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--out', default=str(HERE / 'results' / 'v10_e1_val_seed42'))
    args = p.parse_args()
    if not Path(args.checkpoint).is_file():
        p.error('Checkpoint missing: '+args.checkpoint)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    cfg = manifest['training_args']
    agent, check_meta = DoubleDQNAgent.from_checkpoint(args.checkpoint, device=args.device)
    if agent.q_network != 'set_ddqn':
        p.error('V1.0 prototype requires the frozen Set-aware DDQN checkpoint')
    env_args = dict(
        candidate_limit=int(cfg['candidate_limit']),
        decision_clock=cfg['clock'], period=float(cfg['period']),
        reward_scale=float(cfg['reward_scale']),
        min_gain=float(cfg['min_gain']), candidate_screen=cfg['candidate_screen'],
        budget_features=cfg['budget_features'],value_features=cfg['value_features'])
    spec = PolicySpecification(
        risk=RiskSpecification(args.sigma,args.risk_samples,args.risk_alpha,args.belief),
        lookahead=LookaheadSpecification(
            stages=args.lookahead_stages,step_hours=args.lookahead_step_hours,
            risk_weight=args.lookahead_risk_weight, alpha=args.risk_alpha,
            min_gain=env_args['min_gain'],candidates_per_stage=env_args['candidate_limit']),
        beta_risk=args.beta_risk,beta_budget=args.beta_budget)
    def case_file(raw):
        f = Path(raw)
        if f.is_file():
            return str(f.resolve())
        original_iamu_d3dqn = Path(args.checkpoint).resolve().parents[2]
        fallback = original_iamu_d3dqn / raw
        return str(fallback.resolve()) if fallback.is_file() else str(f)

    cases = [case_file(c) for c in (args.cases[:1] if args.smoke else args.cases)]
    scenarios = args.scenarios.split(',')
    seeds = ints(args.seeds)
    budgets = ints(args.budgets)
    if args.smoke:
        scenarios, seeds, budgets = scenarios[:1], seeds[:1], budgets[:1]
    requested = args.policies.split(',')
    if len(requested) != len(set(requested)) or not set(requested).issubset(POLICIES):
        p.error('Invalid or duplicate policies')
    # Set baseline and periodic are ALWAYS evaluated for paired metrics.
    names = list(dict.fromkeys([*requested, 'keep', 'periodic', 'set_ddqn']))
    for case in cases:
        if not Path(case).is_file():
            p.error('Case file missing: '+case)
    if args.candidate_provider == 'erldde_hint':
        if not (args.erldde_repo and args.erldde_checkpoint and args.erldde_probe):
            p.error('ERL-DDE hint mode requires --erldde-repo/checkpoint/probe')
        from iamu_dynamic.legacy_erldde import FrozenERLDDEHints
        hints = FrozenERLDDEHints(args.erldde_repo, args.erldde_checkpoint,
                                  args.erldde_probe)
        provider = FrozenERLDDEHintProvider(hints)
    else:
        provider = SingleMoveProvider()
    if args.verify_baseline and args.candidate_provider != 'single':
        p.error('--verify-baseline only valid with original single source')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error('Output directory not empty, choose a new --out')
    try:
        git_head = subprocess.check_output(['git','rev-parse','HEAD'],
                                           cwd=HERE,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        git_head = None
    implementation = [
        'iamu_dynamic/v10_risk_features.py',
        'iamu_dynamic/v10_budget_lookahead.py',
        'iamu_dynamic/v10_policy.py',
        'iamu_dynamic/v10_candidate_api.py',
        'iamu_dynamic/v10_candidate_integration.py',
        'experiments/eval_risk_budget_v10.py',
    ]
    rootinfo = {"arguments":vars(args),"v09_training_parameters":env_args,
                "git_head":git_head,
                "dependency_snapshot_sha256": (digest(HERE / 'dependency_snapshot.json')
                                               if (HERE / 'dependency_snapshot.json').is_file() else None),
                "implementation_sha256":{name:digest(HERE/name) for name in implementation},
                "training_seed":cfg['seed'],"checkpoint_sha256":digest(Path(args.checkpoint)),
                "manifest_sha256":digest(Path(args.manifest)),
                "cases_sha256":{p:digest(Path(p)) for p in cases},
                "risk_specification":asdict(spec.risk),
                "lookahead_specification":asdict(spec.lookahead),
                "important":"Validation-only; do not tune on independent test Cases 10-14"}
    if args.dry_run:
        print(json.dumps(rootinfo, ensure_ascii=False, indent=2)); return
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(rootinfo,indent=2,ensure_ascii=False))
    rows, audits = [], []
    for case in cases:
        vessels, port = read_case(case)
        for scenario_name in scenarios:
            for seed in seeds:
                scenario = generate_scenario(vessels,port,label=scenario_name,seed=seed,
                    updates_per_vessel=int(cfg['updates']),
                    uncertainty=float(cfg['validation_uncertainty']))
                for budget in budgets:
                    grouped = []
                    for name in names:
                        result, audit = evaluate_once(
                            scenario,budget,agent,name,spec,env_args,collect_audit=args.audit,
                            provider=provider)
                        grouped.append({"case":Path(case).stem,"scenario":scenario_name,
                                        "eta_seed":seed,"budget":budget,"policy":name,**result})
                        for x in audit:
                            audits.append({"case":Path(case).stem,"scenario":scenario_name,
                                           "eta_seed":seed,"budget":budget,**x})
                    ref = {r['policy']:r for r in grouped}
                    for r in grouped:
                        r['gain_vs_periodic'] = (ref['periodic']['weighted_wait'] - r['weighted_wait'])
                        r['gain_vs_keep'] = (ref['keep']['weighted_wait'] - r['weighted_wait'])
                        r['gain_vs_set_ddqn'] = (ref['set_ddqn']['weighted_wait'] - r['weighted_wait'])
                        rows.append(r)
                    if args.verify_baseline:
                        old = original_episode(scenario,budget,'ddqn',agent=agent,
                            candidate_limit=env_args['candidate_limit'],
                            decision_clock=env_args['decision_clock'],period=env_args['period'],
                            reward_scale=env_args['reward_scale'],
                            candidate_screen=env_args['candidate_screen'],
                            min_gain=env_args['min_gain'],
                            budget_features=env_args['budget_features'],
                            value_features=env_args['value_features'])
                        same = ref['set_ddqn']
                        if abs(old['wait']-same['weighted_wait']) > 1e-6 or old['spent'] != same['spent']:
                            raise AssertionError(f'V0.9 frozen baseline mismatch: {case} {scenario_name} seed={seed} B={budget}')
                    # Durable group-by-group progress; full-run outputs are
                    # committed only after every requested group completes.
                    write_csv(out/'runs.partial.csv', rows)
                    print(f'[v10] case={Path(case).stem} scenario={scenario_name} seed={seed} B={budget}', flush=True)
    write_csv(out/'runs.csv',rows)
    write_csv(out/'paired_summary.csv',summarize(rows))
    if args.audit:
        write_csv(out/'decision_audit.csv',audits)
    print(f'V1.0 complete: {len(rows)} policy trajectories; {out}',flush=True)


if __name__=='__main__':
    main()
