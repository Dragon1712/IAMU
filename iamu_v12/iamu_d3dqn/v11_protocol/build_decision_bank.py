#!/usr/bin/env python3
"""Build a stratified, fixed *actionable* observation bank from dev cases.

Public observation tensors are stored separately from trusted simulator snapshots
which contain realized futures exclusively for ex-post counterfactual evaluation.
A bank is descriptive of its chosen anchor-policy/stratification distribution,
NOT a sample from the natural deployment visitation distribution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from iamu_dynamic import generate_scenario, read_case
from iamu_dynamic.rl_env_v11 import IAMUEventEnv
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
from iamu_dynamic.rl_experiment_v11 import parse_ints, parse_names, write_csv
from v11_protocol.experiment_fingerprint_v12 import digest_bytes, digest_json


def parse_ckpts(values):
    result = {}
    for entry in values:
        if '=' not in entry: raise ValueError('Use --rl-checkpoint name=/path/to/weights.pt')
        name, path = entry.split('=', 1)
        if not name or not path or name in result or name in ('greedy', 'periodic'):
            raise ValueError(f'Duplicate/invalid checkpoint label: {entry}')
        if not Path(path).is_file(): raise FileNotFoundError(path)
        result[name] = str(Path(path).resolve())
    return result


def state_id(source, case, scenario, seed, budget, event, obs, env):
    """Stable identity includes plan/runtime so same feature vectors cannot alias."""
    token = hashlib.sha256()
    token.update(json.dumps([source, str(case), scenario, seed, budget, event],
                            ensure_ascii=False).encode())
    token.update(obs.global_features.tobytes())
    token.update(obs.action_features.tobytes())
    token.update(obs.mask.tobytes())
    token.update(str(sorted((i, a.start, a.pos, a.cranes, a.duration)
                             for i, a in env.plan.items())).encode())
    token.update(str(sorted((i, a.start, a.pos, a.cranes, a.duration)
                             for i, a in env.runtime.started.items())).encode())
    return token.hexdigest()[:24]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--seeds', default='400001:400003')
    p.add_argument('--scenarios', default='shock,repeated,small')
    p.add_argument('--budgets', default='2,5,10')
    p.add_argument('--rl-checkpoint', action='append', default=[])
    p.add_argument('--anchors', default='greedy,periodic')
    p.add_argument('--max-states', type=int, default=200)
    p.add_argument('--per-stratum', type=int, default=6)
    p.add_argument('--sample-seed', type=int, default=1701)
    p.add_argument('--candidate-limit', type=int, default=4)
    p.add_argument('--uncertainty', type=float, default=8.)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if a.max_states < 1 or a.per_stratum < 1: p.error('Invalid sampling counts')
    dest = Path(a.out)
    if dest.exists() and any(dest.iterdir()): p.error('Use a new empty output directory')
    ckpts = parse_ckpts(a.rl_checkpoint)
    anchors = parse_names(a.anchors)
    for anchor in anchors:
        if anchor not in ('greedy', 'periodic') and anchor not in ckpts:
            p.error(f'Unknown anchor {anchor}; specify --rl-checkpoint {anchor}=...')
    agents = {k: DoubleDQNAgent.from_checkpoint(v, 'cpu')[0] for k, v in ckpts.items()}
    for model in agents.values(): model.online.eval()
    rng = random.Random(a.sample_seed)
    group_states = defaultdict(list)
    task_list = []
    for case in a.cases:
        vessels, port = read_case(case)
        for scenario in parse_names(a.scenarios):
            for seed in parse_ints(a.seeds):
                future = generate_scenario(vessels, port, label=scenario, seed=seed,
                                           uncertainty=a.uncertainty, updates_per_vessel=a.updates)
                for budget in parse_ints(a.budgets):
                    for source in anchors:
                        task_list.append(dict(case=case, scenario=scenario, seed=seed,
                                              budget=budget, source_policy=source))
                        env = IAMUEventEnv(future, budget=budget, candidate_limit=a.candidate_limit,
                                           decision_clock='union', period=12., reward_scale=100.,
                                           candidate_screen='positive', min_gain=.1,
                                           budget_features='absolute', value_features='global')
                        obs = env.reset()
                        while True:
                            if bool(obs.mask[1:].any()):
                                phase = min(2, (3 * env.event_index) // max(1, len(env.events) + 1))
                                key = (source, budget, scenario, phase)
                                row = dict(state_id=state_id(source, case, scenario, seed, budget,
                                                             env.event_index, obs, env),
                                           source_policy=source, source_checkpoint=ckpts.get(source, ''),
                                           case=str(case), scenario=scenario, scenario_seed=seed,
                                           budget=budget, event_index=env.event_index,
                                           phase=phase, candidate_ids=list(obs.candidate_ids),
                                           n_legal_edits=int(obs.mask[1:].sum()),
                                           time=float(obs.time))
                                group_states[key].append((row, obs.copy(), copy.deepcopy(env)))
                            if source == 'greedy' or source == 'periodic':
                                action = 0 if source == 'periodic' and env.event_type != 'clock' else env.point_greedy_action()
                            else:
                                action = agents[source].act(obs, epsilon=0.)
                            obs, _, done, _ = env.step(action)
                            if done: break
    sampled = []
    # Per-stratum caps prevent abundant policy/phase groups dominating;
    # round-robin fills to cap without silently duplicating small groups.
    groups = {}
    for key, entries in sorted(group_states.items()):
        shuffled = list(entries)
        rng.shuffle(shuffled)
        groups[key] = shuffled[:a.per_stratum]
    while len(sampled) < a.max_states and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(sampled) < a.max_states:
                sampled.append(groups[key].pop())
    if not sampled:
        raise RuntimeError('No actionable states found; inspect input cases/anchors before creating bank')
    meta = [x[0] for x in sampled]
    if len({r['state_id'] for r in meta}) != len(meta):
        raise ValueError('Duplicate state IDs; inspect bank identity')
    dest.mkdir(parents=True)
    n = len(sampled)
    shapes = (24,), (a.candidate_limit + 1, 24), (a.candidate_limit + 1,)
    if n:
        obs_g = np.stack([x[1].global_features for x in sampled])
        obs_a = np.stack([x[1].action_features for x in sampled])
        obs_m = np.stack([x[1].mask for x in sampled])
    else:
        obs_g = np.empty((0, *shapes[0]), dtype=np.float32)
        obs_a = np.empty((0, *shapes[1]), dtype=np.float32)
        obs_m = np.empty((0, *shapes[2]), dtype=np.bool_)
    np.savez_compressed(dest / 'public_observations.npz', globals=obs_g, actions=obs_a, mask=obs_m)
    # Untrusted snapshots must NEVER be loaded by an online policy or model input.
    torch.save([x[2] for x in sampled], dest / 'realized_future_snapshots.pt')
    write_csv(dest / 'states.csv', [{**x, 'candidate_ids': json.dumps(x['candidate_ids'])} for x in meta])
    (dest / 'states.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (dest / 'task_list.json').write_text(json.dumps(task_list, indent=2))
    manifest = dict(config=vars(a), bank_type='stratified_actionable_v1',
                    reference_type='realized_future_diagnostic',
                    public_observations_sha256=digest_bytes((dest / 'public_observations.npz').read_bytes()),
                    snapshot_sha256=digest_bytes((dest / 'realized_future_snapshots.pt').read_bytes()),
                    task_list_sha256=digest_json(task_list),
                    checkpoint_sha256={k: digest_bytes(Path(v).read_bytes()) for k, v in ckpts.items()},
                    n_available_by_stratum={str(k): len(v) for k, v in group_states.items()},
                    n_selected=n)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    assert obs_g.shape[0] == obs_a.shape[0] == obs_m.shape[0] == len(meta) == n
    (dest / 'complete.json').write_text(json.dumps(dict(states=n, tasks=len(task_list), unique_ids=n)))
    print(f'Bank: {n} distinct actionable states, {len(task_list)} source trajectories')

if __name__ == '__main__': main()
