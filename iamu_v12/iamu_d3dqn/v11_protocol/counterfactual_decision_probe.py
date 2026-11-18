#!/usr/bin/env python3
"""Paired realized-future Keep/Edit comparisons on a trusted decision bank.

NOT a conditional-expectation oracle. Do not feed the simulator snapshots into
an online policy; only public_observations.npz is suitable for model input.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from iamu_dynamic.rl_experiment_v11 import write_csv
from v11_protocol.experiment_fingerprint_v12 import digest_bytes


def rollout_remaining_cost(snapshot, first_action, continuation):
    env = copy.deepcopy(snapshot)
    initial_used = env.used
    initial_budget = env.budget
    initial_history_len = len(env.history)
    first = True
    total_future_reward = 0.
    while True:
        if first:
            action = first_action
            first = False
        else:
            action = 0 if continuation == 'keep' else env.point_greedy_action()
        if not bool(env._observation().mask[action]):
            raise ValueError('Branch selected illegal action')
        _, reward, done, info = env.step(action)
        total_future_reward += float(reward)
        if done: break
    assert env.used <= initial_budget
    assert len(env.history) > initial_history_len
    assert env.used >= initial_used
    if first_action != 0: assert env.used >= initial_used + 1
    # Only post-snapshot costs are counted; shared historical waiting cancels.
    return -total_future_reward * env.reward_scale, int(env.used)


def paired_realized_advantage(snapshot, edit_action, continuation='greedy'):
    if continuation not in ('greedy', 'keep'):
        raise ValueError('Unsupported continuation')
    if edit_action <= 0 or edit_action >= len(snapshot._observation().mask):
        raise ValueError('Invalid Edit index')
    if not bool(snapshot._observation().mask[edit_action]):
        raise ValueError('Edit is masked')
    before = (snapshot.used, len(snapshot.history), tuple(snapshot._candidate_ids),
              float(snapshot.total_raw_reward))
    keep_cost, keep_spent = rollout_remaining_cost(snapshot, 0, continuation)
    edit_cost, edit_spent = rollout_remaining_cost(snapshot, edit_action, continuation)
    after = (snapshot.used, len(snapshot.history), tuple(snapshot._candidate_ids),
             float(snapshot.total_raw_reward))
    if before != after:
        raise AssertionError('Counterfactual changed source snapshot')
    return dict(realized_gain=keep_cost-edit_cost, keep_future_wait=keep_cost,
                edit_future_wait=edit_cost, keep_final_spent=keep_spent,
                edit_final_spent=edit_spent)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', required=True)
    p.add_argument('--continuation', choices=('greedy','keep'), default='greedy')
    p.add_argument('--max-states', type=int, default=0, help='0 means all bank states')
    p.add_argument('--trust-local-snapshots', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if not args.trust_local_snapshots:
        p.error('Only locally trusted Python simulator snapshots may be loaded')
    if args.max_states < 0: p.error('max-states must be >= 0')
    bank = Path(args.bank)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): p.error('Use an empty output directory')
    if not (bank / 'complete.json').is_file(): p.error('Incomplete decision bank')
    meta = json.loads((bank / 'states.json').read_text())
    snapshots = torch.load(bank / 'realized_future_snapshots.pt', map_location='cpu', weights_only=False)
    if not meta: raise RuntimeError('Empty decision bank')
    if len(meta) != len(snapshots): raise RuntimeError('Unmatched bank metadata and simulator snapshots')
    selected = range(min(args.max_states or len(meta), len(meta)))
    rows = []
    for idx in selected:
        state, snap = meta[idx], snapshots[idx]
        obs = snap._observation()
        if not bool(obs.mask[1:].any()): raise ValueError('Non-actionable snapshot')
        if list(obs.candidate_ids) != state['candidate_ids']:
            raise AssertionError('Snapshot candidate identity differs from metadata')
        for action in range(1, len(obs.mask)):
            if not bool(obs.mask[action]): continue
            result = paired_realized_advantage(snap, action, args.continuation)
            rows.append(dict(state_id=state['state_id'], source_policy=state['source_policy'],
                             source_checkpoint=state['source_checkpoint'], case=state['case'],
                             scenario=state['scenario'], scenario_seed=state['scenario_seed'],
                             budget=state['budget'], event_index=state['event_index'],
                             candidate_index=action, candidate_id=int(obs.candidate_ids[action-1]),
                             reference_type='realized_future_diagnostic',
                             continuation_policy=args.continuation, **result))
    expected = sum(int(snapshots[i]._observation().mask[1:].sum()) for i in selected)
    if len(rows) != expected or len({(r['state_id'],r['candidate_index']) for r in rows}) != len(rows):
        raise AssertionError('Missing/duplicate paired counterfactual')
    out.mkdir(parents=True)
    write_csv(out / 'runs.csv', rows)
    summary = []
    for index in selected:
        subset = [r for r in rows if r['state_id'] == meta[index]['state_id']]
        if subset:
            gains = [0.] + [r['realized_gain'] for r in subset]
            summary.append(dict(state_id=meta[index]['state_id'], evaluated_edits=len(subset),
                                best_evaluated_edit_gain=max(gains),
                                beneficial_edits=sum(g>0 for g in gains[1:]),
                                max_wrong_edit_loss=max(max(0., -g) for g in gains[1:])))
    write_csv(out / 'summary.csv', summary)
    task_list = [meta[i]['state_id'] for i in selected]
    (out / 'task_list.json').write_text(json.dumps(task_list, indent=2))
    (out / 'manifest.json').write_text(json.dumps(dict(config=vars(args),
        reference_type='realized_future_diagnostic', continuation_policy=args.continuation,
        snapshot_sha256=digest_bytes((bank / 'realized_future_snapshots.pt').read_bytes())), indent=2))
    (out / 'complete.json').write_text(json.dumps(dict(expected=expected, actual=len(rows),
                                                        states=len(task_list))))
    print('Finished paired realized-future diagnostics:', expected, 'candidate comparisons')

if __name__ == '__main__': main()
