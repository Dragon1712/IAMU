#!/usr/bin/env python3
"""Batch Q(s,a) for the same public bank across checkpoint/margin choices."""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
from iamu_dynamic.rl_experiment_v11 import write_csv
from v11_protocol.experiment_fingerprint_v12 import digest_bytes


def inspect_q_array(values, mask, margin=0.):
    """Exact argmax tie handling: Keep is index 0, as in agent.act()."""
    if not bool(mask[0]) or not bool(mask[1:].any()):
        return None
    q = np.asarray(values, dtype=float).copy()
    q[~mask] = -np.inf
    edit_idx = int(np.argmax(q[1:]) + 1)
    gap = float(q[edit_idx] - q[0])
    scored = q.copy()
    scored[0] += margin
    chosen = int(np.argmax(scored))
    return dict(q_keep=float(q[0]), best_edit=edit_idx,
                q_best_edit=float(q[edit_idx]), best_qgap=gap,
                choose_edit_without_margin=bool(np.argmax(q) != 0),
                choose_edit_with_margin=bool(chosen != 0),
                chosen_action=chosen)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', required=True)
    p.add_argument('--checkpoint', action='append', required=True, help='name=path.pt')
    p.add_argument('--margins', default='0,0.05,0.1,0.15')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if a.batch_size < 1: p.error('Positive batch size required')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): p.error('Use an empty output directory')
    bank = Path(a.bank)
    if not (bank / 'complete.json').is_file(): p.error('Incomplete decision bank')
    metadata = json.loads((bank / 'states.json').read_text())
    blob = np.load(bank / 'public_observations.npz', allow_pickle=False)
    globals_, actions, masks = blob['globals'], blob['actions'], blob['mask']
    if not metadata: raise RuntimeError('Empty decision bank')
    if not len(metadata) == len(globals_) == len(actions) == len(masks):
        raise RuntimeError('Bank contains unequal data lengths')
    margins = [float(x) for x in a.margins.split(',')]
    if any(x < 0 or not np.isfinite(x) for x in margins): p.error('Invalid margin')
    model_rows, candidate_rows, jobs = [], [], []
    for spec in a.checkpoint:
        name, path = spec.split('=', 1)
        if not name or not Path(path).is_file(): raise ValueError(spec)
        agent, _ = DoubleDQNAgent.from_checkpoint(path, 'cpu')
        agent.online.eval()
        jobs.append(dict(name=name, checkpoint=path, sha256=digest_bytes(Path(path).read_bytes())))
        batches = []
        with torch.no_grad():
            for lo in range(0, len(metadata), a.batch_size):
                hi = lo + a.batch_size
                q = agent.online(torch.as_tensor(globals_[lo:hi], dtype=torch.float32),
                                 torch.as_tensor(actions[lo:hi], dtype=torch.float32),
                                 torch.as_tensor(masks[lo:hi], dtype=torch.bool))
                batches.append(q.cpu().numpy())
        all_q = np.concatenate(batches) if batches else np.empty(masks.shape, dtype=float)
        for i, meta in enumerate(metadata):
            for slot in np.flatnonzero(masks[i]):
                candidate_rows.append(dict(state_id=meta['state_id'], checkpoint=name,
                                           action_index=int(slot), candidate_id=(meta['candidate_ids'][slot-1]
                                           if slot else 'Keep'), q=float(all_q[i, slot])))
            for margin in margins:
                report = inspect_q_array(all_q[i], masks[i], margin)
                if report is None: raise RuntimeError('Bank includes forced Keep state')
                model_rows.append(dict(state_id=meta['state_id'], checkpoint=name, margin=margin,
                                       budget=meta['budget'], scenario=meta['scenario'],
                                       source_policy=meta['source_policy'],
                                       event_index=meta['event_index'], **report))
    if len({(r['state_id'], r['checkpoint'], r['margin']) for r in model_rows}) != len(model_rows):
        raise ValueError('Duplicate Q-gap row keys')
    out.mkdir(parents=True)
    write_csv(out / 'runs.csv', model_rows)
    write_csv(out / 'candidate_q.csv', candidate_rows)
    # Group means are diagnostic only; rows are not independent samples.
    summary = []
    for job in jobs:
        for margin in margins:
            subset = [x for x in model_rows if x['checkpoint'] == job['name'] and x['margin'] == margin]
            summary.append(dict(checkpoint=job['name'], margin=margin, states=len(subset),
                                actionable_keep=sum(x['chosen_action'] == 0 for x in subset),
                                mean_qgap=(float(np.mean([x['best_qgap'] for x in subset])) if subset else None)))
    write_csv(out / 'summary.csv', summary)
    # Explicit within-state changes; never mistake different policy trajectories
    # for identical observations. Naming convention: method_ep200/300/400.
    by_model_state_margin = {}
    for r in model_rows:
        match = re.fullmatch(r'(.+)_ep(200|300|400)', r['checkpoint'], flags=re.I)
        if match:
            by_model_state_margin[(match.group(1), r['state_id'], r['margin'],
                                   int(match.group(2)))] = r
    changes = []
    keys = sorted({(base, state, margin) for base,state,margin,ep in by_model_state_margin})
    for base, state, margin in keys:
        triplet = [by_model_state_margin.get((base,state,margin,ep)) for ep in (200,300,400)]
        if not all(triplet): continue
        p200,p300,p400 = triplet
        changes.append(dict(model=base, state_id=state, margin=margin,
            q_keep_delta_200_300=p300['q_keep']-p200['q_keep'],
            q_keep_delta_300_400=p400['q_keep']-p300['q_keep'],
            qgap_delta_200_300=p300['best_qgap']-p200['best_qgap'],
            qgap_delta_300_400=p400['best_qgap']-p300['best_qgap'],
            best_edit_change_200_300=p200['best_edit']!=p300['best_edit'],
            best_edit_change_300_400=p300['best_edit']!=p400['best_edit'],
            edit_keep_edit=bool(p200['chosen_action'] and not p300['chosen_action'] and
                                p400['chosen_action']),
            chosen_200=p200['chosen_action'], chosen_300=p300['chosen_action'],
            chosen_400=p400['chosen_action']))
    write_csv(out / 'checkpoint_changes.csv', changes)
    (out / 'task_list.json').write_text(json.dumps(jobs, indent=2))
    (out / 'manifest.json').write_text(json.dumps(dict(config=vars(a),
        bank_public_sha256=digest_bytes((bank / 'public_observations.npz').read_bytes()),
        bank_metadata_sha256=digest_bytes((bank / 'states.json').read_bytes()),
        checkpoints=jobs), indent=2))
    expected = len(metadata) * len(jobs) * len(margins)
    if len(model_rows) != expected: raise RuntimeError('Incomplete Q-gap matrix')
    (out / 'complete.json').write_text(json.dumps(dict(expected=expected, actual=len(model_rows))))
    print('Q-gap bank evaluated:', len(model_rows), 'checkpoint-margin-state rows')

if __name__ == '__main__': main()
