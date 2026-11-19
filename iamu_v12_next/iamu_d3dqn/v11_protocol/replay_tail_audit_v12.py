#!/usr/bin/env python3
"""Offline raw-vs-collapse accounting from a locally trusted EP200 full fork.

Does not train a network or claim to measure counterfactual policy improvement.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from v11_protocol.research_v12 import collapse_exhaustion_tail,verify_collapse,class_counts
from iamu_dynamic.rl_experiment_v11 import write_csv

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork',required=True)
    p.add_argument('--trust-local-training-state',action='store_true')
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if not a.trust_local_training_state:p.error('Explicit trust required for serialized local Python Replay state')
    dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):p.error('Use fresh output directory')
    data=torch.load(a.fork,map_location='cpu',weights_only=False)
    replay=data['replay_items']; episodes=defaultdict(list)
    for tr in replay:episodes[tr.episode_id].append(tr)
    budgets={int(r['episode']):int(r['budget']) for r in data.get('train_rows',[])}
    rows=[]; skipped=[]
    for episode, trs in sorted(episodes.items()):
        # The historical V1.1 Transition lacked explicit budget_before/after.
        # Recover budgets ONLY from recorded per-episode budget and the complete
        # action sequence, never by decoding a floating point network feature.
        if not trs[-1].done or episode not in budgets or not (trs[0].reason_flags or {}).get('initial_event'):
            skipped.append(episode);continue
        original_budget=budgets[episode]
        remaining=original_budget
        reconstructed=[]
        for tr in trs:
            next_remaining=remaining-int(tr.action!=0)
            if next_remaining<0: raise AssertionError(f'Budget overspend in episode {episode}')
            flags=tr.reason_flags or {}
            if 'budget_exhausted' in flags and bool(flags['budget_exhausted'])!=(remaining==0):
                raise AssertionError(f'Budget label inconsistent with actions: episode {episode}')
            reconstructed.append(replace(tr,budget_before=remaining,budget_after=next_remaining,
                                         event_index=int(flags.get('event_index',0)),
                                         raw_steps=1,transition_version=2))
            remaining=next_remaining
        try:
            collapsed=collapse_exhaustion_tail(reconstructed,gamma=a.gamma)
            verify_collapse(reconstructed,collapsed,gamma=a.gamma)
        except ValueError as exc:
            skipped.append(episode);continue
        counts=class_counts(trs)
        rows.append(dict(episode=episode,raw_steps=len(trs),stored_steps=len(collapsed),
                         removed=len(trs)-len(collapsed),**counts))
    dest.mkdir(parents=True)
    write_csv(dest/'runs.csv',rows)
    summary=dict(total_episodes=len(episodes),verified=len(rows),skipped=len(skipped),
                 skipped_ids=skipped,original_steps=sum(r['raw_steps'] for r in rows),
                 compressed_steps=sum(r['stored_steps'] for r in rows),
                 caution='Physical waiting preserved, but Bellman targets and replay distribution change')
    write_csv(dest/'summary.csv',[{k:v for k,v in summary.items() if not isinstance(v,list)}])
    (dest/'task_list.json').write_text(json.dumps(sorted(episodes),indent=2))
    (dest/'manifest.json').write_text(json.dumps(dict(config=vars(a),fork_version=data.get('version')),indent=2))
    if summary['verified']+summary['skipped'] != summary['total_episodes']:
        raise RuntimeError('Partial episode accounting')
    (dest/'complete.json').write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=='__main__':main()
