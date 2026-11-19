#!/usr/bin/env python3
"""Re-evaluate a selected checkpoint on its *validation* set, never held-out test.

Separate from evaluate_budgeted_ddqn.py, which MUST reject test/validation overlap.
This diagnostic is for interpreting actions and tuning, not for OOS claims.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import evaluate_paired, parse_names, verify_cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--methods',default='keep,greedy,periodic,expected,ddqn')
    p.add_argument('--risk-samples',type=int,default=16)
    p.add_argument('--forecast-std',type=float,default=8.)
    p.add_argument('--device',default='cpu')
    p.add_argument('--save-actions',action='store_true')
    p.add_argument('--out',default='results/v07_validation_diagnostics')
    args=p.parse_args()
    torch.set_num_threads(1)
    agent,cfg=DoubleDQNAgent.from_checkpoint(args.checkpoint,args.device)
    verify_cases(cfg['train_cases'],cfg['val_cases'])
    methods=parse_names(args.methods)
    if 'ddqn' not in methods or 'keep' not in methods:
        p.error('Both keep and ddqn are required')
    if 'periodic' in methods and cfg['decision_clock']!='union':
        p.error('Periodic comparator requires union clock')
    rows,summary=evaluate_paired(
        cases=cfg['val_cases'],seeds=cfg['val_seeds'],
        scenarios=cfg['scenarios'],budgets=cfg['budgets'],policies=methods,
        agent=agent,candidate_limit=cfg['candidate_limit'],
        decision_clock=cfg['decision_clock'],period=cfg['period'],
        reward_scale=cfg['reward_scale'],uncertainty=8.,updates=cfg['updates'],
        risk_samples=args.risk_samples,forecast_std=args.forecast_std,
        candidate_screen=cfg.get('candidate_screen','none'),
        min_gain=cfg.get('min_gain',.1),
        budget_features=cfg.get('budget_features','absolute'),
        value_features=cfg.get('value_features','global'),
        save_actions=args.save_actions,out=args.out)
    out=Path(args.out)
    (out/'diagnostic_manifest.json').write_text(json.dumps({
        'split':'VALIDATION_NOT_HELDOUT_TEST',
        'checkpoint':args.checkpoint,'checkpoint_meta':cfg,
        'args':vars(args),'n_runs':len(rows)},indent=2),encoding='utf-8')
    for r in summary:
        print(f"B={r['budget']} {r['method']}: "
              f"gain={r['mean_gain']:+.3f} bad={r['degraded']}/{r['n']} "
              f"spent={r['mean_spent']:.2f} "
              f"last={r['mean_last_revision_time']:.2f}")


if __name__=='__main__':main()
