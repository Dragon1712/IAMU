#!/usr/bin/env python3
"""Strict held-out paired evaluation of the selected budgeted-DDQN checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import (parse_ints,parse_names,verify_cases,
                                         evaluate_paired)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--test-cases',nargs='+',required=True)
    p.add_argument('--seeds',default='700001:700020')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--methods',default='keep,greedy,periodic,expected,ddqn')
    p.add_argument('--risk-samples',type=int,default=16)
    p.add_argument('--forecast-std',type=float,default=8.)
    p.add_argument('--device',default='cpu')
    p.add_argument('--save-actions',action='store_true',
                   help='Save event-level decisions for diagnosis (large CSV)')
    p.add_argument('--out',default='results/v06_rl_test')
    args=p.parse_args()
    torch.set_num_threads(1)
    agent,config=DoubleDQNAgent.from_checkpoint(args.checkpoint,args.device)
    verify_cases(config['train_cases'],config['val_cases'],args.test_cases)
    seeds=parse_ints(args.seeds)
    if set(seeds)&set(config['val_seeds']):p.error('Test seeds overlap validation')
    first=int(config['train_seed_start'])
    stop=first+int(config['training_args']['episodes'])
    if set(seeds)&set(range(first,stop)):p.error('Test seeds overlap training')
    methods=parse_names(args.methods)
    if 'ddqn' not in methods or 'keep' not in methods:p.error('Methods must include ddqn and keep')
    if 'periodic' in methods and config['decision_clock']!='union':
        p.error('Periodic comparator requires a union clock')
    rows,summary=evaluate_paired(cases=args.test_cases,seeds=seeds,
        scenarios=parse_names(args.scenarios),budgets=parse_ints(args.budgets),
        policies=methods,agent=agent,candidate_limit=config['candidate_limit'],
        decision_clock=config['decision_clock'],period=config['period'],
        reward_scale=config['reward_scale'],uncertainty=args.uncertainty,
        updates=args.updates,risk_samples=args.risk_samples,
        forecast_std=args.forecast_std,out=args.out,
        candidate_screen=config.get('candidate_screen','none'),
        min_gain=config.get('min_gain',.1),save_actions=args.save_actions,
        budget_features=config.get('budget_features','absolute'),
        value_features=config.get('value_features','global'))
    out=Path(args.out)
    (out/'manifest.json').write_text(json.dumps({
        'checkpoint':str(args.checkpoint),'checkpoint_meta':config,
        'evaluation_args':vars(args),'n_runs':len(rows),
        'test_seeds':seeds,'test_cases':args.test_cases,
        'test_not_used_for_training_or_checkpoint_selection':True
    },indent=2),encoding='utf-8')
    print(f'Completed {len(rows)} paired evaluations: {out}')
    for entry in summary:
        print(f'B={entry["budget"]} {entry["method"]:9} '
              f'gain={entry["mean_gain"]:+.3f} median={entry["median_gain"]:+.3f} '
              f'bad={entry["degraded"]}/{entry["n"]} '
              f'CVaR90={entry["cvar90_regret"]:.3f} used={entry["mean_spent"]:.2f}')

if __name__=='__main__':main()
