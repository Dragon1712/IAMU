#!/usr/bin/env python3
"""Train an event-driven, hard-budget DDQN; validation never updates weights.

Run from IAMU_dynamic root. Current code handles ONE ship revision per decision
step. It does not train or modify ERL-DDE or integrate shore-power emissions.
"""
import argparse
import json
import random
from collections import deque
from pathlib import Path
from statistics import mean

import torch

from iamu_dynamic import generate_scenario,read_case
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_ddqn import DoubleDQNAgent,ReplayBuffer
from iamu_dynamic.rl_experiment import (parse_ints,parse_names,verify_cases,
                                         evaluate_paired,write_csv)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cases',nargs='+',required=True)
    p.add_argument('--val-cases',nargs='+',required=True)
    p.add_argument('--episodes',type=int,default=2000)
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--train-seed-start',type=int,default=200000)
    p.add_argument('--val-seeds',default='400001:400010')
    p.add_argument('--uncertainties',default='4,8,12')
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--clock',choices=['eta','union','clock'],default='union')
    p.add_argument('--period',type=float,default=12.)
    p.add_argument('--candidate-limit',type=int,default=4)
    p.add_argument('--budget-features',choices=['legacy','absolute'],default='absolute')
    p.add_argument('--value-features',choices=['legacy','global'],default='global')
    p.add_argument('--candidate-screen',choices=['none','positive'],default='none',
                   help='Public-ETA point-gain heuristic guard; not true-risk safe')
    p.add_argument('--min-gain',type=float,default=.1)
    p.add_argument('--keep-margin',type=float,default=0.,
                   help='Q-space margin added to Keep before argmax, in reward-scale units')
    p.add_argument('--greedy-exploration-prob',type=float,default=0.,
                   help='Among epsilon exploration actions, choose public-ETA Greedy with this probability')
    p.add_argument('--reward-scale',type=float,default=100.)
    p.add_argument('--hidden-dim',type=int,default=96)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--replay-capacity',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--warmup',type=int,default=512)
    p.add_argument('--updates-per-step',type=int,default=1)
    p.add_argument('--target-update',type=int,default=250)
    p.add_argument('--epsilon-start',type=float,default=1.)
    p.add_argument('--epsilon-end',type=float,default=.05)
    p.add_argument('--epsilon-decay-episodes',type=int,default=1000)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='results/v06_budgeted_ddqn')
    args=p.parse_args()
    if args.episodes<1 or args.eval_every<1 or args.batch_size<1:
        p.error('Positive episode, eval and batch counts required')
    if (not 0<=args.greedy_exploration_prob<=1 or args.keep_margin<0 or
            args.min_gain<0):p.error('Invalid exploration, keep margin or minimum gain')
    if args.warmup<args.batch_size or args.epsilon_decay_episodes<1 or args.updates_per_step<0:
        p.error('Invalid replay/epsilon parameters')
    verify_cases(args.train_cases,args.val_cases)
    val_seeds=parse_ints(args.val_seeds)
    if set(val_seeds)&set(range(args.train_seed_start,args.train_seed_start+args.episodes)):
        p.error('Training and validation seeds must not overlap')
    scenarios=parse_names(args.scenarios)
    if not set(scenarios).issubset({'shock','repeated','small'}):p.error('Unknown scenario')
    budgets=parse_ints(args.budgets)
    if min(budgets)<0:p.error('Budgets must be nonnegative')
    uncertainties=[float(x) for x in parse_names(args.uncertainties)]
    if any(x<0 for x in uncertainties):p.error('Negative uncertainty')
    torch.set_num_threads(1)
    rnd=random.Random(args.seed)
    case_cache={x:read_case(x) for x in args.train_cases}
    agent=DoubleDQNAgent(hidden_dim=args.hidden_dim,learning_rate=args.lr,
           gamma=args.gamma,target_update_steps=args.target_update,
           seed=args.seed,device=args.device,keep_margin=args.keep_margin)
    replay=ReplayBuffer(args.replay_capacity,args.seed+100)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train_rows=[];val_rows=[];best_val=float('inf')
    config={'train_cases':args.train_cases,'val_cases':args.val_cases,
            'candidate_limit':args.candidate_limit,'decision_clock':args.clock,
            'period':args.period,'reward_scale':args.reward_scale,
            'candidate_screen':args.candidate_screen,'min_gain':args.min_gain,
            'budget_features':args.budget_features,'value_features':args.value_features,
            'scenarios':scenarios,'budgets':budgets,'uncertainties':uncertainties,
            'updates':args.updates,'train_seed_start':args.train_seed_start,
            'val_seeds':val_seeds,'training_args':vars(args)}
    (out/'manifest.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for ep in range(1,args.episodes+1):
        case=rnd.choice(args.train_cases)
        vessels,port=case_cache[case]
        label=rnd.choice(scenarios)
        uncertainty=rnd.choice(uncertainties)
        budget=rnd.choice(budgets)
        seed=args.train_seed_start+ep-1
        sc=generate_scenario(vessels,port,label,seed,args.updates,uncertainty)
        env=IAMUEventEnv(sc,budget=budget,candidate_limit=args.candidate_limit,
              decision_clock=args.clock,period=args.period,reward_scale=args.reward_scale,
              candidate_screen=args.candidate_screen,min_gain=args.min_gain,
              budget_features=args.budget_features,value_features=args.value_features)
        obs=env.reset();rewards=0.;losses=[];decisions=0
        frac=min(1.,(ep-1)/max(1,args.epsilon_decay_episodes))
        epsilon=args.epsilon_start+(args.epsilon_end-args.epsilon_start)*frac
        while True:
            teacher=(env.point_greedy_action() if args.greedy_exploration_prob>0 else None)
            action=agent.act(obs,epsilon=epsilon,teacher_action=teacher,
                  teacher_prob=args.greedy_exploration_prob)
            nxt,reward,done,info=env.step(action)
            replay.append(obs,action,reward,nxt,done)
            rewards+=reward;decisions+=1
            if len(replay)>=args.warmup:
                for _ in range(args.updates_per_step):
                    losses.append(agent.update(replay.sample(args.batch_size)))
            obs=nxt
            if done:break
        train_rows.append(dict(episode=ep,case=case,scenario=label,seed=seed,
            uncertainty=uncertainty,budget=budget,spent=info['spent'],
            weighted_wait=info['weighted_wait'],reward_sum_scaled=rewards,
            steps=decisions,loss=(mean(losses) if losses else ''),epsilon=epsilon,
            replay_size=len(replay),gradient_steps=agent.gradient_steps,
            screened_proposals=env.screened_proposals,
            last_revision_time=(env.last_revision_time or 0.),
            mean_estimated_edit_gain=(mean(x['estimated_gain'] for x in env.action_audit
                if x['action']!=0) if env.used else 0.)))
        if ep%args.eval_every==0 or ep==args.episodes:
            # No test instance/seed ever occurs in this validation loop.
            policies=['keep','greedy','ddqn']
            if args.clock=='union':policies.append('periodic')
            _,summary=evaluate_paired(cases=args.val_cases,seeds=val_seeds,
                scenarios=scenarios,budgets=budgets,policies=policies,
                agent=agent,candidate_limit=args.candidate_limit,
                decision_clock=args.clock,period=args.period,
                reward_scale=args.reward_scale,uncertainty=8.,updates=args.updates,
                candidate_screen=args.candidate_screen,min_gain=args.min_gain,
                budget_features=args.budget_features,value_features=args.value_features)
            rl=[x for x in summary if x['method']=='ddqn']
            score=mean(x['mean_wait'] for x in rl)
            val_rows.append(dict(episode=ep,mean_wait=score,
                mean_gain=mean(x['mean_gain'] for x in rl),
                worst_gain=min(x['worst_gain'] for x in rl),
                mean_spent=mean(x['mean_spent'] for x in rl)))
            if score<best_val:
                best_val=score
                torch.save(agent.checkpoint({**config,'selected_episode':ep,
                               'selection_score':score}),out/'best.pt')
            print(f'[val] ep={ep} wait={score:.3f} gain={val_rows[-1]["mean_gain"]:.3f} '
                  f'best={best_val:.3f} replay={len(replay)} grad={agent.gradient_steps}',flush=True)
            write_csv(out/'train_curve.csv',train_rows)
            write_csv(out/'validation_curve.csv',val_rows)
    print(f'Finished {args.episodes} episodes; best checkpoint at {out/"best.pt"}',flush=True)

if __name__=='__main__':main()
