#!/usr/bin/env python3
"""Frozen-model execution-margin scan; never changes bootstrap training targets."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
from iamu_dynamic.rl_experiment_v11 import evaluate_paired,parse_ints,parse_names,write_csv

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--margins',default='0,0.05,0.10,0.15')
    p.add_argument('--seeds',default='400001:400003')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--candidate-screen',choices=('positive','none','mixed'),default='positive')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv);dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):p.error('Use new --out')
    dest.mkdir(parents=True,exist_ok=True)
    ag,meta=DoubleDQNAgent.from_checkpoint(a.checkpoint,'cpu');rows=[];summaries=[]
    for margin in [float(x) for x in a.margins.split(',')]:
        if margin<0:p.error('Negative margin')
        ag.execution_margin=margin;ag.keep_margin=margin
        rr,ss=evaluate_paired(cases=a.cases,scenarios=parse_names(a.scenarios),
            seeds=parse_ints(a.seeds),budgets=parse_ints(a.budgets),
            agent=ag,policies=['keep','greedy','ddqn','periodic'],candidate_limit=4,
            decision_clock='union',period=12.,reward_scale=100.,uncertainty=8.,updates=2,
            candidate_screen=a.candidate_screen,min_gain=.1,budget_features='absolute',value_features='global')
        rows.extend(dict(margin=margin,**r) for r in rr)
        summaries.extend(dict(margin=margin,**r) for r in ss)
    write_csv(dest/'runs.csv',rows);write_csv(dest/'summary.csv',summaries)
    (dest/'manifest.json').write_text(json.dumps(dict(checkpoint=a.checkpoint,
        model_meta=meta,argv=vars(a),note='frozen weights; execution-margin scan only; new v11 candidate space is migration diagnosis'),default=str,indent=2))
    print('[DONE]',dest)
if __name__=='__main__':main()
