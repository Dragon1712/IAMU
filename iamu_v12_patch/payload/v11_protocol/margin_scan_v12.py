#!/usr/bin/env python3
"""Multi-checkpoint frozen execution-margin scan with baseline reuse."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from iamu_dynamic import generate_scenario, read_case
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
from iamu_dynamic.rl_experiment_v11 import parse_ints, parse_names, run_episode, write_csv
from v11_protocol.experiment_fingerprint_v12 import digest_bytes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, help='name=/path/checkpoint.pt')
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--scenarios', default='shock,repeated,small')
    p.add_argument('--seeds', default='400001:400003')
    p.add_argument('--budgets', default='2,5,10')
    p.add_argument('--margins', default='0,0.05,0.1,0.15')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    dest = Path(args.out)
    if dest.exists() and any(dest.iterdir()): p.error('Use a fresh output directory')
    margins = [float(m) for m in args.margins.split(',')]
    if any(m < 0 for m in margins): p.error('Negative execution margin')
    checkpoints = {}
    for spec in args.checkpoint:
        key, path = spec.split('=', 1)
        if not key or key in checkpoints or not Path(path).is_file(): p.error(f'Invalid checkpoint {spec}')
        checkpoints[key] = path
    # Run baselines just once each across all checkpoint/margin choices.
    tasks, baseline_rows, rl_rows = [], [], []
    for path in args.cases:
        vessels, port = read_case(path)
        for scenario in parse_names(args.scenarios):
            for seed in parse_ints(args.seeds):
                instance = generate_scenario(vessels, port, label=scenario, seed=seed,
                                             updates_per_vessel=2, uncertainty=8.)
                for budget in parse_ints(args.budgets):
                    tasks.append((path, scenario, seed, budget, instance))
                    for method in ('keep','greedy','periodic'):
                        result = run_episode(instance, budget, method,
                                             candidate_limit=4, decision_clock='union', period=12.,
                                             reward_scale=100., candidate_screen='positive',
                                             min_gain=.1, budget_features='absolute', value_features='global',
                                             return_audit=True)
                        measures, audit = result
                        actionable = [r for r in audit if r['candidate_has_retained_edit']]
                        baseline_rows.append(dict(case=path, scenario=scenario, seed=seed, budget=budget,
                                                  model='baseline', method=method, margin=None,
                                                  actionable_keep=sum(r['action']==0 for r in actionable),
                                                  actionable_states=len(actionable), **measures))
    for label, path in checkpoints.items():
        agent,_ = DoubleDQNAgent.from_checkpoint(path, 'cpu')
        for margin in margins:
            agent.execution_margin = margin
            agent.keep_margin = margin
            for case, scenario, seed, budget, instance in tasks:
                measures, audit = run_episode(instance, budget, 'ddqn', agent=agent,
                    candidate_limit=4, decision_clock='union', period=12., reward_scale=100.,
                    candidate_screen='positive', min_gain=.1, budget_features='absolute',
                    value_features='global', return_audit=True)
                actionable = [r for r in audit if r['candidate_has_retained_edit']]
                rl_rows.append(dict(case=case, scenario=scenario, seed=seed, budget=budget,
                                    model=label, method='ddqn', margin=margin,
                                    actionable_keep=sum(r['action']==0 for r in actionable),
                                    actionable_states=len(actionable), **measures))
    lookup = {(x['case'], x['scenario'], x['seed'], x['budget'], x['method']): x['wait']
              for x in baseline_rows}
    for row in rl_rows:
        key = (row['case'], row['scenario'], row['seed'], row['budget'])
        row['diff_vs_greedy'] = row['wait'] - lookup[(*key, 'greedy')]
        row['diff_vs_periodic'] = row['wait'] - lookup[(*key, 'periodic')]
        row['tail_worse_vs_periodic'] = max(0., row['diff_vs_periodic'])
        row['terminal_unused_budget'] = row['budget'] - row['spent']
    summary = []
    for label in checkpoints:
        for margin in margins:
            subset = [r for r in rl_rows if r['model']==label and r['margin']==margin]
            summary.append(dict(model=label, margin=margin, n=len(subset),
                mean_wait=mean(r['wait'] for r in subset),
                paired_vs_greedy=mean(r['diff_vs_greedy'] for r in subset),
                paired_vs_periodic=mean(r['diff_vs_periodic'] for r in subset),
                mean_edits=mean(r['spent'] for r in subset),
                mean_unused=mean(r['terminal_unused_budget'] for r in subset),
                actionable_keep_rate=(sum(r['actionable_keep'] for r in subset) /
                    max(1, sum(r['actionable_states'] for r in subset)))))
    expected = len(tasks)*len(checkpoints)*len(margins)
    if len(rl_rows) != expected or len(baseline_rows) != 3*len(tasks):
        raise RuntimeError('Incomplete evaluation grid')
    if len({(r['case'],r['scenario'],r['seed'],r['budget'],r['model'],r['margin'])
            for r in rl_rows}) != len(rl_rows): raise RuntimeError('Duplicate RL evaluation key')
    dest.mkdir(parents=True)
    write_csv(dest/'runs.csv', baseline_rows+rl_rows)
    write_csv(dest/'summary.csv', summary)
    write_csv(dest/'paired_comparisons.csv', rl_rows)
    job_list = [dict(case=a, scenario=b, seed=c, budget=d) for a,b,c,d,_ in tasks]
    (dest/'task_list.json').write_text(json.dumps(job_list, indent=2))
    (dest/'manifest.json').write_text(json.dumps(dict(config=vars(args),
        checkpoints={k:digest_bytes(Path(v).read_bytes()) for k,v in checkpoints.items()},
        baselines_evaluated_once=True), indent=2))
    (dest/'complete.json').write_text(json.dumps(dict(expected_rl=expected, actual_rl=len(rl_rows),
                                                          baselines=len(baseline_rows))))
    # Figures are derived entirely from runs/summary, never filled with
    # predetermined expected results. Matplotlib is an optional dependency.
    try:
        import matplotlib.pyplot as plt
        figdir=dest/'figures';figdir.mkdir(exist_ok=True)
        names=list(checkpoints)
        matrix=[[next(x['mean_wait'] for x in summary
                      if x['model']==label and x['margin']==margin)
                 for margin in margins] for label in names]
        fig,ax=plt.subplots(figsize=(max(6,len(margins)*1.3),max(3,len(names)*.7)))
        pic=ax.imshow(matrix,aspect='auto')
        ax.set_xticks(range(len(margins)),[str(x) for x in margins])
        ax.set_yticks(range(len(names)),names)
        ax.set_xlabel('execution margin');ax.set_ylabel('checkpoint')
        ax.set_title('Mean waiting (frozen checkpoint × margin)')
        fig.colorbar(pic,ax=ax,label='Mean waiting')
        fig.tight_layout();fig.savefig(figdir/'checkpoint_margin_heatmap.png',dpi=180)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.5))
        for model in names:
            points=[r for r in summary if r['model']==model]
            ax.plot([r['mean_edits'] for r in points],
                    [r['mean_wait'] for r in points],marker='o',label=model)
        ax.set_xlabel('Mean edits');ax.set_ylabel('Mean waiting')
        ax.set_title('Waiting–revision tradeoff (descriptive)')
        ax.legend(fontsize='small');fig.tight_layout()
        fig.savefig(figdir/'waiting_edits.png',dpi=180);plt.close(fig)
    except ImportError:
        print('[v12 margin] matplotlib unavailable: CSV results complete, figures skipped')
    print(f'Finished {expected} frozen RL evaluations and {len(baseline_rows)} baseline evaluations')

if __name__ == '__main__': main()
