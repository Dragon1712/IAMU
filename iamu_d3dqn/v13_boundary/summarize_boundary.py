"""Aggregate within-state tapes first; bootstrap over source-episode clusters."""
import argparse,csv,random
from collections import defaultdict
from pathlib import Path
from statistics import mean,median
from v13_diagnostics.common import write_csv,atomic_json,digest

def read(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def summarise(rows,boots=2000,seed=34):
    tape_groups=defaultdict(list)
    for r in rows:tape_groups[(r['model'],r['continuation'],r['scenario'],r['state_id'])].append(r)
    states=[]
    for (model,cont,scenario,sid),rr in tape_groups.items():
        row=dict(model=model,continuation=cont,scenario=scenario,state_id=sid,
                 episode_id=rr[0]['episode_id'],n_tapes=len(rr))
        for k in ('keep_bootstrap_error','edit_bootstrap_error','predicted_gap_error',
                  'return_gap','target_gap','keep_cost_understatement','identity_residual',
                  'keep_interval_length','edit_interval_length'):
            row[k]=mean(float(r[k]) for r in rr)
        row['q_gap']=float(rr[0]['predicted_gap'])
        states.append(row)
    buckets=defaultdict(list)
    for row in states:
        for scenario in (row['scenario'],'ALL'):
            buckets[(row['model'],row['continuation'],scenario)].append(row)
    output=[];rng=random.Random(seed)
    for (model,cont,scenario),rs in sorted(buckets.items()):
        episodes=defaultdict(list)
        for r in rs:episodes[r['episode_id']].append(r)
        cluster_keys=list(episodes)
        for metric in ('keep_bootstrap_error','edit_bootstrap_error','predicted_gap_error',
                       'return_gap','target_gap','keep_cost_understatement','q_gap'):
            values=[r[metric] for r in rs];ci=[float('nan'),float('nan')]
            if boots>0 and len(cluster_keys)>1:
                bs=[]
                for _ in range(boots):
                    chosen=[episodes[rng.choice(cluster_keys)] for _ in cluster_keys]
                    bs.append(mean(y[metric] for group in chosen for y in group))
                bs.sort();ci=[bs[int(.025*(boots-1))],bs[int(.975*(boots-1))]]
            output.append(dict(model=model,continuation=cont,scenario=scenario,metric=metric,
               n_states=len(rs),n_episode_clusters=len(episodes),mean=mean(values),
               median=median(values),ci95_low=ci[0],ci95_high=ci[1]))
    return states,output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args(argv)
    rows=[r for path in a.pairs for r in read(path)];seen=set()
    for r in rows:
        key=(r['state_id'],r['model'],r['continuation'],r['tape_index'])
        if key in seen:raise ValueError(f'Duplicate pair {key}')
        seen.add(key)
    if not rows:raise ValueError('Empty paired audit')
    states,groups=summarise(rows,a.bootstrap)
    dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'budget_boundary_states.csv',states)
    write_csv(dest/'budget_boundary_summary.csv',groups)
    atomic_json(dest/'summary.json',dict(n_states=len(states),n_summary_rows=len(groups),
        input_sha256={p:digest(p) for p in a.pairs},
        method='state-level average across tapes; source-episode cluster bootstrap',
        caveat='Synthetic conditional futures, not a calibrated posterior'))
    print('[boundary] summarised',len(states),'state-model-continuation rows')
if __name__=='__main__':main()
