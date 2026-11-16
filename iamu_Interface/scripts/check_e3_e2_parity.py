#!/usr/bin/env python3
"""Check q_fixed==E2 q_joint and q_only==E2 set_ddqn on shared episodes."""
import argparse
import csv
from pathlib import Path

def load(p):
    with Path(p).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def key(r):
    return tuple(str(r[x]) for x in ('case','scenario','eta_seed','budget'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e2',default='results/v10_e2_multiseed_42/runs.csv')
    p.add_argument('--e3',default='results/v10_e3_fusion_seed42/runs.csv')
    args=p.parse_args()
    e2={}
    for r in load(args.e2):e2[(key(r),r['policy'])]=r
    n=0
    for r in load(args.e3):
        baseline={'q_only':'set_ddqn','q_fixed':'q_joint'}.get(r['policy'])
        if baseline is None:continue
        src=e2.get((key(r),baseline))
        if src is None:raise AssertionError(f'Missing E2 parity reference: {r}')
        for v in ('weighted_wait','spent'):
            if abs(float(r[v])-float(src[v]))>1e-6:
                raise AssertionError(f'PARITY FAILED {key(r)} {r["policy"]} vs {baseline} on {v}: {r[v]} != {src[v]}')
        n+=1
    if n==0:raise AssertionError('No matching E3 parity rows')
    print(f'[PASS] {n} matched policy episodes: E3 fixed/Q-only reproduce E2.')

if __name__=='__main__':main()
