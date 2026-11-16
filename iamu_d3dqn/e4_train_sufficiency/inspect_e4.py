#!/usr/bin/env python3
"""Validate E4 checkpoint inventory; never select checkpoints using test cases.

Requires only Python stdlib for file/ledger/hash verification. Optional
--load-test imports the original V0.9 DDQN agent to confirm each checkpoint
can be restored for inference. Reports incomplete runs without inventing rows.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = 100


def digest(f:Path):
    sha=hashlib.sha256()
    with f.open('rb') as reader:
        for chunk in iter(lambda:reader.read(1024*1024),b''):
            sha.update(chunk)
    return sha.hexdigest()


def records(path:Path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path,entries):
    if not entries: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        cols=list(dict.fromkeys(k for row in entries for k in row))
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(entries)


def inspect_run(folder:Path, *,verify_hash:bool=True,load_test:bool=False):
    folder=Path(folder)
    result={'directory':str(folder),'complete':False,'episodes':0,
            'expected_checkpoints':0,'valid_checkpoints':0,'errors':[],
            'inventory':[]}
    manifest=folder/'manifest.json'
    if not manifest.is_file():
        result['errors'].append('missing manifest.json');return result
    try:
        args=json.loads(manifest.read_text(encoding='utf8'))['training_args']
        total=int(args['episodes']); interval=int(args['eval_every'])
        result['episodes']=total
        if interval!=STEPS or total%STEPS!=0:
            result['errors'].append('Not a 100-episode E4 schedule')
            return result
        result['expected_checkpoints']=total//STEPS
        valcsv=folder/'validation_curve.csv'
        ledger_file=folder/'validation_ledger.json'
        if not valcsv.is_file() or not ledger_file.is_file():
            result['errors'].append('missing validation_curve or validation_ledger')
            return result
        scores={int(float(row['completed'])):row for row in records(valcsv)}
        ledger=json.loads(ledger_file.read_text(encoding='utf8'))['records']
        used=set()
        if load_test:
            from iamu_dynamic.rl_ddqn import DoubleDQNAgent
        for episode in range(STEPS,total+1,STEPS):
            vid=f'val_{episode:06d}'
            info=ledger.get(vid)
            val=scores.get(episode)
            entry={'episode':episode,'validation_id':vid,
                   'path':'','status':'missing','gradient_steps':'',
                   'mean_wait':'','evaluated_runs':'','sha256':'',
                   'load_test':'not_requested'}
            if info is None or val is None:
                result['errors'].append(f'{vid}: missing ledger or validation row')
                result['inventory'].append(entry)
                continue
            candidate=Path(info['snapshot'])
            # Server absolute path may differ from copying output elsewhere.
            if not candidate.is_file():
                candidate=folder/'validation_snapshots'/candidate.name
            entry['path']=str(candidate)
            entry['gradient_steps']=int(info['gradient_steps'])
            entry['mean_wait']=float(val['mean_wait'])
            entry['evaluated_runs']=int(info.get('evaluated_runs',0))
            entry['sha256']=info['snapshot_sha256']
            failures=[]
            if info.get('status')!='done':failures.append('validation not done')
            if not candidate.is_file():failures.append('checkpoint missing')
            if int(info.get('evaluated_runs',0))!=int(info.get('expected_runs',-1)):
                failures.append('validation run count mismatch')
            if int(val['gradient_steps'])!=int(info['gradient_steps']):
                failures.append('gradient step mismatch')
            if str(val['snapshot_sha256'])!=info['snapshot_sha256']:
                failures.append('CSV/ledger checkpoint SHA mismatch')
            if candidate.is_file() and verify_hash and digest(candidate)!=info['snapshot_sha256']:
                failures.append('checkpoint SHA invalid')
            if candidate.is_file() and load_test:
                try:
                    agent,meta=DoubleDQNAgent.from_checkpoint(str(candidate),device='cpu')
                    if agent.q_network!='set_ddqn':
                        failures.append('wrong network architecture')
                    if int(meta.get('selected_episode',episode))!=episode:
                        failures.append('loaded selected_episode mismatch')
                    entry['load_test']='passed' if not failures else 'failed'
                except Exception as ex:
                    failures.append('from_checkpoint: '+str(ex))
                    entry['load_test']='failed'
            entry['status']='valid' if not failures else '; '.join(failures)
            result['inventory'].append(entry)
            if failures:
                result['errors'].append(f'{vid}: '+', '.join(failures))
            else:
                result['valid_checkpoints']+=1
        performance=folder/'performance.json'
        if performance.is_file():
            perf=json.loads(performance.read_text(encoding='utf8'))
            if int(perf.get('completed_episodes',-1))!=total:
                result['errors'].append('completed_episodes != target')
            if int(perf.get('completed_validations',-1))!=total//STEPS:
                result['errors'].append('completed_validations != target')
        else:
            result['errors'].append('performance.json missing (training may be active)')
        result['complete']=not result['errors'] and result['valid_checkpoints']==result['expected_checkpoints']
    except Exception as ex:
        result['errors'].append(type(ex).__name__+': '+str(ex))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',default='v10_e4')
    p.add_argument('--conditions',default='a300,b900,c900,d900')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--load-test',action='store_true')
    p.add_argument('--skip-hash',action='store_true')
    p.add_argument('--out',default=None)
    args=p.parse_args()
    cond=args.conditions.split(',');seeds=[int(x) for x in args.seeds.split(',')]
    out=Path(args.out) if args.out else ROOT/'results'/f'{args.prefix}_inspection'
    if out.exists() and any(out.iterdir()):
        p.error(f'Choose a fresh output directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    all_rows=[];reports=[];summary=[]
    for condition in cond:
        for seed in seeds:
            folder=ROOT/'results'/f'{args.prefix}_{condition}_seed{seed}'
            r=inspect_run(folder,verify_hash=not args.skip_hash,load_test=args.load_test)
            reports.append({k:v for k,v in r.items() if k!='inventory'})
            for row in r['inventory']:
                all_rows.append({'condition':condition,'train_seed':seed,**row})
            print(f"[{condition} seed={seed}] {r['valid_checkpoints']}/{r['expected_checkpoints']} "
                  f"snapshots complete={r['complete']}")
            if r['errors']:
                print('  '+ '; '.join(r['errors'][:5]))
    write_csv(out/'checkpoint_inventory.csv',all_rows)
    grouped=defaultdict(list)
    for row in all_rows:
        if row['status']=='valid':grouped[(row['condition'],row['episode'])].append(row)
    for (condition,episode),rr in sorted(grouped.items()):
        scores=[x['mean_wait'] for x in rr]
        summary.append({'condition':condition,'episode':episode,
                        'n_training_seeds':len(scores),
                        'mean_wait':statistics.mean(scores),
                        'sample_std_wait':statistics.stdev(scores) if len(scores)>1 else None,
                        'worst_seed_wait':max(scores),
                        'best_seed_wait':min(scores)})
    write_csv(out/'cross_seed_learning_curve.csv',summary)
    (out/'inspection_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf8')
    bad=[r for r in reports if not r['complete']]
    print(f'[E4] {len(reports)-len(bad)}/{len(reports)} runs fully verified. CSV: {out}')
    return 1 if bad else 0


if __name__=='__main__':sys.exit(main())
