#!/usr/bin/env python3
"""CPU-parallel identical validation for online and offline E4-E snapshots.

Validates original online run parity; every model scored on same 54 public
scenario/ETA/finite budget episodes. No testing on held-out Case 10–14.
"""
from __future__ import annotations
import argparse,csv,json,os,sys,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from statistics import mean
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from iamu_dynamic import generate_scenario,read_case
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import run_episode
from e4_mechanism.telemetry import sha,append_csv


_AGENT_CACHE={};_CASE_CACHE={}

def _do_group(task):
    case,scenario,eta_seed,budget,models,env_kwargs,updates,uncertainty=task
    if case not in _CASE_CACHE:_CASE_CACHE[case]=read_case(case)
    vessels,port=_CASE_CACHE[case]
    instance=generate_scenario(vessels,port,label=scenario,seed=eta_seed,
        updates_per_vessel=updates,uncertainty=uncertainty)
    rows=[]
    for m in models:
        path=m['path']
        if path not in _AGENT_CACHE:
            agent,meta=DoubleDQNAgent.from_checkpoint(path,device='cpu')
            if agent.q_network!='set_ddqn':raise AssertionError('Wrong architecture')
            _AGENT_CACHE[path]=agent
        val=run_episode(instance,budget,'ddqn',agent=_AGENT_CACHE[path],**env_kwargs)
        rows.append(dict(model=m['model'],phase=m['phase'],episode=m['episode'],
             case=Path(case).stem,scenario=scenario,eta_seed=eta_seed,
             budget=budget,weighted_wait=val['wait'],spent=val['spent'],
             runtime_seconds=val['runtime_seconds']))
    return rows


def csv_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def collect_models(online,offline_dirs,episodes):
    ledger=json.loads((online/'validation_ledger.json').read_text(encoding='utf-8'))['records']
    models=[]
    for ep in episodes:
        record=ledger.get(f'val_{ep:06d}')
        if record is None or record['status']!='done':raise ValueError(f'Online missing validated ep{ep}')
        p=online/'validation_snapshots'/Path(record['snapshot']).name
        if not p.exists() or sha(p)!=record['snapshot_sha256']:
            raise ValueError(f'Online snapshot missing/corrupt at ep{ep}')
        models.append(dict(phase='online',model=f'online_ep{ep}',episode=ep,path=str(p)))
    for folder in offline_dirs:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        rows=csv_rows(folder/'checkpoints.csv')
        mode=manifest['args']['mode']
        for ep in episodes:
            matches=[x for x in rows if int(x['episode'])==ep]
            if len(matches)!=1:raise ValueError(f'Offline {mode} missing ep{ep}')
            row=matches[0];p=Path(row['checkpoint'])
            if not p.is_file():p=folder/'checkpoints'/p.name
            if sha(p)!=row['sha256']:raise ValueError(f'Offline snapshot bad SHA {p}')
            models.append(dict(phase=mode,model=f'{mode}_ep{ep}',episode=ep,path=str(p)))
    return models


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--online',required=True,help='Instrumented online training run directory')
    p.add_argument('--offline',nargs='*',default=[],help='One or more E5 offline output folders')
    p.add_argument('--episodes',default='200,300,400')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    online=Path(a.online)
    mf=json.loads((online/'manifest.json').read_text(encoding='utf-8'))
    cfg=mf['training_args']
    episodes=list(map(int,a.episodes.split(',')))
    if len(set(episodes))!=len(episodes):p.error('Duplicate episodes')
    off=[Path(x) for x in a.offline]
    models=collect_models(online,off,episodes)
    if a.workers<1:p.error('workers must be >=1')
    env_kwargs=dict(candidate_limit=int(cfg['candidate_limit']),
        decision_clock=cfg['clock'],period=float(cfg['period']),
        reward_scale=float(cfg['reward_scale']),candidate_screen=cfg['candidate_screen'],
        min_gain=float(cfg['min_gain']),budget_features=cfg['budget_features'],
        value_features=cfg['value_features'])
    scenarios=mf['scenarios'];budgets=mf['budgets'];seeds=mf['val_seeds'];cases=mf['val_cases']
    if a.smoke:
        scenarios=scenarios[:1];budgets=budgets[:1];seeds=seeds[:1];cases=cases[:1]
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Choose empty --out')
    out.mkdir(parents=True,exist_ok=True)
    # Validate test grid exactly matches the original ONLINE run manifest.
    plan=dict(args=vars(a),online=mf['training_args'],
              models=[{k:v for k,v in m.items() if k!='path'}|{'sha256':sha(m['path'])} for m in models],
              scientific_note='Same V0.9 validation Case8/9; includes ORIGINAL simulator; never model selection on Cases10-14')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2,default=str))
    # One group per physical future; models reuse physical scenario constructed once.
    jobs=[(str(case),sc,int(seed),int(b),models,env_kwargs,
           int(cfg['updates']),float(cfg['validation_uncertainty']))
          for case in cases for sc in scenarios for seed in seeds for b in budgets]
    all_rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(_do_group,x) for x in jobs]
        for i,fut in enumerate(as_completed(futures),1):
            all_rows.extend(fut.result())
            print(f'[E5-EVAL] group={i}/{len(jobs)}',flush=True)
    all_rows.sort(key=lambda x:(x['case'],x['scenario'],x['eta_seed'],x['budget'],x['phase'],x['episode']))
    # Baselines are precomputed E5 ONLINE validations, never created anew.
    reference={}
    for ep in episodes:
        original=online/'validation_details'/f'val_{ep:06d}'/'runs.csv'
        for r in csv_rows(original):
            k=(ep,Path(r['case']).stem,r['scenario'],int(r['seed']),int(r['budget']))
            reference.setdefault(k,{})[r['method']]=r
    for r in all_rows:
        k=(r['episode'],r['case'],r['scenario'],r['eta_seed'],r['budget'])
        ref=reference[k]
        if r['phase']=='online':
            if abs(r['weighted_wait']-float(ref['ddqn']['wait']))>1e-5 or r['spent']!=int(ref['ddqn']['spent']):
                raise AssertionError(f'ONLINE VALIDATION PARITY FAILED: {k}')
        r['gain_vs_keep']=float(ref['keep']['wait'])-r['weighted_wait']
        if 'periodic' in ref:
            r['gain_vs_periodic']=float(ref['periodic']['wait'])-r['weighted_wait']
        r['regret_vs_keep']=max(0.,-r['gain_vs_keep'])
    # Use direct CSV util for stable ordering; no duplicate keys.
    from iamu_dynamic.rl_experiment import write_csv
    write_csv(out/'runs.csv',all_rows)
    summary=[]
    for m in models:
        for budget in ['ALL',*sorted(set(r['budget'] for r in all_rows))]:
            z=[r for r in all_rows if r['model']==m['model'] and (budget=='ALL' or r['budget']==budget)]
            if not z:continue
            summary.append(dict(model=m['model'],episode=m['episode'],phase=m['phase'],budget=budget,
                n=len(z),mean_wait=mean(r['weighted_wait'] for r in z),
                mean_spent=mean(r['spent'] for r in z),
                mean_gain_vs_keep=mean(r['gain_vs_keep'] for r in z),
                mean_gain_vs_periodic=mean(r['gain_vs_periodic'] for r in z) if 'gain_vs_periodic' in z[0] else None))
    write_csv(out/'summary.csv',summary)
    (out/'complete.json').write_text(json.dumps(dict(groups=len(jobs),trajectories=len(all_rows),
        online_parity='passed',models=len(models)),indent=2))
    print(f'[DONE] models={len(models)}, trajectories={len(all_rows)}: {out}')

if __name__=='__main__':main()
