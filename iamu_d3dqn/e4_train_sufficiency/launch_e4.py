#!/usr/bin/env python3
"""V1.0-E4: independent Set-aware DDQN training-length ablation.

Uses the EXISTING phase-synchronous V0.9 trainer unchanged. At --eval-every
100 it writes fully loadable, frozen network checkpoints under:
  results/v10_e4_<condition>_seed<seed>/validation_snapshots/
      val_000100_grad_XXXXXXXXX.pt ... val_000900_grad_XXXXXXXXX.pt
plus the paired validation metrics and atomic full training recovery state.

Training starts FROM SCRATCH for every condition and seed. No E2/E3 results,
old checkpoints, or previous replay buffers are used as starting weights.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # iamu_d3dqn/
TRAINER = ROOT / 'train_budgeted_ddqn_parallel.py'
CONDITIONS = {
    'a300': {'episodes': 300, 'lr': 3e-4, 'epsilon_decay_episodes': 300,
             'label': 'fresh 300-episode control'},
    'b900': {'episodes': 900, 'lr': 3e-4, 'epsilon_decay_episodes': 300,
             'label': 'more experience only; epsilon floor after 300'},
    'c900': {'episodes': 900, 'lr': 1e-4, 'epsilon_decay_episodes': 300,
             'label': 'lower constant learning rate'},
    'd900': {'episodes': 900, 'lr': 3e-4, 'epsilon_decay_episodes': 900,
             'label': 'longer exploration schedule'},
}
VALIDATION_INTERVAL = 100


def csv_ints(s: str) -> list[int]:
    a = [int(x) for x in s.split(',')]
    if not a or len(a) != len(set(a)):
        raise ValueError('Use a nonempty, duplicate-free seed list')
    return a


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class Job:
    condition: str
    seed: int
    device: str
    run_dir: str
    log_file: str
    command: tuple[str,...]


def create_job(*, condition: str, seed: int, device: str,
               actors: int, validators: int, prefix: str,
               validation_seeds: str, validation_every: int,
               recovery_every: int) -> Job:
    c=CONDITIONS[condition]
    output=ROOT/'results'/f'{prefix}_{condition}_seed{seed}'
    case_dir=(ROOT.parent/'2024LiERLDDE/case_code').resolve()
    train_cases=[case_dir/f'{i}.txt' for i in range(3,8)]
    val_cases=[case_dir/f'{i}.txt' for i in (8,9)]
    absent=[p for p in [TRAINER,*train_cases,*val_cases] if not p.is_file()]
    if absent:
        raise FileNotFoundError(f'Missing original trainer/case: {absent}')
    cmd=[sys.executable,'-u',str(TRAINER),
         '--train-cases',*[str(p) for p in train_cases],
         '--val-cases',*[str(p) for p in val_cases],
         '--episodes',str(c['episodes']),
         '--workers',str(actors),'--validation-workers',str(validators),
         '--prefetch-per-worker','1','--sync-every','1',
         '--state-every',str(recovery_every),
         '--eval-every',str(validation_every),
         '--scenarios','shock,repeated,small',
         '--budgets','2,5,10',
         '--uncertainties','4,8,12',
         '--validation-uncertainty','8',
         '--train-seed-start','200000',
         '--val-seeds',validation_seeds,
         '--updates','2','--clock','union','--period','12',
         '--candidate-limit','4','--candidate-screen','positive',
         '--reward-scale','100','--min-gain','0.1',
         '--budget-features','absolute','--value-features','global',
         '--keep-margin','0.1','--greedy-exploration-prob','0.75',
         '--hidden-dim','96','--q-network','set_ddqn',
         '--lr',str(c['lr']),'--gamma','1.0',
         '--replay-capacity','50000','--batch-size','64',
         '--warmup','512','--updates-per-step','0.25',
         '--max-updates-per-episode','64','--target-update','250',
         '--epsilon-start','1.0','--epsilon-end','0.05',
         '--epsilon-decay-episodes',str(c['epsilon_decay_episodes']),
         '--seed',str(seed),'--device',device,
         '--out',str(output)]
    log=ROOT/'logs'/f'{prefix}_{condition}_seed{seed}.log'
    return Job(condition,seed,device,str(output),str(log),tuple(cmd))


def _check_completed(job: Job, expected:int) -> bool:
    """A previous run is cached only when its *full* snapshot grid is valid."""
    path=Path(job.run_dir)
    if not (path/'performance.json').is_file():
        return False
    from inspect_e4 import inspect_run
    report=inspect_run(path,verify_hash=False)
    return bool(report['complete'] and report['episodes']==expected)


def run_job(job:Job,*,resume:bool,checkpoints_required:bool):
    path=Path(job.run_dir)
    expected=CONDITIONS[job.condition]['episodes']
    if resume and path.exists() and _check_completed(job,expected):
        return dict(condition=job.condition,seed=job.seed,device=job.device,
                    code=0,status='previously_complete',dir=job.run_dir)
    if path.exists() and any(path.iterdir()):
        if not resume:
            raise FileExistsError(f'Output already exists: {path}; pass --resume after inspecting')
        state=path/'training_state.pt'
        if not state.is_file():
            raise RuntimeError(f'Cannot resume without atomic training state: {state}')
        command=(*job.command,'--resume')
    else:
        command=job.command
    Path(job.log_file).parent.mkdir(parents=True,exist_ok=True)
    start=dt.datetime.now(dt.timezone.utc).isoformat()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    logmode='a' if '--resume' in command else 'w'
    with Path(job.log_file).open(logmode,encoding='utf8') as log:
        log.write(f'\n[E4 start] {start}\n'+subprocess.list2cmdline(command)+'\n'); log.flush()
        p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        log.write(f'\n[E4 finish] exit={p.returncode} {dt.datetime.now(dt.timezone.utc).isoformat()}\n')
    status='completed' if p.returncode==0 else 'failed'
    if p.returncode==0 and checkpoints_required:
        from inspect_e4 import inspect_run
        chk=inspect_run(path,verify_hash=True)
        if not chk['complete']:
            status='checkpoint_validation_failed'
            p=subprocess.CompletedProcess(p.args,1)
    return dict(condition=job.condition,seed=job.seed,device=job.device,
                code=p.returncode,status=status,dir=job.run_dir,
                log=job.log_file)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions',default='a300,b900,c900,d900',
                        help='Comma list of a300,b900,c900,d900')
    parser.add_argument('--seeds',default='42,43,44')
    parser.add_argument('--devices',default='cuda:0',
                        help='Distinct learner devices e.g. cuda:0,cuda:1; one run per device')
    parser.add_argument('--actors',type=int,default=8)
    parser.add_argument('--validators',type=int,default=8)
    parser.add_argument('--validation-seeds',default='400001:400003')
    parser.add_argument('--validation-every',type=int,default=100)
    parser.add_argument('--recovery-every',type=int,default=25)
    parser.add_argument('--prefix',default='v10_e4',
                        help='New prefix under iamu_d3dqn/results/ (never reuse existing)')
    parser.add_argument('--resume',action='store_true',
                        help='Continue interrupted runs from complete original training_state.pt')
    parser.add_argument('--dry-run',action='store_true',
                        help='Print commands and manifest; never start training')
    args=parser.parse_args(argv)
    if args.validation_every!=VALIDATION_INTERVAL:
        parser.error('E4 pre-registered snapshots require --validation-every 100')
    if args.actors<1 or args.validators<1 or args.recovery_every<1:
        parser.error('Invalid workers or recovery frequency')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.prefix):
        parser.error('Prefix must use only letters, digits, hyphen, underscore')
    devices=args.devices.split(',')
    if not devices or len(set(devices))!=len(devices) or any(not (re.fullmatch(r'cuda:\d+',x) or x=='cpu') for x in devices):
        parser.error('Use distinct devices like cuda:0,cuda:1, or cpu')
    conditions=args.conditions.split(',')
    if not conditions or len(set(conditions))!=len(conditions) or not set(conditions)<=set(CONDITIONS):
        parser.error('Unknown or repeated conditions')
    seeds=csv_ints(args.seeds)
    jobs=[]
    for condition in conditions:
        for seed in seeds:
            device=devices[len(jobs)%len(devices)]
            jobs.append(create_job(condition=condition,seed=seed,device=device,
                actors=args.actors,validators=args.validators,prefix=args.prefix,
                validation_seeds=args.validation_seeds,
                validation_every=args.validation_every,
                recovery_every=args.recovery_every))
    config_manifest={
        'experiment':'E4 training sufficiency', 'training_source':str(TRAINER),
        'training_source_sha256':sha256(TRAINER), 'ablation_grid':CONDITIONS,
        'fixed_validation_interval':VALIDATION_INTERVAL,
        'args':vars(args),
        'jobs':[asdict(j) for j in jobs],
        'scientific_note':'All runs start fresh; do not use Case 10-14 for choosing best epoch.',
    }
    mpath=ROOT/'results'/f'{args.prefix}_plan.json'
    if args.resume and mpath.is_file():
        old=json.loads(mpath.read_text(encoding='utf8'))
        if old['training_source_sha256']!=config_manifest['training_source_sha256']:
            parser.error('Trainer source changed; do not --resume under different code')
        if {k:v for k,v in old['args'].items() if k not in ('resume','dry_run')} != {k:v for k,v in vars(args).items() if k not in ('resume','dry_run')}:
            parser.error('Scientific parameters differ from saved plan; use a new --prefix')
    if args.dry_run:
        print(json.dumps(config_manifest,ensure_ascii=False,indent=2));return 0
    if mpath.exists() and not args.resume:
        parser.error(f'Plan exists: {mpath}. Use --resume or a new --prefix')
    if not args.resume:
        for j in jobs:
            p=Path(j.run_dir)
            if p.exists() and any(p.iterdir()):
                parser.error(f'Run directory not empty: {p}')
    mpath.parent.mkdir(parents=True,exist_ok=True)
    mpath.write_text(json.dumps(config_manifest,ensure_ascii=False,indent=2))
    # GPU-serialized queues: no two learners on the SAME device. The original
    # training tool continues to control its own CPU rollout/validation phases.
    queues=[[] for _ in devices]
    for j in jobs:
        queues[devices.index(j.device)].append(j)
    results=[];lock=threading.Lock()
    def worker(queue):
        for j in queue:
            try:r=run_job(j,resume=args.resume,checkpoints_required=True)
            except BaseException as e:
                r=dict(condition=j.condition,seed=j.seed,device=j.device,
                       code=1,status='launcher_error',detail=repr(e),dir=j.run_dir)
            with lock:
                results.append(r)
                print(f"[E4] {r['condition']} seed={r['seed']} {r['device']}: {r['status']}",flush=True)
        return True
    with ThreadPoolExecutor(max_workers=len(queues)) as ex:
        list(ex.map(worker,queues))
    report=ROOT/'results'/f'{args.prefix}_launch_report.json'
    report.write_text(json.dumps(results,ensure_ascii=False,indent=2))
    failures=[x for x in results if x['code']!=0]
    if failures:
        print(f'[FAILED] {len(failures)} runs. Check logs and --resume without changing settings.')
        return 2
    print(f'[DONE] All {len(jobs)} runs completed; inventory with inspect_e4.py')
    return 0


if __name__=='__main__':
    sys.exit(main())
