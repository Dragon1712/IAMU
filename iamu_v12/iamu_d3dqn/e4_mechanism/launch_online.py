#!/usr/bin/env python3
"""Launch an E4-E ONLINE 400-episode training fork using E4 B900 settings.

Each job owns one GPU plus four rollout actors / four validation workers.
The actual 200-episode full fork is archived and checksum-verified by the
instrumented trainer at the completed validation barrier.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from e4_train_sufficiency.launch_e4 import create_job
from e4_mechanism.telemetry import sha


def replace_arg(cmd,key,value):
    i=cmd.index(key)
    cmd[i+1]=str(value)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',default='43',help='comma-separated training seeds, e.g. 42,43,44')
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3')
    p.add_argument('--actors',type=int,default=4)
    p.add_argument('--validators',type=int,default=4)
    p.add_argument('--episodes',type=int,default=400)
    p.add_argument('--fork',type=int,default=200)
    p.add_argument('--validation-every',type=int,default=50)
    p.add_argument('--monitor-every',type=int,default=25)
    p.add_argument('--td-sample',type=int,default=384)
    p.add_argument('--probe-bank',default='results/v10_e5_probe/public_bank.npz')
    p.add_argument('--prefix',default='v10_e5_online')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    seed=[int(x) for x in a.seeds.split(',')]
    dev=a.devices.split(',')
    if len(set(seed))!=len(seed) or len(set(dev))!=len(dev):p.error('Duplicate seeds or devices')
    if not all(d=='cpu' or d.startswith('cuda:') for d in dev):p.error('Bad device')
    if a.actors<1 or a.validators<1 or not (a.fork<a.episodes and a.fork%a.validation_every==0 and a.fork%a.monitor_every==0):
        p.error('Illegal actor/validation/fork grid')
    trainer=ROOT/'train_budgeted_ddqn_e5.py'
    if not trainer.is_file():p.error('Run build_instrumented_trainer.py first')
    bank=(ROOT/a.probe_bank).resolve()
    if not bank.is_file():p.error('Run build_probe_bank.py first')
    if len(seed)>len(dev):p.error('For reproducibility provide one GPU per concurrent training seed, or run seeds sequentially')
    jobs=[]
    for index,s in enumerate(seed):
        base=create_job(condition='b900',seed=s,device=dev[index],
            actors=a.actors,validators=a.validators,prefix='v10_e5_template',
            validation_seeds='400001:400003',validation_every=a.validation_every,
            recovery_every=25)
        cmd=list(base.command)
        cmd[2]=str(trainer)
        folder=ROOT/'results'/f'{a.prefix}_seed{s}'
        replace_arg(cmd,'--episodes',a.episodes)
        replace_arg(cmd,'--out',folder)
        cmd.extend(['--e5-probe-bank',str(bank),
                    '--e5-monitor-every',str(a.monitor_every),
                    '--e5-td-sample',str(a.td_sample),
                    '--e5-fork-episode',str(a.fork)])
        if a.resume:cmd+=['--resume']
        jobs.append(dict(seed=s,device=dev[index],folder=str(folder),command=cmd,
                         log=str(ROOT/'logs'/f'{a.prefix}_seed{s}.log')))
    plan=dict(config=vars(a),trainer_sha256=sha(trainer),probe_bank_sha256=sha(bank),jobs=jobs,
              scientific_note='EP200 archived AFTER validation; no old E4 weights reused as full recovery state')
    path=ROOT/'results'/f'{a.prefix}_launch_plan.json'
    if a.resume and path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        ignore={'resume','dry_run'}
        if ({k:v for k,v in old['config'].items() if k not in ignore}
             != {k:v for k,v in vars(a).items() if k not in ignore}
             or old['trainer_sha256']!=plan['trainer_sha256']
             or old['probe_bank_sha256']!=plan['probe_bank_sha256']):
            p.error('Resume scientific config/hash mismatch')
    if a.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if not a.resume and path.exists():p.error('Plan already exists. Use a NEW --prefix.')
    for j in jobs:
        d=Path(j['folder'])
        if not a.resume and d.exists() and any(d.iterdir()):p.error(f'Output exists: {d}')
        if a.resume and not (d/'training_state.pt').exists():p.error(f'No resumable full state in {d}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    def run(j):
        log=Path(j['log']);log.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with log.open('a' if a.resume else 'w',encoding='utf-8') as f:
            f.write(' '.join(j['command'])+'\n');f.flush()
            result=subprocess.run(j['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(seed=j['seed'],exit_code=result.returncode,log=str(log),out=j['folder'])
    with ThreadPoolExecutor(max_workers=len(dev)) as pool:
        results=list(pool.map(run,jobs))
    report=ROOT/'results'/f'{a.prefix}_launch_report.json'
    report.write_text(json.dumps(results,indent=2))
    for r in results:print('[E5-ONLINE]',r,flush=True)
    if any(r['exit_code'] for r in results):raise SystemExit('At least one monitored online run failed')
    print(f'[DONE] {len(results)} monitored online runs')

if __name__=='__main__':main()
