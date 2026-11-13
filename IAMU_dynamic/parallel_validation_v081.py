"""Independent CPU validation pool with unchanged V0.8 evaluation semantics.

Each independent group (case, scenario, seed, budget) runs V0.8's original
`evaluate_paired` for ALL originally configured policies on a *frozen* Q policy.
A coordinator process runs a spawn-based process pool; the GPU learner never
participates in expensive `SlotPlanner` validation evaluation.

A request is identified by validation_id and an immutable snapshot path.
In phase-synchronous V0.8.2 the learner waits for this frozen snapshot before
starting the next rollout phase. Best checkpoints always use THAT snapshot.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import threading
from typing import Any
from statistics import mean, median

import torch

from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import evaluate_paired, write_csv

_VALIDATION_AGENT: DoubleDQNAgent | None = None
_VALIDATION_SETTINGS: dict | None = None
_VALIDATION_ID: str | None = None
_VALIDATION_REPORT_Q: Any = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _init_validation_worker(snapshot_path: str, settings: dict,
                            validation_id: str, report_q: Any) -> None:
    # The worker does not touch CUDA, even if the learner uses it.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    global _VALIDATION_AGENT, _VALIDATION_SETTINGS, _VALIDATION_ID, _VALIDATION_REPORT_Q
    _VALIDATION_AGENT, meta = DoubleDQNAgent.from_checkpoint(snapshot_path, device='cpu')
    _VALIDATION_AGENT.online.eval()
    _VALIDATION_SETTINGS = settings
    _VALIDATION_ID = validation_id
    _VALIDATION_REPORT_Q = report_q
    _VALIDATION_REPORT_Q.put(('validation-worker-online',validation_id,os.getpid()))
    print(f'[validation-worker-start] pid={os.getpid()} '
          f'snapshot={Path(snapshot_path).name}', flush=True)


def _validate_group(job: tuple) -> tuple[int, int, list[dict], list[dict]]:
    """Evaluate one group with the SAME original evaluator, all policies."""
    index, case, scenario, seed, budget = job
    assert _VALIDATION_AGENT is not None and _VALIDATION_SETTINGS is not None
    if _VALIDATION_REPORT_Q is not None:
        _VALIDATION_REPORT_Q.put(('validation-task-start',_VALIDATION_ID,index,os.getpid(),
                                   case,scenario,seed,budget))
    s = _VALIDATION_SETTINGS
    kw = dict(
        cases=[case], seeds=[seed], scenarios=[scenario], budgets=[budget],
        policies=s['policies'], agent=_VALIDATION_AGENT,
        candidate_limit=s['candidate_limit'], decision_clock=s['clock'],
        period=s['period'], reward_scale=s['reward_scale'],
        uncertainty=s['validation_uncertainty'], updates=s['updates'],
    )
    import inspect
    supported = inspect.signature(evaluate_paired).parameters
    for name in ('candidate_screen', 'min_gain', 'budget_features', 'value_features'):
        if name in supported:
            kw[name] = s[name]
        elif s[name] != {'candidate_screen': 'none', 'min_gain': .1,
                        'budget_features': 'legacy', 'value_features': 'legacy'}[name]:
            raise RuntimeError(f'Installed evaluation does not support {name}')
    rows, summary = evaluate_paired(**kw)
    # Do not silently omit a policy or cut down the 216 evaluations.
    if len(rows) != len(s['policies']) or len(summary) != len(s['policies']):
        raise RuntimeError('Validation group did not execute all requested policies')
    return index, os.getpid(), rows, summary


def validation_jobs(settings: dict) -> list[tuple]:
    """Each job evaluates all policies, preserving the original Cartesian grid."""
    jobs = []
    for case in settings['val_cases']:
        for scenario in settings['scenarios']:
            for seed in settings['val_seeds']:
                for budget in settings['budgets']:
                    jobs.append((len(jobs), case, scenario, seed, budget))
    return jobs


def aggregate_validation_results(results: list[tuple], policies: list[str],
                                 budgets: list[int]) -> tuple[list[dict], list[dict], float]:
    """Recompute the *entire* original evaluate_paired summary from all rows.

    Stable task ordering recreates the exact original case/scenario/seed/budget
    evaluation order; only independent execution timing differs. The same
    weighted_wait/gain/CVaR/median formulas are applied to the same rows.
    """
    results = sorted(results, key=lambda x: x[0])
    rows = [r for _,_,episode_rows,_ in results for r in episode_rows]
    seen_methods = {(r['method'],int(r['budget'])) for r in rows}
    expected_methods={(method,int(b)) for method in policies for b in budgets}
    if seen_methods!=expected_methods:
        raise RuntimeError(f'Missing validation method/budget groups: {expected_methods-seen_methods}')
    base={(r['case'],r['scenario'],r['seed'],r['budget']):float(r['wait'])
          for r in rows if r['method']=='keep'}
    periodic={(r['case'],r['scenario'],r['seed'],r['budget']):float(r['wait'])
              for r in rows if r['method']=='periodic'}
    for r in rows:
        key=(r['case'],r['scenario'],r['seed'],r['budget'])
        if key not in base:raise RuntimeError('Missing Keep baseline for validation group')
        r['gain_vs_keep']=base[key]-float(r['wait'])
        r['gain_vs_periodic']=(periodic[key]-float(r['wait'])) if key in periodic else ''
        r['regret_vs_keep']=max(0.,float(r['wait'])-base[key])
    def cvar90(values):
        items=sorted(map(float,values),reverse=True)
        mass=len(items)*(1.0-.90)
        integral=0.;left=mass
        for value in items:
            weight=min(left,1.)
            integral+=value*weight
            left-=weight
            if left<=1e-12:break
        return integral/mass
    groups=defaultdict(list)
    for r in rows:groups[(r['method'],int(r['budget']))].append(r)
    summary=[]
    for (method,budget),group in sorted(groups.items()):
        gains=[float(x['gain_vs_keep']) for x in group]
        risks=[float(x['regret_vs_keep']) for x in group]
        # Metrics mirror IAMU_dynamic/iamu_dynamic/rl_experiment.py.
        summary.append(dict(method=method,budget=budget,n=len(group),
            mean_wait=mean(float(x['wait']) for x in group),
            mean_gain=mean(gains),median_gain=median(gains),
            improved=sum(x>1e-6 for x in gains),
            degraded=sum(x< -1e-6 for x in gains),
            worst_gain=min(gains),cvar90_regret=cvar90(risks),
            mean_spent=mean(float(x.get('spent',0)) for x in group),
            mean_runtime_seconds=mean(float(x.get('runtime_seconds',0)) for x in group),
            mean_screened_proposals=mean(float(x.get('screened_proposals',0)) for x in group),
            mean_estimated_edit_gain=mean(float(x.get('mean_estimated_edit_gain',0)) for x in group),
            mean_last_revision_time=mean(float(x.get('last_revision_time',0)) for x in group),
            mean_gain_vs_periodic=(mean(float(x['gain_vs_periodic']) for x in group)
                                   if 'periodic' in policies else '')))
    selected=[r for r in summary if r['method']=='ddqn']
    if len(selected)!=len(budgets):raise RuntimeError('Missing DDQN budget group')
    score=sum(r['mean_wait'] for r in selected)/len(selected)
    return rows,summary,score


def validation_coordinator(request_q: Any, report_q: Any,
                           validation_workers: int, output_dir: str,
                           progress_every: int) -> None:
    """Long-lived coordinator; creates a fresh pool for each frozen snapshot.

    Only one validation snapshot is evaluated at a time; requests queue up but
    are NEVER dropped. Using `spawn` avoids inherited CUDA contexts.
    """
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    ctx = mp.get_context('spawn')
    try:
        while True:
            msg = request_q.get()
            if msg[0] == 'stop':
                return
            if msg[0] != 'validate':
                raise ValueError(f'Unexpected validator request: {msg[0]}')
            _, validation_id, snapshot_path, settings = msg
            started = time.perf_counter()
            jobs = validation_jobs(settings)
            expected = len(jobs)*len(settings['policies'])
            report_q.put(('validation-start', validation_id, len(jobs), expected,
                          str(snapshot_path), os.getpid()))
            results = []
            seen_pids = set()
            heartbeat_stop=threading.Event()
            def heartbeat():
                while not heartbeat_stop.wait(15.):
                    count=len(results)
                    report_q.put(('validation-progress',validation_id,count,
                                  len(jobs),count*len(settings['policies']),
                                  expected,round(time.perf_counter()-started,3),
                                  sorted(seen_pids)))
            heartbeat_thread=threading.Thread(target=heartbeat,name='validation-heartbeat',daemon=True)
            heartbeat_thread.start()
            try:
                with ctx.Pool(processes=validation_workers,
                              initializer=_init_validation_worker,
                              initargs=(str(snapshot_path),settings,validation_id,report_q)) as pool:
                    for result in pool.imap_unordered(_validate_group,jobs,chunksize=1):
                        results.append(result)
                        seen_pids.add(result[1])
                        count=len(results)
                        if count % progress_every == 0 or count == len(jobs):
                            report_q.put(('validation-progress',validation_id,count,
                                          len(jobs),count*len(settings['policies']),
                                          expected,round(time.perf_counter()-started,3),
                                          sorted(seen_pids)))
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=2.)
            rows, summary, score = aggregate_validation_results(
                results,settings['policies'],settings['budgets'])
            if len(rows) != expected:
                raise RuntimeError(f'{validation_id}: expected {expected} runs, got {len(rows)}')
            folder = Path(output_dir)/'validation_details'/validation_id
            folder.mkdir(parents=True, exist_ok=True)
            write_csv(folder/'runs.csv', rows)
            write_csv(folder/'summary.csv', summary)
            report_q.put(('validation-finished',validation_id,score,summary,
                          expected,len(jobs),time.perf_counter()-started,
                          str(snapshot_path),file_sha256(Path(snapshot_path)),
                          sorted(seen_pids)))
    except BaseException:
        report_q.put(('validation-error', traceback.format_exc()))
        raise
