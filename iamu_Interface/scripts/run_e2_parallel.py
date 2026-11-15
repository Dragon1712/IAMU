#!/usr/bin/env python3
"""Run IAMU V1.0-E2 experimental functions concurrently, without shared writes.

A shard is one (case, scenario, ETA seed, hard budget, optional H/M). Each
process runs the EXISTING experiment entry point, with exactly the same seeds,
case and evaluation settings as a serial invocation. CPU pinning and output
isolation prevent thread scheduling and output file collisions.

First install V1.0-E2. This launcher refuses old V1.0-E1 entrypoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel_e2_core import (WORKFLOWS, parse_ints, sha256, hash_json,
                              read_rows, write_rows, summarize_runs,
                              summarize_stability, validate_unique,
                              SHARD_PRIMARY, STABILITY_PRIMARY)

DEFAULT_WORKERS = {"multistage": 4, "risk_stability": 6,
                   "risk_stability_joint": 4, "mpc_samples": 3, "h_depth": 3}
DEFAULT_POLICIES = 'set_ddqn,q_risk,q_budget,q_joint,q_budget_heuristic,q_joint_heuristic'


@dataclass(frozen=True)
class Job:
    workflow: str
    variant: str
    shard: str
    args: tuple[str, ...]
    output_rel: str
    log_rel: str

    @property
    def identity(self) -> str:
        return f'{self.workflow}/{self.variant}/{self.shard}'


def resolve_cases(raw: str) -> list[Path]:
    paths = []
    for value in raw.split(','):
        part = value.strip()
        if not part:
            raise ValueError('Empty case')
        if '/' in part or part.endswith('.txt'):
            path = Path(part).expanduser().resolve()
        else:
            local = ROOT / 'assets' / 'cases' / f'{part}.txt'
            fallback = ROOT.parent / '2024LiERLDDE' / 'case_code' / f'{part}.txt'
            path = local if local.is_file() else fallback
        if not path.is_file():
            raise FileNotFoundError(f'Missing case {path}')
        paths.append(path.resolve())
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Case filenames must have unique stems')
    return paths


def resolve_checkpoint(seed: int, checkpoint: str | None, manifest: str | None):
    if checkpoint or manifest:
        if not checkpoint or not manifest:
            raise ValueError('Both --checkpoint and --manifest required together')
        a, b = Path(checkpoint).expanduser().resolve(), Path(manifest).expanduser().resolve()
    else:
        local = ROOT / 'assets' / 'checkpoints' / f'set_ddqn_seed{seed}'
        sibling = ROOT.parent / 'iamu_d3dqn' / 'results' / f'v09b_set_ddqn_seed{seed}_300'
        base = local if (local / 'best.pt').is_file() else sibling
        a, b = (base / 'best.pt').resolve(), (base / 'manifest.json').resolve()
    for path in (a, b):
        if not path.is_file():
            raise FileNotFoundError(path)
    return a, b


def parse_cpu_pool(spec: str | None) -> list[int]:
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    if spec:
        cpus = parse_ints(spec.replace('-', ':'))
        if not set(cpus).issubset(available):
            raise ValueError(f'CPU pool includes unavailable cores: {sorted(set(cpus)-set(available))}')
        return cpus
    return available


def resolve_workers(raw: str, workflows: list[str]) -> dict[str, int]:
    values = dict(DEFAULT_WORKERS)
    if raw:
        for token in raw.split(','):
            key, sep, value = token.strip().partition('=')
            if sep != '=' or key not in WORKFLOWS or int(value) < 1:
                raise ValueError('Workers: workflow=N, e.g. multistage=6,risk_stability=8')
            values[key] = int(value)
    return {w: values[w] for w in workflows}


def scripts_for(workflows: Sequence[str]):
    entry = ROOT / 'experiments' / 'eval_risk_budget_v10.py'
    stability = ROOT / 'experiments' / 'eval_risk_sampling_stability_v10.py'
    policy = ROOT / 'iamu_dynamic' / 'v10_policy.py'
    if not entry.is_file() or not policy.is_file():
        raise FileNotFoundError('E2 policy/evaluation files missing; install V1.0-E2 first')
    if '--mpc-samples' not in entry.read_text(encoding='utf-8') or 'q_joint_heuristic' not in policy.read_text(encoding='utf-8'):
        raise RuntimeError('Old V1.0-E1 code detected. Install the V1.0-E2 patch before parallel execution')
    if any(w.startswith('risk_stability') for w in workflows) and not stability.is_file():
        raise FileNotFoundError('E2 risk stability evaluator is missing')
    return entry, stability


def common_group_args(case: Path, scenario: str, seed: int, budget: int) -> list[str]:
    return ['--cases', str(case), '--scenarios', scenario,
            '--seeds', str(seed), '--budgets', str(budget)]


def build_jobs(args, cases: list[Path], workflows: list[str]):
    jobs = []
    scenarios = args.scenarios.split(',')
    seeds = parse_ints(args.eta_seeds)
    budgets = parse_ints(args.budgets)
    if len(scenarios) != len(set(scenarios)):
        raise ValueError('Duplicate scenarios')
    if not set(scenarios) <= {'shock', 'repeated', 'small'}:
        raise ValueError('Expected scenarios shock,repeated,small')
    if min(budgets) < 0:
        raise ValueError('Negative budgets')
    policy_flags = ['--policies', args.policies]
    base = ['--checkpoint', str(args.checkpoint_path), '--manifest', str(args.manifest_path),
            '--device', 'cpu']
    for case in cases:
        for scenario in scenarios:
            for seed in seeds:
                for budget in budgets:
                    shard = f'case{case.stem}__{scenario}__seed{seed}__B{budget}'
                    group = common_group_args(case, scenario, seed, budget)
                    for workflow in workflows:
                        if workflow in ('multistage', 'mpc_samples', 'h_depth'):
                            if workflow == 'multistage':
                                variants = [('default', args.mpc_samples, args.h, args.policies)]
                            elif workflow == 'mpc_samples':
                                variants = [(f'M{m}', m, args.h, 'set_ddqn,q_joint')
                                            for m in parse_ints(args.mpc_sizes)]
                            else:
                                variants = [(f'H{h}', args.mpc_samples, h, 'set_ddqn,q_joint,lookahead')
                                            for h in parse_ints(args.h_sizes)]
                            for variant, mpc_m, h, policy in variants:
                                if mpc_m > args.risk_samples:
                                    raise ValueError('mpc_samples must be <= risk_samples')
                                flags = (base + group +
                                         ['--risk-samples', str(args.risk_samples),
                                          '--mpc-samples', str(mpc_m),
                                          '--lookahead-stages', str(h),
                                          '--lookahead-step-hours', str(args.step_hours),
                                          '--lookahead-risk-weight', str(args.lookahead_risk_weight),
                                          '--beta-risk', str(args.beta_risk),
                                          '--beta-budget', str(args.beta_budget),
                                          '--legacy-beta-budget', str(args.legacy_beta_budget),
                                          '--risk-alpha', str(args.alpha),
                                          '--sigma', str(args.sigma), '--belief', args.belief,
                                          '--policies', policy, '--verify-baseline', '--audit'])
                                jobs.append(Job(workflow, variant, shard, tuple(flags),
                                                f'shards/{workflow}/{variant}/{shard}',
                                                f'logs/{workflow}/{variant}/{shard}.log'))
                        else:
                            flags = (base + group + ['--sample-sizes', args.risk_sizes,
                                                     '--reference-samples', str(args.reference_samples),
                                                     '--repeats', str(args.repeats),
                                                     '--states-per-group', str(args.states_per_group),
                                                     '--stride', str(args.stride),
                                                     '--sigma', str(args.sigma),
                                                     '--alpha', str(args.alpha),
                                                     '--belief', args.belief,
                                                     '--beta-risk', str(args.beta_risk),
                                                     '--mpc-stages', str(args.h),
                                                     '--mpc-samples', str(args.mpc_samples),
                                                     '--beta-budget', str(args.beta_budget)])
                            if workflow == 'risk_stability_joint':
                                flags += ['--joint']
                            jobs.append(Job(workflow, 'default', shard, tuple(flags),
                                            f'shards/{workflow}/default/{shard}',
                                            f'logs/{workflow}/default/{shard}.log'))
    if len(set(j.identity for j in jobs)) != len(jobs):
        raise AssertionError('Shard names are not unique')
    return jobs


def source_fingerprint(cases, checkpoint, manifest):
    paths = [
        ROOT / 'iamu_dynamic' / x for x in (
            'core.py', 'rl_env.py', 'rl_ddqn.py', 'rl_experiment.py',
            'risk_selection.py', 'v10_risk_features.py',
            'v10_budget_lookahead.py', 'v10_policy.py',
            'v10_candidate_api.py', 'v10_candidate_integration.py')]
    paths += [ROOT / 'experiments' / 'eval_risk_budget_v10.py',
              ROOT / 'experiments' / 'eval_risk_sampling_stability_v10.py']
    paths += [Path(__file__).resolve(), Path(__file__).resolve().parent/'parallel_e2_core.py']
    paths += list(cases) + [checkpoint, manifest]
    return {str(p.resolve()): sha256(p) for p in paths if p.exists()}


def expected_input_manifest(args, jobs, cases, workflows, fingerprint):
    return dict(format='iamu-e2-functional-parallel-v1',
                paths=[str(c) for c in cases],
                checkpoint=str(args.checkpoint_path), manifest=str(args.manifest_path),
                source_fingerprint=fingerprint,
                workflows=workflows,
                # Important: worker counts and scheduling are NOT in this fingerprint.
                # A completed run can be resumed with different worker counts.
                config={k: v for k, v in vars(args).items()
                        if k not in ('workers', 'cpus', 'cores_per_worker', 'out',
                                     'dry_run', 'resume', 'retry_failed', 'merge_only')
                        and not k.endswith('_path')},
                jobs=[asdict(job) for job in jobs])


def validate_shard_output(job: Job, path: Path, checkpoint_hash: str,
                          manifest_hash: str, source_hashes: dict):
    if not (path/'manifest.json').is_file():
        raise ValueError(f'Missing shard manifest: {path}')
    meta = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('checkpoint_sha256') != checkpoint_hash or meta.get('manifest_sha256') != manifest_hash:
        raise ValueError(f'Shard checkpoint/manifest mismatch: {path}')
    if job.workflow.startswith('risk_stability'):
        file = path/'stability_details.csv'
        code_hash = meta.get('source_sha256', {})
        hashes_to_check = {
            'risk': 'iamu_dynamic/v10_risk_features.py',
            'policy': 'iamu_dynamic/v10_policy.py',
            'mpc': 'iamu_dynamic/v10_budget_lookahead.py',
            'stability': 'experiments/eval_risk_sampling_stability_v10.py',
        }
        for field, relative in hashes_to_check.items():
            expected = source_hashes.get(str((ROOT/relative).resolve()))
            if expected is not None and code_hash.get(field) != expected:
                raise ValueError(f'Shard source changed: {path}/{field}')
    else:
        file = path/'runs.csv'
        impl = meta.get('implementation_sha256', {})
        for relative, actual in impl.items():
            expected = source_hashes.get(str((ROOT/relative).resolve()))
            if expected is not None and actual != expected:
                raise ValueError(f'Shard source changed: {path}/{relative}')
    rows = read_rows(file)
    if not rows:
        raise ValueError(f'Empty shard {file}')
    # Verify no extra groups migrated into this shard unexpectedly.
    tokens = job.shard.split('__')
    case, scenario, seed, budget = tokens
    expected = (case.removeprefix('case'), scenario,
                seed.removeprefix('seed'), budget.removeprefix('B'))
    for r in rows:
        actual = (r['case'], r['scenario'], r['eta_seed'], r['budget'])
        if actual != expected:
            raise ValueError(f'Wrong group in {file}: {actual} != {expected}')
    if not job.workflow.startswith('risk_stability'):
        validate_unique(rows, SHARD_PRIMARY, 'shard policy')
        flags = list(job.args)
        policies = set(flags[flags.index('--policies')+1].split(','))
        expected_policies = policies | {'keep', 'periodic', 'set_ddqn'}
        actual_policies = {r['policy'] for r in rows}
        if actual_policies != expected_policies:
            raise ValueError(f'Missing/extra policy in {file}: {actual_policies ^ expected_policies}')
        for r in rows:
            if not 0 <= float(r['spent']) <= float(r['budget']) + 1e-9:
                raise AssertionError(f'Hard revision budget violated in {file}')
    else:
        validate_unique(rows, STABILITY_PRIMARY, 'shard stability replicate')
    return file, rows


def merge_jobs(jobs, out, checkpoint_hash, manifest_hash, source_hashes):
    variants = defaultdict(list)
    for j in jobs:
        variants[(j.workflow, j.variant)].append(j)
    report = []
    for (workflow, variant), group in sorted(variants.items()):
        records, audit, source_counts = [], [], []
        for j in group:
            shard_dir = out / j.output_rel
            complete = shard_dir/'_success.json'
            if not complete.is_file():
                raise RuntimeError(f'Cannot merge incomplete shard: {shard_dir}')
            marker = json.loads(complete.read_text(encoding='utf-8'))
            if marker.get('command_hash') != hash_json(j.args):
                raise RuntimeError(f'Stale success marker: {shard_dir}')
            csv_path, rows = validate_shard_output(j, shard_dir, checkpoint_hash,
                                                    manifest_hash, source_hashes)
            if marker.get('output_sha256') != sha256(csv_path):
                raise RuntimeError(f'Shard output changed since success: {csv_path}')
            source_counts.append(len(rows))
            records.extend(rows)
            audit_path = shard_dir/'decision_audit.csv'
            if audit_path.is_file():
                audit.extend(read_rows(audit_path))
        target = out/'merged'/workflow/variant
        target.mkdir(parents=True, exist_ok=True)
        if workflow.startswith('risk_stability'):
            validate_unique(records, STABILITY_PRIMARY, 'stability sample')
            write_rows(target/'stability_details.csv', records)
            write_rows(target/'stability_summary.csv', summarize_stability(records))
        else:
            validate_unique(records, SHARD_PRIMARY, 'evaluation run')
            write_rows(target/'runs.csv', records)
            write_rows(target/'paired_summary.csv', summarize_runs(records))
            if audit:
                write_rows(target/'decision_audit.csv', audit)
        report.append(dict(workflow=workflow, variant=variant,
                           n_shards=len(group), n_rows=len(records),
                           n_audit_rows=len(audit), shard_rows=source_counts))
    (out/'merge_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workflows', default='multistage,risk_stability')
    p.add_argument('--workers', default='multistage=4,risk_stability=6',
                   help='per-function concurrency, e.g. multistage=6,risk_stability=8,mpc_samples=4')
    p.add_argument('--cpus', default=None, help='Allowed CPU list such as 16-63,68,70-75')
    p.add_argument('--cores-per-worker', type=int, default=1)
    p.add_argument('--cases', default='8,9')
    p.add_argument('--scenarios', default='shock,repeated,small')
    p.add_argument('--eta-seeds', default='400001:400003')
    p.add_argument('--budgets', default='2,5,10')
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--manifest', default=None)
    p.add_argument('--policies', default=DEFAULT_POLICIES)
    p.add_argument('--risk-samples', type=int, default=16)
    p.add_argument('--mpc-samples', type=int, default=8)
    p.add_argument('--h', type=int, default=2, help='future recourse stages')
    p.add_argument('--step-hours', type=float, default=12.)
    p.add_argument('--lookahead-risk-weight', type=float, default=0.,
                   help='Pure lookahead optional CVaR term; Q-joint risk remains separately scored')
    p.add_argument('--beta-risk', type=float, default=.25)
    p.add_argument('--beta-budget', type=float, default=.25)
    p.add_argument('--legacy-beta-budget', type=float, default=.10)
    p.add_argument('--alpha', type=float, default=.9)
    p.add_argument('--sigma', type=float, default=7.12463033710398)
    p.add_argument('--belief', choices=('censored','truncated'), default='censored')
    p.add_argument('--risk-sizes', default='16,64,128')
    p.add_argument('--reference-samples', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--states-per-group', type=int, default=2)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--mpc-sizes', default='4,8,16')
    p.add_argument('--h-sizes', default='0,1,2,3')
    p.add_argument('--out', default='results/v10_e2_function_parallel')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--merge-only', action='store_true')
    args = p.parse_args(argv)
    workflows = [w.strip() for w in args.workflows.split(',')]
    if not workflows or len(workflows) != len(set(workflows)) or not set(workflows).issubset(WORKFLOWS):
        p.error('Unknown or duplicate workflow')
    if args.mpc_samples > args.risk_samples or args.mpc_samples < 1:
        p.error('--mpc-samples must be in [1, --risk-samples]')
    if args.cores_per_worker < 1:
        p.error('--cores-per-worker must be >=1')
    scripts_for(workflows)
    try:
        cases = resolve_cases(args.cases)
        checkpoint, manifest = resolve_checkpoint(args.training_seed, args.checkpoint, args.manifest)
        args.checkpoint_path, args.manifest_path = checkpoint, manifest
        jobs = build_jobs(args, cases, workflows)
        workers = resolve_workers(args.workers, workflows)
        cpus = parse_cpu_pool(args.cpus)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    used = args.cores_per_worker * sum(workers.values())
    if used > len(cpus):
        p.error(f'CPU pool has {len(cpus)} allowed cores, but {used} cores are reserved. '
                'Reduce --workers or --cores-per-worker, or provide a larger --cpus pool.')
    if not shutil.which('taskset'):
        print('[WARNING] taskset unavailable: CPU pinning disabled, numeric outputs still shard-isolated')
    pools, offset = {}, 0
    for workflow in workflows:
        size = workers[workflow] * args.cores_per_worker
        pool = cpus[offset:offset+size]
        pools[workflow] = [pool[k:k+args.cores_per_worker]
                           for k in range(0, len(pool), args.cores_per_worker)]
        offset += size
    fingerprint = source_fingerprint(cases, checkpoint, manifest)
    plan = expected_input_manifest(args, jobs, cases, workflows, fingerprint)
    # --resume and --merge-only do not change the scientific job specification.
    for transient in ('resume', 'retry_failed', 'merge_only', 'dry_run'):
        plan['config'].pop(transient, None)
    plan_hash = hash_json(plan)
    out = (ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        p.error('--out must stay inside iamu_Interface to avoid accidental writes to original project')
    print(f'Parallel plan: {len(jobs)} independent shards, functions={workflows}, '
          f'CPU workers={workers}, cpus/job={args.cores_per_worker}, plan={plan_hash[:12]}')
    if args.dry_run:
        for j in jobs[:12]:
            print(f'  {j.identity} -> {j.output_rel}')
        print('Dry run: no files written')
        return 0
    plan_file = out/'parallel_plan.json'
    if out.exists() and any(out.iterdir()):
        if not args.resume and not args.merge_only:
            p.error(f'Output already exists: {out}. Use --resume with the SAME scientific configuration')
        if not plan_file.is_file():
            p.error('Cannot resume output with no parallel_plan.json')
        old = json.loads(plan_file.read_text(encoding='utf-8'))
        if old.get('plan_hash') != plan_hash:
            p.error('Scientific inputs changed. Use a NEW --out directory, do not resume mixed parameters')
    else:
        out.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(dict(plan=plan, plan_hash=plan_hash), indent=2,
                                        ensure_ascii=False, default=str), encoding='utf-8')
    checkpoint_hash, manifest_hash = sha256(checkpoint), sha256(manifest)
    if args.merge_only:
        report = merge_jobs(jobs, out, checkpoint_hash, manifest_hash, fingerprint)
        print(f'Merged {len(report)} functional result sets in {out}/merged')
        return 0
    import concurrent.futures
    failed = []
    complete = 0
    lock = threading.Lock()

    def run_job(job: Job, cpu_set: list[int]):
        nonlocal complete
        shard_dir = out/job.output_rel
        log_path = out/job.log_rel
        expected_file = shard_dir/('stability_details.csv' if job.workflow.startswith('risk_stability') else 'runs.csv')
        marker_file = shard_dir/'_success.json'
        if marker_file.is_file():
            marker = json.loads(marker_file.read_text(encoding='utf-8'))
            if marker.get('command_hash') == hash_json(job.args) and expected_file.is_file():
                validate_shard_output(job, shard_dir, checkpoint_hash, manifest_hash, fingerprint)
                if sha256(expected_file) == marker.get('output_sha256'):
                    return 'skipped'
            raise RuntimeError(f'Existing success marker mismatches source/command: {shard_dir}')
        if shard_dir.exists() and any(shard_dir.iterdir()):
            if args.resume and args.retry_failed:
                shutil.rmtree(shard_dir)  # only explicitly requested incomplete shard
            else:
                raise RuntimeError(f'Incomplete shard exists: {shard_dir}. Use --resume --retry-failed')
        shard_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = ('eval_risk_sampling_stability_v10.py'
                 if job.workflow.startswith('risk_stability') else 'eval_risk_budget_v10.py')
        cmd = [sys.executable, '-u', str(ROOT/'experiments'/entry),
               *job.args, '--out', str(shard_dir)]
        if shutil.which('taskset'):
            cmd = ['taskset', '-c', ','.join(map(str, cpu_set)), *cmd]
        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
            'PYTHONHASHSEED': '0', 'CUDA_VISIBLE_DEVICES': '',
            'TOKENIZERS_PARALLELISM': 'false',
        })
        with log_path.open('w', encoding='utf-8') as log:
            log.write(f'JOB: {job.identity}\nCPU_PIN: {cpu_set}\nCOMMAND: {cmd!r}\n')
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f'{job.identity}: exit={proc.returncode}; log={log_path}')
        output_csv, records = validate_shard_output(
            job, shard_dir, checkpoint_hash, manifest_hash, fingerprint)
        marker = dict(job=job.identity, command_hash=hash_json(job.args),
                      output_sha256=sha256(output_csv), n_rows=len(records),
                      finish_utc=datetime.now(timezone.utc).isoformat())
        temp = shard_dir/'_success.json.tmp'
        temp.write_text(json.dumps(marker, indent=2), encoding='utf-8')
        temp.replace(marker_file)
        with lock:
            complete += 1
            print(f'[DONE {complete}/{len(jobs)}] {job.identity}', flush=True)
        return 'complete'

    # Disjoint executor-specific CPU pools. Futures from all workflows are
    # scheduled together, allowing each function to progress simultaneously.
    executors, futures = [], {}
    try:
        for workflow in workflows:
            cpu_slots = pools[workflow]
            thread_local = threading.local()
            counter, counter_lock = [0], threading.Lock()
            def init(slot_pool=cpu_slots, local=thread_local, index=counter,
                     ilock=counter_lock):
                with ilock:
                    local.cpu_set = slot_pool[index[0]]
                    index[0] += 1
            executor = ThreadPoolExecutor(max_workers=workers[workflow], initializer=init)
            executors.append(executor)
            for j in jobs:
                if j.workflow == workflow:
                    def dispatch(job=j, local=thread_local):
                        return run_job(job, local.cpu_set)
                    futures[executor.submit(dispatch)] = j
        skipped = 0
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result == 'skipped':
                    skipped += 1
            except Exception as error:
                failed.append(dict(job=job.identity, error=str(error)))
                print(f'[FAILED] {job.identity}: {error}', file=sys.stderr, flush=True)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
    if source_fingerprint(cases, checkpoint, manifest) != fingerprint:
        print('[FAILED] Code/checkpoint/case inputs changed during execution. '
              'Do not merge mixed-source results.', file=sys.stderr)
        return 1
    if failed:
        (out/'parallel_failures.json').write_text(json.dumps(failed, indent=2), encoding='utf-8')
        print(f'{len(failed)} shards failed; completed shards retained. '
              'Fix the error and use --resume --retry-failed.', file=sys.stderr)
        return 1
    report = merge_jobs(jobs, out, checkpoint_hash, manifest_hash, fingerprint)
    print(f'[SUCCESS] {len(jobs)} shards, {skipped} reused, '
          f'{len(report)} merged variants: {out}/merged')
    return 0


if __name__ == '__main__':
    sys.exit(main())
