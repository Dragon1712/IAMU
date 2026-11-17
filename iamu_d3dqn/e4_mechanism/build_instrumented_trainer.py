#!/usr/bin/env python3
"""Generate an additive, fail-closed instrumented fork of the V0.9 trainer.

Does not alter train_budgeted_ddqn_parallel.py. Hooks only run at completed
training episodes and completed validation barriers; they DO NOT sample from
ReplayBuffer.rng, change rollout action selection, or add learner gradients.
"""
from __future__ import annotations
import argparse,hashlib,json,py_compile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def insert_once(code,needle,replacement):
    n=code.count(needle)
    if n!=1:
        raise RuntimeError(f'Unexpected trainer source, expected 1 anchor, found {n}: {needle[:100]!r}')
    return code.replace(needle,replacement,1)


def patch(source):
    code=source
    code=insert_once(code,
        'from parallel_recovery_v081 import atomic_json, save_state, load_state',
        'from parallel_recovery_v081 import atomic_json, save_state, load_state\n'
        'from e4_mechanism.telemetry import MechanismMonitor')
    code=insert_once(code,'    return p.parse_args()\n\n\ndef _atomic_csv',
        '''    p.add_argument('--e5-probe-bank',required=True,
                   help='Fixed PUBLIC observation bank .npz; required for causal matched-state trace')
    p.add_argument('--e5-monitor-every',type=int,default=25)
    p.add_argument('--e5-td-sample',type=int,default=384)
    p.add_argument('--e5-fork-episode',type=int,default=200)
    return p.parse_args()


def _atomic_csv''')
    code=insert_once(code,
        '    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=True)',
        '''    if not (args.e5_monitor_every>0 and args.e5_td_sample>0 and
            1 <= args.e5_fork_episode < args.episodes and
            args.e5_fork_episode % args.eval_every == 0 and
            args.e5_fork_episode % args.e5_monitor_every == 0):
        raise ValueError('E5 fork must be within training at validation & monitor boundary')
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=True)''')
    code=insert_once(code,
        '    if args.resume:\n        # Validation callbacks that arrived after the LAST full state',
        '''    e5_monitor=MechanismMonitor(args,output,learner,replay)
    if args.resume:
        # Validation callbacks that arrived after the LAST full state''')
    code=insert_once(code,
        '    completed=len(completed_ids)\n    # Do not send a single episode beyond',
        '    completed=len(completed_ids)\n    e5_monitor.reconcile(completed)\n    if completed==args.e5_fork_episode and validations.get(f\"val_{completed:06d}\",{}).get(\"status\")==\"done\":\n        e5_monitor.archive_fork(completed,validations,state_file,replay,learner)\n    # Do not send a single episode beyond')
    code=insert_once(code,
        "                persist(save_full=True,reason='validation-finished')",
        "                persist(save_full=True,reason='validation-finished')\n"
        "                e5_monitor.archive_fork(completed,validations,state_file,replay,learner)")
    code=insert_once(code,
        '            # Realtime CSV: no longer wait for hundreds of CPU validation runs.',
        '''            e5_monitor.record(completed,learner,replay,train_rows)
            # Realtime CSV: no longer wait for hundreds of CPU validation runs.''')
    # Avoid hard-coded code version claiming byte-identical E4. Original source
    # is copied, all scientific behavior should remain unchanged except the
    # observational (read-only) hooks and sampling process scheduling noise.
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainer',default='train_budgeted_ddqn_parallel.py')
    p.add_argument('--out',default='train_budgeted_ddqn_e5.py')
    args=p.parse_args(argv)
    inp=(ROOT/args.trainer).resolve();out=(ROOT/args.out).resolve()
    if not inp.is_file():p.error(f'Missing original trainer {inp}')
    if inp==out:p.error('Refusing to overwrite original trainer')
    modified=patch(inp.read_text(encoding='utf-8'))
    if out.exists():
        if out.read_text(encoding='utf-8')!=modified:
            p.error(f'Existing generated trainer differs: {out}. Use new --out; do NOT overwrite running trainer')
    else:
        out.write_text(modified,encoding='utf-8')
    py_compile.compile(str(out),doraise=True)
    digest=dict(original=str(inp),original_sha256=sha(inp),
                instrumented=str(out),instrumented_sha256=sha(out),
                hooks='read-only replay/model monitor + archive full 200ep state after validation')
    out.with_suffix('.e5_build.json').write_text(json.dumps(digest,ensure_ascii=False,indent=2))
    print(f'[BUILT] {out} hash={sha(out)} (original unchanged {sha(inp)})')

if __name__=='__main__':main()
