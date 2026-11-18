#!/usr/bin/env python3
"""Apply the IAMU V1.2 additive research patch to a CLEAN Git checkout.

Never touches pre-existing results, checkpoints or test cases; creates a new
research branch and performs an all-anchor preflight before writing source.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_HEAD = '2d96981a38434014f14e31be5641d422d28835f8'
BRANCH = 'research/limited-budget-replay-v12'
BASE = Path(__file__).resolve().parent


def git(root, *args):
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    if p.returncode: raise RuntimeError(f'git {args}: {p.stderr.strip()}')
    return p.stdout.strip()


def rep(text, old, new, context):
    found = text.count(old)
    if found != 1:
        raise RuntimeError(f'{context}: expected exactly one anchor, got {found}: {old[:120]!r}')
    return text.replace(old, new, 1)


def patch_env(s):
    context = 'rl_env_v11.py'
    s = rep(s, 'import math\nimport numpy as np', 'import math\nimport time\nimport numpy as np', context)
    s = rep(s, "        self.screened_proposals = 0\n", "        self.screened_proposals = 0\n        self.candidate_generation_seconds = 0.\n        self.forecast_projection_seconds = 0.\n", context)
    s = rep(s, '        future=copy.deepcopy(self.runtime)\n', '        projection_started = time.perf_counter()\n        future=copy.deepcopy(self.runtime)\n', context)
    s = rep(s, '        return starts,costs,sum(costs.values())\n',
            '        self.forecast_projection_seconds += time.perf_counter() - projection_started\n        return starts,costs,sum(costs.values())\n', context)
    s = rep(s, '    def _refresh_candidates(self):\n        self._candidates=[]',
            '    def _refresh_candidates(self):\n        generation_started = time.perf_counter()\n        self._candidates=[]', context)
    s = rep(s, "            search_exhausted=True,pending_ships=len(ranked),ships_examined=0,",
            "            search_performed=False,search_exhausted=False,pending_ships=len(ranked),ships_examined=0,", context)
    s = rep(s, '            self._heuristic_mask=self._physical_mask.copy()\n            return\n',
            '            self._heuristic_mask=self._physical_mask.copy()\n            self.candidate_generation_seconds += time.perf_counter() - generation_started\n            return\n', context)
    s = rep(s, '        for vessel_id in ranked:\n            flags[\'ships_examined\']+=1\n',
            '        flags[\'search_performed\'] = True\n        for vessel_id in ranked:\n            flags[\'ships_examined\']+=1\n', context)
    s = rep(s, "        flags['returned_candidates']=len(selected)\n",
            "        flags['search_exhausted'] = flags['ships_examined'] == len(ranked)\n        flags['returned_candidates']=len(selected)\n", context)
    s = rep(s, "            raise AssertionError('Candidate search stopped short of K without exhaustion')\n",
            "            raise AssertionError('Candidate search stopped short of K without exhaustion')\n        self.candidate_generation_seconds += time.perf_counter() - generation_started\n", context)
    return s


def patch_agent(s):
    context = 'rl_ddqn_v11.py'
    anchor = '''    reason_flags: dict | None = None
    episode_id: int | None = None
'''
    s = rep(s, anchor, anchor + '''    budget_before: int | None = None
    budget_after: int | None = None
    event_index: int | None = None
    raw_steps: int = 1
    transition_version: int = 2
''', context)
    return s


def patch_actors(s):
    context = 'v11_actors.py'
    s = rep(s, '    epsilon: float\n', '    epsilon: float\n    exploration_seed: int\n', context)
    s = rep(s, "            agent.rng.seed(args['seed']*1000003+job.episode*10007+job.seed)\n",
            "            agent.rng.seed(job.exploration_seed)\n", context)
    s = rep(s, "                    dict(obs.decision_flags or {}),int(job.episode)))",
            "                    dict(obs.decision_flags or {}),int(job.episode),\n"
            "                    int(obs.remaining_budget), int(nxt.remaining_budget),\n"
            "                    int((obs.decision_flags or {}).get('event_index', 0)), 1, 2))", context)
    s = rep(s, "                'actor_seconds': time.perf_counter()-started,\n",
            "                'actor_seconds': time.perf_counter()-started,\n"
            "                'candidate_generation_seconds': env.candidate_generation_seconds,\n"
            "                'forecast_projection_seconds': env.forecast_projection_seconds,\n"
            "                'worker_idle_seconds': idle_on_last_job,\n", context)
    s = rep(s, "            message = inbox.get()\n",
            "            idle_started = time.perf_counter()\n"
            "            message = inbox.get()\n"
            "            idle_on_last_job = time.perf_counter()-idle_started\n", context)
    return s


def patch_replay(s):
    context = 'indexed_replay.py'
    methods = '''    def state_dict(self):
        """Exact ring/pool order: restoring content alone is NOT sampler-identical."""
        return dict(version=2, capacity=self.capacity, head=self._head,
                    size=self._size, slots=list(self._slots),
                    actionable=list(self.actionable.values),
                    forced=list(self.forced.values), rng=self.rng.getstate())

    def load_state_dict(self, state):
        if state.get('version') != 2 or state.get('capacity') != self.capacity:
            raise ValueError('Unsupported or incompatible IndexedReplay checkpoint')
        head, size = int(state['head']), int(state['size'])
        slots = list(state['slots'])
        if not (0 <= head < self.capacity and 0 <= size <= self.capacity and
                len(slots) == self.capacity):
            raise ValueError('Invalid replay ring metadata')
        active = {(head + i) % self.capacity for i in range(size)}
        if any(slots[i] is None for i in active):
            raise ValueError('Active replay ring slot is empty')
        if any(slots[i] is not None for i in range(self.capacity) if i not in active):
            raise ValueError('Replay has data in inactive slots')
        a, f = list(state['actionable']), list(state['forced'])
        if len(a) + len(f) != size or set(a).intersection(f) or set(a + f) != active:
            raise ValueError('Replay index pools do not cover active ring exactly')
        if any(not self._actionable(slots[i]) for i in a):
            raise ValueError('Actionable pool has forced transition')
        if any(self._actionable(slots[i]) for i in f):
            raise ValueError('Forced pool has actionable transition')
        self._slots = slots
        self._head, self._size = head, size
        self.actionable, self.forced = DensePool(), DensePool()
        for i in a: self.actionable.add(i)
        for i in f: self.forced.add(i)
        self.rng.setstate(state['rng'])

'''
    return rep(s, '    def stats(self):\n', methods + '    def stats(self):\n', context)


def patch_recovery(s):
    context = 'parallel_recovery_v081.py'
    s = rep(s, "    args=dict(config['training_args'])\n",
            "    if 'scientific_config' in config:\n"
            "        from v11_protocol.experiment_fingerprint_v12 import semantic_digest\n"
            "        return semantic_digest(config)\n"
            "    args=dict(config['training_args'])\n", context)
    s = rep(s, '''               best_val: float, gradient_credit: float) -> None:
    payload=dict(version=1,config_digest=config_digest(config),''',
            '''               best_val: float, gradient_credit: float,
               protocol_state: dict | None = None) -> None:
    payload=dict(version=2,config_digest=config_digest(config),''', context)
    s = rep(s, "        replay_items=list(replay.items),replay_rng=replay.rng.getstate(),\n",
            "        replay_items=list(replay.items),replay_rng=replay.rng.getstate(),\n"
            "        replay_state=replay.state_dict(),protocol_state=dict(protocol_state or {}),\n", context)
    s = rep(s, "    if item.get('version')!=1 or item['config_digest']!=config_digest(config):\n"
            "        raise ValueError('Resume config differs from saved training config')\n",
            "    if item.get('version') != 2 or 'replay_state' not in item:\n"
            "        raise ValueError('Legacy state cannot guarantee exact stratified replay sampling; '\n"
            "                         'resume it with the historical trainer, or start a new V1.2 run')\n"
            "    if item['config_digest'] != config_digest(config):\n"
            "        raise ValueError('Resume scientific configuration differs from saved state')\n", context)
    s = rep(s, "    replay.items.clear();replay.items.extend(item['replay_items'])\n    replay.rng.setstate(item['replay_rng'])\n",
            "    replay.load_state_dict(item['replay_state'])\n"
            "    if len(replay) != len(item['replay_items']):\n"
            "        raise AssertionError('Restored Replay length differs from checkpoint')\n", context)
    return s


def patch_fixed_probe(s):
    context='fixed_label_probe.py'
    s=rep(s, "    ref.target.load_state_dict(fork['target'])\n",
       "    ref.target.load_state_dict(fork['target'])\n"
       "    science = fork.get('config', {}).get('scientific_config',\n"
       "                fork.get('config', {}).get('training_args', {}))\n"
       "    ref.gamma = float(science.get('gamma', ref.gamma))\n"
       "    ref.bootstrap_margin = float(science.get('bootstrap_margin', ref.bootstrap_margin))\n"
       "    if not 0 < ref.gamma <= 1 or ref.bootstrap_margin < 0:\n"
       "        raise ValueError('Fork has invalid gamma/bootstrap margin')\n", context)
    s=rep(s, "        selector_evaluator_sha256=sha(a.reference_checkpoint),reference_bootstrap_margin=ref.bootstrap_margin,\n",
       "        selector_online_sha256=sha(a.reference_checkpoint),\n"
       "        evaluator_target_fork_sha256=sha(a.fork),\n"
       "        reference_gamma=ref.gamma,reference_bootstrap_margin=ref.bootstrap_margin,\n", context)
    return s


def patch_train(s):
    context='train_v11.py'
    s=rep(s, 'from collections import deque\n', 'from collections import deque,Counter\n', context)
    s=rep(s, 'from parallel_recovery_v081 import save_state,load_state,atomic_json\n',
       'from parallel_recovery_v081 import atomic_json\n'
       'from v11_protocol.recovery_v12 import save_state,load_state\n', context)
    s=rep(s, 'from v11_protocol.v11_actors import RolloutJob,_actor_loop\n',
       'from v11_protocol.v11_actors import RolloutJob,_actor_loop\n'
       'from v11_protocol.research_v12 import collapse_exhaustion_tail,verify_collapse,class_counts,classify_transition\n'
       'from v11_protocol.experiment_fingerprint_v12 import fingerprint,compare_for_resume,digest_json\n', context)
    s=rep(s, "                args.epsilon_start+(args.epsilon_end-args.epsilon_start)*min(1.,(ep-1)/max(1,args.epsilon_decay_episodes)))\n",
       "                args.epsilon_start+(args.epsilon_end-args.epsilon_start)*min(1.,(ep-1)/max(1,args.epsilon_decay_episodes)),\n"
       "                args.seed*1000003+ep*10007+(args.train_seed_start+ep-1))\n", context)
    s=rep(s, "    p.add_argument('--resume',action='store_true')\n",
       "    p.add_argument('--resume',action='store_true')\n"
       "    p.add_argument('--exhaustion-tail-mode',choices=('raw','collapse'),default='raw')\n", context)
    old='''    code_files=[Path(__file__),Path(__file__).parents[1]/'iamu_dynamic/rl_env_v11.py',
        Path(__file__).parents[1]/'iamu_dynamic/rl_ddqn_v11.py',
        Path(__file__).parents[1]/'iamu_dynamic/core.py']
    cfg=dict(code_sha256={str(f.name):sha(f) for f in code_files},
        training_args=vars(a),protocol_version='V1.1-deterministic-consumption',
        validation_sync_phases=True,round_size=a.round_size,
        candidate_search='scan-until-K-retained',actor_job_seed='independent per episode',
        actor_behavior_version='frozen per round',candidate_screen=a.candidate_screen)
'''
    new='''    cfg=fingerprint(a,Path(__file__).resolve().parents[1])
    cfg['task_list_sha256']=digest_json([vars(j) for j in all_jobs])
    cfg['protocol_version']='V1.2-frozen-round-ordered-consumption'
'''
    s=rep(s,old,new,context)
    s=rep(s, "    state=out/'training_state.pt';rows=[];val_rows=[];validations={};best=float('inf');credit=0.\n",
       "    state=out/'training_state.pt';rows=[];val_rows=[];validations={};best=float('inf');credit=0.\n"
       "    observed_raw_steps=0;cumulative_sample_counts=Counter()\n", context)
    s=rep(s, '''        if json.loads((out/'manifest.json').read_text())!=cfg:
            raise ValueError('Run manifest/source-code hashes differ from saved state')
''',
       '''        changed_runtime=compare_for_resume(json.loads((out/'manifest.json').read_text()),cfg)
        if changed_runtime:
            print('[v12] Runtime device/workers/paths differ; bitwise cross-platform replay not guaranteed',flush=True)
''', context)
    s=rep(s, "        best=float(saved['best_val']);credit=float(saved['gradient_credit'])\n",
       "        best=float(saved['best_val']);credit=float(saved['gradient_credit'])\n"
       "        observed_raw_steps=int(saved.get('protocol_state',{}).get('observed_raw_steps',0))\n"
       "        cumulative_sample_counts=Counter(saved.get('protocol_state',{}).get('cumulative_sample_counts',{}))\n", context)
    s=rep(s, "        atomic_json(cfg,out/'manifest.json')\n",
       "        atomic_json(cfg,out/'manifest.json')\n"
       "        atomic_json([vars(j) for j in all_jobs],out/'task_list.json')\n",context)
    s=rep(s, '''        save_state(state,learner=learner,replay=replay,config=cfg,train_rows=rows,
            val_rows=val_rows,validations=validations,best_val=best,gradient_credit=credit)
''',
       '''        save_state(state,learner=learner,replay=replay,config=cfg,train_rows=rows,
            val_rows=val_rows,validations=validations,best_val=best,gradient_credit=credit,
            protocol_state=dict(observed_raw_steps=observed_raw_steps,\n                cumulative_sample_counts=dict(cumulative_sample_counts)))
''', context)
    s=rep(s, '''    def persist(reason):
''', '''    def persist(reason):
''',context) if False else s
    # nonlocal observed raw counter is at main() scope; no global declaration needed.
    old='''                    old_replay_size=len(replay)
                    for tr in trajectory:replay.items.append(tr)
                    n_ready=sum(min(a.replay_capacity,old_replay_size+j+1)>=a.warmup
                                for j in range(len(trajectory)))
'''
    new='''                    raw_trajectory=trajectory
                    if len(raw_trajectory)!=int(report['steps']):
                        raise AssertionError('Actor raw step count mismatch')
                    if a.exhaustion_tail_mode=='collapse':
                        trajectory=collapse_exhaustion_tail(raw_trajectory,gamma=a.gamma)
                        verify_collapse(raw_trajectory,trajectory,gamma=a.gamma)
                    stored_steps=len(trajectory)
                    old_replay_size=len(replay)
                    for tr in trajectory: replay.items.append(tr)
                    raw_before=observed_raw_steps
                    observed_raw_steps+=len(raw_trajectory)
                    n_ready=max(0,observed_raw_steps-(a.warmup-1))-max(0,raw_before-(a.warmup-1))
'''
    s=rep(s,old,new,context)
    s=rep(s, '''                    if a.max_updates_per_episode:updates=min(updates,a.max_updates_per_episode)
                    credit-=updates
''',
       '''                    if a.max_updates_per_episode:updates=min(updates,a.max_updates_per_episode)
                    if len(replay)<a.batch_size: updates=0
                    credit-=updates
''', context)
    s=rep(s, '''                    losses=[];sample_metas=[]
                    for _ in range(updates):
''',
       '''                    losses=[];sample_metas=[];sampled_classes=Counter()
                    for _ in range(updates):
''', context)
    s=rep(s, '''                        losses.append(learner.update(sampled,weights=weights));sample_metas.append(meta)
''',
       '''                        sampled_classes.update(classify_transition(tr) for tr in sampled)
                        losses.append(learner.update(sampled,weights=weights));sample_metas.append(meta)
''', context)
    s=rep(s, '''                    rows.append(dict(episode=ep,completed=done,round_id=round_id,
''',
       '''                    replay_counts=class_counts(replay.items)
                    cumulative_sample_counts.update(sampled_classes)
                    rows.append(dict(episode=ep,completed=done,round_id=round_id,
''', context)
    s=rep(s, '''                        forced_flags=json.dumps(report.get('forced_flags',{}),sort_keys=True)))
''',
       '''                        forced_flags=json.dumps(report.get('forced_flags',{}),sort_keys=True),
                        raw_events_observed=observed_raw_steps,
                        episode_raw_events=len(raw_trajectory),episode_stored_transitions=stored_steps,
                        exhaustion_tail_mode=a.exhaustion_tail_mode,
                        replay_budget_exhausted=replay_counts['budget_exhausted'],
                        replay_forced_with_budget=replay_counts['forced_with_budget'],
                        replay_actionable_keep=replay_counts['actionable_keep'],
                        replay_actionable_edit=replay_counts['actionable_edit'],
                        sampled_actionable=sampled_classes['actionable'],
                        sampled_budget_exhausted=sampled_classes['budget_exhausted'],
                        sampled_forced_with_budget=sampled_classes['forced_with_budget'],
                        batch_actionable_fraction=(sampled_classes['actionable']/max(1,updates*a.batch_size)),
                        cumulative_sampled_actionable=cumulative_sample_counts['actionable'],
                        cumulative_sampled_budget_exhausted=cumulative_sample_counts['budget_exhausted'],
                        cumulative_sampled_forced_with_budget=cumulative_sample_counts['forced_with_budget'],
                        first_exhaustion_event_index=next((tr.event_index for tr in raw_trajectory
                            if tr.budget_before>0 and tr.budget_after==0),''),
                        remaining_budget_trace=json.dumps([tr.budget_after for tr in raw_trajectory]),
                        candidate_generation_seconds=report.get('candidate_generation_seconds',0.),
                        forecast_projection_seconds=report.get('forecast_projection_seconds',0.),
                        worker_idle_seconds=report.get('worker_idle_seconds',0.)))
''', context)
    s=rep(s, "        atomic_json(dict(completed_episodes=done,gradient_steps=learner.gradient_steps,\n",
       "        if done!=a.episodes or len(rows)!=a.episodes or len({int(r['episode']) for r in rows})!=a.episodes:\n"
       "            raise RuntimeError('Incomplete or duplicate training episodes: cannot write complete.json')\n"
       "        atomic_json(dict(completed_episodes=done,gradient_steps=learner.gradient_steps,\n", context)
    return s


def patch_coordinator(s):
    c = 'train_v11.py coordinator'
    s = rep(s, 'from v11_protocol.research_v12 import collapse_exhaustion_tail,verify_collapse,class_counts,classify_transition\n',
            'from v11_protocol.dispatch_v12 import RoundDispatcher\n'
            'from v11_protocol.research_v12 import collapse_exhaustion_tail,verify_collapse,class_counts,classify_transition\n', c)
    s = rep(s, "    p.add_argument('--exhaustion-tail-mode',choices=('raw','collapse'),default='raw')\n",
            "    p.add_argument('--exhaustion-tail-mode',choices=('raw','collapse'),default='raw')\n"
            "    p.add_argument('--dispatch-coordinator',action='store_true')\n", c)
    s = rep(s, '''            for wid in range(a.workers):assign(wid)
            while next_index<len(order):
''', '''            dispatcher=None
            if a.dispatch_coordinator:
                waiting.clear()
                inflight=set(order)
                dispatcher=RoundDispatcher(result_q,commands,batch)
                dispatcher.start()
            else:
                for wid in range(a.workers):assign(wid)
            while next_index<len(order):
''', c)
    s = rep(s, 'try:msg=result_q.get(timeout=30)\n',
            'try:msg=(dispatcher.get(timeout=30) if dispatcher is not None else result_q.get(timeout=30))\n', c)
    s = rep(s, '''                assign(int(report['worker_id']))
                while next_index<len(order) and order[next_index] in ready:
''', '''                if dispatcher is None: assign(int(report['worker_id']))
                while next_index<len(order) and order[next_index] in ready:
''', c)
    s = rep(s, "            if inflight or waiting or ready:raise AssertionError('Incomplete collection round')\n",
            "            if dispatcher is not None: dispatcher.finish()\n"
            "            if inflight or waiting or ready:raise AssertionError('Incomplete collection round')\n", c)
    s = rep(s, "        for c in commands:\n            try:c.put_nowait(('stop',))\n",
            "        if 'dispatcher' in locals() and dispatcher is not None:\n"
            "            dispatcher.cancel()\n"
            "        for c in commands:\n            try:c.put_nowait(('stop',))\n", c)
    s=rep(s, "        summary=validate(snap,a,out/'validation_details'/key)\n",
          "        validation_started=time.perf_counter()\n"
          "        summary=validate(snap,a,out/'validation_details'/key)\n"
          "        validation_seconds=time.perf_counter()-validation_started\n",c)
    s=rep(s, "            mean_spent=summary['mean_spent'],gradient_steps=learner.gradient_steps,\n",
          "            mean_spent=summary['mean_spent'],validation_seconds=validation_seconds,\n"
          "            gradient_steps=learner.gradient_steps,\n",c)
    s=rep(s, "            round_wall=time.perf_counter()-round_started\n",
          "            round_wall=time.perf_counter()-round_started\n"
          "            for record in rows[-len(batch):]:record['round_wall_seconds']=round_wall\n",c)
    return s

PATCHES = {
    'iamu_dynamic/rl_env_v11.py': patch_env,
    'iamu_dynamic/rl_ddqn_v11.py': patch_agent,
    'v11_protocol/v11_actors.py': patch_actors,
    'v11_protocol/indexed_replay.py': patch_replay,
    'v11_protocol/fixed_label_probe.py': patch_fixed_probe,
    'v11_protocol/train_v11.py': lambda s: patch_coordinator(patch_train(s)),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, help='Path to IAMU Git checkout root')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    p.add_argument('--allow-untracked-results', action='store_true',
                   help='Permit only ?? files under iamu_d3dqn/results/; never alters them')
    p.add_argument('--allow-other-head', action='store_true',
                   help='Permits a newer HEAD ONLY when every source anchor matches')
    args=p.parse_args()
    root=Path(args.repo).resolve()
    if not (root/'iamu_d3dqn'/'v11_protocol'/'train_v11.py').is_file():
        p.error('Invalid IAMU checkout root')
    head=git(root,'rev-parse','HEAD')
    dirty=git(root,'status','--porcelain')
    if dirty:
        status_lines=dirty.splitlines()
        safe_untracked=all(line.startswith('?? iamu_d3dqn/results/') for line in status_lines)
        if not (args.allow_untracked_results and safe_untracked):
            p.error('Working tree contains modifications/untracked files. Commit or stash before applying; nothing written.')
        print('[safety] Untracked result data detected and preserved; not used as patch input')
    if head!=EXPECTED_HEAD and not args.allow_other_head:
        p.error(f'HEAD={head} differs from reviewed {EXPECTED_HEAD}; inspect differences first')
    dest=root/'iamu_d3dqn'
    changes={}
    for rel, fn in PATCHES.items():
        path=dest/rel
        if not path.is_file(): raise FileNotFoundError(path)
        old=path.read_text(encoding='utf-8')
        changes[path]=fn(old)
    for source in sorted((BASE/'payload').rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
            rel=source.relative_to(BASE/'payload')
            target=dest/rel
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite existing payload: {target}')
            changes[target]=source.read_text(encoding='utf-8')
    import ast
    for path,content in changes.items():
        if path.suffix=='.py': ast.parse(content,filename=str(path))
    print(f'All anchors and Python syntax pass: {len(changes)} patched/new files')
    if args.check: return
    if git(root,'branch','--list',BRANCH):
        p.error(f'Branch {BRANCH} already exists. Refusing to overwrite it')
    git(root,'switch','-c',BRANCH)
    # All source checks performed before any write. Keep a local rollback cache
    # in memory, so filesystem errors restore the original files where possible.
    backups={path:path.read_bytes() if path.exists() else None for path in changes}
    try:
        for path,content in changes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp=path.with_name(path.name+'.v12.tmp')
            tmp.write_text(content,encoding='utf-8')
            os.replace(tmp,path)
    except Exception:
        for path,previous in backups.items():
            if previous is None: path.unlink(missing_ok=True)
            else: path.write_bytes(previous)
        raise
    print(f'Applied to {BRANCH}; inspect diff, run tests, then commit manually.')
    print('No training runs or old result files were modified.')

if __name__=='__main__': main()
