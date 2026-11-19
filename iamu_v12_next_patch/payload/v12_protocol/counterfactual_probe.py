"""Pathwise Keep vs legal Edit under a fixed continuation and SAME realized future.

These labels are neither optimal Q* nor conditional expected advantages.
Do not change scenario seeds after a history has been observed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from .baseline_cache import canonical,file_sha
from .probe_bank import candidate_fingerprint,load_probe_bank,restore_probe_env,source_dataset

LABEL_KIND='realized_path_fixed_continuation'


def _future_cost(env, first_action, continuation, reward_scale=100.):
    from iamu_dynamic.rl_experiment_v11 import choose_action
    obs=env._observation()
    if not bool(obs.mask[first_action]):raise ValueError('Illegal first counterfactual action')
    _,reward,done,info=env.step(first_action)
    total=float(reward)
    while not done:
        nxt=int(choose_action(continuation,env))
        _,r,done,info=env.step(nxt)
        total+=float(r)
    if env.used>env.budget:raise AssertionError('Overspent budget')
    return -float(reward_scale)*total


def evaluate_state(probe,dataset,continuation='greedy',reward_scale=100.):
    if continuation not in ('greedy','periodic','keep'):
        raise ValueError('Only frozen public-ETA continuation permitted')
    base=restore_probe_env(probe,dataset)
    fp=candidate_fingerprint(base)
    obs=base._observation()
    legal=[i for i,ok in enumerate(obs.mask) if bool(ok)]
    if len(legal)<2:raise ValueError('Counterfactual probe needs legal Edit')
    costs={}
    for action in legal:
        # An independent deep copy for EACH branch, not a shared mutated plan.
        branch=copy.deepcopy(base)
        if candidate_fingerprint(branch)!=fp:
            raise AssertionError('Deepcopy changed candidate plan semantics')
        costs[action]=_future_cost(branch,action,continuation,reward_scale)
    if candidate_fingerprint(base)!=fp:
        raise AssertionError('Counterfactual branches mutated base state')
    result=[]
    keep=costs[0]
    for action in legal:
        gain=keep-costs[action]
        if action==0 and not math.isclose(gain,0,abs_tol=1e-8):
            raise AssertionError('Keep vs Keep gain !=0')
        result.append(dict(state_id=probe['state_id'],episode_id=probe['episode_id'],
               raw_step_index=probe['raw_step_index'],scenario=probe['scenario'],
               initial_budget=probe['initial_budget'],remaining_budget=probe['remaining_budget'],
               case_sha256=probe['case_sha256'],candidate_fingerprint=fp,
               continuation_policy=continuation,label_kind=LABEL_KIND,
               action_index=action,candidate_id=(0 if action==0 else probe['candidate_ids'][action-1]),
               future_keep_cost=keep,future_action_cost=costs[action],
               delta_cost_path_pi=gain,delta_q_units=gain/reward_scale))
    return result


def evaluate_bank(bank_path,out_dir,*,data=None,continuation='greedy',limit=0):
    from iamu_dynamic.rl_experiment_v11 import write_csv
    bank=load_probe_bank(bank_path,data)
    ds=source_dataset(data or bank['source_path'])
    states=bank['states'] if limit==0 else bank['states'][:limit]
    if limit<0:raise ValueError('Limit must be >=0')
    root=Path(__file__).resolve().parents[1]
    code_paths=[root/'iamu_dynamic/core.py',root/'iamu_dynamic/rl_env_v11.py',
                root/'iamu_dynamic/rl_experiment_v11.py',
                root/'iamu_dynamic/risk_selection.py',Path(__file__)]
    code_sha256={str(p.relative_to(root)):file_sha(p) for p in code_paths}
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    rows=[]
    for i,state in enumerate(states):
        key=dict(schema=LABEL_KIND,state_id=state['state_id'],
            scenario_seed=state['scenario_seed'],
            candidate_fingerprint=state['candidate_fingerprint'],
            observation_sha256=state['observation_sha256'],
            source_sha256=bank['source_sha256'],
            continuation_policy=continuation,code_sha256=code_sha256)
        label_key=hashlib.sha256(canonical(key).encode()).hexdigest()
        group=evaluate_state(state,ds,continuation)
        for row in group:row['reference_fingerprint']=label_key
        rows.extend(group)
        print(f'[pathwise] {i+1}/{len(states)} {state["state_id"][:12]}',flush=True)
    if rows:write_csv(out/'counterfactual_labels.csv',rows)
    else:(out/'counterfactual_labels.csv').write_text('state_id,action_index,delta_cost_path_pi\n')
    meta=dict(label_kind=LABEL_KIND,continuation_policy=continuation,
              state_count=len(states),row_count=len(rows),
              bank_sha256=file_sha(bank_path),dataset_sha256=bank['source_sha256'],
              code_sha256=code_sha256,reference_fingerprint_schema=LABEL_KIND,
              reward_scale=100.,single_realized_future_per_state=True)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    (out/'complete.json').write_text(json.dumps(dict(states=len(states),rows=len(rows))))
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--data');p.add_argument('--out',required=True)
    p.add_argument('--continuation',choices=['greedy','keep','periodic'],default='greedy')
    p.add_argument('--limit',type=int,default=8,help='0 means full bank; default 8-state smoke')
    a=p.parse_args(argv)
    print(json.dumps(evaluate_bank(a.bank,a.out,data=a.data,
       continuation=a.continuation,limit=a.limit),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
