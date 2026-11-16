#!/usr/bin/env python3
"""E4-D: matched-state Keep-collapse and paired counterfactual Q-gap audit.

SERVER REQUIREMENTS: original iamu_dynamic from V0.9 and E4 snapshot .pt files.
No network updates, no state mutations in original trajectories. We use the
original generated hidden future ONLY after selecting public decision states,
for paired ex-post evaluation. Never treat one realized world as a conditional
expected-return target. Re-run across independent ETA seeds to assess that.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0,str(Path(__file__).resolve().parent))

from diagnostic_core import (
    action_diagnostics, choose_evenly, calibration_summary, regret_summary,
    cvar_upper,
)


def imports():
    # Import only when invoking the server experiment, so --help and pure unit
    # tests work without a locally installed CUDA/PyTorch environment.
    import torch
    from iamu_dynamic import read_case, generate_scenario
    from iamu_dynamic.rl_env import IAMUEventEnv
    from iamu_dynamic.rl_ddqn import DoubleDQNAgent
    return torch, read_case, generate_scenario, IAMUEventEnv, DoubleDQNAgent


def digest(f: Path) -> str:
    sha=hashlib.sha256()
    with Path(f).open('rb') as stream:
        for piece in iter(lambda:stream.read(1024*1024),b''):
            sha.update(piece)
    return sha.hexdigest()


def write_csv(path: Path, records):
    if not records:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for row in records for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def ints(s):
    v=[]
    for segment in s.split(','):
        if ':' in segment:
            a,b=map(int,segment.split(':'))
            if b<a:
                raise ValueError('Reversed integer range')
            v.extend(range(a,b+1))
        else:
            v.append(int(segment))
    if not v or len(set(v))!=len(v):
        raise ValueError('Empty or duplicate integer grid')
    return v


@dataclass
class Model:
    label: str
    episode: int
    folder: Path
    snapshot: Path
    checksum: str
    agent: object
    run_args: dict
    validation_rows: dict
    gradient_steps: int


def load_model(spec: str, device: str, *, verify_checksum: bool, Agent):
    if '@' not in spec:
        raise ValueError(f'Expected run-directory@episode, got {spec}')
    path, episode_str = spec.rsplit('@',1)
    episode=int(episode_str)
    folder=Path(path).resolve()
    if episode<=0 or episode%100:
        raise ValueError('E4 snapshot episode must be a positive multiple of 100')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    args=manifest['training_args']
    ledger=json.loads((folder/'validation_ledger.json').read_text(encoding='utf-8'))['records']
    key=f'val_{episode:06d}'
    info=ledger.get(key)
    if not info or info.get('status')!='done':
        raise RuntimeError(f'Validation not marked done: {folder}/{key}')
    if int(info['evaluated_runs'])!=int(info['expected_runs']):
        raise RuntimeError(f'Validation run count mismatch: {folder}/{key}')
    # This is intentionally the on-server E4 .pt snapshot, not the selected best.pt.
    snapshot=folder/'validation_snapshots'/Path(info['snapshot']).name
    if not snapshot.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {snapshot}')
    expected=info['snapshot_sha256']
    if verify_checksum and digest(snapshot)!=expected:
        raise ValueError(f'Checkpoint hash mismatch: {snapshot}')
    agent,meta=Agent.from_checkpoint(str(snapshot),device=device)
    if agent.q_network!='set_ddqn':
        raise ValueError('E4-D requires the original Set-aware DDQN architecture')
    if abs(float(agent.keep_margin)-float(args['keep_margin']))>1e-9:
        raise ValueError('Checkpoint Keep margin differs from manifest')
    if 'selected_episode' in meta and int(meta['selected_episode'])!=episode:
        raise ValueError('Snapshot metadata episode mismatch')
    report_path=folder/'validation_details'/key/'runs.csv'
    if not report_path.is_file():
        raise FileNotFoundError(f'Original paired validation runs not found: {report_path}')
    with report_path.open(encoding='utf-8',newline='') as f:
        records=list(csv.DictReader(f))
    validation={
        (Path(r['case']).stem,str(r['scenario']),int(r['seed']),int(r['budget'])):r
        for r in records if r['method']=='ddqn'
    }
    if not validation:
        raise ValueError('No original DDQN validation rows for '+str(snapshot))
    label=f'{folder.name}_ep{episode:03d}'
    return Model(label,episode,folder,snapshot,expected,agent,args,
                 validation,int(info['gradient_steps']))


def consistent_environment(models):
    # Learning rate/episodes/exploration are experimental changes; the
    # environment, reward, observable state and test scenarios must not change.
    keys=('clock','period','candidate_limit','candidate_screen','reward_scale',
          'min_gain','budget_features','value_features','updates',
          'validation_uncertainty')
    ref=models[0].run_args
    for m in models[1:]:
        delta={k:(ref[k],m.run_args[k]) for k in keys if ref[k]!=m.run_args[k]}
        if delta:
            raise ValueError(f'Incompatible decision/evaluation protocol: {m.label} {delta}')
    return dict(candidate_limit=int(ref['candidate_limit']),
                decision_clock=ref['clock'],period=float(ref['period']),
                candidate_screen=ref['candidate_screen'],
                reward_scale=float(ref['reward_scale']),
                min_gain=float(ref['min_gain']),
                budget_features=ref['budget_features'],
                value_features=ref['value_features'])


def predict_q(agent,obs, torch):
    agent.online.eval()
    with torch.no_grad():
        global_state=torch.as_tensor(obs.global_features,device=agent.device).unsqueeze(0)
        candidates=torch.as_tensor(obs.action_features,device=agent.device).unsqueeze(0)
        mask=torch.as_tensor(obs.mask,device=agent.device,dtype=torch.bool).unsqueeze(0)
        q=agent.online(global_state,candidates,mask).squeeze(0).detach().cpu().tolist()
    if len(q)!=len(obs.mask):
        raise AssertionError('Q vector and legal-action mask mismatch')
    return q


def run_trajectory(scenario,budget,model,env_kwargs,IAMUEventEnv,torch):
    env=IAMUEventEnv(scenario,budget=budget,**env_kwargs)
    obs=env.reset()
    audit=[]
    while True:
        q=predict_q(model.agent,obs,torch)
        d=action_diagnostics(q,obs.mask,model.agent.keep_margin)
        actual=int(model.agent.act(obs,epsilon=0.))
        if actual!=d['action']:
            raise AssertionError(f'Q forward/action mismatch at {model.label} event {env.event_index}: '
                                 f'{actual} vs {d["action"]}')
        n=int(obs.mask[1:].sum())
        audit.append(dict(
            model=model.label,event_index=int(env.event_index),
            event_time=float(env.event_time),event_type=env.event_type,
            remaining_budget=int(env.remaining_budget),budget=int(budget),
            n_legal_edits=n,chosen_action=actual,
            chosen_vessel=(obs.candidate_ids[actual-1] if actual else None),
            action_without_margin=d['action_without_margin'],
            best_edit=d['best_edit'],q_keep=d['q_keep'],
            q_best_edit=d['q_best_edit'],q_gap_best=d['best_gap'],
            keep_margin=model.agent.keep_margin,margin_distance=d['margin_distance'],
            keep_reason=('budget_exhausted' if n==0 and env.remaining_budget==0
                         else d['reason'] if n else 'no_legal_edit'),
            best_point_forecast_gain=max((x['gain'] for x in env._candidate_data),default=None),
            screened_proposals_cumulative=int(env.screened_proposals),
        ))
        obs,_,done,info=env.step(actual)
        if done:
            break
    if env.used>budget:
        raise AssertionError('Hard-budget violation')
    all_eligible=[x for x in audit if x['n_legal_edits']>0]
    margin_only=sum(x['keep_reason']=='keep_margin_only' for x in all_eligible)
    episode=dict(model=model.label,weighted_wait=float(info['weighted_wait']),
                 spent=int(env.used),budget=budget,episodes_snapshot=model.episode,
                 n_events=len(audit),actionable_events=len(all_eligible),
                 edit_events=sum(x['chosen_action']>0 for x in audit),
                 keep_with_legal_edit=sum(x['chosen_action']==0 for x in all_eligible),
                 keep_due_to_positive_margin=margin_only,
                 q_prefers_keep_count=sum(x['keep_reason']=='q_prefers_keep' for x in audit),
                 eligible_gap_positive_count=sum(x['q_gap_best'] is not None and x['q_gap_best']>0
                                                 for x in all_eligible),
                 eligible_gap_over_margin=sum(x['q_gap_best'] is not None and
                                              x['q_gap_best']>model.agent.keep_margin for x in all_eligible),
                 exhausted_steps=sum(x['keep_reason']=='budget_exhausted' for x in audit),
                 mean_best_q_gap=(mean(x['q_gap_best'] for x in all_eligible)
                                  if all_eligible else None),
                 final_screened_proposals=int(env.screened_proposals))
    return episode,audit


def anchor_action(env,obs,mode,anchor_model):
    if mode=='keep':return 0
    if mode=='early':return int(anchor_model.agent.act(obs,epsilon=0.))
    raise ValueError(mode)


def collect_states(scenario,budget,mode,anchor_model,env_kwargs,
                   *,n_states,IAMUEventEnv):
    """First pass finds legal events; second pass replays to selected states.

    Selection uses only event index and the existence of a legal action,
    never a future realized cost or an evaluated model's Q scores.
    """
    env=IAMUEventEnv(scenario,budget=budget,**env_kwargs)
    obs=env.reset()
    legal_events=[]
    while True:
        if bool(obs.mask[1:].any()):
            legal_events.append(env.event_index)
        action=anchor_action(env,obs,mode,anchor_model)
        obs,_,done,_=env.step(action)
        if done:break
    if not legal_events:
        return [],0
    chosen={legal_events[i] for i in choose_evenly(len(legal_events),n_states)}
    env=IAMUEventEnv(scenario,budget=budget,**env_kwargs)
    obs=env.reset()
    snapshots=[]
    while True:
        if env.event_index in chosen:
            snapshots.append(copy.deepcopy(env))
        action=anchor_action(env,obs,mode,anchor_model)
        obs,_,done,_=env.step(action)
        if done:break
    if len(snapshots)!=len(chosen):
        raise AssertionError('State-bank replay is not deterministic')
    return snapshots,len(legal_events)


def counterfactual(env,first_action,continuation,anchor_model):
    """Paired ex-post replay: the agent sees only public observations.

    Same scenario hidden arrivals/updates and same public prefix for each arm.
    ``env.scenario.true_etas`` is read by physical env.step, NOT policy scorer.
    Past incurred waiting is common and cancels in Keep-vs-Edit differences.
    """
    state=copy.deepcopy(env)
    obs=state._observation()
    if not 0<=first_action<len(obs.mask) or not bool(obs.mask[first_action]):
        raise ValueError('Illegal counterfactual initial action')
    obs,_,done,info=state.step(first_action)
    steps=1
    while not done:
        if continuation=='anchor':
            act=int(anchor_model.agent.act(obs,epsilon=0.))
        elif continuation=='greedy':
            act=int(state.point_greedy_action())
        elif continuation=='keep':
            act=0
        else:
            raise ValueError('Unknown continuation '+continuation)
        obs,_,done,info=state.step(act)
        steps+=1
        if steps>20000:
            raise RuntimeError('Counterfactual exceeded event bound')
    return dict(wait=float(info['weighted_wait']),spent=int(state.used),steps=steps)


def group_summaries(episodes, events, calibration, decisions):
    """Aggregate matched, correlated experiments without significance claims."""
    by_ep=defaultdict(list)
    for x in episodes:
        for scen,b in [('ALL','ALL'),(x['scenario'],'ALL'),('ALL',x['budget']),
                       (x['scenario'],x['budget'])]:
            by_ep[(x['model'],scen,b)].append(x)
    ep_summary=[]
    for (m,sc,b),rr in sorted(by_ep.items(),key=lambda y:tuple(map(str,y[0]))):
        ep_summary.append(dict(model=m,scenario=sc,budget=b,n_episodes=len(rr),
            mean_wait=mean(r['weighted_wait'] for r in rr),
            mean_spent=mean(r['spent'] for r in rr),
            mean_actionable_events=mean(r['actionable_events'] for r in rr),
            total_actionable_events=sum(r['actionable_events'] for r in rr),
            actionable_keep_rate=(sum(r['keep_with_legal_edit'] for r in rr)/
                                  max(1,sum(r['actionable_events'] for r in rr))),
            mean_spent_fraction=mean(r['spent']/r['budget'] for r in rr),
            mean_margin_only_keeps=mean(r['keep_due_to_positive_margin'] for r in rr),
            mean_screened_proposals=mean(r['final_screened_proposals'] for r in rr)))
    by_cal=defaultdict(list)
    for r in calibration:
        for scen,b in [('ALL','ALL'),(r['scenario'],'ALL'),('ALL',r['budget']),
                       (r['scenario'],r['budget'])]:
            by_cal[(r['model'],r['anchor'],r['continuation'],scen,b)].append(r)
    cal_summary=[]
    for key,rr in sorted(by_cal.items(),key=lambda x:tuple(map(str,x[0]))):
        cal_summary.append(dict(zip(('model','anchor','continuation','scenario','budget'),key),
                                **calibration_summary(rr,truth_dead_zone=0.001)))
    by_dec=defaultdict(list)
    for r in decisions:
        for scen,b in [('ALL','ALL'),(r['scenario'],'ALL'),('ALL',r['budget']),
                       (r['scenario'],r['budget'])]:
            by_dec[(r['model'],r['anchor'],r['continuation'],scen,b)].append(r)
    dec_summary=[]
    for key,rr in sorted(by_dec.items(),key=lambda x:tuple(map(str,x[0]))):
        dec_summary.append(dict(zip(('model','anchor','continuation','scenario','budget'),key),
                                **regret_summary(rr)))
    return ep_summary,cal_summary,dec_summary


def analyze_bank(env,models,anchor_model,continuations,*,torch,max_edit_actions,
                 case_name,scenario_name,eta_seed,budget,anchor_name):
    obs=env._observation()
    assert obs.mask[0] and obs.mask[1:].any() and env.remaining_budget>0
    event=env.event_index
    key=dict(case=case_name,scenario=scenario_name,eta_seed=eta_seed,
             budget=budget,anchor=anchor_name,event_index=event,
             event_time=env.event_time,event_type=env.event_type,
             remaining_budget=env.remaining_budget,
             n_legal_edits=int(obs.mask[1:].sum()),
             n_started=len(env.runtime.started),
             n_arrived=sum(env.scenario.true_etas[i]<=env.event_time+1e-8 for i in env.ids))
    model_actions={}
    qs=[]
    for model in models:
        q=predict_q(model.agent,obs,torch)
        d=action_diagnostics(q,obs.mask,model.agent.keep_margin)
        act=int(model.agent.act(obs,epsilon=0.))
        if act!=d['action']:
            raise AssertionError('Offline Q forward did not reproduce frozen DDQN act()')
        model_actions[model.label]=dict(d)
        for a,allowed in enumerate(obs.mask):
            if not allowed:continue
            qs.append(dict(**key,model=model.label,action_id=a,
                candidate_vessel=(obs.candidate_ids[a-1] if a else None),
                q_value=float(q[a]),q_keep=float(q[0]),
                q_gap=(float(q[a]-q[0]) if a else 0.),
                selected_action=act,
                keep_margin=model.agent.keep_margin,
                is_best_edit=bool(a==d['best_edit']),
                is_chosen=bool(a==act),
                point_gain=(float(env._candidate_data[a-1]['gain']) if a else 0.)))
    legal=[a for a in range(1,len(obs.mask)) if obs.mask[a]]
    # Limit labels based on *public point-forecast gains*, never on any model Q
    # or any hidden counterfactual gain. If selected model action is outside
    # the evaluated set, mark its regret as NA instead of fabricating labels.
    ranked=sorted(legal,key=lambda a:(-env._candidate_data[a-1]['gain'],a))
    chosen=sorted(ranked[:max_edit_actions])
    candidate_set=[0,*chosen]
    truth=[]; calibration=[];decisions=[]
    for continuation in continuations:
        outcomes={a:counterfactual(env,a,continuation,anchor_model) for a in candidate_set}
        keep=outcomes[0]['wait']
        gains={a:keep-outcomes[a]['wait'] for a in candidate_set}
        best=min(candidate_set,key=lambda a:(outcomes[a]['wait'],a!=0,a))
        for a in candidate_set:
            truth.append(dict(**key,continuation=continuation,action_id=a,
                changed_vessel=(obs.candidate_ids[a-1] if a else None),
                realized_total_wait=outcomes[a]['wait'],
                realized_total_spent=outcomes[a]['spent'],
                gain_vs_keep=gains[a],gain_vs_keep_scaled=gains[a]/env.reward_scale,
                ex_post_best_action=best,ex_post_best_wait=outcomes[best]['wait']))
        for q_row in qs:
            model=q_row['model'];a=q_row['action_id']
            if not a or a not in gains:continue
            calibration.append(dict(**key,model=model,continuation=continuation,
                action_id=a,candidate_vessel=q_row['candidate_vessel'],
                q_keep=q_row['q_keep'],q_edit=q_row['q_value'],
                q_gap=q_row['q_gap'],
                true_gain_scaled=gains[a]/env.reward_scale,
                true_gain_raw=gains[a],
                q_gap_error=(q_row['q_gap']-gains[a]/env.reward_scale),
                predicted_beneficial=bool(q_row['q_gap']>0.),
                realized_beneficial=bool(gains[a]>env.min_gain),
                predicted_over_margin=bool(q_row['q_gap']>q_row['keep_margin']),
                keep_margin=q_row['keep_margin'],
                point_gain=q_row['point_gain']))
        for model in models:
            d=model_actions[model.label]
            a=d['action']
            available=a in outcomes
            regret=(outcomes[a]['wait']-outcomes[best]['wait']) if available else None
            decisions.append(dict(**key,model=model.label,continuation=continuation,
                selected_action=a,ex_post_best_action=best,
                best_true_gain=max(gains.values()),
                selected_true_gain=(gains[a] if available else None),
                first_action_regret=regret,
                action_labeled=available,
                missed_beneficial_keep=(a==0 and max(gains.values())>env.min_gain),
                harmful_edit=(a!=0 and available and gains[a]<-env.min_gain),
                q_best_edit=d['best_edit'],q_best_gap=d['best_gap'],
                q_keep=d['q_keep'],
                margin_distance=d['margin_distance'],
                keep_margin=model.agent.keep_margin,
                keep_reason=d['reason']))
    return key,qs,truth,calibration,decisions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',required=True,
                   help='E4 run directory@episode, e.g. results/v10_e4_b900_seed43@100 (repeatable)')
    p.add_argument('--anchor-model',default=None,
                   help='Anchor run directory@episode; defaults to the FIRST --model')
    p.add_argument('--cases',nargs='+',default=['../2024LiERLDDE/case_code/8.txt',
                                                 '../2024LiERLDDE/case_code/9.txt'])
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--eta-seeds',default='400001:400003')
    p.add_argument('--budgets',default='2,5,10')
    p.add_argument('--anchors',default='early,keep',
                   help='State banks from frozen anchor checkpoint and/or Keep')
    p.add_argument('--continuations',default='anchor,greedy',
                   help='Paired factual-future continuations; optional keep')
    p.add_argument('--states-per-group',type=int,default=3,
                   help='Temporally stratified legal states per case/scenario/seed/budget/anchor')
    p.add_argument('--max-edit-actions',type=int,default=4,
                   help='Max candidates to label, ranked by PUBLIC point gain')
    p.add_argument('--limit-groups',type=int,default=0,
                   help='Limit case×scenario×ETA×budget groups for smoke runs; 0 = ALL')
    p.add_argument('--device',default='cpu')
    p.add_argument('--skip-checkpoint-hash',action='store_true',
                   help='Not recommended; skip SHA256 verification of local model file')
    p.add_argument('--no-parity',action='store_true',
                   help='Disable exact original E4 val-runs replay check (not recommended)')
    p.add_argument('--out',required=True,help='MUST be a new, empty results directory')
    a=p.parse_args(argv)
    if a.states_per_group<1 or a.max_edit_actions<1 or a.limit_groups<0:
        p.error('Invalid states-per-group, edit-action limit, or group limit')
    anchors=a.anchors.split(',')
    continuations=a.continuations.split(',')
    if not anchors or set(anchors)-{'early','keep'} or len(set(anchors))!=len(anchors):
        p.error('Anchors must be early,keep')
    if not continuations or set(continuations)-{'anchor','greedy','keep'} or len(set(continuations))!=len(continuations):
        p.error('Continuations must be anchor,greedy,keep')
    cases=[Path(f).resolve() for f in a.cases]
    for f in cases:
        if not f.is_file():p.error('Case file not found: '+str(f))
    eta_seeds=ints(a.eta_seeds)
    budgets=ints(a.budgets)
    if any(b<0 for b in budgets):p.error('Negative budget')
    scenarios=a.scenarios.split(',')
    if set(scenarios)-{'shock','repeated','small'}:p.error('Invalid scenario')
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output already exists and is nonempty; use unique --out')
    torch,read_case,generate_scenario,IAMUEventEnv,Agent=imports()
    models=[load_model(s,a.device,verify_checksum=not a.skip_checkpoint_hash,Agent=Agent)
            for s in a.model]
    if len(set(m.label for m in models))!=len(models):
        p.error('Duplicate model labels')
    anchor_model=(load_model(a.anchor_model,a.device,
                 verify_checksum=not a.skip_checkpoint_hash,Agent=Agent)
                  if a.anchor_model else models[0])
    env_kwargs=consistent_environment([*models,anchor_model])
    train=models[0].run_args
    out.mkdir(parents=True)
    metadata=dict(config=vars(a),environment=env_kwargs,
        checkpoint_info=[dict(label=m.label,path=str(m.snapshot),sha256=m.checksum,
                              episode=m.episode,gradient_steps=m.gradient_steps,
                              run_manifest_sha256=digest(m.folder/'manifest.json'))
                         for m in models],
        anchor_checkpoint=anchor_model.label,
        script_sha256=digest(Path(__file__)),
        core_sha256=digest(Path(__file__).parent/'diagnostic_core.py'),
        case_sha256={str(f):digest(f) for f in cases},
        evaluation_warning=('Known realized future is used ONLY to evaluate paired labels. '
            'Not a true conditional expectation. All sampled states are selected before '
            'inspection of counterfactual outcomes; training models are frozen.'))
    (out/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    episodes=[];events=[];bank=[];q_rows=[];truth=[];calibration=[];decisions=[]
    coverage=[];parity=[]; groups=0
    for case in cases:
        vessels,port=read_case(case)
        for sc in scenarios:
            for eta_seed in eta_seeds:
                scenario=generate_scenario(vessels,port,label=sc,seed=eta_seed,
                        updates_per_vessel=int(train['updates']),
                        uncertainty=float(train['validation_uncertainty']))
                for budget in budgets:
                    groups+=1
                    if a.limit_groups and groups>a.limit_groups:
                        break
                    key=dict(case=case.stem,scenario=sc,eta_seed=eta_seed,budget=budget)
                    for m in models:
                        episode,event_rows=run_trajectory(
                            scenario,budget,m,env_kwargs,IAMUEventEnv,torch)
                        episode={**key, **episode}
                        episodes.append(episode)
                        events.extend([{**key, **r} for r in event_rows if r['n_legal_edits']>0
                                       or r['keep_reason']=='budget_exhausted'])
                        original=m.validation_rows.get((case.stem,sc,eta_seed,budget))
                        if not a.no_parity and original is None:
                            raise AssertionError(f'E4 original validation row missing: {m.label} {key}')
                        if original:
                            delta=abs(episode['weighted_wait']-float(original['wait']))
                            equal=(delta<=1e-6 and episode['spent']==int(original['spent']))
                            parity.append(dict(**key,model=m.label,wait_diff=delta,
                                  spent_matches=episode['spent']==int(original['spent']),
                                  matches=equal))
                            if not equal and not a.no_parity:
                                raise AssertionError(f'Original E4 validation parity failure: '
                                                     f'{m.label} {key} wait_diff={delta}')
                    for anchor_name in anchors:
                        states,n_eligible=collect_states(scenario,budget,anchor_name,
                            anchor_model,env_kwargs,n_states=a.states_per_group,
                            IAMUEventEnv=IAMUEventEnv)
                        coverage.append(dict(**key,anchor=anchor_name,
                                    n_eligible=n_eligible,n_selected=len(states)))
                        for state in states:
                            state_key,qs,cf,cal,dec=analyze_bank(
                                state,models,anchor_model,continuations,torch=torch,
                                max_edit_actions=a.max_edit_actions,
                                case_name=case.stem,scenario_name=sc,
                                eta_seed=eta_seed,budget=budget,anchor_name=anchor_name)
                            bank.append(state_key)
                            q_rows.extend(qs)
                            truth.extend(cf)
                            calibration.extend(cal)
                            decisions.extend(dec)
                    # Partial output remains inspectable if a long run stops.
                    write_csv(out/'trajectory_episodes.partial.csv',episodes)
                    write_csv(out/'coverage.partial.csv',coverage)
                    write_csv(out/'calibration_pairs.partial.csv',calibration)
                    write_csv(out/'decision_regret.partial.csv',decisions)
                    print(f'[E4-D] group={groups} case={case.stem} {sc} '
                          f'ETA={eta_seed} B={budget} states={len(bank)} '
                          f'parity={len(parity)}',flush=True)
                if a.limit_groups and groups>=a.limit_groups:break
            if a.limit_groups and groups>=a.limit_groups:break
        if a.limit_groups and groups>=a.limit_groups:break
    if not episodes:
        raise RuntimeError('No evaluated episodes')
    write_csv(out/'trajectory_episodes.csv',episodes)
    write_csv(out/'trajectory_events.csv',events)
    write_csv(out/'state_bank.csv',bank)
    write_csv(out/'q_values.csv',q_rows)
    write_csv(out/'counterfactual.csv',truth)
    write_csv(out/'calibration_pairs.csv',calibration)
    write_csv(out/'decision_regret.csv',decisions)
    write_csv(out/'coverage.csv',coverage)
    write_csv(out/'validation_parity.csv',parity)
    valid_decisions=[r for r in decisions if r['action_labeled']]
    ep_summary,cal_summary,dec_summary=group_summaries(episodes,events,calibration,valid_decisions)
    write_csv(out/'trajectory_summary.csv',ep_summary)
    write_csv(out/'calibration_summary.csv',cal_summary)
    write_csv(out/'decision_summary.csv',dec_summary)
    # Matched Q-gap shift against the FIRST model, including every legal Edit.
    by_state=defaultdict(dict)
    for x in q_rows:
        state_key=(x['case'],x['scenario'],x['eta_seed'],x['budget'],
                   x['anchor'],x['event_index'],x['action_id'])
        by_state[state_key][x['model']]=x
    matched=[]
    for k,row_map in by_state.items():
        origin=row_map[models[0].label]
        for m in models[1:]:
            dest=row_map[m.label]
            matched.append(dict(case=k[0],scenario=k[1],eta_seed=k[2],budget=k[3],
                    anchor=k[4],event_index=k[5],action_id=k[6],
                    reference_model=models[0].label,comparison_model=m.label,
                    reference_q_keep=origin['q_keep'],comparison_q_keep=dest['q_keep'],
                    reference_q_gap=origin['q_gap'],comparison_q_gap=dest['q_gap'],
                    q_keep_shift=dest['q_keep']-origin['q_keep'],
                    q_gap_shift=dest['q_gap']-origin['q_gap'],
                    reference_action=origin['selected_action'],
                    comparison_action=dest['selected_action'],
                    action_flipped=origin['selected_action']!=dest['selected_action'],
                    reference_pass_margin=origin['q_gap']>origin['keep_margin'],
                    comparison_pass_margin=dest['q_gap']>dest['keep_margin']))
    write_csv(out/'matched_q_shift.csv',matched)
    report=dict(groups=groups,episodes=len(episodes),actionable_events=len(events),
         selected_states=len(bank),labeled_candidate_pairs=len(calibration),
         matched_q_rows=len(matched),
         selected_by_anchor={name:sum(x['n_selected'] for x in coverage if x['anchor']==name)
                             for name in anchors},
         original_validation_checks=len(parity),
         all_parity_pass=bool(parity) and all(x['matches'] for x in parity),
         generated_files=['trajectory_summary.csv','calibration_summary.csv',
                          'decision_summary.csv','matched_q_shift.csv',
                          'q_values.csv','counterfactual.csv'])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE] '+json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
