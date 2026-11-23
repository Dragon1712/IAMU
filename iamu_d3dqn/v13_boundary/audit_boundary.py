"""Budget-one frozen-critic Keep/Edit SMDP branch value diagnosis.

Every branch starts from the same saved physical snapshot and exogenous tape.
Its rollout suffix starts at that branch's own next macro endpoint. Thus:
(target_edit - target_keep) - (full_edit - full_keep)
= bootstrap_error_edit - bootstrap_error_keep, on every row.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .core import BranchValue,macro_steps,paired_identity
from .utils import boundary_provenance
from v13_diagnostics.common import load_model,load_trusted,verify_v12_scientific_model,write_csv,atomic_json,action_descriptor
from v13_diagnostics.future_tapes import draw_tape,apply_tape,tape_hash
from v13_diagnostics.counterfactual import predict_pair


def greedy_action(model,obs):
    with torch.no_grad():
        g=torch.as_tensor(obs.global_features,dtype=torch.float32)[None]
        acts=torch.as_tensor(obs.action_features,dtype=torch.float32)[None]
        mask=torch.as_tensor(obs.mask,dtype=torch.bool)[None]
        q=model.online(g,acts,mask).masked_fill(~mask,-torch.inf).clone()
        q[:,0]+=model.execution_margin
        return int(q.argmax(dim=1)[0])


def next_bootstrap(model,obs):
    with torch.no_grad():
        g=torch.as_tensor(obs.global_features,dtype=torch.float32)[None]
        a=torch.as_tensor(obs.action_features,dtype=torch.float32)[None]
        mask=torch.as_tensor(obs.mask,dtype=torch.bool)[None]
        scores=model.online(g,a,mask).masked_fill(~mask,-torch.inf).clone()
        scores[:,0]+=model.bootstrap_margin
        ix=int(scores.argmax(dim=1)[0])
        v=float(model.target(g,a,mask)[0,ix])
    return v,ix


def rollout_suffix(env,model,continuation,gamma):
    total=0.;power=1.;steps=0;first_edit=None;last_info=None
    while not env.done:
        obs=env._observation()
        action=greedy_action(model,obs) if continuation=='self' else int(env.point_greedy_action())
        if not obs.mask[action]:raise AssertionError('Illegal continuation edit')
        if action and first_edit is None:
            first_edit=dict(time=float(obs.time),event_index=int(env.event_index),
                            action=action_descriptor(env,action))
        _,r,done,info=env.step(action)
        total+=power*float(r);power*=gamma;steps+=1;last_info=info
    return total,steps,first_edit,last_info


def branch_value(snapshot,tape,action,model,continuation,gamma):
    env=apply_tape(snapshot,tape)
    macro=macro_steps(env,action,model.representation,gamma)
    v,next_act=(0.,None) if macro['done'] else next_bootstrap(model,macro['next_obs'])
    suffix,steps,first_edit,info=rollout_suffix(env,model,continuation,gamma)
    b=BranchValue(float(macro['reward']),float(macro['discount']),v,suffix,
                  bool(macro['done']),int(macro['n']))
    terminal_info=info or macro['last_info']
    return b,dict(next_action=next_act,suffix_raw_steps=steps,
                  first_suffix_edit=first_edit,
                  weighted_wait=terminal_info.get('weighted_wait'))


def restore_matching_target(model,path):
    # Only user-generated trusted local .pt; never deserialize untrusted pickle.
    state=load_trusted(path)
    if state.get('protocol')!=12:raise ValueError('Not a trusted V1.2 full-state checkpoint')
    online=model.online.state_dict()
    if set(state['online'])!=set(online) or any(
          not torch.equal(v,state['online'][k]) for k,v in online.items()):
        raise ValueError('Full-state online differs from model checkpoint; gradient mismatch')
    model.target.load_state_dict(state['target'],strict=True)
    model.target.eval()


def run(probes,models,continuations=('self','greedy'),tapes=8,tape_seed=910000,
        uncertainty=8.,updates=2,gamma=1.,max_states=None,start_index=0,stop_index=0):
    if tapes<1:raise ValueError('Require tapes>=1')
    selected=[p for p in probes if int(p['source_budget'])==1]
    end=stop_index or len(selected)
    if not 0<=start_index<end<=len(selected):raise ValueError('Invalid budget-one probe slice')
    selected=selected[start_index:end]
    if max_states is not None and max_states>0:selected=selected[:max_states]
    branches=[];pairs=[]
    for p in selected:
        snapshot=p['snapshot'];sid=p['state_id'];edit=int(p['selected_edit'])
        obs=snapshot._observation()
        if int(obs.remaining_budget)!=1 or not obs.mask[edit]:raise AssertionError('Stale probe snapshot')
        for j in range(tapes):
            tape=draw_tape(snapshot,sid,j,base_seed=tape_seed,uncertainty=uncertainty,
                           updates_per_vessel=updates)
            th=tape_hash(tape)
            for name,model in models.items():
                q_keep,q_edit=predict_pair(model,obs,edit)
                for cont in continuations:
                    b_keep,kinfo=branch_value(snapshot,tape,0,model,cont,gamma)
                    b_edit,einfo=branch_value(snapshot,tape,edit,model,cont,gamma)
                    difference=paired_identity(b_keep,b_edit)
                    shared=dict(state_id=sid,episode_id=p['episode_id'],raw_step_index=p['raw_step_index'],
                      case=p['case'],scenario=p['scenario'],source_budget=1,tape_index=j,
                      tape_sha256=th,model=name,checkpoint=name,
                      model_representation=model.representation,continuation=cont,
                      source_action_descriptor=json.dumps(p['action_descriptor'],sort_keys=True))
                    for label,b,meta,q in (('Keep',b_keep,kinfo,q_keep),('Edit',b_edit,einfo,q_edit)):
                        branches.append(shared|dict(branch=label,interval_reward=b.interval_reward,
                          bootstrap_discount=b.discount,next_bootstrap_value=b.bootstrap_value,
                          true_suffix_return=b.suffix_return,bootstrap=b.discount*b.bootstrap_value,
                          scaled_suffix=b.discount*b.suffix_return,bootstrap_error=b.bootstrap_error,
                          estimated_wait_cost_understatement=100.*b.bootstrap_error,
                          td_target=b.td_target,full_return=b.full_return,q_taken=q,
                          n_raw_steps=b.n_raw_steps,done=int(b.done),
                          suffix_raw_steps=meta['suffix_raw_steps'],
                          first_suffix_edit=json.dumps(meta['first_suffix_edit'],sort_keys=True),
                          terminal_weighted_wait=meta['weighted_wait']))
                    pairs.append(shared|dict(q_keep=q_keep,q_edit=q_edit,
                      predicted_gap=q_edit-q_keep,**difference,
                      predicted_gap_error=(q_edit-q_keep)-difference['return_gap'],
                      keep_cost_understatement=100.*b_keep.bootstrap_error,
                      edit_cost_understatement=100.*b_edit.bootstrap_error,
                      keep_done=int(b_keep.done),edit_done=int(b_edit.done),
                      keep_interval_length=b_keep.n_raw_steps,
                      edit_interval_length=b_edit.n_raw_steps,
                      keep_suffix_first_edit=json.dumps(kinfo['first_suffix_edit'],sort_keys=True),
                      keep_total_wait=kinfo['weighted_wait'],edit_total_wait=einfo['weighted_wait']))
    return branches,pairs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes',required=True)
    p.add_argument('--checkpoint',nargs='+',required=True,help='LABEL=path, e.g. A2_EP100=...')
    p.add_argument('--full-state',nargs='*',default=[],help='LABEL=trusted training_state.pt')
    p.add_argument('--continuations',default='self,greedy')
    p.add_argument('--tapes',type=int,default=8)
    p.add_argument('--seed',type=int,default=910000)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--max-states',type=int,default=0)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--stop-index',type=int,default=0)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    cont=tuple(a.continuations.split(','))
    if set(cont)-{'self','greedy'} or not 0<a.gamma<=1:p.error('Invalid continuation/gamma')
    full={}
    for item in a.full_state:
        k,sep,v=item.partition('=')
        if not sep or k in full:p.error('Duplicate/malformed full-state')
        full[k]=v
    models={};inputs=[a.probes]
    for item in a.checkpoint:
        name,sep,file=item.partition('=')
        if not sep or not name or name in models:p.error('Duplicate/malformed checkpoint')
        model=load_model(file)
        verify_v12_scientific_model(model,None,a.gamma)
        if model.representation not in ('primitive','budget_tail','decision_epoch'):
            raise ValueError('Invalid checkpoint representation')
        if name in full:restore_matching_target(model,full[name]);inputs.append(full[name])
        inputs.append(file);models[name]=model
    if set(full)-set(models):p.error('Unknown full-state label')
    bank=load_trusted(a.probes)
    branches,pairs=run(bank['probes'],models,cont,a.tapes,a.seed,a.uncertainty,a.updates,
                       a.gamma,a.max_states or None,a.start_index,a.stop_index)
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'branch_values.csv',branches)
    write_csv(out/'paired_gap.csv',pairs)
    zero=[];group={}
    for row in pairs:group.setdefault((row['state_id'],row['model'],row['continuation']),[]).append(row)
    for (sid,model,continuation),rows in group.items():
        if len(rows)==a.tapes and all(abs(r['return_gap'])<1e-9 for r in rows):
            zero.extend(dict(state_id=sid,model=model,continuation=continuation,
                tape_index=r['tape_index'],return_gap=r['return_gap'],
                first_edit_after_keep=r['keep_suffix_first_edit'],
                keep_wait=r['keep_total_wait'],edit_wait=r['edit_total_wait']) for r in rows)
    write_csv(out/'zero_gap_trace.csv',zero,
      columns=['state_id','model','continuation','tape_index','return_gap',
               'first_edit_after_keep','keep_wait','edit_wait'])
    residual=max((abs(x['identity_residual']) for x in pairs),default=0.)
    atomic_json(out/'summary.json',dict(n_states=len({x['state_id'] for x in pairs}),
      n_pairs=len(pairs),n_components=len(branches),max_identity_residual=residual,
      models=list(models),continuations=cont,
      target_kind={k:'matched_historical_target' if k in full else 'target_equal_online'
                   for k in models},
      caveat='Synthetic conditional future; not calibrated posterior or optimal Q'))
    atomic_json(out/'manifest.json',boundary_provenance(a,inputs=inputs))
    print('[boundary]',len(pairs),'pairs, largest Bellman identity residual:',residual)
if __name__=='__main__':main()
