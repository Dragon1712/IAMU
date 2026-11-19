"""Combine common-state Q gaps with single-path fixed-continuation references.

No model ranking by expected Q*: each label is a realized, path-conditional
cost difference. States from the same trajectory are correlated.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from .counterfactual_probe import LABEL_KIND


def read_csv(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields):
    with open(path,'w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def compare(qgap_rows,reference_rows,cost_tol=1e-6):
    if cost_tol<0:raise ValueError('cost_tol must be >=0')
    refs=defaultdict(dict)
    for r in reference_rows:
        if r['label_kind']!=LABEL_KIND:raise ValueError('Invalid reference semantics')
        k=(r['state_id'],r['candidate_fingerprint'],r['continuation_policy'])
        a=int(r['action_index'])
        if a in refs[k]:raise ValueError('Duplicate reference action')
        refs[k][a]=float(r['delta_cost_path_pi'])
    continuations={k[2] for k in refs}
    if len(continuations)!=1:raise ValueError('Mixed continuation policies')
    continuation=next(iter(continuations)) if continuations else None
    by_state=defaultdict(list)
    for r in qgap_rows:
        k=(r['checkpoint_label'],r['state_id'],r['candidate_fingerprint'])
        by_state[k].append(r)
    state_metrics=[];action_metrics=[]
    for (checkpoint,state_id,candidate_fp),rs in sorted(by_state.items()):
        reference_key=(state_id,candidate_fp,continuation)
        if reference_key not in refs:raise ValueError('Missing shared pathwise reference')
        ref=refs[reference_key]
        if 0 not in ref or abs(ref[0])>cost_tol:raise ValueError('No valid Keep baseline')
        if not rs or any(int(r['action_index'])==0 for r in rs):
            raise ValueError('Q-gap rows must contain only Edit actions')
        actions={int(r['action_index']) for r in rs}
        if len(actions)!=len(rs) or set(ref)!={0,*actions}:
            raise ValueError('Action set differs between Q-gap and reference')
        selected=int(rs[0]['selected_action'])
        if selected not in ref:raise ValueError('Selected action absent in reference')
        if any(int(r['selected_action'])!=selected for r in rs):
            raise ValueError('Selected action inconsistent within state')
        best_gain=max([0.,*(ref[a] for a in actions)])
        selected_gain=ref[selected]
        missed_gain=best_gain if selected==0 else 0.
        harm_edit=bool(selected>0 and selected_gain<-cost_tol)
        wrong_keep=bool(selected==0 and best_gain>cost_tol)
        first=rs[0]
        state_metrics.append(dict(checkpoint_label=checkpoint,state_id=state_id,
            continuation_policy=continuation,scenario=first['scenario'],
            initial_budget=int(first['initial_budget']),
            remaining_budget=int(first['remaining_budget']),
            selected_action=selected,best_path_gain=best_gain,selected_path_gain=selected_gain,
            pathwise_wrong_keep=int(wrong_keep),pathwise_harmful_edit=int(harm_edit),
            missed_gain=missed_gain,decision_regret=max(0.,best_gain-selected_gain),
            n_evaluated_edits=len(actions)))
        for r in rs:
            action=int(r['action_index']);gain=ref[action]
            action_metrics.append(dict(checkpoint_label=checkpoint,state_id=state_id,
              action_index=action,scenario=first['scenario'],
              initial_budget=first['initial_budget'],remaining_budget=first['remaining_budget'],
              delta_q=float(r['delta_q']),delta_cost_path_pi=gain,
              gap_path_residual=float(r['delta_q'])-gain/100.,
              label_kind=LABEL_KIND,continuation_policy=continuation))
    return state_metrics,action_metrics


def summarize(states):
    groups=defaultdict(list)
    for s in states:
        for key in (('ALL','ALL','ALL'),
                    (s['scenario'],str(s['initial_budget']),str(s['remaining_budget']))):
            groups[(s['checkpoint_label'],*key)].append(s)
    rows=[]
    for (cp,sc,b,r),g in sorted(groups.items()):
        keep=[x for x in g if x['selected_action']==0]
        edit=[x for x in g if x['selected_action']>0]
        rows.append(dict(checkpoint_label=cp,scenario=sc,initial_budget=b,
            remaining_budget=r,n_states=len(g),n_keep=len(keep),n_edit=len(edit),
            wrong_keep_count=sum(x['pathwise_wrong_keep'] for x in keep),
            wrong_keep_rate=(sum(x['pathwise_wrong_keep'] for x in keep)/len(keep) if keep else ''),
            harmful_edit_count=sum(x['pathwise_harmful_edit'] for x in edit),
            harmful_edit_rate=(sum(x['pathwise_harmful_edit'] for x in edit)/len(edit) if edit else ''),
            mean_missed_gain=mean(x['missed_gain'] for x in g),
            mean_decision_regret=mean(x['decision_regret'] for x in g)))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qgap',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--cost-tol',type=float,default=1e-6);p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    path=Path(a.out)
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True)
    states,actions=compare(read_csv(a.qgap),read_csv(a.labels),cost_tol=a.cost_tol)
    summary=summarize(states)
    write_csv(path/'decision_state_metrics.csv',states,list(states[0]) if states else ['state_id'])
    write_csv(path/'action_path_residuals.csv',actions,list(actions[0]) if actions else ['state_id'])
    write_csv(path/'stratified_summary.csv',summary,list(summary[0]) if summary else ['checkpoint_label'])
    manifest=dict(label_kind=LABEL_KIND,summary_distribution='prestratified_probe_bank',
                  statistical_independence='none assumed among states in same episode',
                  n_state_model_pairs=len(states),n_action_model_pairs=len(actions),
                  cost_tol=a.cost_tol,expected_q_inference=False)
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (path/'complete.json').write_text(json.dumps(dict(completed=True,n_states=len(states))))
    print(f'[diagnostic analysis] states={len(states)} actions={len(actions)}')

if __name__=='__main__':main()
