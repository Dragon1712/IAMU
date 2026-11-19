#!/usr/bin/env python3
"""Multi-label forced Keep audit from trusted local V1.1 Replay checkpoint.

Primary labels are an EXCLUSIVE presentation grouping; all overlapping flags
are retained separately so budget exhaustion + all-started is not hidden.
"""
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork',required=True)
    p.add_argument('--trust-local-training-state',action='store_true')
    p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    if not args.trust_local_training_state:p.error('Only use trusted locally generated pickle state')
    import torch
    state=torch.load(args.fork,map_location='cpu',weights_only=False)
    items=state.get('replay_items',[])
    if not items:raise RuntimeError('No replay transitions')
    primary=Counter();labels=Counter();combinations=Counter();events=defaultdict(list)
    for tr in items:
        flags=getattr(tr,'reason_flags',None)
        if not isinstance(flags,dict):raise RuntimeError('Replay lacks V1.1 reason flags: do not use a legacy fork')
        has_edit=bool(tr.obs_mask[1:].any())
        if has_edit:
            category='actionable_edit' if int(tr.action)!=0 else 'actionable_keep'
        elif flags.get('initial_event'):
            category='initial'
        elif not flags.get('decision_allowed'):
            category='decision_clock_blocked'
        elif flags.get('budget_exhausted'):
            category='budget_exhausted'
        elif flags.get('all_started'):
            category='all_started'
        elif flags.get('has_raw_proposal') and flags.get('screened_proposals',0):
            category='all_point_screened'
        elif flags.get('unchanged_proposals',0):
            category='unchanged_or_no_change'
        else:
            category='no_feasible_proposal'
        primary[category]+=1
        active=tuple(sorted(k for k in ('initial_event','decision_allowed','budget_exhausted',
            'all_started','has_raw_proposal','has_retained_edit','search_exhausted') if bool(flags.get(k))))
        combinations[active]+=1
        for key in ('initial_event','decision_allowed','budget_exhausted','all_started',
                    'has_raw_proposal','has_retained_edit','search_exhausted'):
            if flags.get(key):labels[key]+=1
        events[getattr(tr,'episode_id',-1)].append((int(flags.get('event_index',0)),category,has_edit))
    tails=Counter()
    for ep,episode in events.items():
        max_event=max(i for i,_,_ in episode)
        cutoff=.75*max_event
        for index,category,has_edit in episode:
            if index>=cutoff:
                tails[category]+=1
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):p.error('Output already exists')
    out.mkdir(parents=True,exist_ok=True)
    def write(file,rs):
        with (out/file).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    write('primary_reasons.csv',[dict(reason=k,n=v,fraction=v/len(items),tail_quartile_n=tails.get(k,0)) for k,v in sorted(primary.items())])
    write('overlap_flags.csv',[dict(flag=k,n=v,fraction=v/len(items)) for k,v in sorted(labels.items())])
    write('reason_combinations.csv',[dict(flags='|'.join(k),n=v,fraction=v/len(items)) for k,v in combinations.most_common()])
    (out/'summary.json').write_text(json.dumps(dict(total=len(items),episode_count=len(events),
        forced_keep=sum(v for k,v in primary.items() if not k.startswith('actionable_')),
        budget_exhausted_any=labels['budget_exhausted'],
        all_started_any=labels['all_started'],
        forced_tail_quartile=sum(v for k,v in tails.items() if not k.startswith('actionable_')),
        clarification='Labels are overlapping; primary reasons are for display only'),indent=2))
    print('[DONE]',out,'transitions=',len(items))
if __name__=='__main__':main()
