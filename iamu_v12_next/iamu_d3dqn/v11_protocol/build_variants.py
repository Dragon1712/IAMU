#!/usr/bin/env python3
"""Fail-closed additive source builder for IAMU commit 52dec41ac8ed interfaces.

Generates iamu_dynamic/rl_{env,ddqn,experiment}_v11.py and
parallel_validation_v11.py, without editing historical sources. Edits are
matched against exact anchors; incompatible upstream versions abort.
"""
from pathlib import Path
import argparse,hashlib,json,py_compile

ROOT=Path(__file__).resolve().parents[1]

def change(source, old, new):
    count=source.count(old)
    if count!=1:raise RuntimeError(f'Expected unique source anchor, matched {count}: {old[:95]!r}')
    return source.replace(old,new,1)

def replace_span(source,begin,end,replacement):
    if source.count(begin)!=1 or source.count(end)!=1:raise RuntimeError('Version drift: cannot locate exact method boundaries')
    i=source.index(begin);j=source.index(end,i)
    return source[:i]+replacement+source[j:]

OBS_OLD='''    candidate_ids: tuple[int, ...]      # index 1.. corresponds to valid actions

    def copy(self):
        return RLObservation(self.global_features.copy(),
                             self.action_features.copy(), self.mask.copy(),
                             self.time, self.event_type, self.remaining_budget,
                             self.candidate_ids)
'''
OBS_NEW='''    candidate_ids: tuple[int, ...]      # index 1.. corresponds to valid actions
    decision_flags: dict | None = None
    physical_mask: np.ndarray | None = None
    heuristic_mask: np.ndarray | None = None

    def copy(self):
        return RLObservation(self.global_features.copy(),
                             self.action_features.copy(), self.mask.copy(),
                             self.time, self.event_type, self.remaining_budget,
                             self.candidate_ids,
                             dict(self.decision_flags or {}),
                             None if self.physical_mask is None else self.physical_mask.copy(),
                             None if self.heuristic_mask is None else self.heuristic_mask.copy())
'''

ELIGIBILITY='''    def _eligible(self):
        ranked=[i for i in self.ids if i not in self.runtime.started]
        vid=self.event_vessel_id
        ranked.sort(key=lambda i:(i!=vid,
                       not (self.published_eta[i]>self.plan[i].start+EPS),
                       self.plan[i].start,i))
        return ranked  # Full ranking; stop ONLY after K *retained* proposals.

    def _proposal_with_reason(self, vessel_id):
        # Equivalent to SlotPlanner.single_move, but preserves the reason when
        # the resulting proposal is None. Do not invoke assign twice.
        from .core import assignments_equal
        fixed=[a for j,a in self.plan.items() if j!=vessel_id]
        try:
            proposed=self.planner.assign(vessel_id,
                max(self.event_time,self.published_eta[vessel_id]),fixed)
        except RuntimeError:
            return None,'infeasible'
        if assignments_equal(proposed,self.plan[vessel_id]):
            return None,'unchanged'
        new=dict(self.plan);new[vessel_id]=proposed
        validate_plan(new,self.vessels,self.scenario.port)
        return new,'valid'

'''

CANDIDATES='''    def _refresh_candidates(self):
        self._candidates=[];self._candidate_ids=();self._candidate_data=[]
        initial=self.event_type=='initial'
        clock_allowed=(self.decision_clock!='clock' or self.event_type=='clock')
        eta_allowed=(self.decision_clock!='eta' or self.event_type=='eta')
        allowed=not(initial or self.done) and clock_allowed and eta_allowed
        ranked=self._eligible()
        flags=dict(initial_event=initial,decision_allowed=allowed,
            budget_exhausted=self.remaining_budget==0,all_started=not bool(ranked),
            has_raw_proposal=False,has_retained_edit=False,
            search_exhausted=True,pending_ships=len(ranked),ships_examined=0,
            event_index=int(self.event_index),event_time=float(self.event_time),
            unchanged_proposals=0,infeasible_proposals=0,screened_proposals=0,
            raw_proposals=0,returned_candidates=0)
        self._candidate_point_gains=[]
        if not allowed or self.remaining_budget==0 or not ranked:
            self.decision_flags=flags
            self._physical_mask=np.zeros(self.candidate_limit+1,dtype=np.bool_)
            self._physical_mask[0]=True
            self._heuristic_mask=self._physical_mask.copy()
            return
        base_starts=base_costs=None
        base_score=0.
        positives=[];others=[];selected=[]
        quota_positive=(self.candidate_limit+1)//2
        quota_other=self.candidate_limit-quota_positive
        for vessel_id in ranked:
            flags['ships_examined']+=1
            plan,reason=self._proposal_with_reason(vessel_id)
            if plan is None:
                flags[reason+'_proposals']+=1
                continue
            flags['raw_proposals']+=1
            flags['has_raw_proposal']=True
            if base_starts is None:
                base_starts,base_costs,base_score=self._forecast_projection(self.plan)
            starts,costs,cost=self._forecast_projection(plan)
            gain=base_score-cost
            positive=bool(gain>self.min_gain+1e-12)
            if self.candidate_screen=='positive' and not positive:
                flags['screened_proposals']+=1
                self.screened_proposals+=1
                continue
            affected=[j for j in starts if j!=vessel_id]
            worsened=[j for j in affected if starts[j]>base_starts[j]+EPS]
            delays=max((starts[j]-base_starts[j] for j in worsened),default=0.)
            data=dict(gain=gain,base_score=base_score,new_score=cost,
                external_worse_count=len(worsened),external_max_delay=delays,
                external_worse_cost=sum(max(0.,costs[j]-base_costs[j]) for j in affected),
                external_relief_cost=sum(max(0.,base_costs[j]-costs[j]) for j in affected))
            entry=(vessel_id,plan,data,positive)
            if self.candidate_screen=='mixed':
                (positives if positive else others).append(entry)
                if len(positives)>=quota_positive and len(others)>=quota_other:
                    flags['search_exhausted']=flags['ships_examined']==len(ranked)
                    break
            else:
                selected.append(entry)
                if len(selected)>=self.candidate_limit:
                    flags['search_exhausted']=flags['ships_examined']==len(ranked)
                    break
        if self.candidate_screen=='mixed':
            selected=positives[:quota_positive]+others[:quota_other]
            # If either category is absent, fill remaining slots with available
            # physical proposals from the other category. Rank is stable.
            if len(selected)<self.candidate_limit:
                chosen={i for i,_,_,_ in selected}
                selected.extend(x for x in (positives+others) if x[0] not in chosen)
            selected=selected[:self.candidate_limit]
            rank={v:i for i,v in enumerate(ranked)}
            selected.sort(key=lambda x:rank[x[0]])
        self._candidate_ids=tuple(x[0] for x in selected)
        self._candidates=[x[1] for x in selected]
        self._candidate_data=[x[2] for x in selected]
        self._candidate_point_gains=[float(x[2]['gain']) for x in selected]
        flags['returned_candidates']=len(selected)
        flags['has_retained_edit']=bool(selected)
        self.decision_flags=flags
        self._physical_mask=np.zeros(self.candidate_limit+1,dtype=np.bool_)
        self._physical_mask[:len(selected)+1]=True
        self._heuristic_mask=np.zeros(self.candidate_limit+1,dtype=np.bool_)
        self._heuristic_mask[0]=True
        for idx,entry in enumerate(selected,1):
            self._heuristic_mask[idx]=entry[3]
        if not flags['search_exhausted'] and len(selected)<self.candidate_limit:
            raise AssertionError('Candidate search stopped short of K without exhaustion')

'''

def make_env(src):
    src=change(src,OBS_OLD,OBS_NEW)
    src=change(src,"if candidate_screen not in {'none','positive'}:","if candidate_screen not in {'none','positive','mixed'}:")
    src=change(src,"raise ValueError('candidate_screen must be none or positive')", "raise ValueError('candidate_screen must be none, positive, or mixed')")
    src=replace_span(src,'    def _eligible(self):\n','    def _forecast_projection(self, plan):\n',ELIGIBILITY)
    src=replace_span(src,'    def _refresh_candidates(self):\n','    def _accrue_started(self):\n',CANDIDATES)
    src=change(src,'''        return RLObservation(g,features,mask,t,self.event_type,
                             self.remaining_budget,self._candidate_ids)''','''        if not np.array_equal(mask,self._physical_mask):
            raise AssertionError('Physical action mask/selected proposals diverged')
        return RLObservation(g,features,mask,t,self.event_type,
                             self.remaining_budget,self._candidate_ids,
                             dict(self.decision_flags),self._physical_mask.copy(),
                             self._heuristic_mask.copy())''')
    src=change(src,'''        self.action_audit.append(dict(time=self.event_time,
               event_type=self.event_type,action=action,vessel_id=changed,
               estimated_gain=estimated_gain,spent=self.used,
               remaining_budget=self.remaining_budget))''','''        self.action_audit.append(dict(time=self.event_time,
               event_type=self.event_type,action=action,vessel_id=changed,
               estimated_gain=estimated_gain,spent=self.used,
               remaining_budget=self.remaining_budget,
               **{('candidate_'+k):v for k,v in observation.decision_flags.items()}))''')
    return src


def make_ddqn(src):
    src=change(src,'from .rl_env import GLOBAL_DIM, ACTION_DIM, RLObservation',
               'from .rl_env_v11 import GLOBAL_DIM, ACTION_DIM, RLObservation')
    src=change(src,'''    done: bool


class ReplayBuffer:''','''    done: bool
    reason_flags: dict | None = None
    episode_id: int | None = None


class ReplayBuffer:''')
    src=change(src,"                 seed:int=0,device:str='cpu',keep_margin:float=0.,\n                 q_network:str='ddqn'):",
               "                 seed:int=0,device:str='cpu',keep_margin:float=0.,\n                 q_network:str='ddqn',execution_margin:float|None=None,\n                 bootstrap_margin:float|None=None):")
    src=change(src,"        self.keep_margin=float(keep_margin)\n        self.rng=random.Random(seed+193)",
               "        self.execution_margin=float(keep_margin if execution_margin is None else execution_margin)\n        self.bootstrap_margin=float(keep_margin if bootstrap_margin is None else bootstrap_margin)\n        if not np.isfinite(self.execution_margin) or self.execution_margin<0 or not np.isfinite(self.bootstrap_margin) or self.bootstrap_margin<0:\n            raise ValueError('Invalid margins')\n        self.keep_margin=self.execution_margin  # historical evaluation API\n        self.rng=random.Random(seed+193)")
    src=change(src,'masked[0]+=self.keep_margin','masked[0]+=self.execution_margin')
    src=change(src,'next_scores[:,0]+=self.keep_margin','next_scores[:,0]+=self.bootstrap_margin')
    src=change(src,'    def update(self,transitions:list[Transition]):','    def update(self,transitions:list[Transition],weights=None):')
    src=change(src,'        loss=F.smooth_l1_loss(predicted,bellman)','''        per_item=F.smooth_l1_loss(predicted,bellman,reduction='none')
        if weights is None:
            loss=per_item.mean()
        else:
            w=torch.as_tensor(weights,dtype=torch.float32,device=self.device)
            if w.shape!=per_item.shape or not bool(torch.isfinite(w).all()) or not bool((w>=0).all()):
                raise ValueError('Invalid importance weights')
            loss=(w*per_item).mean()''')
    src=change(src,'''                    action_dim=ACTION_DIM,keep_margin=self.keep_margin,
                    q_network=self.q_network,''','''                    action_dim=ACTION_DIM,keep_margin=self.execution_margin,
                    execution_margin=self.execution_margin,
                    bootstrap_margin=self.bootstrap_margin,
                    protocol_version=11,q_network=self.q_network,''')
    src=change(src,'''        agent=cls(hidden_dim=item['hidden_dim'],device=device,
                  keep_margin=item.get('keep_margin',0.),q_network=architecture)''','''        agent=cls(hidden_dim=item['hidden_dim'],device=device,
                  execution_margin=item.get('execution_margin',item.get('keep_margin',0.)),
                  bootstrap_margin=item.get('bootstrap_margin',item.get('keep_margin',0.)),
                  q_network=architecture)''')
    return src

def make_experiment(src):
    src=change(src,'from .rl_env import IAMUEventEnv', 'from .rl_env_v11 import IAMUEventEnv')
    src=change(src,"budget_features='absolute',value_features='global'):","budget_features='absolute',value_features='global'):") if False else src
    return src

def make_validation(src):
    src=change(src,'from iamu_dynamic.rl_ddqn import DoubleDQNAgent',
               'from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent')
    src=change(src,'from iamu_dynamic.rl_experiment import evaluate_paired, write_csv',
               'from iamu_dynamic.rl_experiment_v11 import evaluate_paired, write_csv')
    return src

def make_actors(src):
    begin='@dataclass(frozen=True)\nclass RolloutJob:'
    end='def _cpu_weights(agent: DoubleDQNAgent) -> dict:'
    if src.count(begin)!=1 or src.count(end)!=1:
        raise RuntimeError('Source actor-loop anchors changed')
    code=src[src.index(begin):src.index(end)]
    code=change(code,"            _, job = message\n", "            _, job = message\n            # Worker assignment does not determine per-job exploration RNG.\n            agent.rng.seed(args['seed']*1000003+job.episode*10007+job.seed)\n")
    code=change(code,"                    nxt.mask.copy(), bool(done)))", "                    nxt.mask.copy(), bool(done),\n                    dict(obs.decision_flags or {}),int(job.episode)))")
    code=change(code, "            report = {", "            flag_counts={}\n            for tr in transitions:\n                for key,value in (tr.reason_flags or {}).items():\n                    if isinstance(value,bool):flag_counts[key]=flag_counts.get(key,0)+int(value)\n            report = {")
    code=change(code, "                'steps': len(transitions),", "                'steps': len(transitions), 'forced_flags': flag_counts,")
    code=change(code, "        agent = DoubleDQNAgent(**agent_kwargs)","        agent_kwargs.update(execution_margin=args['execution_margin'],\n                            bootstrap_margin=args['bootstrap_margin'])\n        agent = DoubleDQNAgent(**agent_kwargs)")
    header=("\"\"\"Generated fork of IAMU CPU actor with job-keyed RNG.\"\"\"\n"
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "from typing import Any\nimport inspect,os,queue,time,math,traceback\nimport torch\n"
        "from iamu_dynamic import read_case,generate_scenario\n"
        "from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent,Transition\n"
        "from iamu_dynamic.rl_env_v11 import IAMUEventEnv\n\n")
    return header+code

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    mapping=[('iamu_dynamic/rl_env.py','iamu_dynamic/rl_env_v11.py',make_env),
             ('iamu_dynamic/rl_ddqn.py','iamu_dynamic/rl_ddqn_v11.py',make_ddqn),
             ('iamu_dynamic/rl_experiment.py','iamu_dynamic/rl_experiment_v11.py',make_experiment),
             ('parallel_validation_v081.py','parallel_validation_v11.py',make_validation),
             ('train_budgeted_ddqn_parallel.py','v11_protocol/v11_actors.py',make_actors)]
    result=[]
    for source,dest,transform in mapping:
        original=ROOT/source;generated=ROOT/dest
        if not original.is_file():raise FileNotFoundError(original)
        text=transform(original.read_text(encoding='utf-8'))
        compile(text,str(generated),'exec')
        if generated.exists() and generated.read_text()!=text:
            raise FileExistsError(f'Refusing overwrite of changed generated file: {generated}')
        if not args.check_only and not generated.exists():
            generated.write_text(text,encoding='utf-8')
            py_compile.compile(str(generated),doraise=True)
        result.append(dict(source=source,source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                           dest=dest,generated_sha256=hashlib.sha256(text.encode()).hexdigest()))
    if not args.check_only:
        path=ROOT/'v11_protocol'/'build_manifest.json'
        if path.exists() and json.loads(path.read_text())!=result:raise RuntimeError('Build manifest differs; start fresh')
        path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
