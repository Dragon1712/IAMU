"""Event-driven, no-training dynamic commitment prototype for BACAP instances.

This is a NEW planning/realization layer. It is NOT the published ERL-DDE C++
physics or the published paper's reward.  Frozen ERL-DDE may supply proposal
ranking/position hints through legacy_erldde.py, but it does not enforce plans.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import csv
import json
import math
from pathlib import Path
import random
from typing import Optional

EPS=1e-8

@dataclass(frozen=True)
class Vessel:
    id: int
    eta0: float
    workload: float
    length: float
    max_cranes: int
    weight: float=1.0

@dataclass(frozen=True)
class Port:
    cranes: int
    quay_length: float=3000.0
    safety_ratio: float=0.1
    max_cranes_per_ship: int=5

@dataclass(frozen=True)
class Assignment:
    vessel_id: int
    start: float
    pos: float
    cranes: int
    duration: float
    @property
    def end(self): return self.start+self.duration
    def as_dict(self):
        return dict(vessel_id=self.vessel_id, start=round(self.start,6),
                    end=round(self.end,6), berth_start_m=round(self.pos,4),
                    cranes=self.cranes, duration=round(self.duration,6))

@dataclass(frozen=True)
class Update:
    time: float
    vessel_id: int
    new_eta: float
    update_id: int

@dataclass
class Scenario:
    vessels: list[Vessel]
    port: Port
    true_etas: dict[int,float]
    updates: list[Update]
    label: str
    seed: int

@dataclass(frozen=True)
class Hint:
    rank: dict[int,int]
    position: dict[int,float]
    backend: str="heuristic"

def read_case(path: str|Path) -> tuple[list[Vessel],Port]:
    """Read original ERL-DDE TSV. Column 5 is read from first row ONLY."""
    vessels=[]
    with open(path,encoding='utf-8-sig',newline='') as f:
        for row in csv.reader(f,delimiter='\t'):
            if not row or all(not x.strip() for x in row): continue
            cols=[x.strip() for x in row]
            if len(cols)<4: raise ValueError(f"Case row {len(vessels)} needs at least 4 columns")
            a,w,length,maxcr=map(float,cols[:4])
            if not all(math.isfinite(x) for x in (a,w,length,maxcr)):raise ValueError('Nonfinite input')
            if a<0 or w<=0 or length<=0 or maxcr<1 or int(maxcr)!=maxcr:
                raise ValueError(f"Invalid vessel row: {row}")
            if not vessels:
                if len(cols)<5: raise ValueError('The first row must include num_cranes in column 5')
                cranes=int(cols[4]);
                if cranes<1: raise ValueError('Number of cranes must be positive')
            vessels.append(Vessel(len(vessels),a,w,length,int(maxcr)))
    if not vessels:raise ValueError('Empty case')
    return vessels,Port(cranes=cranes)

def demo_case() -> tuple[list[Vessel],Port]:
    # All fields have exactly the semantics of ERL-DDE's case_code TSV.
    return [Vessel(i, a, w, l, m) for i,(a,w,l,m) in enumerate([
        (4,60,380,6),(6,80,400,5),(8,75,360,6),
        (10,90,420,5),(11,100,380,6),(15,65,350,4),
        (19,120,390,6),(22,70,390,6),(25,85,360,6),
        (28,110,410,5),(31,90,400,5),(34,80,370,6)])],Port(cranes=12,max_cranes_per_ship=4)

def generate_scenario(vessels:list[Vessel],port:Port, label='repeated',seed=42,
                      updates_per_vessel=3, uncertainty=8.0) -> Scenario:
    """Generate hidden actual arrivals and ordered, causal ETA announcements.

    The policy never sees true_etas; the simulator alone uses them for arrivals.
    Event timestamps are <= actual arrival. Forecast values are >= event time.
    """
    if label not in ('small','shock','repeated'): raise ValueError(label)
    if updates_per_vessel<1: raise ValueError('updates_per_vessel must be >= 1')
    rng=random.Random(seed)
    actual={}; updates=[]; k=0
    for v in vessels:
        s=uncertainty*(.25 if label=='small' else (1.8 if label=='shock' else 1.0))
        deviation = rng.gauss(0,s)
        if label=='shock' and rng.random()<.25: deviation+=rng.uniform(s,3*s)
        truth=max(0.05, v.eta0+deviation)
        actual[v.id]=truth
        # Announcements occur at fractions of *elapsed physical time*; this
        # event generator is the oracle, but event times contain no final ETA.
        prev=v.eta0
        for u in range(updates_per_vessel):
            t=truth*(u+1)/(updates_per_vessel+1)
            alpha=(u+1)/(updates_per_vessel+1)
            noise=rng.gauss(0, s*(.45 if label=='repeated' else .18))
            new=max(t, v.eta0+alpha*(truth-v.eta0)+noise)
            if abs(new-prev)<1e-4: new+=0.01
            updates.append(Update(t,v.id,new,k));k+=1
            prev=new
    updates.sort(key=lambda e:(e.time,e.update_id))
    return Scenario(vessels,port,actual,updates,label,seed)

def _spatial_overlap(a:Assignment,b:Assignment,vessels:dict[int,Vessel],port:Port)->bool:
    aa=vessels[a.vessel_id];bb=vessels[b.vessel_id]
    gap=port.safety_ratio*max(aa.length,bb.length)
    return a.pos < b.pos+bb.length+gap-EPS and b.pos < a.pos+aa.length+gap-EPS

def _time_overlap(a:Assignment,b:Assignment)->bool:
    return a.start < b.end-EPS and b.start < a.end-EPS

def feasible_insert(a:Assignment, fixed:list[Assignment],vessels:dict[int,Vessel],port:Port)->bool:
    v=vessels[a.vessel_id]
    if a.cranes<1 or a.cranes>min(v.max_cranes,port.cranes):return False
    if a.pos < -EPS or a.pos+v.length > port.quay_length+EPS:return False
    if a.start< -EPS:return False
    for b in fixed:
        if _time_overlap(a,b) and _spatial_overlap(a,b,vessels,port):return False
    # Validate concurrent crane sum, rather than summing all pairwise overlaps.
    breakpoints=sorted({a.start,a.end} |
                       {max(b.start,a.start) for b in fixed if _time_overlap(a,b)} |
                       {min(b.end,a.end) for b in fixed if _time_overlap(a,b)})
    for l,r in zip(breakpoints,breakpoints[1:]):
        if r-l<EPS:continue
        mid=(l+r)*.5
        used=a.cranes+sum(b.cranes for b in fixed if b.start<=mid<b.end)
        if used>port.cranes:return False
    return True

def validate_plan(plan:dict[int,Assignment],vessels:dict[int,Vessel],port:Port):
    if set(plan)!=set(vessels):raise AssertionError('Missing/extra vessel assignments')
    fixed=[]
    for a in plan.values():
        if a.vessel_id not in vessels or not feasible_insert(a,fixed,vessels,port):
            raise AssertionError(f'Invalid published plan at vessel {a.vessel_id}')
        fixed.append(a)
    return True

class SlotPlanner:
    def __init__(self,vessels:list[Vessel],port:Port):
        self.vessels={v.id:v for v in vessels};self.port=port
    def _q(self,v:Vessel)->int:
        return min(v.max_cranes,self.port.cranes,self.port.max_cranes_per_ship)
    def assign(self,id:int,release:float,fixed:list[Assignment],preferred_pos:Optional[float]=None,
               cranes:Optional[int]=None)->Assignment:
        v=self.vessels[id];q=self._q(v) if cranes is None else cranes
        if not isinstance(q,int) or not 1 <= q <= self._q(v):
            raise ValueError(f'Invalid assigned crane count for ship {id}: {q}')
        if v.length > self.port.quay_length:raise ValueError('Vessel longer than quay')
        dur=v.workload/q
        # Candidate spots include free quay origins, berth boundaries and any
        # learned position hint. This is a deterministic *heuristic*, not MILP.
        positions=[0.0]
        positions += [float(x) for x in range(0,int(self.port.quay_length-v.length)+1,250)]
        for other in fixed:
            ov=self.vessels[other.vessel_id]
            positions += [other.pos+ov.length+self.port.safety_ratio*max(v.length,ov.length),
                          other.pos-v.length-self.port.safety_ratio*max(v.length,ov.length)]
        if preferred_pos is not None: positions.append(float(preferred_pos))
        positions=sorted({round(min(max(x,0.0),self.port.quay_length-v.length),6) for x in positions})
        # Release and all end times suffice to find the earliest feasible slot
        # for this fixed-duration, no-preemption insertion heuristic.
        times=sorted({max(0.,release)} | {max(release,a.end) for a in fixed if a.end>=release-EPS})
        best=None
        for t in times:
            for p in positions:
                a=Assignment(id,t,p,q,dur)
                if feasible_insert(a,fixed,self.vessels,self.port):
                    # Prefer earlier start; use original ES position only to
                    # break ties between equal-start feasible placements.
                    tie=abs(p-(preferred_pos if preferred_pos is not None else 0.0))
                    key=(t,tie,p)
                    if best is None or key<best[0]:best=(key,a)
            if best is not None and best[0][0]<=t+EPS:break
        if best is None:raise RuntimeError(f'Cannot schedule ship {id}')
        return best[1]
    def initial(self,etas:dict[int,float],hint:Optional[Hint]=None)->dict[int,Assignment]:
        rank=hint.rank if hint else {}
        # Rank comes from frozen policy's observed action selection order.
        order=sorted(self.vessels,key=lambda i:(rank.get(i,10**8),etas[i],i))
        planned={}
        for i in order:
            planned[i]=self.assign(i,etas[i],list(planned.values()),
                                   hint.position.get(i) if hint else None)
        validate_plan(planned,self.vessels,self.port)
        return planned
    def single_move(self,plan:dict[int,Assignment],id:int,now:float,etas:dict[int,float],
                    hint:Optional[Hint]=None)->Optional[dict[int,Assignment]]:
        fixed=[a for j,a in plan.items() if j!=id]
        try:
            a=self.assign(id,max(now,etas[id]),fixed,hint.position.get(id) if hint else None)
        except RuntimeError:return None
        if assignments_equal(a,plan[id]):return None
        new=dict(plan);new[id]=a
        validate_plan(new,self.vessels,self.port)
        return new

def assignments_equal(a:Assignment,b:Assignment,tol=1e-6)->bool:
    return (abs(a.start-b.start)<=tol and abs(a.pos-b.pos)<=tol
            and a.cranes==b.cranes and abs(a.duration-b.duration)<=tol)

class RealizedRuntime:
    """Causal physical dispatcher. Hidden arrival times are used only by simulator.

    A published reservation cannot make an unarrived ship start. On conflict,
    operations wait for a genuinely free quay/crane; this involuntary slip is
    logged separately from a *published commitment revision*.
    """
    def __init__(self,vessels:list[Vessel],port:Port):
        self.vessels={v.id:v for v in vessels};self.port=port
        self.clock=0.0
        self.started:dict[int,Assignment]={}
        self.completed:set[int]=set()
    def _start_ready(self,plan:dict[int,Assignment],actual:dict[int,float]):
        changed=False
        pending=[i for i in self.vessels if i not in self.started
                 and actual[i]<=self.clock+EPS and plan[i].start<=self.clock+EPS]
        for i in sorted(pending,key=lambda j:(plan[j].start,j)):
            p=plan[i];a=Assignment(i,self.clock,p.pos,p.cranes,p.duration)
            active=[r for j,r in self.started.items() if j not in self.completed]
            if feasible_insert(a,active,self.vessels,self.port):
                self.started[i]=a;changed=True
        return changed
    def advance(self,until:float,plan:dict[int,Assignment],actual:dict[int,float]):
        if until+EPS<self.clock:raise ValueError('Time cannot go backwards')
        max_steps=8*len(self.vessels)+10
        for _ in range(max_steps):
            self.completed.update(i for i,a in self.started.items() if a.end<=self.clock+EPS)
            self._start_ready(plan,actual)
            if self.clock>=until-EPS:
                self.clock=max(self.clock,until)
                return
            pending=[i for i in self.vessels if i not in self.started]
            next_times=[a.end for i,a in self.started.items() if i not in self.completed and a.end>self.clock+EPS]
            next_times += [max(plan[i].start,actual[i]) for i in pending
                           if max(plan[i].start,actual[i])>self.clock+EPS]
            if not next_times:
                if pending:raise RuntimeError('Deadlock in dispatcher; check plan feasibility')
                self.clock=until;return
            self.clock=min(until,min(next_times))
        raise RuntimeError('Realized dispatcher exceeded event limit')
    def finish(self,plan:dict[int,Assignment],actual:dict[int,float]):
        horizon=max(max(actual.values()),max(a.end for a in plan.values()))
        # generous, finite bound for any feasible fixed plan
        horizon+=sum(v.workload for v in self.vessels.values())+10
        self.advance(horizon,plan,actual)
        if len(self.started)!=len(self.vessels):raise RuntimeError('Not all ships started')
        return self
    def score(self,plan:dict[int,Assignment],forecast:dict[int,float])->float:
        future=copy.deepcopy(self)
        # Future ETA forecasts are assumed exact for scoring ONLY; this never
        # accesses the simulator's hidden actual ETA values.
        future.finish(plan,forecast)
        return sum(v.weight*max(0,future.started[i].start-forecast[i])
                   for i,v in self.vessels.items() if i not in self.started)

def simulate(scenario:Scenario,policy='budget_aware',budget:Optional[int]=5,
             threshold:float=3.0,period:float=12.0,max_edits_per_event=2,
             min_gain:float=0.1,candidate_limit:int=12,
             hint_provider=None,repair_method:str='single',
             revision_penalty:float=0.0,max_repair_pairs:int=4,
             periodic_horizon:Optional[float]=None)->dict:
    """Event-sourced, budget-constrained, non-anticipative recovery simulation.

    `periodic` now uses an independent clock instead of a gate on ETA notices.
    `alns2` tries deterministic, bounded, two-ship destroy/repair neighborhoods.
    These adapt ideas from published transport recovery work; no aircraft or
    railway code is directly imported.  The original ES checkpoint is frozen.
    """
    if policy not in {'no_replan','always','periodic','threshold','budget_aware'}:
        raise ValueError(policy)
    if repair_method not in {'single','alns2'}:
        raise ValueError('repair_method must be single or alns2')
    if budget is not None and budget<0:raise ValueError('Budget >=0 or None')
    if max_edits_per_event<1 or candidate_limit<1 or max_repair_pairs<0:
        raise ValueError('Invalid candidate or neighborhood limit')
    if period<=0 or not math.isfinite(period):raise ValueError('period must be positive')
    if not math.isfinite(revision_penalty) or revision_penalty<0:
        raise ValueError('revision_penalty must be nonnegative and finite')
    planner=SlotPlanner(scenario.vessels,scenario.port)
    ids=planner.vessels
    eta={v.id:v.eta0 for v in scenario.vessels}
    initial_hint=hint_provider(scenario.vessels,scenario.port,eta) if hint_provider else None
    pub=planner.initial(eta,initial_hint)
    initial_pub={i:a.as_dict() for i,a in pub.items()}
    rt=RealizedRuntime(scenario.vessels,scenario.port)
    used=0; revision_counts={i:0 for i in ids}
    logs=[];revision_log=[]; candidate_evaluations=0

    # Scheduled clock is based ONLY on the initially public instance, not the
    # hidden actual arrivals or future ETA updates. This avoids oracle timing.
    # A bounded operational horizon includes one worst-case serial processing
    # upper bound using initial workloads; late arrivals can still occur later,
    # but will not secretly prolong the preannounced periodic schedule.
    decision_events=[(e.time,0,e.update_id,'eta',e) for e in scenario.updates]
    if policy=='periodic':
        # Published initial makespan is public and independent of hidden ETA.
        horizon=(max(a.end for a in pub.values()) + period
                 if periodic_horizon is None else periodic_horizon)
        if not math.isfinite(horizon) or horizon < 0:
            raise ValueError('periodic_horizon must be finite and nonnegative')
        n_ticks=int(horizon // period)
        if n_ticks>10000:
            raise ValueError('Periodic horizon too long; increase period')
        for k in range(1,n_ticks+1):
            t=k*period
            decision_events.append((t,1,k,'clock',None))
    decision_events.sort(key=lambda e:(e[0],e[1],e[2]))

    for event_time,_,event_id,event_type,event in decision_events:
        rt.advance(event_time,pub,scenario.true_etas)
        if event_type=='eta':
            vessel_id=event.vessel_id
            old_eta=eta[vessel_id]
            eta[vessel_id]=event.new_eta
            trigger=(policy in ('always','budget_aware') or
                     (policy=='threshold' and abs(event.new_eta-old_eta)>=threshold))
            new_eta=event.new_eta
        else:
            vessel_id=None
            old_eta=None
            new_eta=None
            trigger=(policy=='periodic')
        edits=[];forecast_gain=0.0;tried=0;repair_pairs=0
        if trigger and policy!='no_replan' and (budget is None or used<budget):
            hint=hint_provider(scenario.vessels,scenario.port,eta) if hint_provider else None
            remaining=max_edits_per_event if budget is None else min(max_edits_per_event,budget-used)
            eligible=[i for i in ids if i not in rt.started]
            rank=hint.rank if hint else {}
            eligible.sort(key=lambda i:(i!=vessel_id,
                                        not (eta[i]>pub[i].start+EPS),
                                        rank.get(i,10**8),pub[i].start,i))
            eligible=eligible[:candidate_limit]
            from .recovery import pressure_graph, candidate_pairs, repair_subset
            # Multiple sequential moves are allowed, but a vessel can be
            # revised at most once per decision event.  Each changed ship is
            # charged, including when a two-ship repair is selected.
            while remaining>0 and eligible:
                score0=rt.score(pub,eta)
                best=None
                best_key=None
                proposals=[]
                for i in eligible:
                    candidate=planner.single_move(pub,i,event_time,eta,hint)
                    if candidate is not None:
                        proposals.append((candidate,(i,),'single'))
                if repair_method=='alns2' and remaining>=2 and len(eligible)>1:
                    graph=pressure_graph(pub,ids,scenario.port,eta,event_time,eligible)
                    for pair in candidate_pairs(graph,pub,vessel_id,max_repair_pairs):
                        repair_pairs+=1
                        for candidate,changed in repair_subset(
                                planner,pub,pair,event_time,eta,hint,
                                budget_remaining=remaining):
                            if all(i in eligible for i in changed):
                                proposals.append((candidate,changed,'alns2'))
                # Waiting time is a counterfactual based on *current forecasts*;
                # never read hidden future arrivals to score proposed edits.
                for candidate,changed,kind in proposals:
                    cost=len(changed)
                    if cost>remaining:continue
                    tried+=1
                    gain=score0-rt.score(candidate,eta)
                    net_gain=gain-revision_penalty*cost
                    min_required=(min_gain if policy=='budget_aware' else 1e-9)
                    if net_gain<=min_required:continue
                    # Budget-aware ratio prefers improvement per unit of a
                    # finite, globally scarce modification resource.
                    key=(net_gain/cost if policy=='budget_aware' else net_gain,
                         net_gain, -cost, tuple(-i for i in changed))
                    if best_key is None or key>best_key:
                        best_key=key;best=(candidate,changed,gain,kind)
                if best is None:break
                candidate,changed,gain,kind=best
                before=pub
                pub=candidate
                validate_plan(pub,ids,scenario.port)
                for i in changed:
                    revision_counts[i]+=1
                    used+=1
                    revision_log.append(dict(
                        time=round(event_time,5),vessel_id=i,
                        old=before[i].as_dict(),new=pub[i].as_dict(),
                        estimated_gain=round(gain,5),
                        repair_kind=kind,event_type=event_type,
                        revision_penalty=revision_penalty,
                        cumulative_budget=used))
                    eligible.remove(i)
                    edits.append(i)
                remaining-=len(changed)
                forecast_gain+=gain
                assert budget is None or used<=budget
        candidate_evaluations+=tried
        logs.append(dict(event_id=event_id,event_type=event_type,
                         time=round(event_time,5),
                         vessel_id=vessel_id,
                         eta_before=(None if old_eta is None else round(old_eta,5)),
                         eta_after=(None if new_eta is None else round(new_eta,5)),
                         triggered=trigger,evaluated=tried,
                         repair_pairs=repair_pairs,edited=edits,changes=len(edits),
                         budget_used=used,
                         budget_remaining=(None if budget is None else budget-used),
                         estimated_gain=round(forecast_gain,6),
                         started_count=len(rt.started)))
    rt.finish(pub,scenario.true_etas)
    realized={i:a.as_dict() for i,a in rt.started.items()}
    weights=sum(v.weight for v in scenario.vessels)
    wait=sum(ids[i].weight*max(0,rt.started[i].start-scenario.true_etas[i]) for i in ids)
    late_slip=sum(max(0,scenario.true_etas[i]-pub[i].start) for i in ids)
    resource_slip=sum(max(0,rt.started[i].start-max(pub[i].start,scenario.true_etas[i])) for i in ids)
    slip=sum(max(0,rt.started[i].start-pub[i].start) for i in ids)
    assert abs(late_slip+resource_slip-slip)<1e-5*max(1,len(ids))
    metrics=dict(policy=policy,budget=budget,spent=used,number_of_updates=sum(
                     e['event_type']=='eta' for e in logs),
                 number_of_decision_events=len(logs),
                 number_of_periodic_ticks=sum(e['event_type']=='clock' for e in logs),
                 number_of_triggered_updates=sum(x['triggered'] for x in logs),
                 number_of_vessels_revised=sum(v>0 for v in revision_counts.values()),
                 candidate_evaluations=candidate_evaluations,
                 repair_method=repair_method,
                 weighted_wait=round(wait,6),mean_weighted_wait=round(wait/weights,6),
                 total_operational_slip=round(slip,6),
                 arrival_lateness_slip=round(late_slip,6),
                 resource_conflict_slip=round(resource_slip,6),
                 mean_turnaround=round(sum(a.end-scenario.true_etas[i] for i,a in rt.started.items())/len(ids),6),
                 makespan=round(max(a.end for a in rt.started.values()),6))
    assert used<=budget if budget is not None else True
    assert used==sum(revision_counts.values())==len(revision_log)
    return dict(metrics=metrics, initial_published=initial_pub,
                final_published={i:a.as_dict() for i,a in pub.items()},
                realized=realized,
                eta_updates=[r for r in logs if r['event_type']=='eta'],
                decision_events=logs,revisions=revision_log,
                revision_counts=revision_counts)

def scenario_as_dict(sc:Scenario):
    return dict(label=sc.label,seed=sc.seed,port=asdict(sc.port),
                vessels=[asdict(v) for v in sc.vessels],
                true_etas={str(k):v for k,v in sc.true_etas.items()},
                updates=[asdict(e) for e in sc.updates])

def save_result(output:Path,scenario:Scenario,result:dict):
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result['metrics'],indent=2),encoding='utf-8')
    (output/'scenario.json').write_text(json.dumps(scenario_as_dict(scenario),indent=2),encoding='utf-8')
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,rows in [('eta_updates.csv',result['eta_updates']),
                      ('decision_events.csv',result['decision_events']),
                      ('revisions.csv',result['revisions'])]:
        if rows:
            with (output/name).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for kind in ['initial_published','final_published','realized']:
        rows=list(result[kind].values())
        if rows:
            with (output/f'{kind}.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
