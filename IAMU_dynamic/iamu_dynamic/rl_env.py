"""IAMU V0.6 non-anticipative event-driven budgeted RL environment.

Literature-inspired design: budget in observable state (Carrara et al. 2019),
feasible action masking, candidate-action Q learning (van Hasselt et al. 2016).
Only current/public information is passed to an agent. Latent true arrivals are
used by the physical dispatcher and by the reward ledger, never by observations,
candidate selection or public event construction.

ACTIONS: 0 = Keep; 1..K = accept exactly one legal single-ship move.
ACTIONS are generated anew at every public decision event. Keep is always legal.
One revision maximum per event by design, for apples-to-apples comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Optional
import math
import numpy as np

from .core import (Scenario, SlotPlanner, RealizedRuntime, EPS,
                   validate_plan)

# Indices are appended, preserving the first 19/17 V0.6 features. The
# learned V0.6 checkpoint is deliberately incompatible with these dimensions.
GLOBAL_DIM = 24
ACTION_DIM = 24
POINT_GAIN_INDEX = 17
BUDGET_FEATURE_SCALE = 10.0  # Fixed across episodes, never divided only by B.



@dataclass
class RLObservation:
    global_features: np.ndarray        # (GLOBAL_DIM,)
    action_features: np.ndarray        # (K+1, ACTION_DIM)
    mask: np.ndarray                   # (K+1,) bool
    time: float
    event_type: str
    remaining_budget: int
    candidate_ids: tuple[int, ...]      # index 1.. corresponds to valid actions

    def copy(self):
        return RLObservation(self.global_features.copy(),
                             self.action_features.copy(), self.mask.copy(),
                             self.time, self.event_type, self.remaining_budget,
                             self.candidate_ids)


class IAMUEventEnv:
    """Step through observable ETA/clock events, not through hidden future data.

    `decision_clock='eta'` yields actions only when ETA announcements occur;
    `'union'` adds preannounced periodic ticks and permits actions at either;
    `'clock'` observes ETA messages as they arrive but permits actions only
    on periodic ticks. The externally visible event at reset is a no-action
    initial observation at t=0, preventing artificial extra time-zero edits.

    A full episode's *unscaled* step rewards sum to `-weighted_wait`. This
    includes waiting incurred before the first ETA event and after the last.
    """

    def __init__(self, scenario: Scenario, budget: int = 5,
                 candidate_limit: int = 4, decision_clock: str = 'eta',
                 period: float = 12., reward_scale: float = 100.,
                 min_gain: float = 0.1,
                 candidate_screen: str = 'none',
                 budget_features: str = 'absolute',
                 value_features: str = 'global'):
        if not isinstance(budget,int) or budget < 0:
            raise ValueError('Budget must be a nonnegative integer')
        if candidate_limit < 1 or candidate_limit > 128:
            raise ValueError('Candidate limit must lie in [1, 128]')
        if decision_clock not in {'eta','clock','union'}:
            raise ValueError('decision_clock must be eta, clock or union')
        if not math.isfinite(period) or period <= 0:
            raise ValueError('period must be positive and finite')
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('reward_scale must be positive and finite')
        if candidate_screen not in {'none','positive'}:
            raise ValueError('candidate_screen must be none or positive')
        if not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError('min_gain must be finite and nonnegative')
        self.scenario = scenario
        self.budget = budget
        self.candidate_limit = candidate_limit
        self.decision_clock = decision_clock
        self.period = float(period)
        self.reward_scale = reward_scale
        self.min_gain = min_gain
        self.candidate_screen = candidate_screen
        if budget_features not in {'absolute','legacy'}:
            raise ValueError('budget_features must be absolute or legacy')
        if value_features not in {'global','legacy'}:
            raise ValueError('value_features must be global or legacy')
        self.budget_features = budget_features
        self.value_features = value_features
        self.planner = SlotPlanner(scenario.vessels, scenario.port)
        self.vessels = self.planner.vessels
        self.ids = tuple(sorted(self.vessels))
        self.horizon_scale = max(1., max(v.eta0 for v in scenario.vessels)
                                + sum(v.workload/max(1,min(v.max_cranes,scenario.port.cranes,
                                         scenario.port.max_cranes_per_ship)) for v in scenario.vessels))
        self.reset()

    def reset(self) -> RLObservation:
        self.published_eta = {i: float(v.eta0) for i,v in self.vessels.items()}
        self.initial_eta = dict(self.published_eta)
        self.prev_delta = {i:0. for i in self.ids}
        self.update_counts = {i:0 for i in self.ids}
        self.revision_counts = {i:0 for i in self.ids}
        self.plan = self.planner.initial(self.published_eta)
        self.original_plan = dict(self.plan)
        self.runtime = RealizedRuntime(self.scenario.vessels, self.scenario.port)
        self.runtime.advance(0., self.plan, self.scenario.true_etas)
        self.used = 0
        self.last_revision_time = None
        self.screened_proposals = 0
        self.action_audit = []
        self.event_index = 0
        self.event_type = 'initial'
        self.event_vessel_id = None
        self.event_time = 0.
        self.done = False
        self.history = []
        self.decision_events = []
        self.events = [(float(e.time), 0, int(e.update_id), 'eta', e)
                       for e in self.scenario.updates]
        if self.decision_clock in {'clock','union'}:
            # Same publicly determined tick horizon as the original simulator.
            horizon = max(a.end for a in self.original_plan.values()) + self.period
            n_ticks = int(horizon // self.period)
            if n_ticks > 10000:
                raise ValueError('Too many clock ticks')
            self.events.extend((k*self.period,1,k,'clock',None)
                               for k in range(1,n_ticks+1))
        self.events.sort(key=lambda e:(e[0],e[1],e[2]))
        self._candidates = []
        self._candidate_ids = ()
        self._rewarded_started = set()
        self.total_raw_reward = 0.
        self._refresh_candidates()
        return self._observation()

    @property
    def remaining_budget(self):
        return self.budget-self.used

    def _eligible(self):
        ranked=[i for i in self.ids if i not in self.runtime.started]
        vid=self.event_vessel_id
        ranked.sort(key=lambda i:(i!=vid,
                       not (self.published_eta[i]>self.plan[i].start+EPS),
                       self.plan[i].start,i))
        return ranked[:self.candidate_limit]

    def _forecast_projection(self, plan):
        """Only currently public ETA values and the already-realized runtime.

        This is a *point-forecast counterfactual*, not a ground-truth oracle.
        Started ships are deliberately excluded from the remaining-wait score,
        exactly as RealizedRuntime.score() does in V0.6.
        """
        future=copy.deepcopy(self.runtime)
        future.finish(plan,self.published_eta)
        pending=[i for i in self.ids if i not in self.runtime.started]
        starts={i:future.started[i].start for i in pending}
        costs={i:self.vessels[i].weight*max(0.,starts[i]-self.published_eta[i])
               for i in pending}
        return starts,costs,sum(costs.values())

    def _refresh_candidates(self):
        self._candidates=[]
        self._candidate_ids=()
        self._candidate_data=[]
        if (self.event_type=='initial' or self.done or self.remaining_budget==0 or
            (self.decision_clock=='clock' and self.event_type!='clock') or
            (self.decision_clock=='eta' and self.event_type!='eta')):
            return
        ids=[]
        base_starts=base_costs=None
        base_score=0.
        for i in self._eligible():
            candidate=self.planner.single_move(self.plan,i,self.event_time,
                                               self.published_eta)
            if candidate is None:
                continue
            if base_starts is None:
                base_starts,base_costs,base_score=self._forecast_projection(self.plan)
            starts,costs,new_score=self._forecast_projection(candidate)
            gain=base_score-new_score
            if self.candidate_screen=='positive' and gain <= self.min_gain+1e-12:
                # An *heuristic* guard: excludes point-forecast non-improving
                # moves; does not certify true waiting or CVaR improvement.
                self.screened_proposals+=1
                continue
            affected=[j for j in starts if j!=i]
            worsened=[j for j in affected if starts[j]>base_starts[j]+EPS]
            delay=max((starts[j]-base_starts[j] for j in worsened),default=0.)
            worse_cost=sum(max(0.,costs[j]-base_costs[j]) for j in affected)
            relief_cost=sum(max(0.,base_costs[j]-costs[j]) for j in affected)
            self._candidate_data.append(dict(
                gain=gain,base_score=base_score,new_score=new_score,
                external_worse_count=len(worsened),
                external_max_delay=delay,external_worse_cost=worse_cost,
                external_relief_cost=relief_cost))
            ids.append(i)
            self._candidates.append(candidate)
        self._candidate_ids=tuple(ids)

    def _accrue_started(self):
        """Charge newly started vessel waiting exactly once."""
        fresh=set(self.runtime.started).difference(self._rewarded_started)
        cost=sum(self.vessels[i].weight * max(0.,
                   self.runtime.started[i].start-self.scenario.true_etas[i])
                 for i in fresh)
        self._rewarded_started.update(fresh)
        return cost

    def _observation(self) -> RLObservation:
        mask=np.zeros(self.candidate_limit+1,dtype=np.bool_)
        mask[0]=True
        features=np.zeros((self.candidate_limit+1,ACTION_DIM),dtype=np.float32)
        features[0,0]=1.0
        n=max(1,len(self.ids));t=self.event_time;port=self.scenario.port
        started=set(self.runtime.started)
        # Arrived flags are observable physical facts AS OF t, not forecasts
        # of future arrival. Their time stamps never appear in agent features.
        arrived={i for i in self.ids if self.scenario.true_etas[i] <= t + EPS}
        waiting=arrived-started
        rem=[i for i in self.ids if i not in started]
        denom=max(1.,self.horizon_scale)
        forecast_slip=sum(max(0., self.published_eta[i]-self.plan[i].start)
                          for i in rem)
        eta_gaps=[self.published_eta[i]-self.plan[i].start for i in rem]
        g=np.array([
            min(3.,t/denom),
            self.remaining_budget/max(1,self.budget),
            self.used/max(1,self.budget),
            len(started)/n, len(arrived)/n, len(waiting)/n,
            len(rem)/n,
            sum(self.update_counts.values())/max(1,3*n),
            sum(self.revision_counts.values())/max(1,self.budget),
            sum(abs(self.prev_delta[i]) for i in self.ids)/n/20.,
            sum(abs(self.published_eta[i]-self.initial_eta[i]) for i in self.ids)/n/20.,
            forecast_slip/n/20.,
            (sum(eta_gaps)/len(eta_gaps)/20.) if eta_gaps else 0.,
            min(3.,port.cranes/30.),
            min(3.,len(waiting)/max(1,port.cranes)),
            float(self.event_type=='eta'), float(self.event_type=='clock'),
            float(self.event_type=='initial'),
            0. if self.event_vessel_id is None else
                 self.event_vessel_id/max(1,max(self.ids)),
            self.budget/BUDGET_FEATURE_SCALE,
            self.remaining_budget/BUDGET_FEATURE_SCALE,
            len(self._candidate_ids)/self.candidate_limit,
            (min(3.,(t-self.last_revision_time)/denom)
             if self.last_revision_time is not None else min(3.,t/denom)),
            (max((d['gain'] for d in self._candidate_data),default=0.)
             /self.reward_scale),
        ],dtype=np.float32)
        if not self.done:
            for idx,(i,alt) in enumerate(zip(self._candidate_ids,self._candidates),start=1):
                v=self.vessels[i];p=self.plan[i];q=alt[i]
                mask[idx]=True
                d=self._candidate_data[idx-1]
                features[idx]=np.array([
                    0., i/max(1,max(self.ids)),
                    self.published_eta[i]/denom,
                    self.prev_delta[i]/20.,
                    min(3.,self.update_counts[i]/3.),
                    self.revision_counts[i]/max(1,self.budget),
                    p.start/denom,
                    (self.published_eta[i]-p.start)/20.,
                    q.start/denom,
                    (q.start-p.start)/20.,
                    p.pos/max(1.,port.quay_length),
                    q.pos/max(1.,port.quay_length),
                    v.workload/500.,
                    v.length/max(1.,port.quay_length),
                    v.weight/5.,
                    p.cranes/max(1,port.cranes),
                    float(i in arrived),
                    d['gain']/self.reward_scale,
                    d['base_score']/(n*self.reward_scale),
                    d['new_score']/(n*self.reward_scale),
                    d['external_max_delay']/self.reward_scale,
                    d['external_worse_count']/n,
                    d['external_worse_cost']/(n*self.reward_scale),
                    d['external_relief_cost']/(n*self.reward_scale),
                ],dtype=np.float32)
        # Fixed-size features allow controlled, architecture-matched ablations.
        if self.budget_features=='legacy':g[19:21]=0.
        if self.value_features=='legacy':
            g[21:24]=0.
            features[:,17:24]=0.
        return RLObservation(g,features,mask,t,self.event_type,
                             self.remaining_budget,self._candidate_ids)

    def point_greedy_action(self) -> int:
        """V0.6 deterministic, one-move greedy, now reusing cached forecasts."""
        best_gain=self.min_gain
        best_idx=0
        for idx,d in enumerate(self._candidate_data,1):
            gain=d['gain']
            if gain>best_gain+1e-12:
                best_gain=gain;best_idx=idx
        return best_idx

    def step(self, action: int):
        if self.done:raise RuntimeError('Cannot step a finished episode')
        observation=self._observation()
        if not isinstance(action,(int,np.integer)) or not (0<=int(action)<len(observation.mask)):
            raise ValueError('Invalid action ID')
        action=int(action)
        if not observation.mask[action]:
            raise ValueError('Action is infeasible or exceeds remaining revision budget')
        changed=None
        estimated_gain=(self._candidate_data[action-1]['gain'] if action else 0.)
        if action:
            changed=self._candidate_ids[action-1]
            # Started ships cannot be revised. Plan validated by single_move.
            if changed in self.runtime.started or self.remaining_budget<1:
                raise AssertionError('Unsafe action passed the mask')
            self.plan=self._candidates[action-1]
            validate_plan(self.plan,self.vessels,self.scenario.port)
            self.used+=1
            self.last_revision_time=self.event_time
            self.revision_counts[changed]+=1
        cost=self._accrue_started()
        current_event=(self.event_type,self.event_time,self.event_vessel_id,changed)
        self.history.append(current_event)
        self.action_audit.append(dict(time=self.event_time,
               event_type=self.event_type,action=action,vessel_id=changed,
               estimated_gain=estimated_gain,spent=self.used,
               remaining_budget=self.remaining_budget))
        if self.event_index < len(self.events):
            now,_,event_id,typ,event=self.events[self.event_index]
            self.event_index+=1
            self.runtime.advance(now,self.plan,self.scenario.true_etas)
            self.event_type=typ
            self.event_time=now
            self.event_vessel_id=None if event is None else event.vessel_id
            if typ=='eta':
                i=event.vessel_id
                self.prev_delta[i]=event.new_eta-self.published_eta[i]
                self.published_eta[i]=event.new_eta
                self.update_counts[i]+=1
            self._refresh_candidates()
            cost+=self._accrue_started()
        else:
            self.runtime.finish(self.plan,self.scenario.true_etas)
            cost+=self._accrue_started()
            self.done=True
            self.event_type='terminal'
            self.event_vessel_id=None
            self._refresh_candidates()
        reward=-cost/self.reward_scale
        self.total_raw_reward-=cost
        info={'time':self.event_time,'changed_vessel':changed,
              'spent':self.used,'remaining_budget':self.remaining_budget,
              'new_wait_cost':cost,'done':self.done,
              'selected_forecast_gain':estimated_gain,
              'screened_proposals':self.screened_proposals}
        if self.done:
            total=sum(self.vessels[i].weight*max(0.,a.start-self.scenario.true_etas[i])
                      for i,a in self.runtime.started.items())
            assert len(self.runtime.started)==len(self.ids)
            assert abs(-self.total_raw_reward-total)<1e-5*max(1.,total)
            info['weighted_wait']=total
            info['reward_sum_raw']=self.total_raw_reward
            info['n_revisions']=self.used
        return self._observation(),reward,self.done,info
