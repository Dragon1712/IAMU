"""Causal, common-random-number ETA scoring and budgeted two-stage recourse.

IMPORTANT: This module NEVER takes the scenario's hidden ``true_etas`` or future
``updates`` as input. All stochastic forecasts are conditional on the *current*
public ETA vector, the public configuration, and the already-realized runtime.
It is a decision-support *model*, NOT a calibrated port-traffic forecaster.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Assignment, RealizedRuntime, SlotPlanner, Vessel


@dataclass(frozen=True)
class RiskConfig:
    forecast_std: float = 8.0
    n_samples: int = 8
    cvar_alpha: float = 0.75
    risk_aversion: float = 0.5
    lookahead_horizon: float = 12.0
    lookahead_candidates: int = 3
    next_move_candidates: int = 3
    future_signal_strength: float = 0.5

    def validate(self):
        if not math.isfinite(self.forecast_std) or self.forecast_std < 0:
            raise ValueError('forecast_std must be finite and nonnegative')
        if not isinstance(self.n_samples, int) or self.n_samples < 1:
            raise ValueError('n_samples must be >= 1')
        if not 0 < self.cvar_alpha < 1:
            raise ValueError('cvar_alpha must be strictly between 0 and 1')
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0:
            raise ValueError('risk_aversion must be finite and nonnegative')
        if not math.isfinite(self.lookahead_horizon) or self.lookahead_horizon <= 0:
            raise ValueError('lookahead_horizon must be positive and finite')
        if not isinstance(self.lookahead_candidates, int) or self.lookahead_candidates < 1:
            raise ValueError('lookahead_candidates must be >= 1')
        if not isinstance(self.next_move_candidates, int) or self.next_move_candidates < 1:
            raise ValueError('next_move_candidates must be >= 1')
        if not 0 <= self.future_signal_strength <= 1:
            raise ValueError('future_signal_strength must be in [0,1]')
        return self


def _seed(run_seed: int, event_id: int, event_type: str) -> int:
    """Stable seed; independent of candidate actions and hidden future data."""
    x = (int(run_seed) ^ 0xA0761D6478BD642F) & ((1 << 64) - 1)
    for n in (int(event_id), 1 if event_type == 'clock' else 0):
        x ^= ((n + 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
        x ^= x >> 31
    return x


def sample_future_arrivals(
    current_etas: Mapping[int, float],
    now: float,
    already_started: set[int],
    *,
    run_seed: int,
    event_id: int,
    event_type: str,
    config: RiskConfig,
) -> list[dict[int, float]]:
    """Sample a *model belief*, not the episode's hidden actual arrivals.

    A not-yet-started ship is conservatively sampled as arriving no earlier
    than now. This is an approximation: the prototype does not separately
    expose arrived-but-unberthed ships, and hence may understate their queueing.
    For started ships the ETA is irrelevant to counterfactual future waiting.
    The same draw index is used for every candidate (common random numbers).
    """
    config.validate()
    rng = random.Random(_seed(run_seed, event_id, event_type))
    draws = []
    for _ in range(config.n_samples):
        sample = {}
        for i in sorted(current_etas):
            eta = float(current_etas[i])
            if i in already_started:
                sample[i] = eta
            else:
                sampled = rng.gauss(eta, config.forecast_std) if config.forecast_std else eta
                sample[i] = max(now, sampled)
        draws.append(sample)
    return draws


def upper_cvar(values: Sequence[float], alpha: float) -> float:
    """Empirical upper-tail conditional value-at-risk, with fractional tail weight."""
    if not values:
        raise ValueError('CVaR requires at least one observation')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    arr = sorted(map(float, values), reverse=True)
    tail_mass = len(arr) * (1.0 - alpha)
    integral = 0.0
    left = tail_mass
    for value in arr:
        w = min(left, 1.0)
        integral += value * w
        left -= w
        if left <= 1e-12:
            break
    return integral / tail_mass


def forecast_gains(rt: 'RealizedRuntime', before: Mapping[int,'Assignment'],
                   after: Mapping[int,'Assignment'], samples: Sequence[Mapping[int,float]]) -> list[float]:
    return [rt.score(before, s) - rt.score(after, s) for s in samples]


def future_one_step_cost(rt: 'RealizedRuntime', planner: 'SlotPlanner',
                         plan: dict[int,'Assignment'],
                         current_eta: Mapping[int,float], sample: Mapping[int,float],
                         now: float, budget_remaining: int,
                         config: RiskConfig, revision_penalty: float) -> tuple[float,int]:
    """Two-stage recourse proxy: after a public fixed horizon, optionally edit 1 ship.

    Future observations are *synthetically generated from a belief sample*;
    they are not copied from the test episode's future ETA messages. The future
    control selects moves using only the synthetic future forecast, then is
    scored against the sampled latent arrival. The future control may also
    retain the plan; each recourse action consumes one unit of *remaining* B.
    This is a one-recourse-step approximation, not a full multistage optimum.
    """
    if budget_remaining <= 0:
        return rt.score(plan, sample), 0
    t_next = now + config.lookahead_horizon
    shadow = copy.deepcopy(rt)
    shadow.advance(t_next, plan, sample)
    # Account for service already started between now and t_next.  Its waiting
    # cost is irreversible at the hypothetical future decision point.  A
    # prospective future edit must NEVER be applied retroactively at 'now'.
    incurred = sum(v.weight * max(0., shadow.started[i].start - sample[i])
                   for i,v in planner.vessels.items()
                   if i in shadow.started and i not in rt.started)
    future_eta = dict(current_eta)
    eligible = []
    for i in planner.vessels:
        if i in shadow.started:
            continue
        # Correlated, partially informative public observation at hypothetical
        # future event. Strength=0 means no new signal; 1 means perfect signal
        # *inside the simulated branch only*, not from the real test scenario.
        future_eta[i] = max(t_next, current_eta[i] + config.future_signal_strength *
                            (sample[i] - current_eta[i]))
        eligible.append(i)
    # Public ordering; not ranked using hidden / sampled true arrival values.
    eligible.sort(key=lambda i:(plan[i].start, i))
    eligible = eligible[:config.next_move_candidates]
    if not eligible:
        return incurred + shadow.score(plan, sample), 0
    old_forecast_cost = shadow.score(plan, future_eta)
    best = None
    best_improvement = revision_penalty + 1e-8
    for i in eligible:
        proposal = planner.single_move(plan, i, t_next, future_eta)
        if proposal is None:
            continue
        g = old_forecast_cost - shadow.score(proposal, future_eta)
        if g > best_improvement:
            best = proposal
            best_improvement = g
    if best is None:
        return incurred + shadow.score(plan, sample), 0
    # Evaluate ONLY the yet-unstarted ships from the time-t_next physical
    # state.  Applying 'best' via rt.score would retroactively reschedule
    # ships that had already started before t_next.
    return incurred + shadow.score(best, sample), 1


def select_from_proposals(
    rt: 'RealizedRuntime',
    planner: 'SlotPlanner',
    published: dict[int,'Assignment'],
    current_eta: dict[int,float],
    proposals: list[tuple[dict[int,'Assignment'],tuple[int,...],str]],
    *,
    now: float,
    remaining_global: int | None,
    within_event_remaining: int,
    revision_penalty: float,
    min_gain: float,
    method: str,
    config: RiskConfig,
    run_seed: int,
    event_id: int,
    event_type: str,
) -> tuple[tuple|None,dict]:
    """Select a current action using point, expected, or risk+recourse score.

    For risk+recourse, INCLUDE Keep as a candidate with a possible future
    one-ship edit. This prices the option value of a saved budget unit.
    """
    if method not in {'deterministic','expected','risk_lookahead'}:
        raise ValueError(f'Unknown selection method {method}')
    if method != 'deterministic': config.validate()
    meta={'method':method,'evaluations':0,'scenarios_evaluated':0,
          'best_expected_gain':None,'best_tail_downside':None,
          'best_two_stage_gain':None,'kept_by_risk':False}
    old_point=rt.score(published,current_eta)
    if method == 'deterministic':
        key_best=None; chosen=None
        for candidate,changed,kind in proposals:
            cost=len(changed)
            if cost>within_event_remaining:continue
            meta['evaluations']+=1
            gain=old_point-rt.score(candidate,current_eta)
            net=gain-revision_penalty*cost
            if net<=min_gain:continue
            key=(net/cost,net,-cost,tuple(-i for i in changed))
            if key_best is None or key>key_best:
                key_best=key;chosen=(candidate,changed,gain,kind)
        return chosen,meta

    # Identical draws per decision event and per method. No reference to
    # Scenario.true_etas, Scenario.updates, or undisclosed future events.
    samples=sample_future_arrivals(current_eta,now,set(rt.started),
        run_seed=run_seed,event_id=event_id,event_type=event_type,config=config)
    baseline=[rt.score(published,s) for s in samples]
    scored=[]
    for candidate,changed,kind in proposals:
        cost=len(changed)
        if cost>within_event_remaining:continue
        gains=[b-rt.score(candidate,s) for b,s in zip(baseline,samples)]
        meta['evaluations']+=1
        meta['scenarios_evaluated']+=len(samples)
        expected=sum(gains)/len(gains)
        downside=upper_cvar([max(0.,-g) for g in gains],config.cvar_alpha)
        net=expected-revision_penalty*cost
        scored.append((net,expected,downside,candidate,changed,kind,gains))

    if method=='expected':
        best=None;best_key=None
        for net,expected,tail,candidate,changed,kind,gains in scored:
            if net<=min_gain:continue
            cost=len(changed)
            key=(net/cost,net,-cost,tuple(-i for i in changed))
            if best_key is None or key>best_key:
                best_key=key;best=(candidate,changed,old_point-rt.score(candidate,current_eta),kind)
                meta['best_expected_gain']=round(expected,6)
                meta['best_tail_downside']=round(tail,6)
        return best,meta

    # Risk-aware two-stage budget lookahead. Preselect candidates using current
    # stochastic value and downside; then perform the costly future repair
    # rollouts. Avoid bias from comparing one plan against a Keep baseline that
    # has no future opportunity: Keep gets the SAME simulated recourse.
    ranked=sorted(scored,key=lambda r:(r[0]-config.risk_aversion*r[2],r[0],-len(r[4])),reverse=True)
    # Include the best budget-1 and budget-2 proposal when possible, so a
    # bounded shortlist does not systematically exclude joint edits.
    selected=ranked[:config.lookahead_candidates]
    for cost_class in (1,2):
        other=next((x for x in ranked if len(x[4])==cost_class),None)
        if other is not None and other not in selected:
            selected.append(other)
    # Baseline represents Keep now, followed by the identical future recourse.
    available = remaining_global if remaining_global is not None else 10**9
    keep_cost=[]
    for sample in samples:
        outcome, future_cost=future_one_step_cost(rt,planner,published,current_eta,
            sample,now,available,config,revision_penalty)
        keep_cost.append(outcome+revision_penalty*future_cost)
    choice=None;choice_key=None
    for _net,expected,tail,candidate,changed,kind,_ in selected:
        cost_now=len(changed)
        future_budget=available-cost_now
        value=[]
        for b,sample in zip(keep_cost,samples):
            post, future_cost=future_one_step_cost(rt,planner,candidate,current_eta,
                sample,now,future_budget,config,revision_penalty)
            value.append(b-(post+revision_penalty*(cost_now+future_cost)))
        mean_two=sum(value)/len(value)
        cvar=upper_cvar([max(0.,-v) for v in value],config.cvar_alpha)
        objective=mean_two-config.risk_aversion*cvar
        # Compare total two-stage value, not per-unit ratio: budget opportunity
        # cost is already represented by future options in each branch.
        key=(objective,mean_two,-cost_now,tuple(-i for i in changed))
        if objective>min_gain and (choice_key is None or key>choice_key):
            choice_key=key
            choice=(candidate,changed,old_point-rt.score(candidate,current_eta),kind)
            meta['best_expected_gain']=round(expected,6)
            meta['best_tail_downside']=round(cvar,6)
            meta['best_two_stage_gain']=round(mean_two,6)
    meta['kept_by_risk']=choice is None and bool(scored)
    meta['scenarios_evaluated']+=len(samples)*(1+len(selected))
    meta['two_stage_candidates']=len(selected)
    return choice,meta
