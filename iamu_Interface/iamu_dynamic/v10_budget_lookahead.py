"""Non-anticipative, budget-feasible MULTI-STAGE scenario recourse for V1.0 E2.

Keep and each Edit are evaluated on identical latent ETA scenarios. Each action
gets its own remaining hard budget; future actions are selected from SYNTHETIC
PUBLIC forecasts revealed sequentially, never from an episode's future updates.

This is a bounded, greedy rollout approximation to a multistage stochastic
control problem, NOT an exact optimal scenario-tree stochastic program.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence
from .risk_selection import upper_cvar


@dataclass(frozen=True)
class LookaheadSpecification:
    stages: int = 2
    step_hours: float = 12.0
    signal_strength: float = 0.5
    min_gain: float = 0.1
    candidates_per_stage: int = 4
    risk_weight: float = 0.0
    alpha: float = 0.90

    def __post_init__(self):
        if not isinstance(self.stages, int) or not 0 <= self.stages <= 6:
            raise ValueError("stages must be an integer in [0,6]")
        if not math.isfinite(self.step_hours) or self.step_hours <= 0:
            raise ValueError("step_hours must be finite positive")
        if not math.isfinite(self.signal_strength) or not 0 <= self.signal_strength <= 1:
            raise ValueError("signal_strength must be in [0,1]")
        if self.candidates_per_stage < 1 or self.min_gain < 0 or self.risk_weight < 0:
            raise ValueError("Invalid candidate count, threshold or risk weight")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")


@dataclass(frozen=True)
class MultistageEvaluation:
    """Expected future waiting with branch-correct budget and common scenarios."""
    expected_cost: Mapping[int, float]
    cvar_cost: Mapping[int, float]
    objective: Mapping[int, float]
    scenario_costs: Mapping[int, tuple[float, ...]]
    scenario_count: int
    stages: int

    def advantage(self, action: int) -> float:
        """Positive means 'Edit now' improves expected cost over Keep now."""
        return self.expected_cost[0] - self.expected_cost[action]


def _remaining_wait(shadow, original_started: set[int], vessels, sample):
    return sum(vessels[i].weight * max(0., shadow.started[i].start - sample[i])
               for i in vessels if i not in original_started)


def _synthetic_public_eta(public_eta, eta_sample, shadow, t_next, strength):
    """What the simulated controller can observe AT this future stage only.

    The latent sampled arrivals drive the simulated physics. A future ETA
    forecast is gradually nudged towards them; exact not-yet-arrived ETAs are
    never passed to the future repair optimizer (except strength=1, an explicit
    perfect-information sensitivity setting). For arrived ships, only their
    arrived-by-now status is used, not their hidden exact arrival time.
    """
    result = dict(public_eta)
    for i in public_eta:
        if i in shadow.started:
            continue
        if eta_sample[i] <= t_next:
            result[i] = float(t_next)
        else:
            result[i] = max(t_next, public_eta[i] +
                            strength * (eta_sample[i] - public_eta[i]))
    return result


def rollout_future(env, initial_plan, eta_sample, budget_after_action: int,
                   spec: LookaheadSpecification, *, return_trace: bool = False):
    """Replay one sampled future, taking at most one additional edit per stage.

    At every future stage: advance physical time, observe synthetic public ETA,
    score legal single-ship repairs on that ETA, and debit budget if selected.
    The final score uses the sampled latent ETA ONLY as evaluation ground truth.
    No hidden ETA/update from env.scenario is used here.
    """
    if budget_after_action < 0:
        raise ValueError("Budget exceeded")
    if set(initial_plan) != set(env.plan):
        raise ValueError("Candidate plan must contain the original vessel set")
    if set(eta_sample) != set(env.published_eta):
        raise ValueError("ETA scenario vessel set mismatch")
    shadow = copy.deepcopy(env.runtime)
    original_started = set(shadow.started)
    plan = dict(initial_plan)
    public_eta = dict(env.published_eta)
    now = float(env.event_time)
    vessels = env.vessels
    trace = []
    initial_budget = int(budget_after_action)

    for h in range(spec.stages):
        t_next = now + (h + 1) * spec.step_hours
        shadow.advance(t_next, plan, eta_sample)
        if budget_after_action <= 0:
            trace.append({'stage': h + 1, 'time': t_next,
                          'chosen_vessel': None, 'remaining_budget': 0})
            continue
        forecast = _synthetic_public_eta(public_eta, eta_sample, shadow,
                                         t_next, spec.signal_strength)
        eligible = [i for i in vessels if i not in shadow.started]
        eligible.sort(key=lambda i: (plan[i].start, i))
        eligible = eligible[:spec.candidates_per_stage]
        best_plan = None
        best_vessel = None
        best_gain = spec.min_gain
        if eligible:
            baseline = shadow.score(plan, forecast)
            for i in eligible:
                proposal = env.planner.single_move(plan, i, t_next, forecast)
                if proposal is None:
                    continue
                gain = baseline - shadow.score(proposal, forecast)
                if gain > best_gain + 1e-9:
                    best_gain, best_plan, best_vessel = gain, proposal, i
        if best_plan is not None:
            plan = best_plan
            budget_after_action -= 1
        trace.append({'stage': h + 1, 'time': t_next,
                      'chosen_vessel': best_vessel,
                      'remaining_budget': budget_after_action,
                      'forecast_gain': best_gain if best_plan else 0.0})
        public_eta = forecast
    shadow.finish(plan, eta_sample)
    cost = float(_remaining_wait(shadow, original_started, vessels, eta_sample))
    if initial_budget - budget_after_action > spec.stages:
        raise AssertionError('More than one future edit per stage')
    return (cost, trace) if return_trace else cost


def evaluate_multistage(env, candidate_plans: Mapping[int, dict],
                        eta_samples: Sequence[Mapping[int, float]],
                        spec: LookaheadSpecification) -> MultistageEvaluation:
    """Evaluate Keep and every feasible current edit on COMMON future samples.

    Keep starts with b remaining edits; each Edit starts with b-1. The physical
    dispatcher and recourse planner are copied per branch, so no branch can
    consume budget belonging to another branch.
    """
    if not eta_samples:
        raise ValueError("At least one ETA scenario required")
    b = int(env.remaining_budget)
    if b < 0:
        raise ValueError("Negative budget")
    if 0 in candidate_plans:
        raise ValueError("Edit candidate action 0 conflicts with Keep")
    plans = {0: env.plan, **candidate_plans}
    values = {}
    expectations = {}
    tail = {}
    objectives = {}
    for action, plan in plans.items():
        edit_cost = 0 if action == 0 else 1  # V1.0 single-move only
        if edit_cost > b:
            continue
        series = tuple(float(rollout_future(
            env, plan, sample, b-edit_cost, spec)) for sample in eta_samples)
        values[action] = series
        expectations[action] = float(mean(series))
        tail[action] = float(upper_cvar(series, spec.alpha))
        objectives[action] = expectations[action] + spec.risk_weight * tail[action]
    return MultistageEvaluation(expectations, tail, objectives, values,
                                len(eta_samples), spec.stages)


def select_lookahead(env, candidate_plans, eta_samples, spec):
    """Pure model-predictive baseline; this does not use DDQN Q values."""
    estimate = evaluate_multistage(env, candidate_plans, eta_samples, spec)
    chosen = min(estimate.objective,
                 key=lambda a: (estimate.objective[a], a != 0, a))
    if chosen and (estimate.objective[0] - estimate.objective[chosen]
                   <= spec.min_gain):
        chosen = 0
    return chosen, dict(estimate.objective), dict(estimate.scenario_costs)
