"""Bounded multistage budget recourse baseline for V1.0.

This *model-predictive* baseline is deliberately separate from learned Q and
uses a synthetic future public ETA, not real future notifications. Each branch
keeps its own remaining hard budget. Reports actual simulated weighted waiting
for ships not already started at the present decision point.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from statistics import mean
from .risk_selection import upper_cvar


@dataclass(frozen=True)
class LookaheadSpecification:
    stages: int = 1
    step_hours: float = 12.0
    signal_strength: float = 0.5
    min_gain: float = 0.1
    candidates_per_stage: int = 4
    risk_weight: float = 0.0
    alpha: float = 0.90

    def __post_init__(self):
        if not 0 <= self.stages <= 6 or self.step_hours <= 0 or not 0 <= self.signal_strength <= 1:
            raise ValueError("Invalid stages / step_hours / signal_strength")
        if self.candidates_per_stage < 1 or self.min_gain < 0 or self.risk_weight < 0:
            raise ValueError("Invalid candidate count, threshold or risk weight")


def _remaining_wait(shadow, original_started: set[int], vessels, sample):
    return sum(
        vessels[i].weight * max(0., shadow.started[i].start - sample[i])
        for i in vessels if i not in original_started
    )


def rollout_future(env, initial_plan, eta_sample, budget_after_action: int,
                   spec: LookaheadSpecification) -> float:
    if budget_after_action < 0:
        raise ValueError("Budget exceeded")
    shadow = copy.deepcopy(env.runtime)
    original_started = set(shadow.started)
    plan = dict(initial_plan)
    public_eta = dict(env.published_eta)
    now = float(env.event_time)
    vessels = env.vessels

    for h in range(spec.stages):
        t_next = now + (h + 1) * spec.step_hours
        shadow.advance(t_next, plan, eta_sample)
        if budget_after_action == 0:
            continue
        # Future arrival is observed only at this hypothetical future time.
        synthetic_forecast = dict(public_eta)
        for i in vessels:
            if i in shadow.started:
                continue
            if eta_sample[i] <= t_next:
                synthetic_forecast[i] = t_next  # arrived, no leaked exact time
            else:
                synthetic_forecast[i] = max(
                    t_next, public_eta[i] + spec.signal_strength *
                    (eta_sample[i] - public_eta[i])
                )
        available = [i for i in vessels if i not in shadow.started]
        available.sort(key=lambda i: (plan[i].start, i))
        available = available[:spec.candidates_per_stage]
        baseline = shadow.score(plan, synthetic_forecast)
        best_plan = None
        best_gain = spec.min_gain
        for i in available:
            proposal = env.planner.single_move(plan, i, t_next, synthetic_forecast)
            if proposal is None:
                continue
            gain = baseline - shadow.score(proposal, synthetic_forecast)
            if gain > best_gain + 1e-9:
                best_gain, best_plan = gain, proposal
        if best_plan is not None:
            plan = best_plan
            budget_after_action -= 1
        # Future public signals arrive whether or not the controller edits.
        public_eta = synthetic_forecast
    shadow.finish(plan, eta_sample)
    return _remaining_wait(shadow, original_started, vessels, eta_sample)


def select_lookahead(env, candidate_plans: dict[int, dict], eta_samples,
                     spec: LookaheadSpecification):
    """Returns action, per-action objective and cost distribution."""
    if not eta_samples:
        raise ValueError("No scenarios")
    plans = {0: env.plan, **candidate_plans}
    b = int(env.remaining_budget)
    distribution = {}
    objective = {}
    for action, plan in plans.items():
        required = 0 if action == 0 else 1
        if required > b:
            continue
        values = [rollout_future(env, plan, sample, b-required, spec)
                  for sample in eta_samples]
        distribution[action] = values
        objective[action] = mean(values) + spec.risk_weight * upper_cvar(values, spec.alpha)
    chosen = min(objective, key=lambda a: (objective[a], a != 0, a))
    # No extra reward for marginal scores at or below original min-gain.
    if chosen and objective[0] - objective[chosen] <= spec.min_gain:
        chosen = 0
    return chosen, objective, distribution
