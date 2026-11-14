"""Frozen-checkpoint V1.0 action policies on EXACT V0.9 legal candidate sets.

Policy corrections are research ablations, NOT a calibrated replacement for
Q(s,a). Both budget opportunity and risk may already be implicit in learned Q.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import numpy as np
from .v10_risk_features import (
    RiskSpecification, event_seed, evaluate_proposals, public_eta_scenarios,
)
from .v10_budget_lookahead import LookaheadSpecification, select_lookahead

POLICIES = (
    "keep", "periodic", "greedy", "set_ddqn", "q_risk", "q_budget",
    "q_joint", "expected", "lookahead",
)


@dataclass(frozen=True)
class PolicySpecification:
    risk: RiskSpecification = RiskSpecification()
    lookahead: LookaheadSpecification = LookaheadSpecification()
    beta_risk: float = 0.25
    beta_budget: float = 0.10

    def __post_init__(self):
        if not (math.isfinite(self.beta_risk) and math.isfinite(self.beta_budget)):
            raise ValueError("Non-finite beta")
        if self.beta_risk < 0 or self.beta_budget < 0:
            raise ValueError("Negative risk/budget coefficient")


def public_arrival_flags(env) -> set[int]:
    """Only the physical arrived-by-now sensor; never provide actual future ETA.

    V0.9 already exposes this sensor in rl_env._observation(). We intentionally
    do NOT return the hidden actual arrival timestamp, only the arrived flag.
    """
    return {i for i in env.ids if env.scenario.true_etas[i] <= env.event_time + 1e-8}


def budget_scarcity(env) -> tuple[float, float]:
    """Simple public-state heuristic; NOT a learned shadow price.

    Lower future opportunity count / more remaining budget => lower scarcity.
    Ignore future env.events, which may encode information unknown now.
    """
    pending = len(env.ids) - len(env.runtime.started)
    if pending <= 0:
        return 0.0, 0.0
    recent_updates = sum(env.update_counts.values()) / max(1, len(env.ids))
    # Bounded prediction from currently observed update history / pending ships.
    future_opportunities = max(1.0, pending * min(1.0, 0.5 + 0.1 * recent_updates))
    scarce = max(0.0, (future_opportunities - env.remaining_budget) / future_opportunities)
    return scarce, float(future_opportunities)


def frozen_q_values(agent, observation) -> np.ndarray:
    import torch
    agent.online.eval()
    with torch.no_grad():
        g = torch.from_numpy(observation.global_features).to(agent.device).unsqueeze(0)
        a = torch.from_numpy(observation.action_features).to(agent.device).unsqueeze(0)
        mask = torch.from_numpy(observation.mask).to(agent.device).unsqueeze(0)
        q = agent.online(g, a, mask).squeeze(0).cpu().numpy()
    return np.asarray(q, dtype=float)


class V10Policy:
    def __init__(self, name: str, agent, spec: PolicySpecification):
        if name not in POLICIES:
            raise ValueError(f"Unknown V1.0 policy: {name}")
        self.name, self.agent, self.spec = name, agent, spec

    def choose(self, env, obs):
        name = self.name
        q = None
        if name in ("keep", "periodic", "greedy"):
            if name == "keep" or (name == "periodic" and env.event_type != "clock"):
                return 0, {"policy": name}
            return env.point_greedy_action(), {"policy": name}
        if name == "set_ddqn":
            return int(self.agent.act(obs, epsilon=0.0)), {"policy": name}
        if env.remaining_budget == 0 or int(obs.mask.sum()) == 1:
            return 0, {"policy": name, "forced_keep": True}

        candidates = {i: env._candidates[i-1] for i in range(1, len(obs.mask))
                      if bool(obs.mask[i])}
        q = frozen_q_values(self.agent, obs) if name.startswith("q_") else None
        risk_seed = event_seed(env.scenario.seed, env.event_index, env.event_type)
        risk = {}
        samples = None
        if name != "q_budget":
            risk, samples, _ = evaluate_proposals(
                env.runtime, env.plan, candidates,
                env.published_eta, env.event_time, public_arrival_flags(env),
                self.spec.risk, seed=risk_seed)

        if name == "lookahead":
            action, objectives, _ = select_lookahead(
                env, candidates, samples, self.spec.lookahead)
            return action, {"policy": name,
                            "value_of_edit": objectives[0] - objectives[action],
                            "objective_keep": objectives[0],
                            "objective_selected": objectives[action]}
        if name == "expected":
            action = max([0, *candidates], key=lambda i:
                         (0.0 if i == 0 else risk[i].mean_gain, -i))
            if action and risk[action].mean_gain <= env.min_gain:
                action = 0
            r = risk.get(action)
            return action, {"policy": name, "mean_gain": r.mean_gain if r else 0.0}

        scarce, opportunities = budget_scarcity(env)
        q_keep = float(q[0])
        margin = float(self.agent.keep_margin)
        score = {}
        for action in candidates:
            gap = float(q[action]) - q_keep
            tail_delta = risk[action].tail_cost_delta if action in risk else 0.0
            value = gap
            if name in ("q_risk", "q_joint"):
                value -= self.spec.beta_risk * tail_delta / env.reward_scale
            if name in ("q_budget", "q_joint"):
                value -= self.spec.beta_budget * scarce
            score[action] = value
        best_action = max(score, key=lambda i: (score[i], -i))
        action = best_action if score[best_action] > margin else 0
        chosen_risk = risk.get(best_action)
        return action, {
            "policy": name, "best_action": int(best_action),
            "best_qgap": float(q[best_action]-q_keep),
            "adjusted_gap": float(score[best_action]),
            "keep_margin": margin,
            "scarcity": float(scarce), "future_opportunities": opportunities,
            "tail_cost_delta": (chosen_risk.tail_cost_delta if chosen_risk else 0.0),
            "mean_gain": (chosen_risk.mean_gain if chosen_risk else 0.0),
            "harm_probability": (chosen_risk.harm_probability if chosen_risk else 0.0),
        }
