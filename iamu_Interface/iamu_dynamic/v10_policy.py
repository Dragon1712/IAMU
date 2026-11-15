"""V1.0-E2 frozen-checkpoint policies with budget-feasible multistage recourse.

A 'q_budget' score now FUSES the frozen Q-gap with a multi-stage, budget-aware
scenario advantage. It does not add a second heuristic scarcity penalty.
The original heuristic variants remain available as historical controls.
No V0.9 network weights, environment, reward or legal actions are modified.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from time import perf_counter
import numpy as np
from .v10_risk_features import (
    RiskSpecification, event_seed, evaluate_proposals, public_eta_scenarios,
)
from .v10_budget_lookahead import (
    LookaheadSpecification, evaluate_multistage, select_lookahead,
)

POLICIES = (
    'keep', 'periodic', 'greedy', 'set_ddqn', 'q_risk',
    'q_budget', 'q_joint', 'expected', 'lookahead',
    'q_budget_heuristic', 'q_joint_heuristic',
)


@dataclass(frozen=True)
class PolicySpecification:
    risk: RiskSpecification = RiskSpecification()
    lookahead: LookaheadSpecification = LookaheadSpecification()
    beta_risk: float = 0.25
    beta_budget: float = 0.25  # E2: convex MPC-Q fusion weight, not scarcity penalty
    mpc_samples: int = 8
    legacy_beta_budget: float = 0.10

    def __post_init__(self):
        if not math.isfinite(self.beta_risk) or self.beta_risk < 0:
            raise ValueError('Invalid risk weight')
        if not math.isfinite(self.beta_budget) or not 0 <= self.beta_budget <= 1:
            raise ValueError('MPC fusion weight must be in [0,1]')
        if not math.isfinite(self.legacy_beta_budget) or self.legacy_beta_budget < 0:
            raise ValueError('Invalid legacy budget coefficient')
        if not isinstance(self.mpc_samples, int) or self.mpc_samples < 1:
            raise ValueError('mpc_samples must be >=1')


def public_arrival_flags(env) -> set[int]:
    """Physical arrived-by-now flag, not a future ETA oracle."""
    return {i for i in env.ids if env.scenario.true_etas[i] <= env.event_time + 1e-8}


def budget_scarcity(env) -> tuple[float, float]:
    """LEGACY V1.0 heuristic, preserved solely for paired comparison."""
    pending = len(env.ids) - len(env.runtime.started)
    if pending <= 0:
        return 0.0, 0.0
    recent_updates = sum(env.update_counts.values()) / max(1, len(env.ids))
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


def choose_adjusted_action(q, mask, *, margin, reward_scale,
                           risk_metrics=None, mpc_estimate=None,
                           beta_risk=0.0, beta_budget=0.0,
                           scarcity=None, legacy=False):
    """Pure testable action decision; common scoring for E2 and stability tests.

    MPC: score=(1-w)*Qgap+w*(E[C_keep,b]-E[C_edit,b-1])/reward_scale
    Risk: subtract risk-weight*(CVaR(C_edit)-CVaR(C_keep))/reward_scale
    Legacy: subtract heuristic budget scarcity (never mix with MPC).
    """
    q = np.asarray(q, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if q.shape != mask.shape or not mask[0]:
        raise ValueError('Q and legal action mask mismatch')
    if not reward_scale > 0 or not math.isfinite(reward_scale):
        raise ValueError('reward_scale must be positive')
    scores = {}
    for a in range(1, len(q)):
        if not mask[a]:
            continue
        gap = float(q[a] - q[0])
        value = gap
        if legacy:
            if scarcity is not None:
                value -= beta_budget * scarcity
        elif mpc_estimate is not None:
            mpc_gap = mpc_estimate.advantage(a) / reward_scale
            value = (1 - beta_budget) * gap + beta_budget * mpc_gap
        if risk_metrics is not None and a in risk_metrics:
            value -= beta_risk * risk_metrics[a].tail_cost_delta / reward_scale
        scores[a] = value
    if not scores:
        return 0, scores, None
    best = max(scores, key=lambda a: (scores[a], -a))
    return (best if scores[best] > margin else 0), scores, best


class V10Policy:
    def __init__(self, name: str, agent, spec: PolicySpecification):
        if name not in POLICIES:
            raise ValueError(f'Unknown V1.0 policy: {name}')
        self.name, self.agent, self.spec = name, agent, spec

    def choose(self, env, obs):
        name = self.name
        if name in ('keep', 'periodic', 'greedy'):
            if name == 'keep' or (name == 'periodic' and env.event_type != 'clock'):
                return 0, {'policy': name}
            return env.point_greedy_action(), {'policy': name}
        if name == 'set_ddqn':
            return int(self.agent.act(obs, epsilon=0.0)), {'policy': name}
        if env.remaining_budget == 0 or int(obs.mask.sum()) == 1:
            return 0, {'policy': name, 'forced_keep': True}

        candidates = {i: env._candidates[i-1] for i in range(1, len(obs.mask))
                      if bool(obs.mask[i])}
        q = frozen_q_values(self.agent, obs) if name.startswith('q_') else None
        risk_seed = event_seed(env.scenario.seed, env.event_index, env.event_type)
        risk = {}
        samples = None
        risk_seconds = 0.0
        mpc_seconds = 0.0
        needs_risk = name in ('q_risk', 'q_joint', 'q_joint_heuristic', 'expected')
        needs_mpc = (name == 'lookahead' or
                     (name in ('q_budget', 'q_joint') and self.spec.beta_budget > 0))
        if needs_risk:
            start = perf_counter()
            risk, samples, _ = evaluate_proposals(
                env.runtime, env.plan, candidates, env.published_eta,
                env.event_time, public_arrival_flags(env), self.spec.risk,
                seed=risk_seed)
            risk_seconds = perf_counter() - start
        elif needs_mpc:
            # Same public ETA belief, without evaluating unused risk metrics.
            samples = public_eta_scenarios(
                env.published_eta, env.event_time, public_arrival_flags(env),
                self.spec.risk, seed=risk_seed)

        if name == 'expected':
            action = max([0, *candidates], key=lambda i:
                         (0.0 if i == 0 else risk[i].mean_gain, -i))
            if action and risk[action].mean_gain <= env.min_gain:
                action = 0
            r = risk.get(action)
            return action, {'policy': name, 'mean_gain': r.mean_gain if r else 0.,
                            'risk_seconds': risk_seconds}

        mpc = None
        if needs_mpc:
            start = perf_counter()
            # Fixed prefix of the SAME common random scenarios as risk scoring.
            # A smaller MPC sample cap saves CPU but must be sensitivity-tested.
            mpc = evaluate_multistage(
                env, candidates, samples[:self.spec.mpc_samples], self.spec.lookahead)
            mpc_seconds = perf_counter() - start
        if name == 'lookahead':
            action = min(mpc.objective, key=lambda a: (mpc.objective[a], a != 0, a))
            if action and mpc.objective[0] - mpc.objective[action] <= env.min_gain:
                action = 0
            return action, {'policy': name,
                            'mpc_gain': mpc.advantage(action),
                            'objective_keep': mpc.objective[0],
                            'objective_selected': mpc.objective[action],
                            'mpc_scenarios': mpc.scenario_count,
                            'mpc_stages': mpc.stages,
                            'mpc_seconds': mpc_seconds}

        use_risk = name in ('q_risk', 'q_joint', 'q_joint_heuristic')
        legacy = name in ('q_budget_heuristic', 'q_joint_heuristic')
        use_mpc = name in ('q_budget', 'q_joint')
        scarcity, opportunities = budget_scarcity(env) if legacy else (None, None)
        margin = float(self.agent.keep_margin)
        action, scores, best = choose_adjusted_action(
            q, obs.mask, margin=margin, reward_scale=env.reward_scale,
            risk_metrics=risk if use_risk else None,
            mpc_estimate=mpc if use_mpc else None,
            beta_risk=self.spec.beta_risk if use_risk else 0.,
            beta_budget=(self.spec.legacy_beta_budget if legacy else
                         self.spec.beta_budget if use_mpc else 0.),
            scarcity=scarcity, legacy=legacy)
        r = risk.get(best)
        return action, {
            'policy': name, 'best_action': best,
            'best_qgap': float(q[best]-q[0]),
            'adjusted_gap': float(scores[best]),
            'keep_margin': margin,
            'mpc_gap_scaled': (mpc.advantage(best) / env.reward_scale if mpc else None),
            'mpc_mean_keep': (mpc.expected_cost[0] if mpc else None),
            'mpc_mean_edit': (mpc.expected_cost[best] if mpc else None),
            'mpc_scenarios': (mpc.scenario_count if mpc else 0),
            'mpc_stages': (mpc.stages if mpc else 0),
            'risk_seconds': risk_seconds,
            'mpc_seconds': mpc_seconds,
            'scarcity': scarcity,
            'future_opportunities': opportunities,
            'tail_cost_delta': (r.tail_cost_delta if r else None),
            'mean_gain': (r.mean_gain if r else None),
            'harm_probability': (r.harm_probability if r else None),
        }
