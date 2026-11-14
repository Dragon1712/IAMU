"""V1.0 public-information ETA sampling and candidate risk features.

No Scenario or true_etas is accepted. Arrived vessel status is supplied explicitly.
Losses are raw weighted-wait units, not scaled DDQN rewards.
"""
from __future__ import annotations
from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence
import hashlib
import math
import random
import numpy as np
from .risk_selection import upper_cvar


@dataclass(frozen=True)
class RiskSpecification:
    sigma: float = 7.12463033710398
    samples: int = 16
    alpha: float = 0.90
    belief: str = "censored"  # censored max(now, N); truncated N | N>now

    def __post_init__(self):
        if self.samples < 1 or self.sigma < 0 or not math.isfinite(self.sigma):
            raise ValueError("samples>=1 and finite nonnegative sigma required")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if self.belief not in ("censored", "truncated"):
            raise ValueError("belief must be censored or truncated")


@dataclass(frozen=True)
class RiskMetrics:
    mean_gain: float
    gain_std: float
    harm_probability: float
    positive_probability: float
    downside_cvar: float
    tail_cost_delta: float
    n_samples: int


def event_seed(base_seed: int, event_index: int, event_type: str) -> int:
    """Stable across different policies and Python processes, not Python hash()."""
    data = f"iamu-v10|{int(base_seed)}|{int(event_index)}|{event_type}".encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def public_eta_scenarios(
    published_eta: Mapping[int, float], now: float,
    arrived_ids: set[int] | frozenset[int], spec: RiskSpecification,
    *, seed: int,
) -> list[dict[int, float]]:
    """Generate common-random-number scenarios from observable public information.

    Arrived-but-unstarted ships are assigned `now` as a conservative known-
    arrived proxy; the simulator's hidden exact arrival time is NEVER read.
    The actual vessel-arrived indicator must be computed by the environment's
    physical sensors, not from future event lists.
    """
    rng = random.Random(int(seed))
    nd = NormalDist()
    samples = []
    for _ in range(spec.samples):
        scenario = {}
        for vessel_id in sorted(published_eta):
            mu = float(published_eta[vessel_id])
            if not math.isfinite(mu):
                raise ValueError("Invalid published ETA")
            if vessel_id in arrived_ids:
                scenario[vessel_id] = float(now)
                continue
            if spec.sigma == 0:
                scenario[vessel_id] = max(float(now), mu)
                continue
            if spec.belief == "censored":
                scenario[vessel_id] = max(float(now), rng.gauss(mu, spec.sigma))
            else:
                cutoff = nd.cdf((now - mu) / spec.sigma)
                if cutoff >= 1 - 1e-14:
                    # Normal tail approximation for extremely stale forecasts;
                    # exact inverse CDF would underflow to infinity here.
                    scale = spec.sigma**2 / max(spec.sigma, now - mu)
                    scenario[vessel_id] = now + rng.expovariate(1 / max(scale, 1e-12))
                else:
                    p = cutoff + rng.random() * (1 - cutoff)
                    p = min(1 - 1e-15, max(1e-15, p))
                    scenario[vessel_id] = max(now, mu + spec.sigma * nd.inv_cdf(p))
        samples.append(scenario)
    return samples


def summarize_risk(keep_costs: Sequence[float], edit_costs: Sequence[float],
                   alpha: float = .9) -> RiskMetrics:
    keep = np.asarray(keep_costs, dtype=np.float64)
    edit = np.asarray(edit_costs, dtype=np.float64)
    if keep.size == 0 or keep.shape != edit.shape or not np.isfinite(keep).all() or not np.isfinite(edit).all():
        raise ValueError("Same nonempty finite scenario costs required")
    gain = keep - edit
    return RiskMetrics(
        mean_gain=float(gain.mean()), gain_std=float(gain.std(ddof=0)),
        harm_probability=float(np.mean(gain < -1e-9)),
        positive_probability=float(np.mean(gain > 1e-9)),
        downside_cvar=upper_cvar(np.maximum(0.0, -gain).tolist(), alpha),
        tail_cost_delta=upper_cvar(edit.tolist(), alpha)-upper_cvar(keep.tolist(), alpha),
        n_samples=int(gain.size),
    )


def evaluate_proposals(runtime, current_plan, candidate_plans: Mapping[int, dict],
                       public_eta: Mapping[int, float], now: float,
                       arrived_ids: set[int], spec: RiskSpecification, *,
                       seed: int):
    """One set of sampled ETA scenarios is reused for Keep and every Edit.

    `runtime` is the already-realized dispatcher state; no hidden future data
    may be used by the caller to construct candidate_plans or public_eta.
    """
    scenarios = public_eta_scenarios(public_eta, now, arrived_ids, spec, seed=seed)
    keep = [float(runtime.score(current_plan, eta)) for eta in scenarios]
    scored = {}
    for action, plan in candidate_plans.items():
        costs = [float(runtime.score(plan, eta)) for eta in scenarios]
        scored[action] = summarize_risk(keep, costs, alpha=spec.alpha)
    return scored, scenarios, keep
