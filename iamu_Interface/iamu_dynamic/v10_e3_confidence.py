"""V1.0-E3: testable, inference-only confidence gates for Q/MPC fusion.

MPC paired differences MUST use shared scenarios for Keep and every Edit.
The t-style half-width is an approximate diagnostic, not a coverage guarantee
under non-Gaussian heavy-tail ETA shocks or dependent Monte Carlo draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Mapping, Sequence


# Student-t 97.5% quantiles for small samples, asymptotic approximation for large.
_T975 = (0., 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
         2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080,
         2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
         2.045, 2.042)


def t975(df: int) -> float:
    if df < 1:
        raise ValueError("A variance interval requires n>=2")
    if df < len(_T975):
        return _T975[df]
    z = 1.95996398454
    # Cornish-Fisher approximation for degrees of freedom >=31.
    return z + (z**3+z)/(4*df) + (5*z**5+16*z**3+3*z)/(96*df**2)


@dataclass(frozen=True)
class PairedConfidence:
    mean: float
    sd: float
    se: float
    half_width: float
    lower: float
    upper: float
    signal_strength: float
    n: int


def paired_confidence(keep_costs: Sequence[float], edit_costs: Sequence[float],
                      reward_scale: float = 100.) -> PairedConfidence:
    """Paired improvement (Keep - Edit) in the SAME normalized Q reward units."""
    k, e = tuple(map(float, keep_costs)), tuple(map(float, edit_costs))
    if len(k) != len(e) or not k or reward_scale <= 0 or not isfinite(reward_scale):
        raise ValueError("Matching, nonempty paired samples and positive scale required")
    if not all(isfinite(v) for v in (*k, *e)):
        raise ValueError("Nonfinite scenario cost")
    vals = [(a-b)/reward_scale for a,b in zip(k,e)]
    n = len(vals)
    m = mean(vals)
    sd = stdev(vals) if n>=2 else float('inf')
    se = sd / sqrt(n)
    half = t975(n-1)*se if n>=2 else float('inf')
    # Gate ranges [0,1]; zero when the confidence interval crosses zero.
    # For deterministic paired differences, we can trust the simulated signal
    # only as much as this *model* allows, never as empirical calibration.
    strength = max(0., min(1., 1.-half/max(abs(m), 1e-12))) if n>=2 else 0.
    return PairedConfidence(m,sd,se,half,m-half,m+half,strength,n)


def summarize_mpc(mpc, reward_scale: float) -> dict[int, PairedConfidence]:
    reference = mpc.scenario_costs[0]
    return {a: paired_confidence(reference, series, reward_scale)
            for a,series in mpc.scenario_costs.items() if a != 0}


def directions_agree(qgap: float, mpc_mean: float, neutral: float = 0.) -> bool:
    """Neutral signals do not count as a contradiction; threshold in Q units."""
    if neutral < 0 or not isfinite(neutral):
        raise ValueError("Invalid neutrality tolerance")
    if abs(qgap) <= neutral or abs(mpc_mean) <= neutral:
        return True
    return (qgap > 0) == (mpc_mean > 0)


MODES = ('q_only', 'q_risk_only', 'pure_mpc', 'pure_mpc_risk',
         'q_fixed', 'q_confidence', 'q_agreement', 'q_adaptive')


def fused_scores(q: Sequence[float], mask: Sequence[bool],
                 mpc_stats: Mapping[int, PairedConfidence], *,
                 mode: str, beta_budget: float, beta_risk: float,
                 risk_metrics: Mapping[int, object] | None,
                 reward_scale: float, keep_margin: float,
                 neutral: float = 0.) -> tuple[int, dict[int, dict]]:
    """All fusion variants consume identical Q/MPC/risk estimates per state.

    q_only: original frozen Keep margin, no risk or MPC.
    q_risk_only: Q-gap with the SAME risk penalty as the fused policies.
    pure_mpc: ONLY MPC expected reward advantage, no Q and NO risk term.
    pure_mpc_risk: MPC expected advantage + the SAME risk penalty.
    q_fixed: E2's constant (1-beta)Q + beta*MPC.
    q_confidence: shrink beta by CI stability only.
    q_agreement: shrink beta by Q/MPC directional consistency only.
    q_adaptive: use BOTH gates; risk term is held constant across variants.

    The Edit decision uses original Keep margin except pure MPC, which uses
    a zero-edit-cost threshold explicitly supplied via keep_margin=0 caller.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if not 0. <= beta_budget <= 1. or beta_risk < 0. or reward_scale <= 0.:
        raise ValueError("Invalid mixing, risk weight, or reward scale")
    if not isfinite(keep_margin):
        raise ValueError("Nonfinite keep margin")
    if len(q) != len(mask) or not mask[0]:
        raise ValueError("Q / action mask mismatch")
    details = {}
    for a in range(1,len(q)):
        if not mask[a]:
            continue
        qgap = float(q[a]) - float(q[0])
        if not isfinite(qgap):
            raise ValueError("Nonfinite Q gap")
        stats = mpc_stats.get(a)
        if mode != 'q_only' and stats is None:
            raise ValueError(f"MPC statistics absent for legal action {a}")
        confidence = stats.signal_strength if stats else 0.
        agreement = directions_agree(qgap, stats.mean, neutral) if stats else True
        if mode in ('q_only','q_risk_only'):
            w = 0.
        elif mode in ('pure_mpc','pure_mpc_risk'):
            w = 1.
        elif mode == 'q_fixed':
            w = beta_budget
        elif mode == 'q_confidence':
            w = beta_budget * confidence
        elif mode == 'q_agreement':
            w = beta_budget * float(agreement)
        else:
            w = beta_budget * confidence * float(agreement)
        fused = (1.-w)*qgap + w*(stats.mean if stats else 0.)
        tail = (float(risk_metrics[a].tail_cost_delta)/reward_scale
                if risk_metrics is not None and a in risk_metrics and mode not in ('q_only','pure_mpc')
                else 0.)
        score = fused - beta_risk*tail
        if not isfinite(score):
            raise ValueError("Nonfinite policy score")
        details[a] = dict(q_gap=qgap, mpc_mean=(stats.mean if stats else None),
                          mpc_se=(stats.se if stats else None),
                          mpc_ci_half=(stats.half_width if stats else None),
                          mpc_lcb=(stats.lower if stats else None),
                          mpc_ucb=(stats.upper if stats else None),
                          mpc_strength=confidence,
                          agree=bool(agreement), effective_weight=w,
                          tail_delta_scaled=tail, score=score)
    if not details:
        return 0, details
    best = max(details, key=lambda a: (details[a]['score'], -a))
    return (best if details[best]['score'] > keep_margin else 0), details
