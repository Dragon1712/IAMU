"""Pure, independently unit-testable E4 Q-gap and Keep-collapse diagnostics.

The reward scale is the original environment's reward_scale, not an arbitrary
calibration coefficient. A predicted Q-gap is compared to a *paired* realized
Keep-vs-Edit improvement in the same units. A single realized future is NOT a
conditional expected return or an exact action-value label.
"""
from __future__ import annotations

from math import sqrt, isfinite
from statistics import mean


def action_diagnostics(q, mask, keep_margin):
    """Match DoubleDQNAgent.act(epsilon=0): Keep wins ties after +margin.

    Return an action, the best legal Edit's raw Q-gap and a Keep explanation.
    Illegal actions must never enter comparisons (even if a stale Q is huge).
    """
    q = list(map(float, q))
    mask = list(map(bool, mask))
    if len(q) != len(mask) or not q or not mask[0]:
        raise ValueError('Malformed Q vector or Keep mask')
    if not isfinite(keep_margin) or keep_margin < 0:
        raise ValueError('Invalid Keep margin')
    legal = [a for a in range(1, len(q)) if mask[a]]
    if not isfinite(q[0]) or any(not isfinite(q[a]) for a in legal):
        raise ValueError('Non-finite legal Q value')
    if not legal:
        return dict(action=0, action_without_margin=0,
                    best_edit=None, best_gap=None, q_keep=q[0],
                    q_best_edit=None, margin_distance=None,
                    reason='no_legal_edit')
    best = max(legal, key=lambda a: (q[a], -a))
    gap = q[best] - q[0]
    action = best if gap > keep_margin else 0
    action_zero = best if gap > 0. else 0
    return dict(action=action, action_without_margin=action_zero,
                best_edit=best, best_gap=gap, q_keep=q[0],
                q_best_edit=q[best], margin_distance=gap - keep_margin,
                reason=('edit' if action else
                        'keep_margin_only' if gap > 0. else 'q_prefers_keep'))


def choose_evenly(n, k):
    """Deterministic temporal-stratum indices, selected without future costs.

    Select midpoints of k equal contiguous bins; no duplicate indices.
    Useful when some trajectories have <k actionable events.
    """
    if n < 0 or k < 1:
        raise ValueError('Invalid population or sample size')
    m = min(n, k)
    return [min(n-1, (2*i + 1)*n // (2*m)) for i in range(m)]


def cvar_upper(values, alpha=0.90):
    """Fractional upper-tail CVaR, without reliance on scipy."""
    v = sorted((float(x) for x in values), reverse=True)
    if not v or not (0. < alpha < 1.):
        return None
    remain = len(v)*(1.-alpha)
    total=0.
    width=remain
    for x in v:
        take=min(1.,width)
        total+=take*x
        width-=take
        if width<=1e-12:
            break
    return total/remain


def pearson(x, y):
    """Pearson r; undefined for constant predictions or observations."""
    if len(x)!=len(y) or len(x)<2:
        return None
    a, b = mean(x), mean(y)
    sx=sum((v-a)**2 for v in x)
    sy=sum((v-b)**2 for v in y)
    if sx<=1e-18 or sy<=1e-18:
        return None
    return sum((u-a)*(v-b) for u,v in zip(x,y))/sqrt(sx*sy)


def calibration_summary(rows, *, truth_dead_zone=0.):
    """One row per legal candidate Edit, not per episode.

    truth_dead_zone is in *normalized reward units*. Zero true gains are
    excluded from sign accuracy because they are not directional successes.
    """
    if not rows:
        return dict(n_pairs=0)
    if truth_dead_zone<0:
        raise ValueError('truth_dead_zone must be nonnegative')
    x=[float(r['q_gap']) for r in rows]
    y=[float(r['true_gain_scaled']) for r in rows]
    errors=[a-b for a,b in zip(x,y)]
    directional=[((a>0)==(b>0)) for a,b in zip(x,y) if abs(b)>truth_dead_zone]
    near=[r for r in rows if abs(float(r['q_gap'])-float(r['keep_margin']))<=0.05]
    return dict(n_pairs=len(rows),
                n_directional=len(directional),
                q_gap_mae=mean(abs(e) for e in errors),
                q_gap_rmse=sqrt(mean(e*e for e in errors)),
                q_gap_signed_bias=mean(errors),
                mean_q_gap=mean(x), mean_realized_gain_scaled=mean(y),
                pearson_r=pearson(x,y),
                direction_accuracy=(mean(directional) if directional else None),
                n_near_margin=len(near),
                near_margin_mae=(mean(abs(float(r['q_gap'])-
                                          float(r['true_gain_scaled']))
                                      for r in near) if near else None))


def regret_summary(rows):
    if not rows:
        return dict(n_states=0)
    regrets=[float(r['first_action_regret']) for r in rows]
    actionable=[r for r in rows if int(r['n_legal_edits'])>0]
    return dict(n_states=len(rows),
                n_with_legal_edits=len(actionable),
                mean_first_action_regret=mean(regrets),
                cvar90_first_action_regret=cvar_upper(regrets),
                max_first_action_regret=max(regrets),
                missed_beneficial_keep=sum(bool(r['missed_beneficial_keep']) for r in rows),
                harmful_edit=sum(bool(r['harmful_edit']) for r in rows),
                total_keep=sum(int(r['selected_action'])==0 for r in rows),
                mean_budget_at_state=mean(int(r['remaining_budget']) for r in rows))
