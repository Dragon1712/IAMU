"""Budgeted local repair, inspired by railway disruption-recovery neighborhoods.

This is a fresh Python implementation for IAMU, NOT copied source code from
INFORMS, Robust_Railway or RecovAir.  Reference commitments remain fixed;
only selected, not-yet-started vessels may move.  The dynamic controller is
still untrained and the original ERL-DDE policy remains frozen.
"""
from __future__ import annotations

from itertools import permutations, product
from typing import Iterator

from .core import Assignment, Hint, Port, SlotPlanner, Vessel, assignments_equal, validate_plan, EPS


def pressure_graph(plan: dict[int, Assignment], vessels: dict[int, Vessel],
                   port: Port, etas: dict[int, float], now: float,
                   eligible: list[int]) -> dict[int, list[int]]:
    """Build a deterministic *pressure* graph, not a railway conflict graph.

    Edges indicate potentially coupled reservations: nearby occupancy of the
    same berth zone, or overlapping time with high joint crane usage.  The
    published plan remains feasible; this graph only selects repair pairs.
    """
    graph: dict[int, list[int]] = {i: [] for i in eligible}
    for k, i in enumerate(eligible):
        a = plan[i]
        for j in eligible[k + 1:]:
            b = plan[j]
            va, vb = vessels[i], vessels[j]
            gap = port.safety_ratio * max(va.length, vb.length)
            # Physical berth intervals are adjacent/overlapping including the
            # original safety gap; scheduled times are at most one service
            # duration apart.  This is *pressure*, not an actual violation.
            berth_near = (a.pos < b.pos + vb.length + gap + EPS and
                          b.pos < a.pos + va.length + gap + EPS)
            time_gap = max(0.0, max(a.start, b.start) - min(a.end, b.end))
            near_time = time_gap <= max(a.duration, b.duration) + EPS
            active_overlap = a.start < b.end - EPS and b.start < a.end - EPS
            crane_pressure = a.cranes + b.cranes >= max(2, port.cranes - 1)
            # Forecast-delayed ships near their commitments can also interact.
            delayed = etas[i] > a.start + EPS or etas[j] > b.start + EPS
            if (berth_near and near_time) or (active_overlap and crane_pressure) or (delayed and berth_near):
                graph[i].append(j)
                graph[j].append(i)
    for i in eligible:
        graph[i].sort(key=lambda j: (abs(plan[j].start - plan[i].start), j))
    return graph


def candidate_pairs(graph: dict[int, list[int]], plan: dict[int, Assignment],
                    updated_id: int | None, limit: int = 4) -> list[tuple[int, int]]:
    """Deterministic ALNS-style destroy neighborhoods from the pressure graph."""
    pairs = set()
    for i, neighbors in graph.items():
        for j in neighbors:
            pairs.add(tuple(sorted((i, j))))
    ranked = sorted(pairs, key=lambda p: (
        updated_id not in p,
        abs(plan[p[0]].start - plan[p[1]].start),
        p,
    ))
    return ranked[:max(0, limit)]


def repair_subset(planner: SlotPlanner, published: dict[int, Assignment],
                  subset: tuple[int, ...], now: float, etas: dict[int, float],
                  hint: Hint | None = None,
                  budget_remaining: int = 2,
                  try_reduced_cranes: bool = True) -> Iterator[tuple[dict[int, Assignment], tuple[int, ...]]]:
    """Destroy a tiny subset, repair it while freezing the complement.

    Feasibility is evaluated at every insertion, and the final proposal must
    differ at no more than the remaining budget's ship-level commitments.
    Candidate enumeration is bounded and deterministic; no MILP dependency.
    """
    if not subset or budget_remaining < 1:
        return
    fixed = [a for j, a in published.items() if j not in subset]
    q_options: dict[int, list[int]] = {}
    for i in subset:
        qmax = planner._q(planner.vessels[i])
        keep = published[i].cranes
        q_options[i] = list(dict.fromkeys(
            [keep, qmax, max(1, qmax // 2)] if try_reduced_cranes else [keep, qmax]
        ))
    for order in permutations(subset):
        for choices in product(*(q_options[i] for i in order)):
            placed = list(fixed)
            candidate = dict(published)
            try:
                for i, cranes in zip(order, choices):
                    a = planner.assign(i, max(now, etas[i]), placed,
                                       hint.position.get(i) if hint else None,
                                       cranes=cranes)
                    candidate[i] = a
                    placed.append(a)
            except (RuntimeError, ValueError):
                continue
            changed = tuple(i for i in sorted(subset)
                            if not assignments_equal(candidate[i], published[i]))
            if not changed or len(changed) > budget_remaining:
                continue
            validate_plan(candidate, planner.vessels, planner.port)
            yield candidate, changed
