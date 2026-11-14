"""Read-only V1.0 candidate boundary for SingleMove and frozen ERL-DDE hints.

Full-plan proposals are validated against current *published* commitments;
actual starts need not equal published starts.  This module never commits edits.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence
from .core import Assignment, assignments_equal, validate_plan


@dataclass(frozen=True)
class PlanProposal:
    proposal_id: str
    source: str
    assignments: Mapping[int, Assignment]
    changed_ids: tuple[int, ...]
    edit_cost: int
    metadata: dict = field(default_factory=dict)


class CandidateProvider(Protocol):
    def propose(self, env) -> Sequence[PlanProposal]: ...


def compute_changes(current: Mapping[int, Assignment],
                    alternative: Mapping[int, Assignment]) -> tuple[int, ...]:
    if set(current) != set(alternative):
        raise ValueError("Candidate must assign exactly the same vessels")
    return tuple(sorted(i for i in current
                        if not assignments_equal(current[i], alternative[i])))


def build_proposal(current, alternative, source: str, proposal_id: str,
                   metadata: dict | None = None) -> PlanProposal:
    changed = compute_changes(current, alternative)
    return PlanProposal(proposal_id=proposal_id, source=source,
                        assignments=dict(alternative), changed_ids=changed,
                        edit_cost=len(changed), metadata=dict(metadata or {}))


def validate_proposal(proposal: PlanProposal, current, *,
                      planner, port, started_ids, now: float,
                      remaining_budget: int, published_eta=None,
                      arrived_ids=frozenset()) -> None:
    """Fail closed; never trust changed_ids/edit_cost supplied by an optimizer."""
    plan = proposal.assignments
    if any(int(i) != int(a.vessel_id) for i,a in plan.items()):
        raise ValueError("Assignment vessel_id must match plan dictionary key")
    actual_changes = compute_changes(current, plan)
    if not actual_changes:
        raise ValueError("No-op plan is Keep, not an Edit candidate")
    if actual_changes != proposal.changed_ids or len(actual_changes) != proposal.edit_cost:
        raise ValueError("Incorrect changed_ids or edit_cost")
    if proposal.edit_cost > remaining_budget:
        raise ValueError("Insufficient hard revision budget")
    if set(actual_changes).intersection(started_ids):
        raise ValueError("Cannot revise a vessel whose service already started")
    for i in actual_changes:
        assignment = plan[i]
        if assignment.start + 1e-8 < now:
            raise ValueError("Revised plan starts in the past")
        if published_eta is not None and i not in arrived_ids:
            if assignment.start + 1e-8 < published_eta[i]:
                raise ValueError("Revised plan precedes published ETA")
    validate_plan(dict(plan), planner.vessels, port)


class SingleMoveProvider:
    """Wrap *cached* environment actions, guaranteeing candidate parity V0.9."""
    def propose(self, env) -> list[PlanProposal]:
        items = []
        for index, (vessel, plan) in enumerate(
                zip(env._candidate_ids, env._candidates), 1):
            prop = build_proposal(
                env.plan, plan, "single_move", f"single-{vessel}",
                metadata={"action_index": index, "vessel_id": int(vessel)})
            validate_proposal(prop, env.plan, planner=env.planner,
                              port=env.scenario.port,
                              started_ids=set(env.runtime.started),
                              now=env.event_time,
                              remaining_budget=env.remaining_budget,
                              published_eta=env.published_eta)
            items.append(prop)
        return items


class FrozenERLDDEHintProvider:
    """Hint-only adapter; does NOT claim to be a full constrained ERL-DDE solver.

    The hint provider has interface FrozenERLDDEHints(vessels, port, etas).
    Plan feasibility and budget are checked after generation.
    """
    def __init__(self, hint_provider):
        self.hint_provider = hint_provider

    def propose(self, env) -> list[PlanProposal]:
        if env.done or env.remaining_budget <= 0 or env.event_type == "initial":
            return []
        if env.decision_clock == "eta" and env.event_type != "eta":
            return []
        if env.decision_clock == "clock" and env.event_type != "clock":
            return []
        hint = self.hint_provider(env.scenario.vessels, env.scenario.port,
                                  dict(env.published_eta))
        result = []
        baseline = None
        for i in env._eligible():
            new = env.planner.single_move(env.plan, i, env.event_time,
                                          env.published_eta, hint=hint)
            if new is None:
                continue
            if env.candidate_screen == "positive":
                if baseline is None:
                    baseline = env.runtime.score(env.plan, env.published_eta)
                gain = baseline - env.runtime.score(new, env.published_eta)
                if gain <= env.min_gain + 1e-12:
                    continue
            proposal = build_proposal(env.plan, new, "frozen_erldde_hint",
                                      f"erldde-hint-{i}",
                                      metadata={"vessel_id": int(i)})
            validate_proposal(proposal, env.plan, planner=env.planner,
                              port=env.scenario.port,
                              started_ids=set(env.runtime.started),
                              now=env.event_time,
                              remaining_budget=env.remaining_budget,
                              published_eta=env.published_eta)
            result.append(proposal)
        return result
