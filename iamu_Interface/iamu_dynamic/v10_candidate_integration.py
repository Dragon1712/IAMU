"""Optional ERL-DDE hint candidates in V0.9's single-ship action space.

Experiment-only injection into newly constructed local IAMUEventEnv objects.
Never patch running worker processes, trainer files or original environments.
One candidate still changes one vessel and costs exactly one budget unit.
"""
from __future__ import annotations
from .v10_candidate_api import SingleMoveProvider, FrozenERLDDEHintProvider


def install_candidate_source(env, source) -> None:
    """Replace cached candidates before _observation(), never during step().

    The 'single' source validates but does not change the V0.9 cached actions.
    """
    if source is None:
        return
    if isinstance(source, SingleMoveProvider):
        source.propose(env)  # parity and feasibility checks; no mutations
        return
    proposals = source.propose(env)
    if len(proposals) > env.candidate_limit:
        raise AssertionError('Provider exceeded candidate limit')
    if any(p.edit_cost != 1 or len(p.changed_ids) != 1 for p in proposals):
        raise ValueError('V1.0 environment supports single-vessel proposals only')
    if not proposals:
        env._candidates, env._candidate_ids, env._candidate_data = [], (), []
        return
    base_starts, base_costs, base_score = env._forecast_projection(env.plan)
    new_candidates, new_ids, new_details = [], [], []
    for proposal in proposals:
        plan = dict(proposal.assignments)
        i = proposal.changed_ids[0]
        starts, costs, new_score = env._forecast_projection(plan)
        affected = [j for j in starts if j != i]
        worsened = [j for j in affected if starts[j] > base_starts[j] + 1e-8]
        d = dict(
            gain=base_score-new_score, base_score=base_score,
            new_score=new_score,
            external_worse_count=len(worsened),
            external_max_delay=max((starts[j]-base_starts[j] for j in worsened),default=0.0),
            external_worse_cost=sum(max(0.,costs[j]-base_costs[j]) for j in affected),
            external_relief_cost=sum(max(0.,base_costs[j]-costs[j]) for j in affected),
        )
        new_candidates.append(plan)
        new_ids.append(i)
        new_details.append(d)
    env._candidates = new_candidates
    env._candidate_ids = tuple(new_ids)
    env._candidate_data = new_details
