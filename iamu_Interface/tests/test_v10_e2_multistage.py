"""V1.0 E2: pure multistage-recourse and Q/MPC composition regression tests.

Run after snapshot installation:
  python -m unittest discover -s tests -p 'test_v10_e2*.py' -v
"""
from __future__ import annotations
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from iamu_dynamic.v10_budget_lookahead import (
    LookaheadSpecification, evaluate_multistage,
    rollout_future, select_lookahead,
)
from iamu_dynamic.v10_policy import (PolicySpecification,
                                      choose_adjusted_action)
from iamu_dynamic.v10_risk_features import (
    RiskSpecification, public_eta_scenarios, summarize_risk)


@dataclass(frozen=True)
class Assignment:
    start: float


class TwoVesselRuntime:
    """Independent-vessel toy physical dispatcher; no resource interactions."""
    def __init__(self):
        self.started = {}
        self.clock = 0.0

    def advance(self, until, plan, actual):
        if until < self.clock:
            raise ValueError('Backwards time')
        for i, assignment in plan.items():
            realized = max(assignment.start, actual[i])
            if realized <= until and i not in self.started:
                self.started[i] = Assignment(realized)
        self.clock = until

    def finish(self, plan, actual):
        for i, a in plan.items():
            if i not in self.started:
                self.started[i] = Assignment(max(a.start, actual[i]))
        return self

    def score(self, plan, forecast):
        return sum(max(0., max(a.start, forecast[i]) - forecast[i])
                   for i, a in plan.items() if i not in self.started)


class TwoVesselPlanner:
    def single_move(self, plan, i, now, forecast):
        new_start = max(now, forecast[i])
        if new_start >= plan[i].start:
            return None
        return {**plan, i: Assignment(new_start)}


def make_env(budget=1):
    return SimpleNamespace(
        plan={0: Assignment(150.), 1: Assignment(80.)},
        runtime=TwoVesselRuntime(),
        planner=TwoVesselPlanner(),
        published_eta={0: 100., 1: 30.},
        event_time=0., remaining_budget=budget,
        vessels={0: SimpleNamespace(weight=1.),
                 1: SimpleNamespace(weight=1.)})


class MultistageTests(unittest.TestCase):
    def test_keep_uses_saved_future_budget(self):
        env = make_env(budget=1)
        samples = [{0: 100., 1: 30.}] * 4
        edits = {1: {0: Assignment(105.), 1: Assignment(80.)}}
        zero = evaluate_multistage(env, edits, samples,
                                   LookaheadSpecification(stages=0))
        one = evaluate_multistage(env, edits, samples,
                                  LookaheadSpecification(stages=1))
        self.assertAlmostEqual(zero.expected_cost[0], 100.)
        self.assertAlmostEqual(zero.expected_cost[1], 55.)
        self.assertAlmostEqual(one.expected_cost[0], 50.)
        self.assertAlmostEqual(one.expected_cost[1], 55.)
        self.assertGreater(zero.advantage(1), 0.)
        self.assertLess(one.advantage(1), 0.)
        choice, _, _ = select_lookahead(env, edits, samples,
                                         LookaheadSpecification(stages=1))
        self.assertEqual(choice, 0)

    def test_two_future_revisions_respect_budget_and_stage(self):
        env = make_env(budget=2)
        values = [{0: 100., 1: 30.}]
        spec = LookaheadSpecification(stages=2, step_hours=12.)
        evaluation = evaluate_multistage(env, {}, values, spec)
        self.assertAlmostEqual(evaluation.expected_cost[0], 0.)
        cost, trace = rollout_future(env, env.plan, values[0], 2, spec,return_trace=True)
        self.assertEqual(len(trace), 2)
        self.assertEqual([x['remaining_budget'] for x in trace], [1, 0])
        self.assertAlmostEqual(cost, 0.)

    def test_mpc_budget_fusion_can_reverse_q_edit(self):
        env = make_env(budget=1)
        estimate = evaluate_multistage(
            env, {1: {0: Assignment(105.), 1: Assignment(80.)}},
            [{0:100.,1:30.}], LookaheadSpecification(stages=1))
        # Frozen Q alone prefers Edit; multistage evaluation correctly prefers Keep.
        q = np.array([0., 0.2])
        mask = np.array([True, True])
        raw, _, _ = choose_adjusted_action(q, mask,margin=.10,reward_scale=100.)
        mixed, scores, _ = choose_adjusted_action(
            q, mask,margin=.10,reward_scale=100.,
            mpc_estimate=estimate,beta_budget=.5)
        self.assertEqual(raw, 1)
        self.assertEqual(mixed, 0)
        self.assertAlmostEqual(scores[1], .5*.2 + .5*(-5./100.))

    def test_hard_budget_zero_blocks_edit(self):
        env = make_env(budget=0)
        estimate = evaluate_multistage(
            env,{1: {0: Assignment(105.), 1: Assignment(80.)}},
            [{0:100.,1:30.}],LookaheadSpecification(stages=2))
        self.assertEqual(tuple(estimate.expected_cost), (0,))
        with self.assertRaises(ValueError):
            rollout_future(env, env.plan, {0:100.,1:30.}, -1,
                           LookaheadSpecification())

    def test_recourse_reads_only_sample_and_public_state(self):
        env = make_env(budget=1)
        # A fake hidden actual arrival vector is attached to env. It MUST NOT
        # influence the simulated recourse, which accepts a separate ETA sample.
        env.scenario = SimpleNamespace(true_etas={0: 900.,1: 900.},updates=[999])
        sample = {0:100.,1:30.}
        one = rollout_future(env, env.plan, sample, 1,LookaheadSpecification(stages=2))
        env.scenario.true_etas = {0:1.,1:2.}
        env.scenario.updates=[-7]
        two = rollout_future(env, env.plan, sample, 1,LookaheadSpecification(stages=2))
        self.assertEqual(one,two)

    def test_risk_draws_nested_across_sample_sizes(self):
        eta = {0:30.,1:60.}
        small = public_eta_scenarios(eta,10.,set(),
            RiskSpecification(sigma=7.,samples=16),seed=41)
        large = public_eta_scenarios(eta,10.,set(),
            RiskSpecification(sigma=7.,samples=128),seed=41)
        self.assertEqual(small,large[:16])
        self.assertNotEqual(small,public_eta_scenarios(
            eta,10.,set(),RiskSpecification(sigma=7.,samples=16),seed=42))

    def test_joint_policy_end_to_end_toy_state(self):
        from unittest.mock import patch
        from iamu_dynamic.v10_policy import V10Policy
        env = make_env(budget=1)
        env.ids = [0,1]
        env.scenario = SimpleNamespace(seed=41,true_etas={0:100.,1:30.})
        env.event_index = 7
        env.event_type = 'eta'
        env._candidates = [{0: Assignment(105.), 1: Assignment(80.)}]
        env.reward_scale = 100.
        env.min_gain = .1
        obs = SimpleNamespace(mask=np.array([True,True]))
        agent = SimpleNamespace(keep_margin=.1)
        spec = PolicySpecification(
            risk=RiskSpecification(sigma=0.,samples=4),
            lookahead=LookaheadSpecification(stages=1),
            beta_risk=.25,beta_budget=.5,mpc_samples=4)
        with patch('iamu_dynamic.v10_policy.frozen_q_values',
                   return_value=np.array([0.,.2])):
            action,audit = V10Policy('q_joint',agent,spec).choose(env,obs)
        # Tail-risk relief can outweigh the MPC budget opportunity penalty.
        self.assertEqual(action,1)
        self.assertEqual(audit['mpc_stages'],1)
        self.assertEqual(audit['mpc_scenarios'],4)
        self.assertAlmostEqual(audit['mpc_gap_scaled'],-.05)
        self.assertEqual(audit['policy'],'q_joint')

    def test_legacy_and_mpc_scoring_distinct(self):
        q = [0., 0.16]
        mask = [True, True]
        legacy, scores, _ = choose_adjusted_action(
            q,mask,margin=.1,reward_scale=100.,legacy=True,
            scarcity=.7,beta_budget=.10)
        self.assertEqual(legacy,0)
        self.assertAlmostEqual(scores[1],.09)
        with self.assertRaises(ValueError):
            PolicySpecification(beta_budget=1.3)


if __name__ == '__main__':
    unittest.main()
