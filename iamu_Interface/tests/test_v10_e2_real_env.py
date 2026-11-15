"""Server-only real IAMU integration: snapshot/physical budget/non-oracle E2.

Requires `python setup_interface.py --source ../iamu_d3dqn` first.
"""
from __future__ import annotations
import unittest

from iamu_dynamic.core import Vessel, Port, generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.v10_budget_lookahead import LookaheadSpecification, evaluate_multistage
from iamu_dynamic.v10_policy import public_arrival_flags
from iamu_dynamic.v10_risk_features import RiskSpecification,public_eta_scenarios


class E2RealEnvironmentTests(unittest.TestCase):
    def test_budget_branch_and_no_hidden_future_events(self):
        vessels=[Vessel(0,5.,20.,100.,2),Vessel(1,9.,30.,110.,2),
                 Vessel(2,16.,25.,120.,2)]
        scenario=generate_scenario(vessels,Port(cranes=4,quay_length=600.),
                                   label='shock',seed=117,updates_per_vessel=2)
        env=IAMUEventEnv(scenario,budget=2,candidate_limit=3,
                         decision_clock='union',candidate_screen='none')
        env.reset()
        for _ in range(60):
            if env._candidates:
                break
            _,_,done,_=env.step(0)
            if done:
                self.skipTest('No eligible Edit in synthetic scenario')
        candidates={i+1:v for i,v in enumerate(env._candidates)}
        samples=public_eta_scenarios(env.published_eta,env.event_time,
            public_arrival_flags(env),RiskSpecification(sigma=4.,samples=3),seed=91)
        spec=LookaheadSpecification(stages=2,step_hours=12.)
        old=evaluate_multistage(env,candidates,samples,spec)
        self.assertEqual(old.scenario_count,3)
        self.assertEqual(set(old.expected_cost),set([0,*candidates]))
        self.assertTrue(all(v>=0. for v in old.expected_cost.values()))
        # The simulated future cannot see real test-episode updates/truths.
        original_true=dict(env.scenario.true_etas)
        original_updates=list(env.scenario.updates)
        try:
            env.scenario.true_etas={i:x+1000. for i,x in original_true.items()}
            env.scenario.updates=[]
            new=evaluate_multistage(env,candidates,samples,spec)
            self.assertEqual(old.expected_cost,new.expected_cost)
            self.assertEqual(old.scenario_costs,new.scenario_costs)
        finally:
            env.scenario.true_etas=original_true
            env.scenario.updates=original_updates


if __name__=='__main__':
    unittest.main()
