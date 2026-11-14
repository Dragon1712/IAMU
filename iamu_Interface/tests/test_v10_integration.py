"""Real simulator parity and non-anticipation checks; run on IAMU server."""
from __future__ import annotations
import unittest
from iamu_dynamic.core import Vessel,Port,generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.v10_candidate_api import SingleMoveProvider
from iamu_dynamic.v10_policy import public_arrival_flags
from iamu_dynamic.v10_risk_features import RiskSpecification,evaluate_proposals


class V10RealEnvironmentTests(unittest.TestCase):
    def setUp(self):
        vessels=[Vessel(0,5.,20.,100.,2),Vessel(1,9.,30.,110.,2),Vessel(2,16.,25.,120.,2)]
        port=Port(cranes=4,quay_length=600.)
        self.scenario=generate_scenario(vessels,port,label='shock',seed=17,updates_per_vessel=2)

    def test_single_source_matches_cached_environment(self):
        env=IAMUEventEnv(self.scenario,budget=2,candidate_limit=3,
                         decision_clock='union',candidate_screen='positive')
        obs=env.reset()
        for _ in range(35):
            source=SingleMoveProvider()
            proposals=source.propose(env)
            self.assertEqual(len(proposals),len(env._candidates))
            self.assertEqual(tuple(x.changed_ids[0] for x in proposals),env._candidate_ids)
            obs,reward,done,info=env.step(0)
            if done:
                break

    def test_future_eta_changes_not_used_in_risk(self):
        env=IAMUEventEnv(self.scenario,budget=2,candidate_limit=3,
                         decision_clock='union',candidate_screen='none')
        env.reset()
        for _ in range(25):
            if env._candidates:
                break
            _,_,done,_=env.step(0)
            if done:
                self.skipTest('No candidate')
        ids=public_arrival_flags(env)
        plans={i+1:p for i,p in enumerate(env._candidates)}
        spec=RiskSpecification(samples=8,sigma=3.)
        old,_,_=evaluate_proposals(env.runtime,env.plan,plans,env.published_eta,
                                    env.event_time,ids,spec,seed=44)
        # Never change physical history: only the latent future arrival of a
        # vessel that has NOT yet arrived at this public decision time.
        not_arrived=[i for i in env.ids if i not in ids]
        if not not_arrived:
            self.skipTest('All ships arrived')
        i=not_arrived[0]
        env.scenario.true_etas[i]=max(env.event_time+100.,env.scenario.true_etas[i]+100.)
        new,_,_=evaluate_proposals(env.runtime,env.plan,plans,env.published_eta,
                                    env.event_time,ids,spec,seed=44)
        self.assertEqual(old,new)

if __name__=='__main__':
    unittest.main()
