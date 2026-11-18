"""Server-only tests: run AFTER build_variants.py against actual IAMU physics."""
import unittest
import numpy as np
try:
    from iamu_dynamic import generate_scenario,demo_case
    from iamu_dynamic.rl_env import IAMUEventEnv as OldEnv
    from iamu_dynamic.rl_env_v11 import IAMUEventEnv as NewEnv
    from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
    HAVE=True
except ImportError:
    HAVE=False

@unittest.skipUnless(HAVE,'Install in IAMU repository and run builder first')
class TestRealEnv(unittest.TestCase):
    def setUp(self):
        vessels,port=demo_case()
        self.future=generate_scenario(vessels,port,label='shock',seed=887,
                    updates_per_vessel=2,uncertainty=8.)
        self.kw=dict(budget=2,candidate_limit=4,decision_clock='union',period=12.,
            reward_scale=100.,min_gain=.1,candidate_screen='positive',
            budget_features='absolute',value_features='global')
    def test_keep_trajectory_physical_wait_is_unchanged(self):
        waits=[]
        for cls in (OldEnv,NewEnv):
            env=cls(self.future,**self.kw);obs=env.reset();rewards=[]
            while True:
                obs,r,done,info=env.step(0)
                rewards.append(r)
                if done:break
            self.assertAlmostEqual(-sum(rewards)*100.,info['weighted_wait'],places=4)
            waits.append(info['weighted_wait'])
        self.assertAlmostEqual(waits[0],waits[1],places=5)
    def test_proposal_plan_parity_with_core(self):
        env=NewEnv(self.future,**self.kw);obs=env.reset()
        while env.event_type=='initial':obs,_,done,_=env.step(0)
        for vid in env._eligible():
            expected=env.planner.single_move(env.plan,vid,env.event_time,env.published_eta)
            actual,reason=env._proposal_with_reason(vid)
            self.assertEqual(expected,actual)
    def test_legality_heuristic_flags_and_feature_shape(self):
        env=NewEnv(self.future,**self.kw);obs=env.reset()
        while True:
            self.assertEqual(obs.mask.shape,(5,))
            self.assertEqual(obs.action_features.shape,(5,24))
            self.assertTrue(np.array_equal(obs.mask,obs.physical_mask))
            self.assertTrue(np.all(~obs.heuristic_mask | obs.physical_mask))
            self.assertEqual(obs.mask.sum()-1,obs.decision_flags['returned_candidates'])
            self.assertGreaterEqual(obs.decision_flags['pending_ships'],obs.decision_flags['ships_examined'])
            obs,r,done,info=env.step(0)
            if done:break
    def test_mixed_does_not_expose_illegal_actions(self):
        env=NewEnv(self.future,**dict(self.kw,candidate_screen='mixed'))
        obs=env.reset()
        while True:
            self.assertTrue(np.array_equal(obs.mask,obs.physical_mask))
            self.assertTrue(np.all(~obs.heuristic_mask | obs.physical_mask))
            obs,r,done,info=env.step(0)
            if done:break
    def test_old_checkpoint_margin_compat(self):
        agent=DoubleDQNAgent(hidden_dim=32,keep_margin=.1,q_network='set_ddqn',device='cpu')
        self.assertEqual(agent.execution_margin,.1)
        self.assertEqual(agent.bootstrap_margin,.1)
        agent=DoubleDQNAgent(hidden_dim=32,execution_margin=.1,bootstrap_margin=0.,q_network='set_ddqn')
        self.assertEqual(agent.execution_margin,.1)
        self.assertEqual(agent.bootstrap_margin,0.)

if __name__=='__main__':unittest.main()
