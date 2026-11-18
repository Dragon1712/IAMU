"""Run on the real IAMU simulator after applying the patch."""
import unittest
try:
    from iamu_dynamic import demo_case,generate_scenario
    from iamu_dynamic.rl_env_v11 import IAMUEventEnv
    from v11_protocol.counterfactual_decision_probe import paired_realized_advantage
    HAVE=True
except ImportError:
    HAVE=False

@unittest.skipUnless(HAVE,'Requires real IAMU environment')
class TestRealEnvV12(unittest.TestCase):
    def setUp(self):
        vessels,port=demo_case()
        self.scenario=generate_scenario(vessels,port,label='repeated',seed=113,
                                        updates_per_vessel=2,uncertainty=8.)
    def test_no_search_event_not_marked_exhausted(self):
        for budget in (0,2):
            env=IAMUEventEnv(self.scenario,budget=budget,candidate_limit=4,
                             decision_clock='union',candidate_screen='positive')
            obs=env.reset()
            self.assertTrue(obs.decision_flags['initial_event'])
            self.assertFalse(obs.decision_flags['search_performed'])
            self.assertFalse(obs.decision_flags['search_exhausted'])
            if budget==0:
                obs,_,done,_=env.step(0)
                self.assertFalse(obs.decision_flags['search_performed'])
    def test_paired_branches_unchanged_source_and_budget(self):
        env=IAMUEventEnv(self.scenario,budget=2,candidate_limit=4,
                         decision_clock='union',candidate_screen='positive')
        obs=env.reset()
        for _ in range(len(env.events)+1):
            if bool(obs.mask[1:].any()):
                prior=(env.used,len(env.history))
                result=paired_realized_advantage(env,1,'greedy')
                self.assertEqual((env.used,len(env.history)),prior)
                self.assertTrue(result['keep_final_spent']<=2)
                self.assertTrue(result['edit_final_spent']<=2)
                return
            obs,_,done,_=env.step(0)
            if done:break
        self.skipTest('No actionable observations in small deterministic fixture')

if __name__=='__main__': unittest.main()
