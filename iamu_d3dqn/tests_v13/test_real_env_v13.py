"""One physical-engine snapshot replay smoke; skipped only if IAMU V1.1 absent."""
import copy
import math
import unittest


class RealEngineTests(unittest.TestCase):
    def test_real_engine_snapshot_replay_and_common_future(self):
        try:
            from iamu_dynamic.core import demo_case,generate_scenario
            from iamu_dynamic.rl_env_v11 import IAMUEventEnv
        except ImportError as exc:
            self.skipTest(f'Full V1.1 physical engine not installed here: {exc}')
        from v13_diagnostics.future_tapes import draw_tape,apply_tape
        vessels,port=demo_case()
        sc=generate_scenario(vessels,port,label='shock',seed=231,
                             updates_per_vessel=2,uncertainty=8.)
        env=IAMUEventEnv(sc,budget=2,candidate_limit=4,decision_clock='union',
                         period=12.,reward_scale=100.,candidate_screen='positive',
                         min_gain=.1,budget_features='absolute',value_features='global')
        obs=env.reset();saved=None;actions=[];rewards=[]
        while True:
            if saved is None and obs.mask[1:].any():
                saved=copy.deepcopy(env)
            action=0  # simple, deterministic original replay
            if saved is not None:
                actions.append(action)
            obs,r,done,_=env.step(action)
            if saved is not None:rewards.append(r)
            if done:break
        if saved is None:
            self.skipTest('Chosen scenario produced no actionable state')
        replay=copy.deepcopy(saved)
        for i,action in enumerate(actions):
            obs,r,done,_=replay.step(action)
            self.assertAlmostEqual(r,rewards[i],places=8)
            self.assertEqual(done,i==len(actions)-1)
        tape=draw_tape(saved,'real:0',0,updates_per_vessel=2)
        a=apply_tape(saved,tape);b=apply_tape(saved,tape)
        self.assertEqual(a.events,b.events)
        self.assertEqual(a.scenario.true_etas,b.scenario.true_etas)
        # The public source observation is unchanged by synthetic future retaping.
        self.assertTrue((a._observation().global_features==
                         saved._observation().global_features).all())
        obs_a,r_a,_,_=a.step(0)
        obs_b,r_b,_,_=b.step(0)
        self.assertAlmostEqual(r_a,r_b,places=8)

if __name__=='__main__':unittest.main()
