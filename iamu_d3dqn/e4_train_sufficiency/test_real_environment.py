"""Server integration: genuine IAMU physics; no E4 checkpoint required.

Run from iamu_d3dqn root:
  python -m unittest discover -s e4_diagnostics -p 'test_real_environment.py' -v

The package is purposely a small incremental patch; V0.9 simulator source is
already on the server and is not included in the archive.
"""
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from iamu_dynamic.core import Vessel,Port,generate_scenario
    from iamu_dynamic.rl_env import IAMUEventEnv
    from iamu_dynamic.rl_ddqn import DoubleDQNAgent
    from iamu_dynamic.rl_experiment import run_episode
    INSTALLED=True
except ImportError:
    INSTALLED=False

if INSTALLED:
    from diagnose_keep_qgap import (
        Model, run_trajectory, collect_states, analyze_bank, counterfactual)


@unittest.skipUnless(INSTALLED,'Requires original IAMU V0.9 simulator')
class E4RealEnvironmentTests(unittest.TestCase):
    def setUp(self):
        vessels=[Vessel(0,5.,20.,100.,2),Vessel(1,9.,30.,110.,2),
                 Vessel(2,16.,25.,120.,2)]
        port=Port(cranes=4,quay_length=600.)
        self.scenario=generate_scenario(vessels,port,label='shock',seed=117,
                                        updates_per_vessel=2,uncertainty=8.)
        self.kw=dict(candidate_limit=3,decision_clock='union',period=12.,
                     reward_scale=100.,min_gain=.1,
                     candidate_screen='none',budget_features='absolute',
                     value_features='global')
        self.agent=DoubleDQNAgent(q_network='set_ddqn',device='cpu',keep_margin=.1)
        self.model=Model('random_untrained_test',0,Path('.'),Path('.'),'',self.agent,{}, {},0)

    def test_true_environment_trajectory_matches_official_episode(self):
        ours,events=run_trajectory(self.scenario,2,self.model,self.kw,IAMUEventEnv,torch)
        official=run_episode(self.scenario,2,'ddqn',agent=self.agent,
                           candidate_limit=3,decision_clock='union',period=12.,
                           reward_scale=100.,min_gain=.1,
                           candidate_screen='none',budget_features='absolute',
                           value_features='global')
        self.assertAlmostEqual(ours['weighted_wait'],official['wait'],places=6)
        self.assertEqual(ours['spent'],official['spent'])
        self.assertLessEqual(ours['spent'],2)
        self.assertTrue(events)

    def test_real_counterfactual_preserves_public_state_and_budget(self):
        states,eligible=collect_states(self.scenario,2,'keep',self.model,self.kw,
                                      n_states=1,IAMUEventEnv=IAMUEventEnv)
        if not states:
            self.skipTest('Toy physical case has no legal candidate under this seed')
        state=states[0]
        obs=state._observation()
        start_public=dict(state.published_eta)
        old_used=state.used
        key,qs,cf,cal,dec=analyze_bank(state,[self.model],self.model,
                 ['anchor','greedy'],torch=torch,max_edit_actions=3,
                 case_name='toy',scenario_name='shock',eta_seed=117,budget=2,
                 anchor_name='keep')
        self.assertEqual(state.published_eta,start_public)
        self.assertEqual(state.used,old_used)
        self.assertEqual(obs.mask.tolist(),state._observation().mask.tolist())
        self.assertGreaterEqual(len(cal),1)
        self.assertEqual(len(cf),2*(1+len(state._candidate_ids)))
        for row in cf:
            self.assertTrue(row['realized_total_wait']>=0.)
            self.assertLessEqual(row['realized_total_spent'],2)


if __name__=='__main__':unittest.main()
