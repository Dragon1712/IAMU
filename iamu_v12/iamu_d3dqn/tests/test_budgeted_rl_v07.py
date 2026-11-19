"""V0.7: absolute budget, public-ETA plan externalities, optional action guard."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iamu_dynamic import Vessel, Port, Scenario, Update, demo_case, generate_scenario
from iamu_dynamic.rl_env import (IAMUEventEnv, GLOBAL_DIM, ACTION_DIM,
                                 POINT_GAIN_INDEX, BUDGET_FEATURE_SCALE)
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_experiment import run_episode


class RevisedBudgetedRLTests(unittest.TestCase):
    def setUp(self):
        ships, port = demo_case()
        self.scenario = generate_scenario(ships, port, 'shock', seed=42,
                                          updates_per_vessel=2)

    def test_absolute_budget_distinguished_even_with_same_ratio(self):
        a=IAMUEventEnv(self.scenario,budget=2)
        b=IAMUEventEnv(self.scenario,budget=10)
        x,y=a._observation(),b._observation()
        self.assertEqual(x.global_features.shape,(GLOBAL_DIM,))
        self.assertEqual(x.action_features.shape,(5,ACTION_DIM))
        self.assertEqual(x.global_features[1],y.global_features[1]) # old ratio = 1
        self.assertNotEqual(x.global_features[19],y.global_features[19])
        self.assertNotEqual(x.global_features[20],y.global_features[20])
        self.assertAlmostEqual(x.global_features[19],.2)
        self.assertAlmostEqual(y.global_features[19],1.)
        a.step(0);b.step(0)
        self.assertEqual(a._observation().global_features[19],.2)
        self.assertEqual(b._observation().global_features[19],1.)

    def test_architecture_matched_ablation_switches_zero_only_new_features(self):
        new=IAMUEventEnv(self.scenario,2,budget_features='absolute',
                         value_features='global')
        old=IAMUEventEnv(self.scenario,2,budget_features='legacy',
                         value_features='legacy')
        for e in (new,old):e.step(0)
        x,y=new._observation(),old._observation()
        np.testing.assert_array_equal(x.global_features[:19],y.global_features[:19])
        np.testing.assert_array_equal(x.action_features[:,:17],y.action_features[:,:17])
        self.assertTrue(np.all(y.global_features[19:]==0))
        self.assertTrue(np.all(y.action_features[:,17:]==0))
        self.assertTrue(np.any(x.global_features[19:]!=0))
        self.assertTrue(np.any(x.action_features[:,17:]!=0))
        np.testing.assert_array_equal(x.mask,y.mask)

    def test_invalid_feature_and_screen_modes_rejected(self):
        for kw in ({'candidate_screen':'oracle'},
                   {'budget_features':'oracle'},
                   {'value_features':'oracle'}):
            with self.assertRaises(ValueError):IAMUEventEnv(self.scenario,2,**kw)

    def test_point_gain_exactly_matches_current_public_eta_forecast_score(self):
        env=IAMUEventEnv(self.scenario,budget=5,candidate_limit=4)
        env.step(0)
        self.assertTrue(env._candidates)
        before=env.runtime.score(env.plan,env.published_eta)
        for j,plan in enumerate(env._candidates):
            gain=before-env.runtime.score(plan,env.published_eta)
            self.assertAlmostEqual(env._candidate_data[j]['gain'],gain,places=6)
            self.assertAlmostEqual(env._observation().action_features[j+1,POINT_GAIN_INDEX],
                                   gain/env.reward_scale,places=6)
            self.assertGreaterEqual(env._candidate_data[j]['external_worse_count'],0)
            self.assertTrue(np.isfinite(env._observation().action_features[j+1]).all())

    def test_optional_screen_removes_nonpositive_candidates(self):
        raw=IAMUEventEnv(self.scenario,budget=5,candidate_limit=4,
                         candidate_screen='none',decision_clock='eta')
        guarded=IAMUEventEnv(self.scenario,budget=5,candidate_limit=4,
                             candidate_screen='positive',decision_clock='eta')
        for _ in range(4):
            raw.step(0);guarded.step(0)
            self.assertTrue(set(guarded._candidate_ids).issubset(raw._candidate_ids))
            for d in guarded._candidate_data:
                self.assertGreater(d['gain'],guarded.min_gain)
            self.assertEqual(raw.point_greedy_action()==0,
                             guarded.point_greedy_action()==0)
        self.assertGreater(guarded.screened_proposals,0)

    def test_no_future_arrival_leakage_in_new_global_and_candidate_features(self):
        v,port=demo_case()
        updates=[Update(1.,0,9.,0),Update(2.,1,12.,1)]
        true1={x.id:100.+x.id for x in v}
        true2={x.id:200.+x.id for x in v}
        e1=IAMUEventEnv(Scenario(v,port,true1,updates,'shock',7),3,
                        candidate_screen='positive')
        e2=IAMUEventEnv(Scenario(v,port,true2,updates,'shock',7),3,
                        candidate_screen='positive')
        for _ in range(3):
            a,b=e1._observation(),e2._observation()
            np.testing.assert_array_equal(a.global_features,b.global_features)
            np.testing.assert_array_equal(a.action_features,b.action_features)
            np.testing.assert_array_equal(a.mask,b.mask)
            e1.step(0);e2.step(0)

    def test_keep_is_always_legal_when_screening(self):
        e=IAMUEventEnv(self.scenario,5,candidate_screen='positive')
        ob=e.reset()
        self.assertTrue(ob.mask[0]);self.assertEqual(sum(ob.mask),1)
        while True:
            ob,reward,done,info=e.step(0)
            self.assertTrue(ob.mask[0])
            if done:break
        self.assertEqual(e.used,0)
        self.assertEqual(info['weighted_wait'],-e.total_raw_reward)

    def test_budget_hard_limit_with_guarded_training_actions(self):
        e=IAMUEventEnv(self.scenario,2,candidate_screen='positive')
        agent=DoubleDQNAgent(seed=10,hidden_dim=16,keep_margin=.2)
        ob=e.reset();rewards=0.
        while True:
            teacher=e.point_greedy_action()
            a=agent.act(ob,epsilon=1.,teacher_action=teacher,teacher_prob=1.)
            self.assertEqual(a,teacher)
            ob,reward,done,info=e.step(a)
            rewards+=reward
            if done:break
        self.assertLessEqual(e.used,2)
        self.assertAlmostEqual(-rewards*100,info['weighted_wait'],places=5)

    def test_teacher_exploration_is_only_during_exploration(self):
        e=IAMUEventEnv(self.scenario,5)
        e.step(0)
        ob=e._observation()
        agent=DoubleDQNAgent(seed=18,hidden_dim=16)
        self.assertEqual(agent.act(ob,epsilon=1,teacher_action=0,teacher_prob=1),0)
        self.assertEqual(agent.act(ob,epsilon=0,teacher_action=100,teacher_prob=1),
                         agent.act(ob,epsilon=0))
        with self.assertRaises(ValueError):agent.act(ob,epsilon=.4,teacher_prob=1.1)

    def test_keep_margin_cannot_allow_illegal_actions(self):
        e=IAMUEventEnv(self.scenario,0)
        obs=e.reset()
        for margin in (0.,.5,2.):
            agent=DoubleDQNAgent(seed=33,hidden_dim=16,keep_margin=margin)
            self.assertEqual(agent.act(obs),0)
            self.assertEqual(agent.act(obs,epsilon=1),0)

    def test_keep_margin_checkpoint_roundtrip(self):
        agent=DoubleDQNAgent(seed=7,hidden_dim=16,keep_margin=.15)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt'
            torch.save(agent.checkpoint({'candidate_screen':'positive'}),p)
            restored,meta=DoubleDQNAgent.from_checkpoint(str(p))
            self.assertAlmostEqual(restored.keep_margin,.15)
            self.assertEqual(meta['candidate_screen'],'positive')
            with torch.no_grad():
                for x,y in zip(agent.online.parameters(),restored.online.parameters()):
                    torch.testing.assert_close(x,y)

    def test_guarded_action_trace_and_episode_metrics(self):
        x,trace=run_episode(self.scenario,2,'greedy',candidate_screen='positive',
                            return_audit=True)
        self.assertEqual(x['spent'],sum(z['action']!=0 for z in trace))
        self.assertTrue(all(z['estimated_gain']>.1 for z in trace if z['action']!=0))
        self.assertGreaterEqual(x['screened_proposals'],0)
        self.assertTrue(x['wait']>=0)

    def test_same_greedy_under_public_eta_positive_screen(self):
        a=run_episode(self.scenario,5,'greedy',candidate_screen='none')
        b=run_episode(self.scenario,5,'greedy',candidate_screen='positive')
        self.assertAlmostEqual(a['wait'],b['wait'],places=6)
        self.assertEqual(a['spent'],b['spent'])

if __name__=='__main__': unittest.main()
