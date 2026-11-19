"""V0.6: causal action state, hard edit budget, replay and baseline parity."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from iamu_dynamic import (Vessel,Port,Update,Scenario,demo_case,
                          generate_scenario,simulate)
from iamu_dynamic.rl_env import IAMUEventEnv,GLOBAL_DIM,ACTION_DIM
from iamu_dynamic.rl_ddqn import DoubleDQNAgent,ReplayBuffer
from iamu_dynamic.rl_experiment import run_episode,verify_cases


class BudgetedRLEnvTests(unittest.TestCase):
    def scenario(self):
        vessels,port=demo_case()
        return generate_scenario(vessels,port,'shock',seed=44,updates_per_vessel=2)

    def test_budget_zero_all_actions_masked_except_keep(self):
        e=IAMUEventEnv(self.scenario(),budget=0,candidate_limit=4)
        o=e.reset()
        while True:
            self.assertEqual(o.mask.tolist(),[True,False,False,False,False])
            o,reward,done,info=e.step(0)
            if done:break
        self.assertEqual(e.used,0)

    def test_illegal_action_is_rejected_and_state_unchanged(self):
        e=IAMUEventEnv(self.scenario(),budget=2,candidate_limit=4)
        o=e.reset()
        self.assertFalse(o.mask[1])
        with self.assertRaises(ValueError):e.step(1)
        self.assertEqual(e.event_time,0.)
        self.assertEqual(e.used,0)

    def test_observation_does_not_reveal_future_true_arrivals(self):
        ships=[Vessel(0,10.,40.,200.,2),Vessel(1,20.,40.,200.,2)]
        port=Port(cranes=4,max_cranes_per_ship=2)
        updates=[Update(1.,0,14.,0),Update(2.,1,19.,1)]
        a=Scenario(ships,port,{0:20.,1:29.},updates,'synthetic',99)
        b=Scenario(ships,port,{0:35.,1:49.},updates,'synthetic',99)
        env1=IAMUEventEnv(a,2,candidate_limit=2)
        env2=IAMUEventEnv(b,2,candidate_limit=2)
        for _ in range(3):
            o1=env1._observation();o2=env2._observation()
            np.testing.assert_array_equal(o1.global_features,o2.global_features)
            np.testing.assert_array_equal(o1.action_features,o2.action_features)
            np.testing.assert_array_equal(o1.mask,o2.mask)
            if o1.event_type=='eta' and o1.time>=2:break
            env1.step(0);env2.step(0)

    def test_full_reward_ledger_equals_final_wait(self):
        for scenario_name in ('shock','repeated','small'):
            vessels,port=demo_case()
            sc=generate_scenario(vessels,port,scenario_name,seed=44,updates_per_vessel=2)
            env=IAMUEventEnv(sc,budget=3,candidate_limit=4,reward_scale=100.)
            obs=env.reset();r_sum=0.
            while True:
                act=env.point_greedy_action()
                obs,r,done,info=env.step(act)
                r_sum+=r
                if done:break
            self.assertAlmostEqual(-100*r_sum,info['weighted_wait'],places=5)
            self.assertEqual(env.used,len([x for x in env.history if x[3] is not None]))
            self.assertLessEqual(env.used,3)

    def test_matches_original_greedy_and_periodic_at_one_edit_per_event(self):
        vessels,port=demo_case()
        for typ in ('shock','repeated'):
            sc=generate_scenario(vessels,port,typ,seed=44,updates_per_vessel=2)
            for B in (0,2,5):
                for method,policy,clock in (
                    ('greedy','budget_aware','eta'),
                    ('periodic','periodic','union'),
                    ('keep','no_replan','eta')):
                    ref=simulate(sc,policy,B,max_edits_per_event=1,candidate_limit=4)
                    ours=run_episode(sc,B,method,candidate_limit=4,decision_clock=clock)
                    self.assertAlmostEqual(ref['metrics']['weighted_wait'],ours['wait'],places=5)
                    self.assertEqual(ref['metrics']['spent'],ours['spent'])

    def test_true_nonarrived_known_after_arrival_only(self):
        ships=[Vessel(0,5.,20.,200.,1),Vessel(1,9.,20.,200.,1)]
        port=Port(cranes=2,max_cranes_per_ship=1)
        sc=Scenario(ships,port,{0:2.,1:15.},[Update(3.,1,16.,1)],'synthetic',2)
        e=IAMUEventEnv(sc,budget=2,candidate_limit=2)
        self.assertEqual(e._observation().global_features[4],0.)
        e.step(0)
        self.assertGreater(e._observation().global_features[4],0.)

    def test_ddqn_mask_and_update(self):
        sc=self.scenario();env=IAMUEventEnv(sc,budget=2,candidate_limit=2)
        buffer=ReplayBuffer(100,seed=7);agent=DoubleDQNAgent(seed=8,hidden_dim=32,
                                target_update_steps=2)
        o=env.reset()
        self.assertEqual(o.global_features.shape,(GLOBAL_DIM,))
        self.assertEqual(o.action_features.shape,(3,ACTION_DIM))
        while True:
            action=agent.act(o,epsilon=1.)
            self.assertTrue(o.mask[action])
            nxt,r,done,info=env.step(action)
            buffer.append(o,action,r,nxt,done)
            o=nxt
            if done:break
        self.assertGreater(len(buffer),2)
        for _ in range(3):
            loss=agent.update(buffer.sample(2))
            self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.gradient_steps,3)
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'check.pt')
            torch.save(agent.checkpoint({'seed':8}),path)
            restored,meta=DoubleDQNAgent.from_checkpoint(path)
            self.assertEqual(meta['seed'],8)
            x=env.reset()
            self.assertEqual(agent.act(x),restored.act(x))

    def test_duplicate_split_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'case0.txt';b=Path(d)/'same_other_name.txt'
            a.write_text('1\t20\t200\t1\t2\n')
            b.write_bytes(a.read_bytes())
            with self.assertRaisesRegex(ValueError,'disjoint'):
                verify_cases([str(a)],[str(b)])

if __name__=='__main__':unittest.main()
