import unittest
from types import SimpleNamespace
import numpy as np
import torch
from v13_diagnostics.audit_targets import inspect_batch


class FakeNet(torch.nn.Module):
    def __init__(self,offset):super().__init__();self.offset=offset
    def forward(self,g,a,m):
        return torch.stack([g[:,0]+self.offset,g[:,0]+2+self.offset,
                            g[:,0]-1+self.offset],dim=1)


class TargetTests(unittest.TestCase):
    def test_manual_bellman_decomposition(self):
        agent=SimpleNamespace(online=FakeNet(0.),target=FakeNet(10.),bootstrap_margin=0.)
        t=SimpleNamespace(obs_global=np.zeros(24,dtype=np.float32),
            next_global=np.full(24,2,dtype=np.float32),
            obs_actions=np.zeros((3,24),dtype=np.float32),
            next_actions=np.zeros((3,24),dtype=np.float32),
            obs_mask=np.array([1,1,0],bool),next_mask=np.array([1,1,0],bool),
            action=1,reward=-7.94,bootstrap_discount=.729,episode_id=9,
            raw_start_index=2,raw_end_index=4,source_budget=1,next_budget=1,
            n_raw_steps=3,elapsed_hours=3.,done=False)
        z=inspect_batch(agent,[t])[0]
        self.assertEqual(z['next_selected_action'],1)
        self.assertAlmostEqual(z['next_target_value'],14)
        self.assertAlmostEqual(z['q_taken'],2)
        self.assertAlmostEqual(z['bootstrap'],.729*14,places=5)
        self.assertAlmostEqual(z['td_target'],-7.94+.729*14,places=5)
        self.assertAlmostEqual(z['td_error'],-7.94+.729*14-2,places=5)

    def test_terminal_ignores_target_q(self):
        agent=SimpleNamespace(online=FakeNet(0.),target=FakeNet(1000.),bootstrap_margin=0.)
        t=SimpleNamespace(obs_global=np.zeros(24,dtype=np.float32),
            next_global=np.full(24,2,dtype=np.float32),
            obs_actions=np.zeros((3,24),dtype=np.float32),
            next_actions=np.zeros((3,24),dtype=np.float32),
            obs_mask=np.array([1,1,0],bool),next_mask=np.array([1,0,0],bool),
            action=1,reward=-200.,bootstrap_discount=0.,episode_id=9,
            raw_start_index=2,raw_end_index=40,source_budget=1,next_budget=0,
            n_raw_steps=39,elapsed_hours=30.,done=True)
        z=inspect_batch(agent,[t])[0]
        self.assertEqual(z['bootstrap'],0.)
        self.assertEqual(z['td_target'],-200.)

if __name__=='__main__':unittest.main()
