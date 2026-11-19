"""CPU uninterrupted vs full-state resumed learner, using V1.1 network if installed."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tests_v12.test_transitions_v12 import episode
from v12_protocol.compress import compress_episode
from v12_protocol.replay import EpisodeReplay
try:
    from v12_protocol.agent import V12Agent
    from v12_protocol.train import save_full_state,restore_full_state
except ImportError:
    V12Agent=None

@unittest.skipUnless(V12Agent is not None,'Requires complete V1.1 source for trainer import')
class RecoveryTests(unittest.TestCase):
    def new_agent(self):
        return V12Agent(hidden_dim=32,seed=77,device='cpu',gamma=1.,
            q_network='set_ddqn',representation='primitive',target_update_steps=5,
            execution_margin=0.,bootstrap_margin=0.)

    def test_complete_state_restores_replay_rng_and_optimizer(self):
        a=self.new_agent()
        ra=EpisodeReplay(2,101,'primitive')
        for eid in [1,2]:
            raw=episode([-1.,-2.,-3.],[True,False,True],[2,2,2],eid=eid)
            ra.append_episode(eid,compress_episode(raw,'primitive',1.).transitions,3)
        b,sm=ra.sample(2)
        a.update(b)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'full.pt'
            config={'science_hash':'test-v12'}
            save_full_state(p,a,ra,config,rows=[{'episode':1}],validation_rows=[],
                ledger={},best=float('inf'),credit=.125,raw_count=6,action_count=4)
            ar=self.new_agent()
            rb=EpisodeReplay(2,0,'primitive')
            saved=restore_full_state(p,ar,rb,config)
            self.assertEqual(saved['credit'],.125)
            self.assertEqual(ra.stats(),rb.stats())
            self.assertEqual(a.gradient_steps,ar.gradient_steps)
            xa,_=ra.sample(2);xb,_=rb.sample(2)
            self.assertEqual([(r.episode_id,r.raw_start_index) for r in xa],
                             [(r.episode_id,r.raw_start_index) for r in xb])
            loss_a=a.update(xa)
            loss_b=ar.update(xb)
            self.assertAlmostEqual(loss_a,loss_b,places=7)
            for k,v in a.online.state_dict().items():
                self.assertTrue(torch.equal(v,ar.online.state_dict()[k]))
            self.assertEqual(a.gradient_steps,ar.gradient_steps)

    def test_scientific_config_mismatch_rejected(self):
        a=self.new_agent();r=EpisodeReplay(2,10,'primitive')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'full.pt'
            save_full_state(p,a,r,{'science_hash':'good'},rows=[],validation_rows=[],
                ledger={},best=0.,credit=0.,raw_count=0,action_count=0)
            with self.assertRaises(ValueError):
                restore_full_state(p,a,r,{'science_hash':'bad'})

if __name__=='__main__':unittest.main()
