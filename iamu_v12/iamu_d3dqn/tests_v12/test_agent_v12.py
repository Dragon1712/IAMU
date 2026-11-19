"""Tests use installed V1.1 physical/network implementation if accessible."""
import unittest
import torch
from v12_protocol.compress import compress_episode

class IntervalTargets(unittest.TestCase):
    def test_gamma_half_two_step_and_no_double_gamma(self):
        # gamma=.5, segment rewards 1 and 2 => 2; next Q=8, gamma**2=.25 => 4.
        r=torch.tensor([1.+.5*2, 2.])
        discount=torch.tensor([.5**2,0.])
        nxt=torch.tensor([8., 999.])
        from v12_protocol.targets import ddqn_interval_target
        y=ddqn_interval_target(r,discount,nxt)
        self.assertTrue(torch.allclose(y,torch.tensor([4.,2.])))

try:
    from v12_protocol.agent import V12Agent
    from iamu_dynamic.rl_env_v11 import GLOBAL_DIM,ACTION_DIM
except ImportError:
    V12Agent=None

@unittest.skipUnless(V12Agent is not None,'Requires full V1.1 source on PYTHONPATH')
class CheckpointAndAgentTests(unittest.TestCase):
    def test_checkpoint_metadata_round_trip(self):
        import tempfile
        from pathlib import Path
        a=V12Agent(hidden_dim=96,seed=10,device='cpu',gamma=.5,q_network='set_ddqn',
                   representation='decision_epoch',reward_scale=100.,
                   execution_margin=0.,bootstrap_margin=.12)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            torch.save(a.checkpoint({'selected_episode':100}),path)
            b,meta=V12Agent.from_checkpoint(path)
            self.assertEqual((b.gamma,b.bootstrap_margin,b.execution_margin),(.5,.12,0.))
            self.assertEqual(b.representation,'decision_epoch')
            self.assertEqual(meta['selected_episode'],100)
            for k,v in a.online.state_dict().items():
                self.assertTrue(torch.equal(v,b.online.state_dict()[k]))

    def test_independent_macro_update_with_two_step_discount(self):
        import numpy as np
        from v12_protocol.transitions import MacroTransition
        a=V12Agent(hidden_dim=32,seed=42,device='cpu',gamma=.5,q_network='set_ddqn',
                   representation='decision_epoch')
        g=np.zeros(GLOBAL_DIM,dtype=np.float32)
        x=np.zeros((3,ACTION_DIM),dtype=np.float32)
        mask=np.array([True,True,False])
        tr=MacroTransition(g,x,mask,1,-3.,g.copy(),x.copy(),mask.copy(),False,
                .25,2,0.,2,1,42,0,0,1,{})
        loss=a.update([tr,tr])
        self.assertTrue(0<=loss<1e9)
        self.assertEqual(a.gradient_steps,1)
        bad=MacroTransition(g,x,mask,1,-3.,g.copy(),x.copy(),mask.copy(),True,
                .25,2,0.,2,1,42,0,0,1,{})
        with self.assertRaises(ValueError):a.update([bad])

if __name__=='__main__':unittest.main()
