import unittest,copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from v13_boundary.audit_boundary import run

class ToyNetwork(torch.nn.Module):
    def forward(self,g,acts,mask):
        return torch.tensor([[8.,1.]]).expand(g.shape[0],-1)
class ToyAgent:
    representation='budget_tail';execution_margin=0.;bootstrap_margin=0.
    online=ToyNetwork();target=ToyNetwork()
class ToyEnv:
    def __init__(self):self.i=0;self.b=1;self.done=False
    def _observation(self):
        return SimpleNamespace(mask=np.array([True,self.i==0 and self.b>0]),
                 remaining_budget=self.b,global_features=np.array([0.],dtype=np.float32),
                 action_features=np.zeros((2,1),dtype=np.float32))
    def point_greedy_action(self):return 0
    def step(self,action):
        if not self._observation().mask[action]:raise ValueError('Illegal')
        if action:self.b-=1
        r=[-1.,-2.,-3.][self.i]
        self.i+=1;self.done=self.i==3
        return self._observation(),r,self.done,dict(weighted_wait=600. if self.done else None)
class BranchTests(unittest.TestCase):
    def test_same_state_dual_branch(self):
        probe=dict(state_id='1:0',episode_id=1,raw_step_index=0,case='toy',
            scenario='shock',source_budget=1,selected_edit=1,
            action_descriptor=dict(kind='edit'),snapshot=ToyEnv())
        with patch('v13_boundary.audit_boundary.draw_tape',return_value=object()), \
             patch('v13_boundary.audit_boundary.tape_hash',return_value='tape-hash'), \
             patch('v13_boundary.audit_boundary.apply_tape',side_effect=lambda e,t:copy.deepcopy(e)):
            branches,pairs=run([probe],{'A2':ToyAgent()},('greedy',),tapes=1)
        self.assertEqual(len(branches),2)
        self.assertEqual(len(pairs),1)
        keep=next(x for x in branches if x['branch']=='Keep')
        edit=next(x for x in branches if x['branch']=='Edit')
        self.assertEqual(keep['n_raw_steps'],1)
        self.assertEqual(keep['suffix_raw_steps'],2)
        self.assertEqual(edit['n_raw_steps'],3)
        self.assertEqual(edit['bootstrap_error'],0.)
        self.assertAlmostEqual(keep['bootstrap_error'],13.)
        self.assertAlmostEqual(pairs[0]['target_gap']-pairs[0]['return_gap'],-13.)
        self.assertAlmostEqual(pairs[0]['identity_residual'],0.)
