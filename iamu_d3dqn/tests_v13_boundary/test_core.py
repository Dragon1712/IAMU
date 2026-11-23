import unittest
from types import SimpleNamespace
import numpy as np
from v13_boundary.core import BranchValue,paired_identity,macro_steps,grouped_state_mean

class ToyEnv:
    def __init__(self,rewards,masks,budget=1):
        self.rewards=rewards;self.masks=masks;self.budget=budget;self.done=False;self.i=0
    def _observation(self):
        return SimpleNamespace(mask=np.asarray(self.masks[self.i],dtype=bool),remaining_budget=self.budget)
    def step(self,action):
        if not self._observation().mask[action]:raise ValueError('illegal')
        if action:self.budget-=1
        r=self.rewards[self.i];self.i+=1;self.done=self.i==len(self.rewards)
        return self._observation(),r,self.done,dict(weighted_wait=-100*sum(self.rewards))

class CoreTests(unittest.TestCase):
    def test_keep_cost_optimism_exact(self):
        k=BranchValue(-2.,1.,8.,-4.,False,1)
        e=BranchValue(-6.,0.,0.,0.,True,4)
        result=paired_identity(k,e)
        self.assertEqual(result['keep_bootstrap_error'],12.)
        self.assertEqual(result['edit_bootstrap_error'],0.)
        self.assertEqual(result['target_gap']-result['return_gap'],-12.)
        self.assertEqual(result['identity_residual'],0.)
    def test_gamma_under_one(self):
        a=BranchValue(-2-.9*3,.9**2,-12.,-10.,False,2)
        b=BranchValue(-6.,0.,0.,0.,True,1)
        self.assertAlmostEqual(paired_identity(a,b)['identity_residual'],0)
    def test_terminal_discount_rejected(self):
        with self.assertRaises(ValueError):BranchValue(1.,.9,0.,0.,True,1)
    def test_budget_last_edit_covers_forced_tail(self):
        e=ToyEnv([-1,-2,-3,-4],[[1,1],[1,0],[1,0],[1,0],[1,0]])
        m=macro_steps(e,1,'budget_tail',.9)
        self.assertEqual(m['n'],4);self.assertEqual(m['discount'],0.)
        self.assertAlmostEqual(m['reward'],-1-.9*2-.9**2*3-.9**3*4)
    def test_a2_keep_single_step(self):
        e=ToyEnv([-1,-2,-3],[[1,1],[1,1],[1,1],[1,1]])
        m=macro_steps(e,0,'budget_tail',1.)
        self.assertEqual(m['n'],1);self.assertEqual(m['discount'],1.)
    def test_a3_keep_to_next_edit(self):
        e=ToyEnv([-1,-2,-3,-4],[[1,1],[1,0],[1,0],[1,1],[1,1]])
        m=macro_steps(e,0,'decision_epoch',.9)
        self.assertEqual(m['n'],3)
        self.assertAlmostEqual(m['reward'],-1-.9*2-.9**2*3)
        self.assertAlmostEqual(m['discount'],.9**3)
    def test_primitive_one_event(self):
        e=ToyEnv([-1,-2,-3],[[1,1],[1,0],[1,0],[1,0]])
        self.assertEqual(macro_steps(e,1,'primitive',1.)['n'],1)
    def test_group_state_means_not_tapes(self):
        rows=[dict(model='a',checkpoint='a',continuation='self',source_budget=1,
                    scenario='shock',state_id='1:0',z=1.),
              dict(model='a',checkpoint='a',continuation='self',source_budget=1,
                    scenario='shock',state_id='1:0',z=3.),
              dict(model='a',checkpoint='a',continuation='self',source_budget=1,
                    scenario='shock',state_id='2:0',z=4.)]
        group=grouped_state_mean(rows,'z')
        self.assertEqual(group[0]['n_states'],2)
        self.assertEqual(group[0]['mean'],3.)
