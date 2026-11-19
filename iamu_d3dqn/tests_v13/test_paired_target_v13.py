import unittest
from types import SimpleNamespace
import numpy as np
from v13_diagnostics.audit_paired_targets import first_macro


class FakeStepEnv:
    def __init__(self,budget=1):
        self.i=0;self.budget=budget;self.done=False
    def _observation(self):
        # If budget is exhausted, there is no legal Edit.
        mask=np.array([1,int(self.budget>0)],dtype=bool)
        return SimpleNamespace(mask=mask,remaining_budget=self.budget)
    def step(self,action):
        if action: self.budget-=1
        self.i+=1
        self.done=self.i==3
        obs=self._observation()
        # Make i=1 forced Keep regardless of remaining budget, by masking edit.
        if self.i==1:obs.mask[1]=False
        return obs, -float(action+1), self.done,{}


class PairedMacroTests(unittest.TestCase):
    def test_keep_stops_at_next_actionable(self):
        env=FakeStepEnv(budget=1)
        m=first_macro(env,0,gamma=.9)
        self.assertEqual(m['n_raw_steps'],2)
        self.assertFalse(m['done'])
        self.assertAlmostEqual(m['interval_reward'],-1-.9)
        self.assertAlmostEqual(m['bootstrap_discount'],.81)

    def test_last_edit_consumes_full_forced_tail(self):
        env=FakeStepEnv(budget=1)
        m=first_macro(env,1,gamma=.9)
        self.assertEqual(m['n_raw_steps'],3)
        self.assertTrue(m['done'])
        self.assertEqual(m['bootstrap_discount'],0.)
        self.assertAlmostEqual(m['interval_reward'],-2-.9-.81)

if __name__=='__main__':unittest.main()
