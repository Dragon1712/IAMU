"""Fast, deterministic tests: no external case data, no GPU."""
from __future__ import annotations
import unittest
from dataclasses import dataclass
import numpy as np
from v11_protocol.research_v12 import (classify_transition, class_counts,
    collapse_exhaustion_tail, discounted_return, verify_collapse)

@dataclass
class Fake:
    reward: float
    action: int
    obs_mask: np.ndarray
    done: bool=False
    budget_before: int=1
    budget_after: int=1
    reason_flags: dict|None=None
    next_global: np.ndarray|None=None
    next_actions: np.ndarray|None=None
    next_mask: np.ndarray|None=None
    raw_steps: int=1
    transition_version: int=2

def tr(reward, before, after, action=0, done=False):
    mask=np.array([True, before>0],dtype=bool)
    return Fake(reward,action,mask,done,before,after,{'budget_exhausted':before==0},
                np.array([0.],dtype=np.float32),np.zeros((2,1),dtype=np.float32),
                np.array([True,False],dtype=bool) if done else mask)

class TestResearchV12(unittest.TestCase):
    def test_three_strata_disjoint(self):
        seq=[tr(-1,1,1),tr(-2,0,0),tr(-3,1,1)]
        seq[2].obs_mask[:]=[True,False]
        self.assertEqual([classify_transition(x) for x in seq],
                         ['actionable','budget_exhausted','forced_with_budget'])
        self.assertEqual(sum(class_counts(seq)[x] for x in
                             ('actionable','budget_exhausted','forced_with_budget')),3)
    def test_budget_becomes_zero(self):
        raw=[tr(-1,2,2),tr(-3,2,1,action=1),tr(-4,1,0,action=1),
             tr(-5,0,0),tr(-6,0,0,done=True)]
        joined=collapse_exhaustion_tail(raw,gamma=1.)
        self.assertEqual(len(joined),3)
        self.assertEqual(joined[-1].raw_steps,3)
        self.assertEqual(joined[-1].reward,-15)
        verify_collapse(raw,joined,1.)
        self.assertEqual([t.reward for t in raw],[-1,-3,-4,-5,-6])
    def test_discounted_gamma(self):
        raw=[tr(-1,1,0,action=1),tr(-2,0,0),tr(-3,0,0,done=True)]
        joined=collapse_exhaustion_tail(raw,gamma=.9)
        self.assertAlmostEqual(discounted_return(raw,.9),discounted_return(joined,.9))
        self.assertTrue(joined[-1].done)
    def test_zero_initial_budget(self):
        raw=[tr(-1,0,0),tr(-2,0,0,done=True)]
        joined=collapse_exhaustion_tail(raw)
        self.assertEqual(len(joined),1)
        self.assertEqual(joined[0].raw_steps,2)
    def test_never_exhausted_keeps_raw(self):
        raw=[tr(-1,2,2),tr(-2,2,1,action=1,done=True)]
        self.assertEqual(collapse_exhaustion_tail(raw),raw)
    def test_rejects_future_edit(self):
        raw=[tr(-1,1,0,action=1),tr(-2,0,0,done=True)]
        raw[1].obs_mask[1]=True
        with self.assertRaises(ValueError):collapse_exhaustion_tail(raw)
    def test_rejects_partial_episode(self):
        with self.assertRaises(ValueError):collapse_exhaustion_tail([tr(-1,1,0,action=1)])

if __name__=='__main__':unittest.main()
