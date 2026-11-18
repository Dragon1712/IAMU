"""Q-gap boundary and tie tests against the deployed evaluator."""
from __future__ import annotations
import unittest
import numpy as np
try:
    from v11_protocol.evaluate_qgap_bank import inspect_q_array
    HAVE=True
except ImportError:
    HAVE=False

@unittest.skipUnless(HAVE,'Requires IAMU network module and its dependencies')
class TestQGapV12(unittest.TestCase):
    def test_margin_and_mask(self):
        mask=np.array([True,True,True,False])
        q=np.array([-2.,-1.92,-2.2,100.])
        self.assertEqual(inspect_q_array(q,mask,margin=0.)['chosen_action'],1)
        self.assertEqual(inspect_q_array(q,mask,margin=.1)['chosen_action'],0)
        self.assertAlmostEqual(inspect_q_array(q,mask)['best_qgap'],.08)
    def test_exact_tie_keep_wins(self):
        q=np.array([1.,1.,.9]);mask=np.array([True,True,True])
        self.assertEqual(inspect_q_array(q,mask)['chosen_action'],0)
    def test_non_actionable_not_defined(self):
        self.assertIsNone(inspect_q_array(np.array([1.,2.]),np.array([True,False])))

if __name__=='__main__':unittest.main()
