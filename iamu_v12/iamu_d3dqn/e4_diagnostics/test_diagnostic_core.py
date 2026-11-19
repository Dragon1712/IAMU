"""Pure diagnostic regression tests; no physical simulator or checkpoint needed."""
import unittest
from diagnostic_core import (action_diagnostics, choose_evenly,
                             calibration_summary, regret_summary,
                             cvar_upper, pearson)


class TestActionDiagnostics(unittest.TestCase):
    def test_matches_margin_keep_rule(self):
        d=action_diagnostics([.2,.27,.45],[True,True,False],.1)
        self.assertEqual(d['action'],0)
        self.assertEqual(d['action_without_margin'],1)
        self.assertEqual(d['reason'],'keep_margin_only')
        self.assertAlmostEqual(d['best_gap'],.07)

    def test_negative_q_gap_is_not_margin_effect(self):
        d=action_diagnostics([.2,-1.,.18],[True,True,True],.1)
        self.assertEqual(d['action'],0)
        self.assertEqual(d['action_without_margin'],0)
        self.assertEqual(d['reason'],'q_prefers_keep')
        self.assertAlmostEqual(d['best_gap'],-.02)

    def test_legal_edit_with_large_q(self):
        d=action_diagnostics([.4,.8,999],[True,True,False],.1)
        self.assertEqual(d['action'],1)
        self.assertEqual(d['best_edit'],1)
        self.assertAlmostEqual(d['margin_distance'],.3)

    def test_tie_favors_keep_and_lower_edit_index(self):
        self.assertEqual(action_diagnostics([0.,.1],[True,True],.1)['action'],0)
        self.assertEqual(action_diagnostics([0.,.5,.5],[True,True,True],.1)['action'],1)

    def test_no_legal_candidate(self):
        d=action_diagnostics([0.,500.],[True,False],.1)
        self.assertEqual(d['reason'],'no_legal_edit')
        self.assertIsNone(d['best_gap'])

    def test_invalid_and_nonfinite(self):
        for q,mask in [([0,1],[False,True]),([0,float('nan')],[True,True]),
                       ([0,1],[True])]:
            with self.assertRaises(ValueError):
                action_diagnostics(q,mask,.1)


class TestSamplingAndCalibration(unittest.TestCase):
    def test_spread_sampling(self):
        self.assertEqual(choose_evenly(0,3),[])
        self.assertEqual(choose_evenly(2,4),[0,1])
        self.assertEqual(choose_evenly(9,3),[1,4,7])
        self.assertEqual(len(set(choose_evenly(100,17))),17)

    def test_cvar_fractional_tail(self):
        # For 5 samples and alpha=.9, upper 10% is the worst half observation.
        self.assertAlmostEqual(cvar_upper([0,0,0,0,10],.9),10.)
        self.assertAlmostEqual(cvar_upper([0,0,0,4,10],.6),7.)

    def test_correlated_gap_calibration(self):
        rr=[dict(q_gap=0.,true_gain_scaled=.1,keep_margin=.1),
            dict(q_gap=.1,true_gain_scaled=.2,keep_margin=.1),
            dict(q_gap=.2,true_gain_scaled=.3,keep_margin=.1)]
        res=calibration_summary(rr)
        self.assertEqual(res['n_pairs'],3)
        self.assertAlmostEqual(res['q_gap_mae'],.1)
        self.assertAlmostEqual(res['q_gap_signed_bias'],-.1)
        self.assertAlmostEqual(res['pearson_r'],1.)

    def test_constant_gap_r_is_undefined(self):
        self.assertIsNone(pearson([1.,1.],[2.,3.]))

    def test_one_counterfactual_future_is_not_oracle_expectation(self):
        # The diagnostic measures regret relative to a *known realization*.
        r=[dict(first_action_regret=7.,missed_beneficial_keep=True,
                harmful_edit=False,selected_action=0,n_legal_edits=2,remaining_budget=2),
           dict(first_action_regret=0.,missed_beneficial_keep=False,
                harmful_edit=False,selected_action=1,n_legal_edits=2,remaining_budget=1)]
        res=regret_summary(r)
        self.assertAlmostEqual(res['mean_first_action_regret'],3.5)
        self.assertEqual(res['missed_beneficial_keep'],1)


if __name__=='__main__':
    unittest.main()
