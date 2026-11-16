"""Pure E3 tests: do not require checkpoints or the physical port simulator."""
import unittest
from types import SimpleNamespace
from iamu_dynamic.v10_e3_confidence import (
    paired_confidence,summarize_mpc,fused_scores,directions_agree,t975)


class E3ConfidenceTests(unittest.TestCase):
    def test_paired_common_random_numbers_cancel_shared_shock(self):
        st=paired_confidence([10,100,1000,10000],[8,98,998,9998],100)
        self.assertAlmostEqual(st.mean,.02)
        self.assertAlmostEqual(st.sd,0.)
        self.assertAlmostEqual(st.signal_strength,1.)
        self.assertLessEqual(st.lower,st.mean)

    def test_high_variance_or_zero_mean_blocks_confidence(self):
        st=paired_confidence([100]*4,[0,200,0,200],100)
        self.assertAlmostEqual(st.mean,0.)
        self.assertEqual(st.signal_strength,0.)
        self.assertGreater(st.half_width,0)

    def test_t_small_samples_more_conservative(self):
        self.assertGreater(t975(2),t975(30))
        self.assertGreater(t975(30),t975(1000))

    def test_mpc_uses_paired_costs_per_action(self):
        m=SimpleNamespace(scenario_costs={0:(100,200,300),1:(90,190,290),2:(80,220,290)})
        stats=summarize_mpc(m,100)
        self.assertAlmostEqual(stats[1].mean,.1)
        self.assertAlmostEqual(stats[1].sd,0)
        self.assertAlmostEqual(stats[2].mean,10/300)

    def test_fixed_matches_known_e2_scoring(self):
        st=paired_confidence([200]*4,[160]*4,100)
        risk={1:SimpleNamespace(tail_cost_delta=-20)}
        chosen,z=fused_scores([0,.08],[True,True],{1:st},mode='q_fixed',
             beta_budget=.25,beta_risk=.25,risk_metrics=risk,
             reward_scale=100,keep_margin=.05)
        expected=.75*.08+.25*.4-.25*(-20/100)
        self.assertEqual(chosen,1)
        self.assertAlmostEqual(z[1]['score'],expected)

    def test_conflict_falls_back_to_frozen_q(self):
        st=paired_confidence([100]*4,[160]*4,100)
        # Q says beneficial Edit, stable MPC says harmful Edit.
        chosen,z=fused_scores([0,.2],[True,True],{1:st},mode='q_adaptive',
             beta_budget=.4,beta_risk=0,risk_metrics=None,
             reward_scale=100,keep_margin=.05)
        self.assertFalse(z[1]['agree'])
        self.assertEqual(z[1]['effective_weight'],0.)
        self.assertEqual(chosen,1)
        _,fixed=fused_scores([0,.2],[True,True],{1:st},mode='q_fixed',
             beta_budget=.4,beta_risk=0,risk_metrics=None,
             reward_scale=100,keep_margin=.05)
        self.assertNotAlmostEqual(fixed[1]['score'],z[1]['score'])

    def test_confidence_suppresses_unstable_mpc(self):
        st=paired_confidence([100]*4,[0,200,0,200],100)
        _,z=fused_scores([0,.2],[True,True],{1:st},mode='q_confidence',
             beta_budget=.4,beta_risk=0,risk_metrics=None,
             reward_scale=100,keep_margin=0.)
        self.assertEqual(z[1]['effective_weight'],0.)
        self.assertAlmostEqual(z[1]['score'],.2)

    def test_agreeing_stable_mpc_keeps_beta(self):
        st=paired_confidence([100]*8,[80]*8,100)
        _,z=fused_scores([0,.1],[True,True],{1:st},mode='q_adaptive',
             beta_budget=.4,beta_risk=0,risk_metrics=None,
             reward_scale=100,keep_margin=0.)
        self.assertEqual(z[1]['effective_weight'],.4)

    def test_pure_mpc_does_not_use_q_but_keeps_physical_threshold(self):
        st=paired_confidence([100]*4,[90]*4,100)
        chosen,z=fused_scores([0,-1.0],[True,True],{1:st},mode='pure_mpc',
             beta_budget=.4,beta_risk=0,risk_metrics=None,
             reward_scale=100,keep_margin=.001)
        self.assertEqual(chosen,1)
        self.assertAlmostEqual(z[1]['score'],.1)

    def test_pure_mpc_omits_risk_while_risk_matched_includes_it(self):
        st=paired_confidence([100.]*4,[90.]*4,100)
        r={1:SimpleNamespace(tail_cost_delta=50.)}
        _,raw=fused_scores([0,-.5],[True,True],{1:st},mode='pure_mpc',
              beta_budget=.25,beta_risk=.5,risk_metrics=r,
              reward_scale=100,keep_margin=0.)
        _,matched=fused_scores([0,-.5],[True,True],{1:st},mode='pure_mpc_risk',
              beta_budget=.25,beta_risk=.5,risk_metrics=r,
              reward_scale=100,keep_margin=0.)
        self.assertAlmostEqual(raw[1]['score'],.1)
        self.assertAlmostEqual(matched[1]['score'],.1-.5*.5)

    def test_q_risk_only_has_same_risk_adjustment_as_fused(self):
        st=paired_confidence([100.]*4,[90.]*4,100)
        r={1:SimpleNamespace(tail_cost_delta=20.)}
        _,baseline=fused_scores([0,.2],[True,True],{1:st},mode='q_only',
              beta_budget=.25,beta_risk=.5,risk_metrics=r,
              reward_scale=100,keep_margin=0.)
        _,risk=fused_scores([0,.2],[True,True],{1:st},mode='q_risk_only',
              beta_budget=.25,beta_risk=.5,risk_metrics=r,
              reward_scale=100,keep_margin=0.)
        self.assertAlmostEqual(baseline[1]['score'],.2)
        self.assertAlmostEqual(risk[1]['score'],.1)
        self.assertEqual(risk[1]['effective_weight'],0.)

    def test_masks_and_ties(self):
        st=paired_confidence([100]*4,[80]*4,100)
        action,z=fused_scores([0,.1,999],[True,True,False],{1:st},mode='q_fixed',
             beta_budget=.25,beta_risk=0,risk_metrics=None,
             reward_scale=100,keep_margin=.0)
        self.assertEqual(action,1)
        self.assertNotIn(2,z)
        with self.assertRaises(ValueError):
            fused_scores([0,.1],[False,True],{1:st},mode='q_fixed',
                 beta_budget=.25,beta_risk=0,risk_metrics=None,
                 reward_scale=100,keep_margin=0.)

    def test_single_sample_is_not_trusted(self):
        st=paired_confidence([100],[20],100)
        self.assertEqual(st.signal_strength,0.)


if __name__=='__main__':unittest.main()
