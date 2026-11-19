"""V0.4 tests: hard revision budget, stochastic scoring, and causal recourse."""
import unittest
from dataclasses import replace

from iamu_dynamic import Vessel, Port, Update, Scenario, demo_case, generate_scenario, simulate
from iamu_dynamic.core import SlotPlanner, RealizedRuntime
from iamu_dynamic.risk_selection import (RiskConfig, upper_cvar,
    sample_future_arrivals, future_one_step_cost)


class RiskSelectionV04Tests(unittest.TestCase):
    def test_upper_cvar_tail_and_fractional_sample(self):
        self.assertAlmostEqual(upper_cvar([1, 2, 3, 4], .75), 4)
        self.assertAlmostEqual(upper_cvar([1, 2, 3, 4], .5), 3.5)
        self.assertAlmostEqual(upper_cvar([1, 2, 3, 4], .625), (4 + .5*3)/1.5)
        with self.assertRaises(ValueError): upper_cvar([], .75)

    def test_sampler_is_reproducible_and_public_state_only(self):
        config=RiskConfig(n_samples=4,forecast_std=5)
        etas={4:17., 3:20.}
        common=dict(run_seed=123,event_id=31,event_type='eta',config=config)
        a=sample_future_arrivals(etas,10,{4},**common)
        b=sample_future_arrivals(dict(reversed(list(etas.items()))),10,{4},**common)
        self.assertEqual(a,b)
        self.assertTrue(all(x[3]>=10 for x in a))
        self.assertTrue(all(x[4]==17 for x in a))

    def test_invalid_risk_configuration_rejected(self):
        with self.assertRaises(ValueError): RiskConfig(forecast_std=-1).validate()
        with self.assertRaises(ValueError): RiskConfig(cvar_alpha=1).validate()
        with self.assertRaises(ValueError): RiskConfig(lookahead_horizon=0).validate()
        with self.assertRaises(ValueError): RiskConfig(n_samples=0).validate()

    def test_future_recourse_cannot_erase_waiting_already_incurred(self):
        ships=[Vessel(0,0,20,300,1),Vessel(1,0,20,300,1)]
        port=Port(cranes=1,max_cranes_per_ship=1,quay_length=800)
        planner=SlotPlanner(ships,port)
        plan=planner.initial({0:0,1:0})
        rt=RealizedRuntime(ships,port)
        sample={0:10.,1:0.}
        cost,edits=future_one_step_cost(rt,planner,plan,{0:0.,1:0.},
            sample,now=0,budget_remaining=1,
            config=RiskConfig(lookahead_horizon=12,n_samples=2,forecast_std=0,
                              future_signal_strength=.5), revision_penalty=0)
        # Ship 0 is committed at time 0 but arrives at 10. At t=12 its
        # 10 units of waiting are a sunk physical cost, even with recourse.
        self.assertGreaterEqual(cost,10.-1e-8)
        self.assertIn(edits,(0,1))

    def test_risk_budget_and_no_oracle_action_selection(self):
        ships=[Vessel(0,10,30,300,2),Vessel(1,12,30,300,2),Vessel(2,16,50,280,2)]
        port=Port(cranes=2,max_cranes_per_ship=2,quay_length=600)
        updates=[Update(1,0,14,0),Update(2,1,17,1)]
        a=Scenario(ships,port,{0:21,1:22,2:23},updates,'test',7)
        b=Scenario(ships,port,{0:40,1:42,2:43},updates,'test',7)
        for method in ('expected','risk_lookahead'):
            x=simulate(a,'budget_aware',2,selection_method=method,
                risk_samples=3,forecast_std=4,lookahead_horizon=3,
                candidate_limit=3,lookahead_candidates=2,next_move_candidates=2)
            y=simulate(b,'budget_aware',2,selection_method=method,
                risk_samples=3,forecast_std=4,lookahead_horizon=3,
                candidate_limit=3,lookahead_candidates=2,next_move_candidates=2)
            self.assertEqual(x['revisions'],y['revisions'])
            self.assertEqual(x['eta_updates'],y['eta_updates'])
            self.assertLessEqual(x['metrics']['spent'],2)
            self.assertEqual(len(x['revisions']),x['metrics']['spent'])
            self.assertTrue(all(log['selection_method']==method for log in x['eta_updates']))

    def test_all_zero_budget_equal_no_replan(self):
        ships,port=demo_case()
        sc=generate_scenario(ships,port,'shock',seed=19,updates_per_vessel=2)
        ref=simulate(sc,'no_replan',0)
        for method in ('deterministic','expected','risk_lookahead'):
            result=simulate(sc,'budget_aware',0,selection_method=method,
                risk_samples=2,lookahead_candidates=2,next_move_candidates=2)
            self.assertEqual(result['realized'],ref['realized'])
            self.assertEqual(result['revisions'],[])

    def test_original_deterministic_selection_unchanged_by_defaults(self):
        ships,port=demo_case()
        sc=generate_scenario(ships,port,'shock',seed=16,updates_per_vessel=2)
        default=simulate(sc,'budget_aware',2)
        explicit=simulate(sc,'budget_aware',2,selection_method='deterministic')
        self.assertEqual(default['revisions'],explicit['revisions'])
        self.assertEqual(default['realized'],explicit['realized'])
        self.assertEqual(default['metrics'],explicit['metrics'])

if __name__=='__main__':unittest.main()
