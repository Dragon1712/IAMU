"""V0.2 causal clock, recovery, and event-audit regression tests."""
import json
import tempfile
import unittest
from pathlib import Path

from iamu_dynamic import Vessel, Port, Update, Scenario, SlotPlanner, simulate, save_result
from iamu_dynamic.core import Assignment, validate_plan
from iamu_dynamic.recovery import pressure_graph, candidate_pairs, repair_subset


class RecoveryV02Tests(unittest.TestCase):
    def setUp(self):
        # 2 ships competing for the same short quay, 2 cranes total.
        # Ship 0 is late, ship 1 can start earlier only when BOTH promises move.
        self.vessels = [Vessel(0, 0, 40, 300, 2), Vessel(1, 0, 40, 300, 2)]
        self.port = Port(cranes=2, quay_length=350, max_cranes_per_ship=2)
        self.scenario = Scenario(self.vessels, self.port, {0:30, 1:0},
                                 [Update(1, 0, 30, 0)], 'crafted', 42)

    def test_pair_repair_overcomes_single_ship_local_optimum(self):
        baseline = simulate(self.scenario, 'budget_aware', budget=2, repair_method='single')
        joint = simulate(self.scenario, 'budget_aware', budget=2, repair_method='alns2')
        self.assertEqual(baseline['metrics']['spent'], 0)
        self.assertEqual(baseline['metrics']['weighted_wait'], 30)
        self.assertEqual(joint['metrics']['spent'], 2)
        self.assertEqual(joint['metrics']['weighted_wait'], 1)
        self.assertEqual([e['vessel_id'] for e in joint['revisions']], [0,1])
        self.assertTrue(all(e['repair_kind']=='alns2' for e in joint['revisions']))
        planner=SlotPlanner(self.vessels,self.port)
        published={int(i):Assignment(int(i),x['start'],x['berth_start_m'],x['cranes'],x['duration'])
                   for i,x in joint['final_published'].items()}
        self.assertTrue(validate_plan(published,planner.vessels,self.port))

    def test_budget_one_forbids_joint_two_ship_edit(self):
        out=simulate(self.scenario, 'budget_aware', budget=1, repair_method='alns2')
        self.assertLessEqual(out['metrics']['spent'],1)
        self.assertEqual(out['metrics']['spent'],len(out['revisions']))

    def test_penalty_prevents_uneconomic_edit(self):
        out=simulate(self.scenario,'budget_aware',budget=2,repair_method='alns2',
                     revision_penalty=100)
        self.assertEqual(out['metrics']['spent'],0)

    def test_clock_is_independent_of_eta_notifications(self):
        # No ETA notices at all: periodic must still create independent decisions.
        silent=Scenario(self.vessels,self.port,{0:0,1:0},[],'silent',44)
        out=simulate(silent,'periodic',budget=0,period=3,periodic_horizon=10)
        self.assertEqual(out['metrics']['number_of_updates'],0)
        self.assertEqual(out['metrics']['number_of_periodic_ticks'],3)
        self.assertEqual([e['time'] for e in out['decision_events']],[3,6,9])
        self.assertEqual(out['eta_updates'],[])
        self.assertEqual(out['metrics']['spent'],0)

    def test_clock_uses_only_initial_public_information(self):
        s1=Scenario(self.vessels,self.port,{0:30,1:0},[], 'a',1)
        s2=Scenario(self.vessels,self.port,{0:50,1:3},[], 'b',1)
        a=simulate(s1,'periodic',budget=0,period=6)
        b=simulate(s2,'periodic',budget=0,period=6)
        self.assertEqual([x['time'] for x in a['decision_events']],
                         [x['time'] for x in b['decision_events']])

    def test_realized_late_and_resource_slip_decomposition(self):
        out=simulate(self.scenario,'no_replan',budget=0)
        m=out['metrics']
        self.assertAlmostEqual(m['total_operational_slip'],
                               m['arrival_lateness_slip']+m['resource_conflict_slip'])
        self.assertEqual(m['arrival_lateness_slip'],30)
        self.assertEqual(m['resource_conflict_slip'],10)

    def test_initial_commitment_and_event_audit_files(self):
        out=simulate(self.scenario,'budget_aware',budget=2,repair_method='alns2')
        with tempfile.TemporaryDirectory() as d:
            save_result(Path(d),self.scenario,out)
            self.assertTrue((Path(d)/'decision_events.csv').is_file())
            self.assertTrue((Path(d)/'eta_updates.csv').is_file())
            self.assertTrue((Path(d)/'revisions.csv').is_file())
            self.assertEqual(json.loads((Path(d)/'metrics.json').read_text())['spent'],2)
            self.assertEqual(len(out['eta_updates']),1)
            self.assertEqual(len(out['decision_events']),1)

    def test_repair_never_changes_completed_or_started_vessels(self):
        # At t=11 vessel 1 is already physically in service under both true ETAs;
        # only the other ship may be revised.
        v=[Vessel(0,8,40,300,2),Vessel(1,0,40,300,2)]
        s=Scenario(v,self.port,{0:30,1:0},[Update(11,0,30,0)],'later',1)
        out=simulate(s,'budget_aware',budget=2,repair_method='alns2')
        self.assertNotIn(1,[r['vessel_id'] for r in out['revisions']])

    def test_pressure_pairs_deterministic(self):
        planner=SlotPlanner(self.vessels,self.port)
        p=planner.initial({0:0,1:0})
        graph=pressure_graph(p,planner.vessels,self.port,{0:30,1:0},1,[0,1])
        self.assertEqual(candidate_pairs(graph,p,0),[(0,1)])
        results=list(repair_subset(planner,p,(0,1),1,{0:30,1:0},budget_remaining=1))
        self.assertTrue(all(len(changed)==1 for _,changed in results))


if __name__=='__main__':
    unittest.main()
