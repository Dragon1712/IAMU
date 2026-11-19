import tempfile
import unittest
from pathlib import Path

from iamu_dynamic import (
    Vessel,Port,Update,Scenario,read_case,demo_case,generate_scenario,simulate,
    SlotPlanner,RealizedRuntime,
)
from iamu_dynamic.core import validate_plan,assignments_equal

class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vessels,cls.port=demo_case()
        cls.scenario=generate_scenario(cls.vessels,cls.port,'shock',seed=14,updates_per_vessel=3)
    def test_file_format_original(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'case.txt'
            p.write_text('0.41\t105\t329\t13\t30\n18.72\t439\t365\t15\n',encoding='utf-8')
            ships,port=read_case(p)
            self.assertEqual(port.cranes,30)
            self.assertEqual([v.id for v in ships],[0,1])
            self.assertEqual(ships[0].eta0,.41)
            self.assertEqual(ships[1].length,365)
    def test_event_causality(self):
        s=self.scenario
        self.assertEqual(s.updates,sorted(s.updates,key=lambda e:(e.time,e.update_id)))
        self.assertTrue(all(e.time<=s.true_etas[e.vessel_id]+1e-9 for e in s.updates))
        self.assertTrue(all(e.new_eta>=e.time-1e-9 for e in s.updates))
        s2=generate_scenario(s.vessels,s.port,'shock',seed=14,updates_per_vessel=3)
        self.assertEqual(s.true_etas,s2.true_etas)
        self.assertEqual(s.updates,s2.updates)
    def test_initial_plan_feasibility(self):
        s=self.scenario
        p=SlotPlanner(s.vessels,s.port)
        plan=p.initial({v.id:v.eta0 for v in s.vessels})
        self.assertTrue(validate_plan(plan,p.vessels,s.port))
        self.assertTrue(all(a.start>=s.vessels[i].eta0 for i,a in plan.items()))
    def test_budget_and_no_replan(self):
        s=self.scenario
        base=simulate(s,'no_replan',budget=0)
        self.assertEqual(base['metrics']['spent'],0)
        self.assertEqual(base['initial_published'],base['final_published'])
        self.assertEqual(base['revisions'],[])
        for b in [0,1,2,5]:
            r=simulate(s,'budget_aware',budget=b)
            self.assertLessEqual(r['metrics']['spent'],b)
            self.assertEqual(r['metrics']['spent'],len(r['revisions']))
            self.assertEqual(r['metrics']['spent'],sum(r['revision_counts'].values()))
            self.assertTrue(all(x['new']!=x['old'] for x in r['revisions']))
            self.assertEqual(set(r['realized']),set(base['realized']))
    def test_realized_resource_constraints(self):
        s=self.scenario
        r=simulate(s,'threshold',budget=5)
        p=SlotPlanner(s.vessels,s.port)
        for name in ['initial_published','final_published','realized']:
            from iamu_dynamic.core import Assignment
            plan={int(i):Assignment(int(i),x['start'],x['berth_start_m'],
                                    x['cranes'],x['duration']) for i,x in r[name].items()}
            self.assertTrue(validate_plan(plan,p.vessels,s.port))
        self.assertTrue(all(r['realized'][i]['start']+1e-6>=s.true_etas[int(i)]
                            for i in r['realized']))
    def test_event_time_freeze(self):
        s=self.scenario
        r=simulate(s,'always',budget=5)
        for x in r['revisions']:
            # A revision is never applied to a ship physically started earlier.
            actual=r['realized'][x['vessel_id']]['start']
            self.assertGreaterEqual(actual+1e-8,x['time'])
    def test_no_future_arrival_leak(self):
        vessels=[Vessel(0,10,30,300,2),Vessel(1,12,30,300,2)]
        port=Port(4,max_cranes_per_ship=2)
        upd=[Update(1,0,14,0)]
        a=Scenario(vessels,port,{0:14,1:13},upd,'custom',1)
        b=Scenario(vessels,port,{0:16,1:18},upd,'custom',1)
        r1=simulate(a,'budget_aware',budget=2)
        r2=simulate(b,'budget_aware',budget=2)
        self.assertEqual(r1['initial_published'],r2['initial_published'])
        self.assertEqual(r1['revisions'],r2['revisions'])
    def test_published_same_zero_budget(self):
        r0=simulate(self.scenario,'always',budget=0)
        r1=simulate(self.scenario,'no_replan',budget=5)
        self.assertEqual(r0['realized'],r1['realized'])
        self.assertEqual(r0['metrics']['weighted_wait'],r1['metrics']['weighted_wait'])

if __name__=='__main__':unittest.main()
