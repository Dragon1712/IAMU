"""Run against REAL IAMU package: python -m unittest tests.test_v10_modules -v"""
from __future__ import annotations
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from iamu_dynamic.v10_risk_features import (
    RiskSpecification, public_eta_scenarios, summarize_risk, event_seed)
from iamu_dynamic.v10_budget_lookahead import (
    LookaheadSpecification, select_lookahead)
from iamu_dynamic.v10_candidate_api import (
    build_proposal, compute_changes, validate_proposal)
from iamu_dynamic.v10_policy import budget_scarcity
from iamu_dynamic.core import Assignment, Port, Vessel, SlotPlanner


class RiskTests(unittest.TestCase):
    def test_repeatable_and_no_hidden_future_input(self):
        eta = {1:12., 2:20.}
        spec = RiskSpecification(samples=50, sigma=5., belief='censored')
        a = public_eta_scenarios(eta, 10., set(), spec, seed=11)
        b = public_eta_scenarios(eta, 10., set(), spec, seed=11)
        self.assertEqual(a,b)
        self.assertTrue(all(row[1]>=10. and row[2]>=10. for row in a))

    def test_arrived_but_waiting_no_future_arrival(self):
        eta={1:50., 2:20.}
        s=RiskSpecification(samples=24,sigma=9.)
        draws=public_eta_scenarios(eta, 30., {1}, s, seed=8)
        self.assertTrue(all(d[1]==30. for d in draws))
        self.assertTrue(all(d[2]>=30. for d in draws))

    def test_truncated_tail_and_deterministic_zero_sigma(self):
        a=public_eta_scenarios({0:3.}, 30., set(),
                               RiskSpecification(samples=100,sigma=2.,belief='truncated'),seed=1)
        self.assertTrue(all(d[0]>=30. for d in a))
        b=public_eta_scenarios({0:3.},30.,set(),RiskSpecification(samples=4,sigma=0.),seed=1)
        self.assertEqual([d[0] for d in b],[30.]*4)

    def test_cvar_and_probabilities(self):
        r=summarize_risk([10.,10.,10.,10.], [9.,11.,9.,13.], alpha=.75)
        self.assertAlmostEqual(r.mean_gain,-.5)
        self.assertAlmostEqual(r.harm_probability,.5)
        self.assertAlmostEqual(r.downside_cvar,3.)
        self.assertAlmostEqual(r.tail_cost_delta,3.)

    def test_event_seed_stable(self):
        self.assertEqual(event_seed(42,15,'eta'),event_seed(42,15,'eta'))
        self.assertNotEqual(event_seed(42,15,'eta'),event_seed(42,16,'eta'))


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.vessels=[Vessel(0,5.,20.,100.,2),Vessel(1,8.,20.,100.,2)]
        self.port=Port(cranes=4,quay_length=1200.)
        self.planner=SlotPlanner(self.vessels,self.port)
        self.current={0:Assignment(0,20.,0.,2,10.),
                      1:Assignment(1,20.,500.,2,10.)}

    def test_unchanged_no_candidate(self):
        self.assertEqual(compute_changes(self.current,dict(self.current)),())

    def test_correct_changed_vessel_cost(self):
        new=dict(self.current)
        new[1]=Assignment(1,25.,500.,2,10.)
        proposal=build_proposal(self.current,new,'test','one')
        self.assertEqual(proposal.changed_ids,(1,))
        self.assertEqual(proposal.edit_cost,1)
        validate_proposal(proposal,self.current,planner=self.planner,
                          port=self.port,started_ids={0},now=10.,
                          remaining_budget=1,published_eta={0:5.,1:8.})
        with self.assertRaises(ValueError):
            validate_proposal(proposal,self.current,planner=self.planner,
                              port=self.port,started_ids={1},now=10.,
                              remaining_budget=1)
        with self.assertRaises(ValueError):
            validate_proposal(proposal,self.current,planner=self.planner,
                              port=self.port,started_ids=set(),now=10.,
                              remaining_budget=0)

    def test_no_mutable_plan_alias(self):
        new=dict(self.current)
        new[1]=Assignment(1,25.,500.,2,10.)
        proposal=build_proposal(self.current,new,'test','one')
        new[1]=self.current[1]
        self.assertNotEqual(proposal.assignments[1],new[1])


@dataclass(frozen=True)
class MiniAssignment:
    start: float


class MiniRuntime:
    def __init__(self):
        self.started={}
    def advance(self,t,plan,actual):
        return None
    def score(self,plan,forecast):
        return max(0.,plan[0].start-forecast[0])
    def finish(self,plan,actual):
        self.started={0:MiniAssignment(plan[0].start)}
        return self


class MiniPlanner:
    def __init__(self):
        self.vessels={0:SimpleNamespace(weight=1.)}
    def single_move(self,plan,i,now,etas):
        if plan[i].start<=102.:
            return None
        return {0:MiniAssignment(102.)}


class BudgetLookaheadTests(unittest.TestCase):
    def test_keep_can_preserve_future_repair(self):
        env=SimpleNamespace(runtime=MiniRuntime(),planner=MiniPlanner(),
            vessels={0:SimpleNamespace(weight=1.)},
            published_eta={0:100.},event_time=0.,remaining_budget=1,
            plan={0:MiniAssignment(110.)})
        edits={1:{0:MiniAssignment(107.)}}
        eta_samples=[{0:100.}]*4
        choice,objective,_=select_lookahead(
            env,edits,eta_samples,LookaheadSpecification(stages=1,step_hours=12.,
                                                         signal_strength=0.5))
        self.assertEqual(choice,0)  # Keep reserves budget for improvement 10->2
        self.assertLess(objective[0],objective[1])

    def test_budget_scarcity_finite(self):
        env=SimpleNamespace(ids=[0,1,2,3],runtime=SimpleNamespace(started={0:1}),
                            update_counts={0:2,1:1,2:0,3:0},remaining_budget=1)
        value,op=budget_scarcity(env)
        self.assertGreaterEqual(value,0.)
        self.assertLessEqual(value,1.)
        self.assertGreater(op,0.)

if __name__=='__main__':
    unittest.main()
