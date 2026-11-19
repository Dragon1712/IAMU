import copy
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from v13_diagnostics.common import balanced_jobs,state_id,stable_digest
from v13_diagnostics.future_tapes import draw_tape,apply_tape,tape_hash
from v13_diagnostics.counterfactual import rollout_branch


@dataclass(frozen=True)
class FakeUpdate:
    time:float
    vessel_id:int
    new_eta:float
    update_id:int


@dataclass
class FakeScenario:
    vessels:list
    port:object
    true_etas:dict
    updates:list
    label:str
    seed:int


class FakeEnv:
    def __init__(self):
        self.event_time=1.
        self.event_index=1
        self.events=[(.5,0,0,'eta',FakeUpdate(.5,1,2.5,0)),
                     (2.,1,1,'clock',None),(3.,0,2,'eta',FakeUpdate(3.,2,6.8,2))]
        self.scenario=FakeScenario([SimpleNamespace(id=0,eta0=.7),
            SimpleNamespace(id=1,eta0=4.),SimpleNamespace(id=2,eta0=8.)],
            object(),{0:.5,1:5.,2:9.},[], 'shock',42)
        self.runtime=SimpleNamespace(started={0:object()})
        self.published_eta={0:.5,1:5.,2:8.}


class BranchEnv:
    def __init__(self):self.i=0;self.deltas=[]
    def _observation(self):
        return SimpleNamespace(mask=[True,True])
    def step(self,action):
        self.deltas.append(action)
        self.i+=1
        return SimpleNamespace(mask=[True,True]), float(-2-action),self.i==3,{}
    def point_greedy_action(self):return 1


class PairingTests(unittest.TestCase):
    def test_pre_generated_tape_is_reproducible(self):
        fake=types.ModuleType('iamu_dynamic.core');fake.Update=FakeUpdate;fake.Scenario=FakeScenario
        with patch.dict(sys.modules,{'iamu_dynamic':types.ModuleType('iamu_dynamic'),
                                      'iamu_dynamic.core':fake}):
            env=FakeEnv()
            tape=draw_tape(env,'probe:3',2,910000,uncertainty=2.,updates_per_vessel=2)
            self.assertEqual(tape,tape_hash and draw_tape(env,'probe:3',2,910000,2.,2))
            self.assertEqual(tape.true_etas[0],.5)
            a=apply_tape(env,tape);b=apply_tape(env,tape)
            self.assertEqual(a.events,b.events)
            self.assertEqual(a.scenario.true_etas,b.scenario.true_etas)
            self.assertEqual(len(env.events),3)
            self.assertNotEqual(id(a),id(b))
            self.assertTrue(all(x[0]>1. for x in a.events[1:]))
            self.assertEqual(tape_hash(tape),tape_hash(draw_tape(env,'probe:3',2,910000,2.,2)))

    def test_two_branches_same_exogenous_tape_different_reward(self):
        a=BranchEnv();b=copy.deepcopy(a)
        keep,_=rollout_branch(a,0,'greedy',gamma=.9)
        edit,_=rollout_branch(b,1,'greedy',gamma=.9)
        self.assertAlmostEqual(keep,-2-.9*3-.9**2*3)
        self.assertAlmostEqual(edit,-3-.9*3-.9**2*3)
        self.assertAlmostEqual(edit-keep,-1.)

    def test_pending_same_timestamp_event_is_not_lost(self):
        fake=types.ModuleType('iamu_dynamic.core');fake.Update=FakeUpdate;fake.Scenario=FakeScenario
        with patch.dict(sys.modules,{'iamu_dynamic':types.ModuleType('iamu_dynamic'),
                                      'iamu_dynamic.core':fake}):
            env=FakeEnv()
            env.event_time=1.
            env.events=[(1.,0,0,'eta',FakeUpdate(1.,1,3.,0)),
                        (1.,0,1,'eta',FakeUpdate(1.,2,7.,1)),
                        (1.,1,1,'clock',None),(2.,1,2,'clock',None)]
            env.event_index=1 # First ETA at t=1 already consumed
            tape=draw_tape(env,'same-timestamp',0,910000,updates_per_vessel=2)
            copy_env=apply_tape(env,tape)
            self.assertEqual(copy_env.events[:3],env.events[:3])
            self.assertEqual(copy_env.event_index,1)

    def test_balanced_jobs_no_repeated_external_seed(self):
        jobs=balanced_jobs(200,['one','two'],['shock','repeated','small'],[2,5,10],7)
        self.assertEqual(len({x['external_seed'] for x in jobs}),200)
        self.assertEqual(len({x['episode_id'] for x in jobs}),200)
        counts={}
        for job in jobs:
            k=job['scenario'],job['budget'];counts[k]=counts.get(k,0)+1
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_state_id_is_stable(self):
        self.assertEqual(state_id(17,9),'17:9')
        self.assertEqual(stable_digest({'x':1,'y':2}),stable_digest({'y':2,'x':1}))

if __name__=='__main__':unittest.main()
