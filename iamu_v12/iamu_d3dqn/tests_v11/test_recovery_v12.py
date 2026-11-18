"""End-to-end short round-state save/load without the IAMU simulator."""
from __future__ import annotations
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from v11_protocol.indexed_replay import IndexedReplay
from v11_protocol.recovery_v12 import save_state,load_state

@dataclass
class DemoTransition:
    obs_mask: np.ndarray
    action: int
    episode_id: int

class Learner:
    def __init__(self,seed):
        torch.manual_seed(seed)
        self.online=torch.nn.Linear(2,1)
        self.target=torch.nn.Linear(2,1)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=1e-3)
        self.gradient_steps=0
        self.rng=random.Random(seed)

class TestStrictResumeV12(unittest.TestCase):
    def test_save_restore_optimizer_rng_and_pool_sequence(self):
        config=dict(scientific_config={'candidate_screen':'positive','gamma':1.,
                                       'replay_rho':.25,'seed':43},
                    code_sha256={'a':'hash'},data_sha256={'case1':'hash'},
                    task_list_sha256='tasks',runtime_config={'resume':False})
        a=Learner(5)
        replay=IndexedReplay(17,11)
        for i in range(51):
            legal=np.array([True, i%5!=0],dtype=bool)
            replay.items.append(DemoTransition(legal, int(i%5!=0 and i%2==0),i))
        for _ in range(3):
            x=torch.ones((4,2))
            loss=a.online(x).square().mean();a.optimizer.zero_grad();loss.backward();a.optimizer.step()
            a.gradient_steps+=1
        a.target.load_state_dict(a.online.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full_state.pt'
            save_state(path,learner=a,replay=replay,config=config,
                       train_rows=[{'episode':1}],val_rows=[],validations={},
                       best_val=10.,gradient_credit=.25,
                       protocol_state={'observed_raw_steps':51})
            b=Learner(500)
            replay_b=IndexedReplay(17,500)
            saved=load_state(path,learner=b,replay=replay_b,config=config)
            self.assertEqual(saved['protocol_state']['observed_raw_steps'],51)
            self.assertEqual(a.gradient_steps,b.gradient_steps)
            for x,y in zip(a.online.parameters(),b.online.parameters()):
                self.assertTrue(torch.equal(x,y))
            self.assertEqual(a.optimizer.state_dict()['state'].keys(),
                             b.optimizer.state_dict()['state'].keys())
            for _ in range(20):
                left,_,_=replay.sample_stratified(8,.25)
                right,_,_=replay_b.sample_stratified(8,.25)
                self.assertEqual([t.episode_id for t in left],
                                 [t.episode_id for t in right])
            mutated=dict(config,scientific_config={'gamma':.9})
            with self.assertRaises(ValueError):
                load_state(path,learner=b,replay=replay_b,config=mutated)

    def test_continuous_vs_resume_training_parameters(self):
        config=dict(scientific_config={'gamma':1.0,'seed':17,'round_size':6},
                    code_sha256={'code':'stable'}, data_sha256={'data':'stable'},
                    task_list_sha256='same-tasks', runtime_config={'resume':False})
        def consume(agent, replay, beginning, end):
            for event in range(beginning,end):
                legal=np.array([True, event%3!=0])
                replay.items.append(DemoTransition(legal,int(bool(legal[1]) and event%2==1),event))
                if len(replay)<8: continue
                selected,_,_=replay.sample_stratified(8,.25)
                # Synthetic deterministic update whose gradient depends on sampled IDs.
                x=torch.tensor([[float(t.episode_id%13)/13.,1.] for t in selected])
                labels=torch.tensor([[float(t.action)] for t in selected])
                loss=(agent.online(x)-labels).square().mean()
                agent.optimizer.zero_grad(); loss.backward(); agent.optimizer.step()
                agent.gradient_steps+=1
                if agent.gradient_steps%10==0:
                    agent.target.load_state_dict(agent.online.state_dict())
        uninterrupted=Learner(17)
        r0=IndexedReplay(19,808)
        consume(uninterrupted,r0,0,60)
        interrupted=Learner(17)
        r1=IndexedReplay(19,808)
        consume(interrupted,r1,0,30)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'round_state.pt'
            save_state(path,learner=interrupted,replay=r1,config=config,
                       train_rows=[{'episode':30}],val_rows=[],validations={},
                       best_val=0.,gradient_credit=0.,
                       protocol_state={'observed_raw_steps':30})
            resumed=Learner(999)
            r2=IndexedReplay(19,999)
            load_state(path,learner=resumed,replay=r2,
                       config=dict(config,runtime_config={'resume':True}))
            consume(resumed,r2,30,60)
        self.assertEqual(uninterrupted.gradient_steps,resumed.gradient_steps)
        for lhs,rhs in zip(uninterrupted.online.parameters(),resumed.online.parameters()):
            self.assertTrue(torch.equal(lhs,rhs))
        for lhs,rhs in zip(uninterrupted.target.parameters(),resumed.target.parameters()):
            self.assertTrue(torch.equal(lhs,rhs))
        self.assertEqual([x.episode_id for x in r0.items],
                         [x.episode_id for x in r2.items])
        self.assertEqual(r0.actionable.values,r2.actionable.values)
        self.assertEqual(r0.forced.values,r2.forced.values)
        for _ in range(8):
            lhs,_,_=r0.sample_stratified(8,.25)
            rhs,_,_=r2.sample_stratified(8,.25)
            self.assertEqual([x.episode_id for x in lhs],
                             [x.episode_id for x in rhs])

if __name__=='__main__':unittest.main()
