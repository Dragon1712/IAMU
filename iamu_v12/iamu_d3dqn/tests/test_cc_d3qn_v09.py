"""CC-D3QN V0.9-A regression/architecture tests.

Run at IAMU_dynamic root: python -m unittest tests.test_cc_d3qn_v09 -v
Requires the real IAMU source package. Uses no real port simulations or GPU.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iamu_dynamic.rl_ddqn import (
    CandidateDuelingQNet, CandidateQNet, DoubleDQNAgent,
    ReplayBuffer, Transition,
)
from iamu_dynamic.rl_env import GLOBAL_DIM, ACTION_DIM, RLObservation
from parallel_recovery_v081 import config_digest, load_state, save_state


class CandidateDuelingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.g = torch.randn(3, GLOBAL_DIM)
        self.a = torch.randn(3, 5, ACTION_DIM)
        self.mask = torch.tensor([
            [True, True, False, True, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ], dtype=torch.bool)
        self.net = CandidateDuelingQNet(hidden_dim=24)

    def test_parameters_and_shapes(self):
        full = CandidateDuelingQNet(hidden_dim=96)
        baseline = CandidateQNet(hidden_dim=96)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 23522)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 14113)
        v,adv,q = self.net.forward_components(self.g,self.a,self.mask)
        self.assertEqual(tuple(v.shape),(3,1))
        self.assertEqual(tuple(adv.shape),(3,5))
        self.assertEqual(tuple(q.shape),(3,5))
        self.assertTrue(torch.isfinite(q).all().item())

    def test_value_matches_masked_mean(self):
        v,adv,q = self.net.forward_components(self.g,self.a,self.mask)
        legal = self.mask.to(q.dtype)
        mean_q=(q*legal).sum(1,keepdim=True)/legal.sum(1,keepdim=True)
        torch.testing.assert_close(mean_q,v,rtol=1e-5,atol=1e-6)

    def test_invalid_fill_invariance(self):
        before=self.net(self.g,self.a,self.mask)
        edited=self.a.clone()
        edited[~self.mask] = 99_999.0
        after=self.net(self.g,edited,self.mask)
        torch.testing.assert_close(before[self.mask],after[self.mask],rtol=0,atol=0)

    def test_candidate_permutation_equivariance(self):
        perm=[0,3,2,1,4]
        original=self.net(self.g,self.a,self.mask)
        swapped=self.net(self.g,self.a[:,perm],self.mask[:,perm])
        torch.testing.assert_close(original[:,perm],swapped,rtol=0,atol=1e-6)

    def test_only_keep_legal_and_advantage_gradient(self):
        self.net.zero_grad(set_to_none=True)
        v,a,q=self.net.forward_components(self.g[1:2],self.a[1:2],self.mask[1:2])
        torch.testing.assert_close(q[:,0:1],v,rtol=0,atol=0)
        q[0,0].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in self.net.value_head.parameters()))
        self.assertTrue(all(p.grad is None or p.grad.abs().sum()==0
                            for p in self.net.advantage_head.parameters()))

    def test_bad_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            self.net(self.g,self.a,None)
        with self.assertRaises(TypeError):
            self.net(self.g,self.a,self.mask.float())
        bad=self.mask.clone();bad[1,0]=False
        with self.assertRaises(ValueError):
            self.net(self.g,self.a,bad)
        with self.assertRaises(ValueError):
            self.net(self.g,self.a,self.mask[:,:-1])

    def test_gradient_is_finite_for_mixed_masks(self):
        _,_,q=self.net.forward_components(self.g,self.a,self.mask)
        loss=(q[self.mask]**2).mean()
        loss.backward()
        for p in self.net.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())


class AgentCompatibilityTests(unittest.TestCase):
    def sample(self,action=1):
        mask=np.array([True,True,False,False,False],dtype=np.bool_)
        nxtmask=np.array([True,False,True,False,False],dtype=np.bool_)
        return Transition(
            np.random.default_rng(1).normal(size=GLOBAL_DIM).astype('float32'),
            np.random.default_rng(2).normal(size=(5,ACTION_DIM)).astype('float32'),
            mask,action,-0.25,
            np.random.default_rng(3).normal(size=GLOBAL_DIM).astype('float32'),
            np.random.default_rng(4).normal(size=(5,ACTION_DIM)).astype('float32'),
            nxtmask,False,
        )

    def test_original_ddqn_outputs_exactly_unchanged(self):
        agent=DoubleDQNAgent(hidden_dim=32,seed=100,q_network='ddqn')
        g=torch.randn(2,GLOBAL_DIM)
        a=torch.randn(2,5,ACTION_DIM)
        mask=torch.tensor([[1,1,0,0,0],[1,0,1,0,0]],dtype=torch.bool)
        legacy=agent.online(g,a)
        new=agent.online(g,a,mask)
        torch.testing.assert_close(legacy,new,rtol=0,atol=0)

    def test_d3qn_update_and_target_sync(self):
        agent=DoubleDQNAgent(hidden_dim=24,seed=101,q_network='d3qn',target_update_steps=1)
        before={k:v.detach().clone() for k,v in agent.online.state_dict().items()}
        loss=agent.update([self.sample() for _ in range(4)])
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.gradient_steps,1)
        self.assertTrue(any(not torch.equal(v,agent.online.state_dict()[k]) for k,v in before.items()))
        for k,v in agent.online.state_dict().items():
            torch.testing.assert_close(v,agent.target.state_dict()[k],rtol=0,atol=0)

    def test_illegal_replayed_action_rejected(self):
        agent=DoubleDQNAgent(hidden_dim=24,seed=102,q_network='d3qn')
        with self.assertRaises(ValueError):
            agent.update([self.sample(action=3)])

    def test_d3qn_checkpoint_and_old_ddqn_checkpoint(self):
        for architecture in ('d3qn','ddqn'):
            agent=DoubleDQNAgent(hidden_dim=24,seed=104,q_network=architecture,keep_margin=.1)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'best.pt'
                item=agent.checkpoint({'selected_episode':25})
                if architecture=='ddqn':
                    # A real pre-V0.9 checkpoint has no network type fields.
                    item.pop('q_network')
                    item.pop('network_architecture_version')
                torch.save(item,path)
                loaded,meta=DoubleDQNAgent.from_checkpoint(str(path),device='cpu')
                self.assertEqual(loaded.q_network,architecture)
                self.assertEqual(meta['selected_episode'],25)
                for k,v in agent.online.state_dict().items():
                    torch.testing.assert_close(loaded.online.state_dict()[k],v,rtol=0,atol=0)

    def test_d3qn_actor_action_with_mask(self):
        agent=DoubleDQNAgent(hidden_dim=24,seed=105,q_network='d3qn',keep_margin=.1)
        t=self.sample()
        ob=RLObservation(t.obs_global,t.obs_actions,t.obs_mask,0.,'eta',2,(1,))
        for _ in range(10):
            action=agent.act(ob,epsilon=0.)
            self.assertTrue(t.obs_mask[action])

    def test_resume_config_and_optimizer(self):
        old={'training_args':{'episodes':8,'device':'cpu','out':'old'}}
        updated={'training_args':{'episodes':8,'device':'cpu','out':'new','q_network':'ddqn'}}
        d3={'training_args':{'episodes':8,'device':'cpu','out':'new','q_network':'d3qn'}}
        self.assertEqual(config_digest(old),config_digest(updated))
        self.assertNotEqual(config_digest(old),config_digest(d3))
        agent=DoubleDQNAgent(hidden_dim=24,seed=106,q_network='d3qn')
        replay=ReplayBuffer(30,seed=30)
        for _ in range(6):replay.items.append(self.sample())
        agent.update(replay.sample(4))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'training_state.pt'
            save_state(path,learner=agent,replay=replay,config=d3,
                       train_rows=[{'episode':1}],val_rows=[],validations={},
                       best_val=float('inf'),gradient_credit=0.)
            new_agent=DoubleDQNAgent(hidden_dim=24,seed=999,q_network='d3qn')
            new_replay=ReplayBuffer(30,seed=999)
            restored=load_state(path,learner=new_agent,replay=new_replay,config=d3)
            self.assertEqual(new_agent.gradient_steps,1)
            self.assertEqual(len(new_replay),6)
            self.assertEqual(restored['train_rows'][0]['episode'],1)
            self.assertEqual(agent.optimizer.state_dict()['state'].keys(),
                             new_agent.optimizer.state_dict()['state'].keys())
            with self.assertRaises(ValueError):
                load_state(path,learner=DoubleDQNAgent(hidden_dim=24,seed=7),
                           replay=ReplayBuffer(30,seed=7),config=updated)


if __name__=='__main__':
    unittest.main()
