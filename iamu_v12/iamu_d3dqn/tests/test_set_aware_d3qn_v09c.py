"""V0.9-C Set-aware D3QN tests. Run: python -m unittest tests.test_set_aware_d3qn_v09c -v"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iamu_dynamic.rl_env import GLOBAL_DIM, ACTION_DIM, RLObservation
from iamu_dynamic.rl_ddqn import (
    CandidateSetDuelingQNet, CandidateSetQNet, CandidateDuelingQNet,
    CandidateQNet, DoubleDQNAgent, ReplayBuffer, Transition,
)
from parallel_recovery_v081 import config_digest, load_state, save_state


class SetDuelingMathematicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.g = torch.randn(3, GLOBAL_DIM)
        self.x = torch.randn(3, 5, ACTION_DIM)
        self.mask = torch.tensor([
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ], dtype=torch.bool)
        self.net = CandidateSetDuelingQNet(96)

    def test_param_count_and_output_shapes(self):
        self.assertEqual(sum(p.numel() for p in self.net.parameters()), 33346)
        v, adv, q = self.net.forward_components(self.g, self.x, self.mask)
        self.assertEqual(tuple(v.shape), (3, 1))
        self.assertEqual(tuple(adv.shape), (3, 5))
        self.assertEqual(tuple(q.shape), (3, 5))
        self.assertTrue(torch.isfinite(q).all().item())

    def test_advantage_mean_includes_keep(self):
        v, a, q = self.net.forward_components(self.g, self.x, self.mask)
        weights = self.mask.to(q.dtype)
        q_mean = (q * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
        torch.testing.assert_close(q_mean, v, rtol=1e-5, atol=1e-6)
        # Keep advantage is included in the dueling mean even though Keep is
        # excluded from the set encoder.
        for b in (0, 2):
            expected = v[b, 0] + a[b, 0] - a[b, self.mask[b]].mean()
            torch.testing.assert_close(q[b, 0], expected)

    def test_set_context_excludes_keep(self):
        before = self.net.set_encoder(self.x, self.mask)
        x = self.x.clone()
        x[:, 0] += 1000
        after = self.net.set_encoder(x, self.mask)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_permutation_invariance_and_q_equivariance(self):
        perm = [0, 3, 2, 1, 4]
        z0 = self.net.set_encoder(self.x, self.mask)
        z1 = self.net.set_encoder(self.x[:, perm], self.mask[:, perm])
        torch.testing.assert_close(z0, z1, rtol=0, atol=1e-6)
        q0 = self.net(self.g, self.x, self.mask)
        q1 = self.net(self.g, self.x[:, perm], self.mask[:, perm])
        torch.testing.assert_close(q0[:, perm], q1, rtol=0, atol=1e-6)

    def test_padding_changes_do_not_modify_legal_q(self):
        original = self.net(self.g, self.x, self.mask)
        for padding in (100_000., float('inf'), float('nan')):
            changed = self.x.clone()
            changed[~self.mask] = padding
            q = self.net(self.g, changed, self.mask)
            torch.testing.assert_close(q[self.mask], original[self.mask], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(q[self.mask]).all().item())

    def test_keep_only_value_and_zero_set_context(self):
        v, a, q = self.net.forward_components(self.g[1:2], self.x[1:2], self.mask[1:2])
        z = self.net.set_encoder(self.x[1:2], self.mask[1:2])
        torch.testing.assert_close(z, torch.zeros_like(z), rtol=0, atol=0)
        torch.testing.assert_close(q[:, 0:1], v, rtol=1e-6, atol=1e-7)
        q_one = self.net(self.g[:1], self.x[:1, :1], torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(tuple(q_one.shape), (1, 1))

    def test_keep_only_adv_and_set_gradients_zero(self):
        small = CandidateSetDuelingQNet(24)
        loss = small(self.g[1:2], self.x[1:2], self.mask[1:2])[0, 0]
        loss.backward()
        for stream in (small.set_encoder, small.advantage_head):
            self.assertTrue(all(p.grad is None or p.grad.abs().sum().item() == 0
                                for p in stream.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for p in small.value_head.parameters()))

    def test_valid_edit_trains_both_heads_and_set_encoder(self):
        small = CandidateSetDuelingQNet(24)
        q = small(self.g[:1], self.x[:1], self.mask[:1])
        q[0, 1].backward()
        for stream in (small.state_encoder, small.set_encoder,
                       small.value_head, small.advantage_head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                                for p in stream.parameters()), stream.__class__.__name__)
            for p in stream.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_context_sensitive_q_and_action_preference(self):
        changed = self.x.clone()
        changed[0, 3] += 8.
        old_context = self.net.set_encoder(self.x, self.mask)
        new_context = self.net.set_encoder(changed, self.mask)
        self.assertFalse(torch.allclose(old_context[0], new_context[0]))
        old_q = self.net(self.g, self.x, self.mask)
        new_q = self.net(self.g, changed, self.mask)
        self.assertFalse(torch.allclose(old_q[0, :2], new_q[0, :2]))

    def test_reject_missing_wrong_mask_and_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            self.net(self.g, self.x, None)
        with self.assertRaises(TypeError):
            self.net(self.g, self.x, self.mask.float())
        invalid = self.mask.clone()
        invalid[0, 0] = False
        with self.assertRaises(ValueError):
            self.net(self.g, self.x, invalid)
        with self.assertRaises(ValueError):
            self.net(self.g, self.x, self.mask[:, :-1])
        with self.assertRaises(ValueError):
            self.net(torch.zeros(3, GLOBAL_DIM-1), self.x, self.mask)


class SetDuelingAgentTests(unittest.TestCase):
    @staticmethod
    def transition(action=1):
        rng = np.random.default_rng(42)
        mask = np.array([1, 1, 0, 1, 0], dtype=np.bool_)
        nxt_mask = np.array([1, 0, 1, 0, 0], dtype=np.bool_)
        return Transition(rng.normal(size=GLOBAL_DIM).astype('float32'),
                          rng.normal(size=(5, ACTION_DIM)).astype('float32'),
                          mask, action, -.5,
                          rng.normal(size=GLOBAL_DIM).astype('float32'),
                          rng.normal(size=(5, ACTION_DIM)).astype('float32'),
                          nxt_mask, False)

    def test_all_four_network_modes_exist(self):
        mapping = [('ddqn', CandidateQNet), ('d3qn', CandidateDuelingQNet),
                   ('set_ddqn', CandidateSetQNet),
                   ('set_d3qn', CandidateSetDuelingQNet)]
        for label, cls in mapping:
            with self.subTest(label=label):
                agent = DoubleDQNAgent(hidden_dim=24, q_network=label)
                self.assertIsInstance(agent.online, cls)
                self.assertIsInstance(agent.target, cls)

    def test_legal_actions_and_target_sync(self):
        agent = DoubleDQNAgent(hidden_dim=24, seed=8, q_network='set_d3qn',
                                keep_margin=.1, target_update_steps=1)
        t = self.transition()
        ob = RLObservation(t.obs_global, t.obs_actions, t.obs_mask, 0., 'eta', 2, (1, 3))
        for _ in range(5):
            self.assertTrue(ob.mask[agent.act(ob, epsilon=0.)])
        initial = {k: v.clone() for k, v in agent.online.state_dict().items()}
        loss = agent.update([self.transition() for _ in range(4)])
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.gradient_steps, 1)
        self.assertTrue(any(not torch.equal(initial[k], v)
                            for k, v in agent.online.state_dict().items()))
        for k, v in agent.online.state_dict().items():
            torch.testing.assert_close(v, agent.target.state_dict()[k], rtol=0, atol=0)

    def test_replay_illegal_action_rejected(self):
        agent = DoubleDQNAgent(hidden_dim=24, seed=9, q_network='set_d3qn')
        with self.assertRaises(ValueError):
            agent.update([self.transition(action=2)])

    def test_checkpoint_architecture_and_prior_versions(self):
        for label in ('ddqn', 'd3qn', 'set_ddqn', 'set_d3qn'):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                agent = DoubleDQNAgent(hidden_dim=24, seed=7, q_network=label, keep_margin=.1)
                saved = agent.checkpoint({'selected_episode': 25})
                self.assertEqual(saved['network_architecture_version'],
                                 2 if label.startswith('set_') else 1)
                if label == 'ddqn':
                    # Simulate V0.8 checkpoint without architecture field.
                    saved.pop('q_network')
                    saved.pop('network_architecture_version')
                path = Path(tmp) / 'best.pt'
                torch.save(saved, path)
                loaded, meta = DoubleDQNAgent.from_checkpoint(str(path))
                self.assertEqual(loaded.q_network, label)
                self.assertEqual(meta['selected_episode'], 25)
                for name, weight in agent.online.state_dict().items():
                    torch.testing.assert_close(weight, loaded.online.state_dict()[name],
                                               rtol=0, atol=0)

    def test_mismatched_checkpoint_version_rejected(self):
        state = DoubleDQNAgent(hidden_dim=24, q_network='set_d3qn').checkpoint({})
        state['network_architecture_version'] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wrong.pt'
            torch.save(state, path)
            with self.assertRaises(ValueError):
                DoubleDQNAgent.from_checkpoint(str(path))

    def test_learner_state_recovery_and_cross_architecture_block(self):
        a = {'training_args': {'episodes': 5, 'device': 'cpu', 'q_network': 'set_d3qn'}}
        b = {'training_args': {'episodes': 5, 'device': 'cpu', 'q_network': 'set_ddqn'}}
        self.assertNotEqual(config_digest(a), config_digest(b))
        agent = DoubleDQNAgent(hidden_dim=24, seed=7, q_network='set_d3qn')
        replay = ReplayBuffer(10, 10)
        for _ in range(6): replay.items.append(self.transition())
        agent.update(replay.sample(4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training_state.pt'
            save_state(path, learner=agent, replay=replay, config=a, train_rows=[{'episode': 1}],
                       val_rows=[], validations={}, best_val=float('inf'), gradient_credit=0.)
            restored = DoubleDQNAgent(hidden_dim=24, q_network='set_d3qn')
            restored_replay = ReplayBuffer(10, 100)
            payload = load_state(path, learner=restored, replay=restored_replay, config=a)
            self.assertEqual(restored.gradient_steps, 1)
            self.assertEqual(len(restored_replay), 6)
            self.assertEqual(payload['train_rows'][0]['episode'], 1)
            for name, value in agent.online.state_dict().items():
                torch.testing.assert_close(value, restored.online.state_dict()[name])
            with self.assertRaises(ValueError):
                load_state(path, learner=DoubleDQNAgent(hidden_dim=24, q_network='set_ddqn'),
                           replay=ReplayBuffer(10, 1), config=b)


if __name__ == '__main__':
    unittest.main()
