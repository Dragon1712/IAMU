"""V0.9-B Set-aware DDQN: mathematical, agent and checkpoint regression tests.

From the iamu_d3dqn directory:
    python -m unittest tests.test_set_aware_ddqn_v09b -v
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from iamu_dynamic.rl_env import ACTION_DIM, GLOBAL_DIM, RLObservation
from iamu_dynamic.rl_ddqn import (
    CandidateQNet, CandidateDuelingQNet, CandidateSetQNet,
    MaskedCandidateSetEncoder, DoubleDQNAgent, ReplayBuffer, Transition,
)
from parallel_recovery_v081 import config_digest, save_state, load_state


class SetEncoderMathematicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.g = torch.randn(3, GLOBAL_DIM)
        self.x = torch.randn(3, 5, ACTION_DIM)
        self.mask = torch.tensor([
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ], dtype=torch.bool)
        self.net = CandidateSetQNet(96)

    def test_parameter_count_and_shape(self):
        self.assertEqual(sum(p.numel() for p in self.net.parameters()), 20865)
        self.assertEqual(sum(p.numel() for p in CandidateQNet(96).parameters()), 14113)
        self.assertEqual(sum(p.numel() for p in CandidateDuelingQNet(96).parameters()), 23522)
        h, z, q = self.net.forward_components(self.g, self.x, self.mask)
        self.assertEqual(tuple(h.shape), (3, 96))
        self.assertEqual(tuple(z.shape), (3, 32))
        self.assertEqual(tuple(q.shape), (3, 5))
        self.assertTrue(torch.isfinite(q).all().item())

    def test_set_encoding_is_permutation_invariant(self):
        perm = [0, 3, 2, 1, 4]  # Keep stays in slot 0
        before = self.net.set_encoder(self.x, self.mask)
        after = self.net.set_encoder(self.x[:, perm], self.mask[:, perm])
        torch.testing.assert_close(before, after, rtol=0, atol=1e-6)

    def test_q_values_follow_candidate_permutation(self):
        perm = [0, 3, 2, 1, 4]
        before = self.net(self.g, self.x, self.mask)
        after = self.net(self.g, self.x[:, perm], self.mask[:, perm])
        torch.testing.assert_close(before[:, perm], after, rtol=0, atol=1e-6)

    def test_invalid_fill_does_not_affect_legal_q(self):
        a = self.x.clone()
        a[~self.mask] = 100_000.0
        reference = self.net(self.g, self.x, self.mask)
        changed = self.net(self.g, a, self.mask)
        torch.testing.assert_close(reference[self.mask], changed[self.mask], rtol=0, atol=0)
        z0 = self.net.set_encoder(self.x, self.mask)
        z1 = self.net.set_encoder(a, self.mask)
        torch.testing.assert_close(z0, z1, rtol=0, atol=0)
        # Invalid slots must not contaminate legal outputs even when corrupt.
        corrupted = self.x.clone()
        corrupted[~self.mask] = float('inf')
        q_corrupt = self.net(self.g, corrupted, self.mask)
        torch.testing.assert_close(reference[self.mask], q_corrupt[self.mask],
                                   rtol=0, atol=0)

    def test_keep_only_has_zero_set_context(self):
        _, z, q = self.net.forward_components(self.g, self.x, self.mask)
        torch.testing.assert_close(z[1], torch.zeros_like(z[1]), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(q[1]).all().item())
        # A single Keep slot (N=1) is a valid model input as well.
        one_mask = torch.ones((1, 1), dtype=torch.bool)
        one_q = self.net(self.g[:1], self.x[:1, :1], one_mask)
        self.assertEqual(tuple(one_q.shape), (1, 1))
        self.assertTrue(torch.isfinite(one_q).all().item())

    def test_no_edit_gradient_to_set_encoder(self):
        net = CandidateSetQNet(24)
        m = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
        q = net(self.g[:1], self.x[:1], m)
        q[0, 0].backward()
        self.assertTrue(all(
            p.grad is None or p.grad.abs().sum().item() == 0
            for p in net.set_encoder.parameters()
        ))

    def test_valid_edit_trains_set_encoder(self):
        net = CandidateSetQNet(24)
        q = net(self.g[:1], self.x[:1], self.mask[:1])
        q[0, 1].backward()
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in net.set_encoder.parameters()
        ))
        for p in net.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_context_changes_as_other_edit_changes(self):
        changed = self.x.clone()
        changed[0, 3] += 3.0
        z0 = self.net.set_encoder(self.x, self.mask)
        z1 = self.net.set_encoder(changed, self.mask)
        self.assertFalse(torch.allclose(z0[0], z1[0]))
        # Keep feature unchanged, but set context lets its value respond.
        q0 = self.net(self.g, self.x, self.mask)
        q1 = self.net(self.g, changed, self.mask)
        self.assertFalse(torch.allclose(q0[0, :1], q1[0, :1]))

    def test_mask_validation(self):
        with self.assertRaises(TypeError):
            self.net(self.g, self.x, self.mask.float())
        with self.assertRaises(ValueError):
            self.net(self.g, self.x, None)
        invalid = self.mask.clone()
        invalid[0, 0] = False
        with self.assertRaises(ValueError):
            self.net(self.g, self.x, invalid)
        with self.assertRaises(ValueError):
            self.net(self.g, self.x, self.mask[:, :-1])
        with self.assertRaises(ValueError):
            self.net(self.g, self.x[:, :0], self.mask[:, :0])


class SetAgentIntegrationTests(unittest.TestCase):
    def make_transition(self, action=1):
        rng = np.random.default_rng(34)
        mask = np.array([1, 1, 0, 1, 0], dtype=np.bool_)
        nextmask = np.array([1, 0, 1, 0, 0], dtype=np.bool_)
        return Transition(
            rng.normal(size=GLOBAL_DIM).astype('float32'),
            rng.normal(size=(5, ACTION_DIM)).astype('float32'),
            mask, action, -.5,
            rng.normal(size=GLOBAL_DIM).astype('float32'),
            rng.normal(size=(5, ACTION_DIM)).astype('float32'),
            nextmask, False,
        )

    def test_ddqn_and_dueling_modes_preserved(self):
        for net_type, network_cls in [('ddqn', CandidateQNet),
                                      ('d3qn', CandidateDuelingQNet),
                                      ('set_ddqn', CandidateSetQNet)]:
            agent = DoubleDQNAgent(hidden_dim=32, seed=8, q_network=net_type)
            self.assertIsInstance(agent.online, network_cls)
            self.assertIsInstance(agent.target, network_cls)

    def test_action_mask_and_training_target_sync(self):
        agent = DoubleDQNAgent(hidden_dim=24, seed=9,
                                q_network='set_ddqn', keep_margin=.1,
                                target_update_steps=1)
        t = self.make_transition()
        observation = RLObservation(t.obs_global, t.obs_actions, t.obs_mask,
                                    0., 'eta', 2, (1, 3))
        for _ in range(5):
            chosen = agent.act(observation, epsilon=0.)
            self.assertTrue(observation.mask[chosen])
        first = {k: x.clone() for k, x in agent.online.state_dict().items()}
        loss = agent.update([self.make_transition() for _ in range(4)])
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.gradient_steps, 1)
        self.assertTrue(any(not torch.equal(first[k], p)
                            for k, p in agent.online.state_dict().items()))
        for k, p in agent.online.state_dict().items():
            torch.testing.assert_close(p, agent.target.state_dict()[k], rtol=0, atol=0)

    def test_illegal_replay_action_rejected(self):
        agent = DoubleDQNAgent(hidden_dim=24, seed=11, q_network='set_ddqn')
        with self.assertRaises(ValueError):
            agent.update([self.make_transition(action=2)])

    def test_checkpoint_backward_compatibility(self):
        for net_type in ('ddqn', 'd3qn', 'set_ddqn'):
            agent = DoubleDQNAgent(hidden_dim=24, q_network=net_type, seed=12, keep_margin=.1)
            saved = agent.checkpoint({'selected_episode': 25})
            self.assertEqual(saved['network_architecture_version'],
                             2 if net_type == 'set_ddqn' else 1)
            if net_type == 'ddqn':
                saved.pop('q_network')
                saved.pop('network_architecture_version')
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'best.pt'
                torch.save(saved, path)
                loaded, metadata = DoubleDQNAgent.from_checkpoint(str(path))
                self.assertEqual(loaded.q_network, net_type)
                self.assertEqual(metadata['selected_episode'], 25)
                for key, value in agent.online.state_dict().items():
                    torch.testing.assert_close(loaded.online.state_dict()[key], value,
                                               rtol=0, atol=0)

    def test_checkpoint_wrong_version_rejected(self):
        agent = DoubleDQNAgent(hidden_dim=24, q_network='set_ddqn')
        state = agent.checkpoint({})
        state['network_architecture_version'] = 1
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'broken.pt'
            torch.save(state, path)
            with self.assertRaises(ValueError):
                DoubleDQNAgent.from_checkpoint(str(path))

    def test_resume_rejects_cross_architecture(self):
        config = {'training_args': {'episodes': 3, 'q_network': 'set_ddqn',
                                    'out': 'temporary', 'device': 'cpu'}}
        ddqn_config = {'training_args': {'episodes': 3, 'q_network': 'ddqn',
                                         'out': 'temporary', 'device': 'cpu'}}
        self.assertNotEqual(config_digest(config), config_digest(ddqn_config))
        source = DoubleDQNAgent(hidden_dim=24, seed=13, q_network='set_ddqn')
        source.update([self.make_transition() for _ in range(4)])
        replay = ReplayBuffer(30, seed=5)
        for _ in range(5):
            replay.items.append(self.make_transition())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'training_state.pt'
            save_state(path, learner=source, replay=replay, config=config,
                       train_rows=[{'episode': 1}], val_rows=[],
                       validations={}, best_val=float('inf'), gradient_credit=0.)
            restored = DoubleDQNAgent(hidden_dim=24, seed=14, q_network='set_ddqn')
            new_replay = ReplayBuffer(30, seed=6)
            load_state(path, learner=restored, replay=new_replay, config=config)
            self.assertEqual(restored.gradient_steps, 1)
            self.assertEqual(len(new_replay), 5)
            with self.assertRaises(ValueError):
                load_state(path,
                           learner=DoubleDQNAgent(hidden_dim=24, q_network='ddqn'),
                           replay=ReplayBuffer(30, 1), config=ddqn_config)


if __name__ == '__main__':
    unittest.main()
