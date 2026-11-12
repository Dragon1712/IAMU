"""Synthetic unit/integration test; does NOT test original C++ environment."""
import os
import sys
import types
import tempfile
import shutil
import torch
import numpy as np

# Stubs for original network/environment while preserving the training API.
model_module = types.ModuleType('model')
class ES(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(16) * 0.1)
        self.b = torch.nn.Parameter(torch.zeros(4))
    def es_params(self):
        return [(k, v) for k, v in self.state_dict().items()]
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    def init_weight(self):
        with torch.no_grad():
            self.w.normal_(0, .02)
            self.b.zero_()
    def forward(self, x):
        return np.array([0., 0., 0., 0., 0.])
model_module.ES = ES
sys.modules['model'] = model_module

env_module = types.ModuleType('env')
class Bay_Env:
    def __init__(self, _): self.step_id = 0
    def reset(self): return np.zeros((30,8),dtype=np.float32)
    def step(self, action):
        self.step_id += 1
        return np.zeros((30,8),dtype=np.float32), -float(self.step_id), self.step_id>=2, None, None
    def close(self): return None
env_module.Bay_Env = Bay_Env
sys.modules['env'] = env_module

import kde_fixed
import train_fixed
import main_kde_fixed


def test_kde():
    torch.manual_seed(3)
    refs = [ES() for _ in range(120)]
    bank = kde_fixed.ReferenceBank(refs, 20 ** 0.5)
    m = ES()
    score = bank.score(m)
    assert 0 <= score <= 1
    # Correct normalization against original pairwise-distance formulation.
    x = kde_fixed.model_vector(m)
    R = torch.stack([kde_fixed.model_vector(ref) for ref in refs])
    expected = torch.exp(-0.5*(torch.nn.functional.pairwise_distance(x.unsqueeze(0),R)/(20**.5))**2).mean().item()
    assert abs(score - expected) < 5e-7, (score, expected)

    from types import SimpleNamespace
    p = SimpleNamespace(sigma=[0.05], c_cov=.02, device=torch.device('cpu'), seed_rng=np.random.default_rng(12),
                        p=[[torch.zeros_like(v) for _, v in m.es_params()]])
    slot = ES()
    seed, selected, density, tries, forced = kde_fixed.bounded_candidate(
        p, 0, slot, m, bank, train_fixed.generate_seeds_and_models,
        train_fixed.perturb_model, 4, 1e-8)
    assert forced and tries==4
    # The least-dense seed was restored to the candidate slot.
    slot2 = ES()
    train_fixed.perturb_model(p, m, slot2, seed, 0)
    assert torch.equal(kde_fixed.model_vector(slot2), kde_fixed.model_vector(slot))
    assert abs(bank.score(slot) - density) < 1e-7
    s, t, f = kde_fixed.bounded_reinitialize(slot, bank, 3, 1e-8)
    assert f and t==3 and 0 <= s <= 1
    print('KDE NORMALIZATION + FALLBACK: PASS')


def test_training_and_resume():
    import argparse
    with tempfile.TemporaryDirectory() as d:
        saved_argv = sys.argv
        try:
            os.chdir(d)
            original_checkpoint_save = train_fixed._checkpoint_save
            def capture_midpoint(*args, **kwargs):
                original_checkpoint_save(*args, **kwargs)
                if args[2] == 2:
                    cp = os.path.join(args[0].folder_path, 'checkpoint_full_latest.pt')
                    shutil.copy2(cp, os.path.join(args[0].folder_path, 'midpoint.pt'))
            train_fixed._checkpoint_save = capture_midpoint
            sys.argv=['main_kde_fixed.py', '--cases', '3', '--populations', '2',
                      '--lambda-size', '4', '--kde-r1', '0.00001',
                      '--kde-r2', '0.00001', '--kde-max-attempts', '3',
                      '--kde-restart-attempts', '2', '--full-checkpoint-every', '1',
                      '--output-prefix', 'test']
            main_kde_fixed.main()
            check = os.path.join(d,'test_checkpoints','0','checkpoint_full_latest.pt')
            assert os.path.isfile(check)
            c = torch.load(check,map_location='cpu',weights_only=False)
            assert c['next_iter']==3
            assert len(c['history_theta'])==2
            assert all(len(h)>0 for h in c['history_theta'])
            print('THREE-ITERATION FULL TRAINING + CHECKPOINT: PASS')
            # Resume from an intermediate checkpoint rather than a completed one.
            mid = os.path.join(d,'test_checkpoints','0','midpoint.pt')
            assert os.path.exists(mid)
            train_fixed._checkpoint_save = original_checkpoint_save
            sys.argv=['main_kde_fixed.py', '--cases', '3', '--populations', '2',
                      '--lambda-size', '4', '--kde-r1', '0.00001',
                      '--kde-r2', '0.00001', '--kde-max-attempts', '3',
                      '--kde-restart-attempts', '2', '--full-checkpoint-every', '1',
                      '--resume', mid]
            main_kde_fixed.main()
            after = torch.load(os.path.join(d,'test_checkpoints','0','checkpoint_full_latest.pt'), map_location='cpu',weights_only=False)
            assert after['next_iter'] == 3
            print('MID-TRAINING FULL-STATE RESUME: PASS')
        finally:
            train_fixed._checkpoint_save = original_checkpoint_save
            sys.argv=saved_argv


if __name__=='__main__':
    test_kde()
    test_training_and_resume()
    print('ALL SYNTHETIC TESTS PASSED')
