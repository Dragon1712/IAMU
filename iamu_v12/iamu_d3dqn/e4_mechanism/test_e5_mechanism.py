"""Focused E5 unit tests: scientific invariants and fail-closed trainer builder."""
from __future__ import annotations
import csv,json,math,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from e4_mechanism.mechanism_core import (replay_stats,balanced_batch,
                                    deterministic_probe,assert_immutable_replay,actionable_batch)
from e4_mechanism.td_probe import td_rows,td_summary
from e4_mechanism.telemetry import eval_qgap,sha
from e4_mechanism.build_instrumented_trainer import patch
from e4_mechanism.offline_branch import read_schedule,load_fork


def tr(action,legal=True,budget=5,reward=-.5,done=False):
    obs=np.zeros(24,dtype='float32');obs[19]=budget/10;obs[20]=budget/10
    mask=np.array([True,legal],dtype=bool)
    nxt=np.array([True,legal],dtype=bool)
    return SimpleNamespace(obs_global=obs,obs_actions=np.zeros((2,24),'float32'),
        obs_mask=mask,action=action,reward=reward,
        next_global=obs.copy(),next_actions=np.zeros((2,24),'float32'),
        next_mask=nxt,done=done)


class MockQ(torch.nn.Module):
    def __init__(self,keep=-1,edit=0):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([float(keep),float(edit)]))
    def forward(self,g,a,mask):
        return self.weight.unsqueeze(0).expand(g.shape[0],-1)


class MechanismTests(unittest.TestCase):
    def test_replay_forced_keep_separate(self):
        r=[tr(0,False) for _ in range(30)]+[tr(0) for _ in range(5)]+[tr(1) for _ in range(5)]
        s=replay_stats(r)
        self.assertEqual(s['n_forced_keep'],30)
        self.assertAlmostEqual(s['keep_fraction_all'],.875)
        self.assertAlmostEqual(s['keep_fraction_actionable'],.5)
        self.assertEqual(s['b5_n'],40)

    def test_invalid_action_rejected(self):
        with self.assertRaises(AssertionError):replay_stats([tr(1,False)])

    def test_balanced_sampling_and_unique(self):
        items=[tr(0) for _ in range(100)]+[tr(1) for _ in range(6)]+[tr(0,False) for _ in range(100)]
        batch,n=balanced_batch(items,64,random.Random(37))
        self.assertEqual(len(batch),64)
        self.assertEqual(sum(t.action==1 for t in batch),32)
        self.assertLess(n,64)  # Edit oversampled with replacement.
        with self.assertRaises(ValueError):balanced_batch([tr(0)]*10,64,random.Random(7))

    def test_actionable_uniform_preserves_natural_choice_ratio(self):
        items=[tr(0) for _ in range(90)]+[tr(1) for _ in range(10)]+[tr(0,False) for _ in range(900)]
        b,n=actionable_batch(items,100,random.Random(3))
        self.assertEqual(n,100)
        self.assertEqual(sum(t.action!=0 for t in b),10)
        self.assertTrue(all(t.obs_mask[1:].any() for t in b))

    def test_probe_does_not_consume_training_replay_rng(self):
        from collections import deque
        items=[tr(i%2) for i in range(50)]
        state=random.Random(2026).getstate()
        out=deterministic_probe(items,k=20)
        self.assertEqual(len(out),20)
        self.assertEqual(random.Random(2026).getstate(),state)

    def test_td_residual_matches_original_mask_margin_and_done(self):
        online=MockQ(keep=0,edit=2)
        target=MockQ(keep=4,edit=3)
        learner=SimpleNamespace(online=online,target=target,device=torch.device('cpu'),
                                gamma=1.,keep_margin=3.)
        x=[tr(1,reward=-1),tr(0,reward=-2,done=True)]
        rows=td_rows(learner,x,batch_size=2)
        # At next state online chooses Keep: 0+3=3 > Edit 2.
        # target Keep=4, while selected Edit current Q=2.
        self.assertAlmostEqual(rows[0]['target'],3.)
        self.assertAlmostEqual(rows[0]['delta'],1.)
        self.assertAlmostEqual(rows[1]['delta'],-2.)
        self.assertEqual(rows[0]['kind'],'edit')
        self.assertEqual(td_summary(rows)[0]['n'],2)

    def test_qgap_respects_masks_and_margin(self):
        net=MockQ(keep=-2,edit=-1.8)
        learner=SimpleNamespace(online=net,device=torch.device('cpu'),keep_margin=.1)
        bank={'g':np.zeros((2,24),'float32'),
              'actions':np.zeros((2,2,24),'float32'),
              'mask':np.array([[True,True],[True,True]])}
        score=eval_qgap(learner,bank)
        self.assertAlmostEqual(score['mean_best_q_gap'],.2,places=6)
        self.assertAlmostEqual(score['fraction_keep_with_legal'],0.)
        learner.keep_margin=.3
        score=eval_qgap(learner,bank)
        self.assertAlmostEqual(score['fraction_keep_with_legal'],1.)

    def test_schedule_matched_per_completed_episode(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'train.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['completed','learner_updates','gradient_steps']);w.writeheader()
                for i,n,grad in [(2,2,12),(3,5,17),(4,0,17)]:w.writerow(dict(completed=i,learner_updates=n,gradient_steps=grad))
            self.assertEqual(read_schedule(p,2,4,12),[(3,5),(4,0)])
            with self.assertRaises(ValueError):read_schedule(p,2,4,100)

    def test_builder_fails_closed_and_preserves_original(self):
        source='''from parallel_recovery_v081 import atomic_json, save_state, load_state

def _arguments():
    p.add_argument('--out',default='foo')
    return p.parse_args()


def _atomic_csv(x):pass

def main():
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        # Validation callbacks that arrived after the LAST full state
        pass
    completed=len(completed_ids)
    # Do not send a single episode beyond
    pass
                persist(save_full=True,reason='validation-finished')
            # Realtime CSV: no longer wait for hundreds of CPU validation runs.
'''
        m=patch(source)
        for target in ('MechanismMonitor','e5_monitor.record','e5_monitor.archive_fork',
                       'e5_monitor.reconcile','e5-fork-episode','e5-probe-bank'):
            self.assertIn(target,m)
        self.assertNotIn('e5_monitor.record',source)
        with self.assertRaises(RuntimeError):patch(source.replace('return p.parse_args()','return p.parse_known_args()'))

    def test_hash_fail_closed_on_full_state(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'full_state_ep0200.pt'
            torch.save(dict(version=1,gradient_steps=6,train_rows=[{}]*200,
                replay_items=[tr(0)],validations={'val_000200':{'status':'done','snapshot':'foo'}},
                online={}),p)
            p.with_suffix('.metadata.json').write_text(json.dumps(dict(
                full_state_sha256='a'*64,completed=200,replay_size=1,gradient_steps=6,
                source_out=t,validation_sha256='b'*64)))
            with self.assertRaises(PermissionError):load_fork(p,must_trust=False)
            with self.assertRaises(ValueError):load_fork(p,must_trust=True)


if __name__=='__main__':unittest.main()
