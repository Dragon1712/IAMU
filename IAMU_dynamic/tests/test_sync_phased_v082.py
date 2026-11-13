"""V0.8.2 phase-synchronous acceptance: all 216 comparisons, no rollouts during validation.

Usage: python -m unittest tests.test_sync_phased_v082 -v
Requires real IAMU_dynamic package; uses only tiny, generated case files.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from parallel_validation_v081 import file_sha256


class SynchronousPhaseTests(unittest.TestCase):
    def test_resume_unfinished_validation_before_training(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'train_budgeted_ddqn_parallel.py'
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            train=p/'train.txt'; val=p/'val.txt'; out=p/'out'
            train.write_text('4\t60\t380\t3\t8\n6\t80\t400\t3\n9\t75\t360\t3\n')
            val.write_text('5\t62\t360\t3\t8\n7\t80\t380\t3\n10\t70\t350\t3\n')
            cmd=[sys.executable,'-u',str(script),'--train-cases',str(train),
                 '--val-cases',str(val),'--episodes','4','--eval-every','4',
                 '--workers','4','--validation-workers','4',
                 '--prefetch-per-worker','1','--scenarios','small',
                 '--budgets','2','--val-seeds','400001', '--uncertainties','8',
                 '--candidate-limit','2','--batch-size','4','--warmup','4',
                 '--updates-per-step','0','--hidden-dim','16',
                 '--device','cpu','--out',str(out)]
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1')
            first=subprocess.run(cmd,cwd=root,env=env,capture_output=True,
                                 text=True,timeout=180)
            self.assertEqual(first.returncode,0,first.stdout+'\n'+first.stderr)
            state_path=out/'training_state.pt'
            state=torch.load(state_path,map_location='cpu',weights_only=False)
            self.assertEqual(len(state['train_rows']),4)
            self.assertEqual(state['validations']['val_000004']['status'],'done')
            # Emulate a crash after freezing and saving the training state,
            # but before receiving the validation result.
            state['validations']['val_000004']['status']='pending'
            state['val_rows']=[]
            state['best_val']=float('inf')
            torch.save(state,state_path)
            (out/'best.pt').unlink()
            before=(out/'train_curve.csv').read_text()
            resume=subprocess.run(cmd+['--resume'],cwd=root,env=env,
                                  capture_output=True,text=True,timeout=180)
            self.assertEqual(resume.returncode,0,resume.stdout+'\n'+resume.stderr)
            self.assertIn('[validation-resumed] val_000004',resume.stdout)
            self.assertIn('[validation-finished] val_000004',resume.stdout)
            self.assertNotIn('[actor-start]',resume.stdout)
            self.assertEqual((out/'train_curve.csv').read_text(),before)
            self.assertTrue((out/'best.pt').exists())
            with (out/'validation_curve.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(int(rows[0]['evaluated_runs']),4)

    def test_four_actors_then_four_validators_no_overlap(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'train_budgeted_ddqn_parallel.py'
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            train = base / 'train.txt'
            val1 = base / 'val1.txt'
            val2 = base / 'val2.txt'
            train.write_text('4\t60\t380\t3\t8\n6\t80\t400\t3\n9\t75\t360\t3\n11\t90\t420\t3\n')
            val1.write_text('5\t62\t360\t3\t8\n7\t80\t380\t3\n10\t70\t350\t3\n13\t85\t400\t3\n')
            val2.write_text('6\t62\t360\t3\t8\n8\t80\t380\t3\n12\t70\t350\t3\n14\t85\t400\t3\n')
            out = base / 'out'
            cmd = [sys.executable, '-u', str(script),
                   '--train-cases', str(train),
                   '--val-cases', str(val1), str(val2),
                   '--episodes', '8', '--eval-every', '4',
                   '--workers', '4', '--validation-workers', '4',
                   '--prefetch-per-worker', '1',
                   '--scenarios', 'shock,repeated,small',
                   '--budgets', '2,5,10', '--val-seeds', '400001:400003',
                   '--uncertainties', '8', '--clock', 'union',
                   '--candidate-limit', '2', '--batch-size', '4',
                   '--warmup', '4', '--updates-per-step', '.25',
                   '--hidden-dim', '16', '--device', 'cpu', '--out', str(out)]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1')
            first = subprocess.run(cmd, cwd=root, env=env, capture_output=True,
                                   text=True, timeout=360)
            self.assertEqual(first.returncode, 0, first.stdout + '\n' + first.stderr)
            with (out / 'train_curve.csv').open() as stream:
                train_rows = list(csv.DictReader(stream))
            with (out / 'validation_curve.csv').open() as stream:
                val_rows = list(csv.DictReader(stream))
            self.assertEqual(len(train_rows), 8)
            self.assertEqual(len(val_rows), 2)
            self.assertEqual([int(r['completed']) for r in val_rows], [4, 8])
            self.assertEqual([int(r['evaluated_runs']) for r in val_rows], [216, 216])
            self.assertEqual([int(r['gradient_steps']) for r in val_rows],
                             [int(r['learner_gradient_steps_at_receive']) for r in val_rows])
            for episode in (4, 8):
                folder = out / 'validation_details' / f'val_{episode:06d}'
                with (folder / 'runs.csv').open() as stream:
                    runs = list(csv.DictReader(stream))
                self.assertEqual(len(runs), 216)
                self.assertEqual({r['method'] for r in runs},
                                 {'keep', 'greedy', 'ddqn', 'periodic'})
            events = [json.loads(x) for x in (out / 'events.jsonl').read_text().splitlines()]
            start_indices = [i for i,e in enumerate(events) if e['event']=='phase_validation_start']
            self.assertEqual(len(start_indices), 2)
            for idx in start_indices:
                start = events[idx]
                end_idx = next(j for j in range(idx+1,len(events))
                               if events[j]['event']=='phase_validation_end')
                intervening = events[idx+1:end_idx]
                self.assertFalse(any(e['event'] in ('actor_start','train_completed')
                                     for e in intervening))
                self.assertEqual(start['gradient_steps'], events[end_idx]['gradient_steps'])
                self.assertEqual(start['outstanding_rollouts'], 0)
                self.assertTrue(any(e['event']=='validation_finished' for e in intervening))
            for i,e in enumerate(events):
                if e['event']=='actor_start':
                    ep=int(e['episode'])
                    self.assertTrue(ep<=4 or i>start_indices[0],
                                    f'Episode {ep} started before first validation: {e}')
            perf=json.loads((out/'performance.json').read_text())
            self.assertEqual(perf['workers_requested'], 4)
            self.assertEqual(perf['validation_workers_requested'], 4)
            self.assertEqual(perf['total_validated_runs'], 432)
            self.assertEqual(perf['completed_validations'], 2)
            ledger=json.loads((out/'validation_ledger.json').read_text())['records']
            best=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
            selection=best['meta']['validation_id']
            record=ledger[selection]
            self.assertEqual(record['status'],'done')
            self.assertEqual(best['meta']['validated_snapshot_sha256'],
                             file_sha256(Path(record['snapshot'])))
            frozen=torch.load(record['snapshot'],map_location='cpu',weights_only=True)
            for k in best['online']:
                torch.testing.assert_close(best['online'][k],frozen['online'][k],rtol=0,atol=0)
            # A completed run can be resumed as a no-op without extra validations.
            resumed=subprocess.run(cmd+['--resume'],cwd=root,env=env,
                                   capture_output=True,text=True,timeout=100)
            self.assertEqual(resumed.returncode,0,resumed.stdout+'\n'+resumed.stderr)
            perf2=json.loads((out/'performance.json').read_text())
            self.assertEqual(perf2['total_validated_runs'],432)


if __name__=='__main__':
    unittest.main()
