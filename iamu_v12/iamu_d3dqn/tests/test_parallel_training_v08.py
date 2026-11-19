"""IAMU parallel trainer end-to-end test (real small scenarios, CPU only).

Run from IAMU_dynamic root: python -m unittest tests.test_parallel_training_v08 -v
Also works in unittest discovery. Does not need a CUDA GPU.
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


class ParallelTrainingIntegration(unittest.TestCase):
    def test_two_cpu_actors_learner_and_checkpoint(self):
        root=Path(__file__).resolve().parents[1]
        script=root/'train_budgeted_ddqn_parallel.py'
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)
            case_train=work/'train.txt'
            case_val=work/'val.txt'
            # Small, but valid original IAMU tab-separated case format.
            case_train.write_text('4\t60\t380\t3\t8\n6\t80\t400\t3\n9\t75\t360\t3\n11\t90\t420\t3\n')
            case_val.write_text('5\t62\t360\t3\t8\n7\t80\t380\t3\n10\t70\t350\t3\n13\t85\t400\t3\n')
            out=work/'results'
            cmd=[sys.executable,'-u',str(script),
                '--train-cases',str(case_train),
                '--val-cases',str(case_val),
                '--workers','2','--prefetch-per-worker','1',
                '--sync-every','2','--episodes','6',
                '--scenarios','small','--budgets','1,2',
                '--uncertainties','8','--val-seeds','440001:440002',
                '--clock','eta','--candidate-limit','2',
                '--batch-size','4','--warmup','4',
                '--updates-per-step','.2',
                '--max-updates-per-episode','4',
                '--target-update','4','--eval-every','3',
                '--hidden-dim','16','--device','cpu',
                '--worker-timeout','30','--out',str(out)]
            # Optional V0.7 parameters are supported by the repository at
            # the time of authoring, but the script also handles old V0.6.
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            proc=subprocess.run(cmd,cwd=root,env=env,timeout=120,
                                capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,msg=proc.stdout+'\n'+proc.stderr)
            self.assertTrue((out/'best.pt').is_file())
            self.assertTrue((out/'latest.pt').is_file())
            with (out/'performance.json').open() as f:perf=json.load(f)
            self.assertEqual(perf['completed_episodes'],6)
            self.assertEqual(perf['workers_used'],2)
            self.assertGreater(perf['gradient_steps'],0)
            with (out/'train_curve.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),6)
            self.assertEqual(len({r['seed'] for r in rows}),6)
            self.assertEqual(len({r['worker_pid'] for r in rows}),2)
            for r in rows:
                self.assertLessEqual(int(r['spent']),int(r['budget']))
                self.assertAlmostEqual(float(r['reward_sum_scaled'])*-100,
                                       float(r['weighted_wait']),places=3)
            with (out/'validation_curve.csv').open() as f:
                v=list(csv.DictReader(f))
            self.assertEqual(len(v),2)


if __name__=='__main__': unittest.main()
