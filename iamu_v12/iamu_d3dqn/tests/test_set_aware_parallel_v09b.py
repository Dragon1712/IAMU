"""V0.9-B integration: 4 train actors -> frozen Set-aware DDQN -> 4 validators, full 216 grid.

Runs original IAMU environment with tiny case files (not the 50-ship benchmark).
Execute from IAMU_dynamic root:
  python -m unittest tests.test_cc_d3qn_parallel_v09 -v
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
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from parallel_validation_v081 import file_sha256


def read_rows(path: Path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


class SetAwarePipelineTests(unittest.TestCase):
    def test_phase_synchronous_set_ddqn_full_validation_and_resume(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            train=p/'train.txt'; v8=p/'val8.txt';v9=p/'val9.txt';out=p/'run'
            train.write_text('4\t60\t380\t3\t8\n6\t80\t400\t3\n9\t75\t360\t3\n')
            v8.write_text('5\t62\t360\t3\t8\n7\t80\t380\t3\n10\t70\t350\t3\n')
            v9.write_text('6\t64\t365\t3\t8\n8\t83\t385\t3\n12\t76\t355\t3\n')
            cmd=[sys.executable,'-u',str(root/'train_budgeted_ddqn_parallel.py'),
                 '--train-cases',str(train),'--val-cases',str(v8),str(v9),
                 '--episodes','4','--eval-every','4',
                 '--workers','4','--validation-workers','4',
                 '--prefetch-per-worker','1','--sync-every','1',
                 '--scenarios','shock,repeated,small','--budgets','2,5,10',
                 '--val-seeds','400001:400003','--uncertainties','8',
                 '--clock','union','--candidate-limit','2',
                 '--batch-size','4','--warmup','4','--updates-per-step','.25',
                 '--hidden-dim','24','--q-network','set_ddqn',
                 '--device','cpu','--out',str(out)]
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1')
            p1=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=240)
            self.assertEqual(p1.returncode,0,p1.stdout+'\n'+p1.stderr)
            trainrows=read_rows(out/'train_curve.csv')
            vals=read_rows(out/'validation_curve.csv')
            runs=read_rows(out/'validation_details/val_000004/runs.csv')
            self.assertEqual(len(trainrows),4)
            self.assertEqual(len(vals),1)
            self.assertEqual(len(runs),216)
            self.assertEqual(int(vals[0]['evaluated_runs']),216)
            self.assertEqual(int(vals[0]['gradient_steps']),
                             int(vals[0]['learner_gradient_steps_at_receive']))
            self.assertEqual({r['method'] for r in runs},
                             {'keep','greedy','ddqn','periodic'})
            perf=json.loads((out/'performance.json').read_text())
            self.assertEqual(perf['workers_used'],4)
            self.assertEqual(perf['validation_workers_requested'],4)
            self.assertEqual(perf['total_validated_runs'],216)
            best=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
            self.assertEqual(best['q_network'],'set_ddqn')
            self.assertEqual(best['network_architecture_version'],2)
            agent,meta=DoubleDQNAgent.from_checkpoint(str(out/'best.pt'),device='cpu')
            self.assertEqual(agent.q_network,'set_ddqn')
            ledger=json.loads((out/'validation_ledger.json').read_text())['records']
            record=ledger['val_000004']
            frozen=torch.load(record['snapshot'],weights_only=True,map_location='cpu')
            self.assertEqual(meta['validated_snapshot_sha256'],file_sha256(Path(record['snapshot'])))
            for k,w in frozen['online'].items():
                torch.testing.assert_close(w,best['online'][k],rtol=0,atol=0)
            events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(x['event']=='phase_validation_start' for x in events))
            self.assertTrue(any(x['event']=='phase_validation_end' for x in events))
            resumed=subprocess.run(cmd+['--resume'],cwd=root,env=env,capture_output=True,
                                   text=True,timeout=120)
            self.assertEqual(resumed.returncode,0,resumed.stdout+'\n'+resumed.stderr)
            self.assertEqual(len(read_rows(out/'validation_curve.csv')),1)


if __name__=='__main__':
    unittest.main()
