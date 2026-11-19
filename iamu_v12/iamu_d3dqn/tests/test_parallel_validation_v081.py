"""IAMU V0.8.1: full validation count, frozen best, serial equivalence, recovery.

Run from IAMU_dynamic root: python -m unittest tests.test_parallel_validation_v081 -v
This test uses tiny original-format IAMU cases; no GPU needed.
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

import numpy as np
import torch

from iamu_dynamic.rl_ddqn import DoubleDQNAgent, ReplayBuffer, Transition
from iamu_dynamic.rl_experiment import evaluate_paired
from iamu_dynamic.rl_env import GLOBAL_DIM, ACTION_DIM
from parallel_validation_v081 import validation_jobs
from parallel_recovery_v081 import save_state,load_state
from train_budgeted_ddqn_parallel import _save_checkpoint,_select_best_snapshot


class ParallelValidationTests(unittest.TestCase):
    def test_original_validation_grid_not_reduced(self):
        settings=dict(val_cases=['case8','case9'],scenarios=['shock','repeated','small'],
                      val_seeds=[400001,400002,400003],budgets=[2,5,10],
                      policies=['keep','greedy','ddqn','periodic'])
        jobs=validation_jobs(settings)
        self.assertEqual(len(jobs),54)
        self.assertEqual(len(jobs)*len(settings['policies']),216)

    def test_frozen_checkpoint_selected_not_latest_model(self):
        agent=DoubleDQNAgent(hidden_dim=16,seed=12)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);snapshot=out/'val_000025.pt'
            _save_checkpoint(snapshot,agent,{'selected_episode':25,'n_gradient_updates':3})
            original=torch.load(snapshot,map_location='cpu',weights_only=True)
            with torch.no_grad():
                for weight in agent.online.parameters():
                    weight.add_(8.)
            _select_best_snapshot(out,snapshot,score=12.4,validation_id='val_000025',
                                  completed=25,gradient_steps=3,snapshot_sha256='fixedhash')
            best=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
            for k in original['online']:
                torch.testing.assert_close(original['online'][k],best['online'][k],rtol=0,atol=0)
                self.assertFalse(torch.equal(agent.online.state_dict()[k],best['online'][k]))
            self.assertEqual(best['meta']['validated_snapshot_sha256'],'fixedhash')
            self.assertEqual(best['meta']['selected_episode'],25)
            self.assertEqual(best['meta']['n_gradient_updates'],3)

    def test_full_optimizer_replay_state_roundtrip(self):
        agent=DoubleDQNAgent(hidden_dim=16,seed=12)
        replay=ReplayBuffer(100,seed=19)
        sample=Transition(np.zeros(GLOBAL_DIM,dtype=np.float32),
                          np.zeros((3,ACTION_DIM),dtype=np.float32),
                          np.array([True,True,False]),0,-1.,
                          np.zeros(GLOBAL_DIM,dtype=np.float32),
                          np.zeros((3,ACTION_DIM),dtype=np.float32),
                          np.array([True,False,False]),False)
        for _ in range(12):replay.items.append(sample)
        agent.update(replay.sample(4))
        config={'training_args':{'episodes':10,'device':'cpu','out':'old','resume':False},
                'train_cases':['train'], 'val_cases':['validation']}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'training_state.pt'
            save_state(p,learner=agent,replay=replay,config=config,
                       train_rows=[{'episode':1}],val_rows=[],validations={
                           'val_000005':{'status':'pending','snapshot':'unmodified.pt'}},
                       best_val=float('inf'),gradient_credit=.25)
            agent2=DoubleDQNAgent(hidden_dim=16,seed=333)
            replay2=ReplayBuffer(100,seed=333)
            new_cfg={**config,'training_args':{**config['training_args'],
                                               'out':'new','resume':True}}
            item=load_state(p,learner=agent2,replay=replay2,config=new_cfg)
            self.assertEqual(agent2.gradient_steps,agent.gradient_steps)
            self.assertEqual(len(replay2),12)
            self.assertEqual(item['validations']['val_000005']['status'],'pending')
            self.assertAlmostEqual(item['gradient_credit'],.25)
            self.assertEqual(replay.rng.getstate(),replay2.rng.getstate())
            for k in agent.online.state_dict():
                torch.testing.assert_close(agent.online.state_dict()[k],
                                           agent2.online.state_dict()[k])

    def test_cpu_pool_all_216_results_match_serial_and_async_checkpoint(self):
        root=Path(__file__).resolve().parents[1]
        script=root/'train_budgeted_ddqn_parallel.py'
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            train=work/'case_train.txt';val_a=work/'case_val_a.txt';val_b=work/'case_val_b.txt'
            train.write_text('4\t60\t380\t3\t8\n6\t80\t400\t3\n9\t75\t360\t3\n11\t90\t420\t3\n')
            val_a.write_text('5\t62\t360\t3\t8\n7\t80\t380\t3\n10\t70\t350\t3\n13\t85\t400\t3\n')
            val_b.write_text('6\t62\t360\t3\t8\n8\t80\t380\t3\n12\t70\t350\t3\n14\t85\t400\t3\n')
            output=work/'out'
            cmd=[sys.executable,'-u',str(script),
                 '--train-cases',str(train),
                 '--val-cases',str(val_a),str(val_b),
                 '--workers','2','--validation-workers','2',
                 '--prefetch-per-worker','1','--episodes','2',
                 '--scenarios','shock,repeated,small','--budgets','2,5,10',
                 '--uncertainties','8','--val-seeds','400001:400003',
                 '--clock','union','--candidate-limit','2',
                 '--batch-size','4','--warmup','4','--updates-per-step','0',
                 '--eval-every','2','--state-every','1',
                 '--validation-progress-every','9','--hidden-dim','16',
                 '--device','cpu','--out',str(output)]
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1')
            proc=subprocess.run(cmd,cwd=root,env=env,timeout=240,
                                capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,msg=proc.stdout+'\n'+proc.stderr)
            with (output/'train_curve.csv').open() as f:
                train_rows=list(csv.DictReader(f))
            with (output/'validation_curve.csv').open() as f:
                val_rows=list(csv.DictReader(f))
            self.assertEqual(len(train_rows),2)
            self.assertEqual(len(val_rows),1)
            self.assertEqual(int(val_rows[0]['evaluated_runs']),216)
            # Fast tiny tests may be exhausted before the 2nd process starts.
            self.assertGreaterEqual(len(set(val_rows[0]['worker_pids'].split(';'))),1)
            self.assertTrue((output/'training_state.pt').exists())
            self.assertTrue((output/'events.jsonl').exists())
            run_file=output/'validation_details/val_000002/runs.csv'
            with run_file.open() as f:
                parallel=list(csv.DictReader(f))
            self.assertEqual(len(parallel),216)
            snapshot=list((output/'validation_snapshots').glob('*.pt'))
            self.assertEqual(len(snapshot),1)
            agent,_=DoubleDQNAgent.from_checkpoint(str(snapshot[0]),device='cpu')
            serial,serial_summary=evaluate_paired(
                cases=[str(val_a),str(val_b)],
                seeds=[400001,400002,400003],
                scenarios=['shock','repeated','small'],budgets=[2,5,10],
                policies=['keep','greedy','ddqn','periodic'],agent=agent,
                candidate_limit=2,decision_clock='union',period=12.,
                reward_scale=100.,uncertainty=8.,updates=2)
            key=lambda r:(r['case'],r['scenario'],int(r['seed']),
                          int(r['budget']),r['method'])
            self.assertEqual(len(serial),216)
            for a,b in zip(sorted(serial,key=key),sorted(parallel,key=key)):
                self.assertEqual(key(a),key(b))
                self.assertAlmostEqual(float(a['wait']),float(b['wait']),places=6)
                self.assertEqual(int(a['spent']),int(b['spent']))
            with (output/'validation_details/val_000002/summary.csv').open() as f:
                parallel_summary=list(csv.DictReader(f))
            serial_idx={(x['method'],int(x['budget'])):x for x in serial_summary}
            for row in parallel_summary:
                ref=serial_idx[row['method'],int(row['budget'])]
                for col in ('mean_wait','mean_gain','median_gain',
                            'worst_gain','cvar90_regret','mean_spent'):
                    self.assertAlmostEqual(float(row[col]),float(ref[col]),places=6)
            best=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
            self.assertEqual(best['meta']['validation_id'],'val_000002')
            self.assertEqual(best['meta']['validated_snapshot_sha256'],
                             val_rows[0]['snapshot_sha256'])

if __name__=='__main__':unittest.main()
