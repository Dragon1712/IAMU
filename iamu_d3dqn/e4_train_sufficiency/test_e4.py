"""Run: python -m unittest discover -s e4_train_sufficiency -p 'test_e4.py' -v"""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from launch_e4 import create_job,CONDITIONS,sha256
from inspect_e4 import inspect_run


class E4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)/'iamu_d3dqn'
        self.root.mkdir()
        (self.root/'train_budgeted_ddqn_parallel.py').write_text('print("fake")')
        case=self.root.parent/'2024LiERLDDE'/'case_code'
        case.mkdir(parents=True)
        for i in range(3,10):
            (case/f'{i}.txt').write_text('synthetic test fixture')

    def tearDown(self):self.tmp.cleanup()

    def test_all_commands_keep_frozen_ablation_config(self):
        import launch_e4
        with patch.object(launch_e4,'ROOT',self.root),patch.object(launch_e4,'TRAINER',self.root/'train_budgeted_ddqn_parallel.py'):
            jobs={name:create_job(condition=name,seed=43,device='cuda:1',actors=8,
                  validators=8,prefix='test_e4',validation_seeds='400001:400003',
                  validation_every=100,recovery_every=25) for name in CONDITIONS}
        def arg(job,key):
            a=list(job.command);return a[a.index('--'+key)+1]
        for name,job in jobs.items():
            self.assertEqual(arg(job,'seed'),'43')
            self.assertEqual(arg(job,'device'),'cuda:1')
            self.assertEqual(arg(job,'eval-every'),'100')
            self.assertEqual(arg(job,'q-network'),'set_ddqn')
            self.assertEqual(arg(job,'candidate-screen'),'positive')
            self.assertEqual(arg(job,'keep-margin'),'0.1')
            self.assertEqual(arg(job,'train-seed-start'),'200000')
            self.assertEqual(arg(job,'epsilon-end'),'0.05')
            self.assertEqual(arg(job,'val-seeds'),'400001:400003')
            self.assertEqual(arg(job,'episodes'),str(CONDITIONS[name]['episodes']))
            self.assertEqual(arg(job,'epsilon-decay-episodes'),str(CONDITIONS[name]['epsilon_decay_episodes']))
            self.assertAlmostEqual(float(arg(job,'lr')),CONDITIONS[name]['lr'])
            self.assertTrue(job.run_dir.endswith('test_e4_'+name+'_seed43'))
        self.assertEqual(len(set(j.run_dir for j in jobs.values())),4)

    def build_fixture(self,episodes=900):
        out=self.root/'results'/'v10_e4_b900_seed42';out.mkdir(parents=True)
        (out/'manifest.json').write_text(json.dumps({'training_args':{
             'episodes':episodes,'eval_every':100,'q_network':'set_ddqn'}}))
        (out/'validation_snapshots').mkdir()
        ledger={}
        rows=[]
        for ep in range(100,episodes+1,100):
            vid=f'val_{ep:06d}'
            checkpoint=out/'validation_snapshots'/f'{vid}_grad_{ep*10:09d}.pt'
            checkpoint.write_bytes(b'This is a fixture, not a model'+str(ep).encode())
            checksum=sha256(checkpoint)
            ledger[vid]=dict(snapshot=str(checkpoint),status='done',
                snapshot_sha256=checksum,expected_runs=216,evaluated_runs=216,
                gradient_steps=ep*10,completed=ep)
            rows.append(dict(completed=ep,mean_wait=2000-ep/10,gradient_steps=ep*10,
                    snapshot_sha256=checksum))
        (out/'validation_ledger.json').write_text(json.dumps({'records':ledger}))
        with (out/'validation_curve.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        (out/'performance.json').write_text(json.dumps({'completed_episodes':episodes,
                  'completed_validations':episodes//100}))
        return out

    def test_nine_valid_inference_ready_snapshots_verified(self):
        folder=self.build_fixture()
        r=inspect_run(folder,verify_hash=True)
        self.assertTrue(r['complete'],r['errors'])
        self.assertEqual(r['expected_checkpoints'],9)
        self.assertEqual(r['valid_checkpoints'],9)
        self.assertEqual([x['episode'] for x in r['inventory']],list(range(100,901,100)))
        self.assertTrue(all(x['status']=='valid' for x in r['inventory']))

    def test_incomplete_validation_is_not_a_valid_checkpoint(self):
        folder=self.build_fixture(episodes=300)
        path=folder/'validation_ledger.json';ledger=json.loads(path.read_text())
        ledger['records']['val_000200']['status']='pending'
        path.write_text(json.dumps(ledger))
        r=inspect_run(folder)
        self.assertFalse(r['complete'])
        self.assertEqual(r['valid_checkpoints'],2)
        self.assertTrue(any('not done' in x for x in r['errors']))

    def test_corrupted_checkpoint_detected(self):
        folder=self.build_fixture(episodes=300)
        x=next((folder/'validation_snapshots').glob('val_000100*'))
        x.write_bytes(b'CORRUPTED')
        r=inspect_run(folder,verify_hash=True)
        self.assertFalse(r['complete'])
        self.assertTrue(any('SHA invalid' in x for x in r['errors']))


if __name__=='__main__':unittest.main()
