"""Synthetic hidden-state regeneration contract; NOT a real-port smoke test."""
from __future__ import annotations
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from v12_protocol.transitions import RawStep
from v12_protocol import probe_bank,counterfactual_probe
import iamu_dynamic
import iamu_dynamic.rl_env_v11 as env_module


class StubEnv:
    def __init__(self,scenario,budget,**kwargs):
        self.scenario=scenario;self.budget=budget;self.used=0
        self.event_index=0;self.done=False
        self.published_eta={1:0.};self.plan={1:dict(start=0)}
        self.refresh()

    def reset(self):
        self.event_index=0;self.used=0;self.done=False
        self.refresh();return self._observation()

    def refresh(self):
        self._candidate_ids=() if self.event_index in (0,3,4) else (1,)
        self._candidates=[{1:dict(start=self.event_index)}] if self._candidate_ids else []
        self._candidate_data=[dict(gain=1.)] if self._candidate_ids else []

    def _observation(self):
        mask=np.array([True,bool(self._candidate_ids)],dtype=bool)
        g=np.full(24,self.event_index,dtype=np.float32)
        a=np.full((2,24),self.event_index,dtype=np.float32)
        return NS(mask=mask,global_features=g,action_features=a,
                  remaining_budget=self.budget-self.used,
                  time=float(self.event_index),candidate_ids=self._candidate_ids)

    def step(self,action):
        self.used+=int(action!=0);self.event_index+=1
        self.done=self.event_index>=4
        self.refresh()
        return self._observation(),-.2,self.done,{}


def raw_fixture(eid):
    env=StubEnv(None,2);ob=env.reset();raw=[]
    for i in range(4):
        nxt,reward,done,_=env.step(0)
        tr=NS(obs_global=ob.global_features.copy(),obs_actions=ob.action_features.copy(),
              obs_mask=ob.mask.copy(),action=0,done=done,
              next_global=nxt.global_features.copy(),next_actions=nxt.action_features.copy(),
              next_mask=nxt.mask.copy(),reward=reward,episode_id=eid,reason_flags={})
        raw.append(RawStep(tr,2,2,i,i+1,float(i),float(i+1),eid,0))
        ob=nxt
    return raw


class ProbeReconstructionTests(unittest.TestCase):
    def test_prefix_replay_and_candidate_plan_fingerprint(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);case=root/'1.txt';case.write_text('synthetic input')
            data=root/'raw.pt';bank=root/'bank.pt';bank2=root/'bank2.pt'
            ds=dict(version=12,
               episodes=[dict(episode_id=1,case=str(case),scenario='shock',
                   seed=300000,budget=2,raw=raw_fixture(1),wait=80.)],
               config=dict(seed=42,scenarios='shock',budgets='2',uncertainties='4',
                           cases=[str(case)],episodes=1,train_seed_start=300000,updates=2))
            torch.save(ds,data)
            previous=random.getstate()
            with patch.object(iamu_dynamic,'read_case',return_value=([],None),create=True),\
                 patch.object(iamu_dynamic,'generate_scenario',return_value=None,create=True),\
                 patch.object(env_module,'IAMUEventEnv',StubEnv,create=True):
                built=probe_bank.build_probe_bank(data,bank,seed=1729,per_stratum=8,max_states=144)
                self.assertEqual(built['states'],2)
                states=probe_bank.load_probe_bank(bank,data)['states']
                for state in states:
                    env=probe_bank.restore_probe_env(state,ds)
                    self.assertEqual(env.event_index,state['source_event_index'])
                probe_bank.build_probe_bank(data,bank2,seed=1729,per_stratum=8,max_states=144)
                again=probe_bank.load_probe_bank(bank2,data)['states']
                self.assertEqual([s['state_id'] for s in states],[s['state_id'] for s in again])
                self.assertEqual([s['candidate_fingerprint'] for s in states],
                                 [s['candidate_fingerprint'] for s in again])
                # Deliberately corrupt proposal-plan fingerprint: restore must fail.
                changed=dict(states[0],candidate_fingerprint='wrong')
                with self.assertRaises(AssertionError):
                    probe_bank.restore_probe_env(changed,ds)
            self.assertEqual(random.getstate(),previous)


    def test_counterfactual_reuses_same_path_independent_branches(self):
        import iamu_dynamic.rl_experiment_v11 as experiment_module
        with TemporaryDirectory() as folder:
            root=Path(folder);case=root/'1.txt';case.write_text('fake')
            data=root/'raw.pt';bank=root/'probe.pt'
            ds=dict(version=12,
               episodes=[dict(episode_id=1,case=str(case),scenario='shock',
                   seed=300000,budget=2,raw=raw_fixture(1),wait=80.)],
               config=dict(seed=42,scenarios='shock',budgets='2',uncertainties='4',
                           cases=[str(case)],episodes=1,train_seed_start=300000,updates=2))
            torch.save(ds,data)
            with patch.object(iamu_dynamic,'read_case',return_value=([],None),create=True),\
                 patch.object(iamu_dynamic,'generate_scenario',return_value=None,create=True),\
                 patch.object(env_module,'IAMUEventEnv',StubEnv,create=True),\
                 patch.object(experiment_module,'choose_action',side_effect=lambda policy,env:0,create=True):
                probe_bank.build_probe_bank(data,bank,seed=1729)
                state=probe_bank.load_probe_bank(bank,data)['states'][0]
                from v12_protocol.counterfactual_probe import evaluate_state
                rows=evaluate_state(state,ds,'greedy')
                self.assertEqual(len(rows),2)
                self.assertTrue(all(abs(r['delta_cost_path_pi'])<1e-8 for r in rows))
                again=evaluate_state(state,ds,'greedy')
                self.assertEqual(rows,again)

    def test_short_fixed_f1_creates_true_td_and_checkpoint_artifacts(self):
        from v12_protocol.fixed_f1 import train_f1
        with TemporaryDirectory() as folder:
            root=Path(folder);case=root/'1.txt';case.write_text('fake')
            data=root/'train.pt';probe=root/'probe.pt';bank=root/'bank.pt';out=root/'f1_A0'
            def dataset(start):
                return dict(version=12,
                  episodes=[dict(episode_id=1,case=str(case),scenario='shock',
                      seed=start,budget=2,raw=raw_fixture(1),wait=80.)],
                  config=dict(seed=42,scenarios='shock',budgets='2',uncertainties='4',
                              cases=[str(case)],episodes=1,train_seed_start=start,updates=2))
            torch.save(dataset(200000),data);torch.save(dataset(300000),probe)
            with patch.object(iamu_dynamic,'read_case',return_value=([],None),create=True),\
                 patch.object(iamu_dynamic,'generate_scenario',return_value=None,create=True),\
                 patch.object(env_module,'IAMUEventEnv',StubEnv,create=True):
                probe_bank.build_probe_bank(probe,bank,seed=1729)
            a=NS(data=str(data),probe_bank=str(bank),group='A0',gradient_updates=4,
                 batch_size=2,checkpoint_every=2,probe_every=2,diagnostics=True,
                 diagnostic_every=2,seed=42,device='cpu',gamma=1.,replay_episodes=200,
                 skip_validation=True,val_cases=None,out=str(out))
            manifest=train_f1(a)
            self.assertEqual(manifest['gradient_updates_actual'],4)
            self.assertFalse(manifest['validation_executed'])
            for name in ('manifest.json','conversion_audit.csv','train_curve.csv',
                         'td_diagnostics.csv','probe_qgap.csv','probe_metrics.csv',
                         'checkpoints/grad_000002.pt','checkpoints/grad_000004.pt',
                         'final.pt','complete.json'):
                self.assertTrue((out/name).is_file(),name)
            with self.assertRaises(FileExistsError):train_f1(a)


if __name__=='__main__':unittest.main()
