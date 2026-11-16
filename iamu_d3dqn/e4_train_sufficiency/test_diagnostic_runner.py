"""Runner tests using deterministic toy public-state transitions, not IAMU physics."""
import copy
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from diagnose_keep_qgap import (
    Model, analyze_bank, collect_states, counterfactual, load_model, main,
    predict_q, run_trajectory, consistent_environment,
)


class TinyOnline(torch.nn.Module):
    def forward(self,g,a,mask):
        # Both Edit/Keep are valued in normalized reward units.
        # At state index 1, a should be chosen if it is legal.
        result=torch.stack([torch.zeros_like(g[:,0]),.4+0.*g[:,0]],dim=1)
        return result


class TinyAgent:
    device=torch.device('cpu')
    keep_margin=.1
    q_network="set_ddqn"
    def __init__(self):self.online=TinyOnline()
    def act(self,obs,epsilon=0.):
        vals=predict_q(self,obs,torch)
        from diagnostic_core import action_diagnostics
        return action_diagnostics(vals,obs.mask,self.keep_margin)['action']


class TinyEnv:
    def __init__(self,scenario,budget=2,**kw):
        self.scenario=copy.deepcopy(scenario)
        self.budget=budget
        self.reward_scale=100.
        self.min_gain=.1
        self.ids=(0,)
        self.vessels={0:SimpleNamespace(weight=1.)}
        self.reset()

    @property
    def remaining_budget(self):return self.budget-self.used

    def reset(self):
        self.used=0
        self.event_index=0
        self.event_time=0.
        self.event_type='initial'
        self._candidates=[]
        self._candidate_ids=()
        self._candidate_data=[]
        self.screened_proposals=0
        self.runtime=SimpleNamespace(started={})
        self.revised=False
        return self._observation()

    def _observation(self):
        import numpy as np
        actionable=1<=self.event_index<=2 and self.remaining_budget>0
        self._candidate_ids=(0,) if actionable else ()
        self._candidates=[1] if actionable else []
        self._candidate_data=[dict(gain=5.)] if actionable else []
        return SimpleNamespace(global_features=np.array([self.event_index],dtype='float32'),
            action_features=np.zeros((2,1),dtype='float32'),
            mask=np.array([True,actionable],dtype=bool),candidate_ids=self._candidate_ids)

    def point_greedy_action(self):return 1 if self._candidates else 0

    def step(self,action):
        obs=self._observation()
        if not obs.mask[action]:raise ValueError('Illegal')
        if action==1:self.used+=1;self.revised=True
        self.event_index+=1
        self.event_type='eta'
        self.event_time=float(self.event_index)
        done=self.event_index>=3
        wait=200.-(10. if self.revised else 0.)
        info={'weighted_wait':wait,'spent':self.used}
        return self._observation(),-wait/self.reward_scale,done,info


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.scenario=SimpleNamespace(true_etas={0:100.},updates=[],seed=41)
        self.agent=TinyAgent()
        self.model=Model('fake_ep100',100,Path('.'),Path('fake'),'',self.agent,{}, {},0)

    def test_trajectory_keep_and_edit_and_replay(self):
        ep,events=run_trajectory(self.scenario,2,self.model,{},TinyEnv,torch)
        self.assertAlmostEqual(ep['weighted_wait'],190.)
        self.assertEqual(ep['spent'],2)  # Both public decision events accepted.
        self.assertEqual(ep['edit_events'],2)
        self.assertEqual(ep['actionable_events'],2)
        self.assertEqual(ep['keep_due_to_positive_margin'],0)

    def test_counterfactual_pairs_original_future(self):
        e=TinyEnv(self.scenario,budget=1)
        e.step(0)  # Reach first actionable event
        k=counterfactual(e,0,'keep',self.model)
        v=counterfactual(e,1,'keep',self.model)
        self.assertAlmostEqual(k['wait']-v['wait'],10.)
        self.assertEqual(k['spent'],0)
        self.assertEqual(v['spent'],1)
        self.assertEqual(e.used,0)  # Original state not mutated.

    def test_state_selection_and_matched_scores(self):
        states,n=collect_states(self.scenario,2,'keep',self.model,{},
                                n_states=2,IAMUEventEnv=TinyEnv)
        self.assertEqual(n,2)
        self.assertEqual(len(states),2)
        key,q,truth,cal,dec=analyze_bank(states[0],[self.model],self.model,
               ['anchor','greedy','keep'],torch=torch,max_edit_actions=4,
               case_name='8',scenario_name='shock',eta_seed=41,budget=2,
               anchor_name='keep')
        self.assertEqual(len(q),2)
        self.assertEqual(len(truth),6)
        self.assertEqual(len(cal),3)
        self.assertEqual(len(dec),3)
        self.assertAlmostEqual(next(x for x in cal if x['continuation']=='keep')['true_gain_scaled'],.1)

    def test_snapshot_ledger_checks_checksum_and_missing_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'validation_snapshots').mkdir()
            ck=root/'validation_snapshots/val_000100_grad_000000001.pt'
            ck.write_bytes(b'fake-model-weight-bytes')
            import hashlib
            sha=hashlib.sha256(ck.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'training_args':{'keep_margin':.1}}))
            (root/'validation_ledger.json').write_text(json.dumps({'records':{'val_000100':{
                'snapshot':str(ck),'snapshot_sha256':sha,'status':'done',
                'evaluated_runs':216,'expected_runs':216,'gradient_steps':1}}}))
            p=root/'validation_details/val_000100';p.mkdir(parents=True)
            (p/'runs.csv').write_text('case,scenario,seed,budget,method,wait,spent\n'
                                      '8,shock,400001,2,ddqn,190,1\n')
            class Loader:
                @classmethod
                def from_checkpoint(cls,path,device='cpu'):
                    return TinyAgent(),{'selected_episode':100}
            m=load_model(str(root)+'@100','cpu',verify_checksum=True,Agent=Loader)
            self.assertEqual(m.episode,100)
            self.assertEqual(len(m.validation_rows),1)
            ck.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                load_model(str(root)+'@100','cpu',verify_checksum=True,Agent=Loader)


    def test_full_main_pipeline_on_toy_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            case=root/'8.txt';case.write_text('synthetic case')
            run=root/'b900_seed43';(run/'validation_snapshots').mkdir(parents=True)
            args=dict(keep_margin=.1,clock='union',period=12.,
                      candidate_limit=1,candidate_screen='positive',
                      reward_scale=100.,min_gain=.1,
                      budget_features='absolute',value_features='global',
                      updates=2,validation_uncertainty=8.)
            (run/'manifest.json').write_text(json.dumps({'training_args':args}))
            ledgers={}
            for ep in [100,900]:
                val=f'val_{ep:06d}'
                ck=run/'validation_snapshots'/f'{val}_grad_000000001.pt'
                ck.write_bytes(f'fake episode {ep}'.encode())
                import hashlib
                ledgers[val]=dict(snapshot=str(ck),
                     snapshot_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
                     status='done',expected_runs=216,evaluated_runs=216,
                     gradient_steps=1)
                p=run/'validation_details'/val;p.mkdir(parents=True)
                (p/'runs.csv').write_text('case,scenario,seed,budget,method,wait,spent\n'
                                          '8,shock,400001,2,ddqn,190,2\n')
            (run/'validation_ledger.json').write_text(json.dumps({'records':ledgers}))
            class Loader:
                @classmethod
                def from_checkpoint(cls,path,device='cpu'):
                    a=TinyAgent()
                    ep=100 if 'val_000100' in path else 900
                    return a,{'selected_episode':ep}
            def mock_imports():
                def read_case(p):return [object()],object()
                def make_scenario(vessels,port,label,seed,updates_per_vessel,uncertainty):
                    return SimpleNamespace(true_etas={0:100.},updates=[],seed=seed)
                return torch,read_case,make_scenario,TinyEnv,Loader
            out=root/'results'
            with patch('diagnose_keep_qgap.imports',new=mock_imports):
                main(['--model',str(run)+'@100','--model',str(run)+'@900',
                      '--cases',str(case),'--scenarios','shock',
                      '--eta-seeds','400001','--budgets','2',
                      '--anchors','early,keep','--continuations','anchor,greedy',
                      '--states-per-group','1','--out',str(out)])
            self.assertEqual(json.loads((out/'summary.json').read_text())['original_validation_checks'],2)
            self.assertEqual(json.loads((out/'summary.json').read_text())['selected_states'],2)
            for name in ('calibration_summary.csv','trajectory_events.csv',
                         'matched_q_shift.csv','decision_regret.csv',
                         'counterfactual.csv','validation_parity.csv'):
                self.assertTrue((out/name).exists(),name)
            from summarize_diagnostic import main as summarize
            from contextlib import redirect_stdout
            import io
            with redirect_stdout(io.StringIO()):
                summarize(['--input',str(out),'--out',str(out/'diagnosis_report.md')])
            self.assertTrue((out/'diagnosis_report.md').exists())


if __name__=='__main__':unittest.main()
