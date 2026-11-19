"""P0–P4 acceptance tests; run from iamu_d3dqn with unittest discover.

No network/GitHub connection; no 400-episode training. The real-environment
probe smoke is a separate explicit CLI, not hidden in this unit suite.
"""
from __future__ import annotations
import random
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from v12_protocol.baseline_cache import baseline_cache_key,load_or_compute_baselines
from v12_protocol.diagnostics import TDRecorder,category,length_bin
from v12_protocol.replay import EpisodeReplay
from v12_protocol.transitions import MacroTransition
from v12_protocol.probe_bank import observation_fingerprint,candidate_fingerprint
from v12_protocol.probe_eval import evaluate_agent_on_probes
from v12_protocol.analyze_diagnostics import compare,summarize
from v12_protocol.agent import V12Agent


def step(ep,i,edit_allowed=True):
    mask=np.array([True,edit_allowed,False,False,False],dtype=bool)
    return MacroTransition(obs_global=np.zeros(24,dtype=np.float32),
      obs_actions=np.zeros((5,24),dtype=np.float32),obs_mask=mask,
      action=0,reward=-.2,next_global=np.zeros(24,dtype=np.float32),
      next_actions=np.zeros((5,24),dtype=np.float32),next_mask=mask.copy(),
      done=True,bootstrap_discount=0.,n_raw_steps=1,elapsed_hours=1.,
      source_budget=1,next_budget=1,episode_id=ep,policy_version=0,
      raw_start_index=i,raw_end_index=i)


class CacheTests(unittest.TestCase):
    def test_cache_hits_content_hash_and_lock(self):
        key=baseline_cache_key(b'case',{'min_gain':.1},{'seed':1},{'core':'aaa'})
        keys=[baseline_cache_key(b'case',{'min_gain':v},{'seed':1},{'core':'aaa'}) for v in (.1,.2)]
        self.assertNotEqual(keys[0],keys[1])
        self.assertNotEqual(key,baseline_cache_key(b'case2',{'min_gain':.1},{'seed':1},{'core':'aaa'}))
        self.assertNotEqual(key,baseline_cache_key(b'case',{'min_gain':.1},{'seed':2},{'core':'aaa'}))
        self.assertNotEqual(key,baseline_cache_key(b'case',{'min_gain':.1},{'seed':1},{'core':'bbb'}))
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            def compute():
                calls.append(1)
                return [dict(method=n,wait=i,spent=0) for i,n in enumerate(('keep','greedy','periodic'))]
            with ThreadPoolExecutor(max_workers=5) as pool:
                result=list(pool.map(lambda _:load_or_compute_baselines(root,key,compute),range(5)))
            self.assertEqual(len(calls),1)
            self.assertEqual(sum(hit for _,hit in result),4)
            self.assertTrue(all([x[0]==result[0][0] for x in result]))
            p=Path(root)/(key+'.json')
            p.write_text('corrupted')
            with self.assertRaises((ValueError,Exception)):load_or_compute_baselines(root,key,compute)

    def test_candidate_fingerprint_includes_proposal_plan(self):
        from types import SimpleNamespace
        p=SimpleNamespace(_candidate_ids=(1,2),_candidates=[{1:dict(start=1)},{2:dict(start=3)}],
             _candidate_data=[{'gain':1},{'gain':2}],plan={1:dict(start=0)},published_eta={1:0.})
        a=candidate_fingerprint(p)
        p._candidates[0][1]['start']=2
        self.assertNotEqual(a,candidate_fingerprint(p))


class ReplayTests(unittest.TestCase):
    def test_index_optimization_same_sample_sequence_and_restore(self):
        from types import MethodType
        fast=EpisodeReplay(max_episodes=2,seed=19)
        old=EpisodeReplay(max_episodes=2,seed=19)
        def legacy(self):
            self._indices=[];self._items=[];self.actionable_indices=[];self.forced_indices=[]
            for eid,ts,_ in self.episodes:
                for j,t in enumerate(ts):
                    pos=len(self._items)
                    self._indices.append((eid,j));self._items.append(t)
                    (self.actionable_indices if bool(t.obs_mask[1:].any()) else self.forced_indices).append(pos)
            self.sample_counts={k:v for k,v in self.sample_counts.items() if k in self._indices}
        old._reindex=MethodType(legacy,old)
        for eid in (1,2,3):
            data=[step(eid,j,j%2==0) for j in range(8)]
            fast.append_episode(eid,data,raw_steps=8)
            old.append_episode(eid,data,raw_steps=8)
            if eid==2:
                for _ in range(30):
                    a=fast.sample(4,'stratified',.25)[0]
                    b=old.sample(4,'stratified',.25)[0]
                    self.assertEqual([(x.episode_id,x.raw_start_index) for x in a],
                                     [(x.episode_id,x.raw_start_index) for x in b])
        self.assertEqual(fast.state_dict()['indices'],old.state_dict()['indices'])
        for _ in range(100):
            a=fast.sample(4,'uniform')[0];b=old.sample(4,'uniform')[0]
            self.assertEqual([(x.episode_id,x.raw_start_index) for x in a],
                             [(x.episode_id,x.raw_start_index) for x in b])
        self.assertEqual(fast.sample_counts,old.sample_counts)
        restored=EpisodeReplay(max_episodes=2,seed=999)
        restored.load_state_dict(fast.state_dict())
        for _ in range(100):
            a=fast.sample(4,'stratified',.25)[0];b=restored.sample(4,'stratified',.25)[0]
            self.assertEqual([(x.episode_id,x.raw_start_index) for x in a],
                             [(x.episode_id,x.raw_start_index) for x in b])


class TDDiagnosticsTests(unittest.TestCase):
    def test_recorder_bins_and_real_zero_vs_missing(self):
        x=step(1,0,True)
        self.assertEqual(category(x),'choice_keep')
        self.assertEqual(length_bin(65),'65+')
        r=TDRecorder(sample_every=1,window=2)
        r(step=1,transitions=[x],q_taken=np.array([1.]),target=np.array([2.]),
          td_error=np.array([1.]),grad_norm_preclip=2.,target_sync=False)
        r(step=2,transitions=[x],q_taken=np.array([1.]),target=np.array([3.]),
          td_error=np.array([2.]),grad_norm_preclip=2.5,target_sync=True)
        a=next(y for y in r.completed if y['group']=='choice_keep' and y['macro_length_bin']=='1')
        self.assertEqual(a['n'],2)
        self.assertAlmostEqual(a['abs_td_mean'],1.5)
        self.assertAlmostEqual(a['abs_td_p90'],1.9)
        self.assertEqual(a['target_sync_count'],1)
        missing=next(y for y in r.completed if y['group']=='edit_other' and y['macro_length_bin']=='65+')
        self.assertEqual(missing['n'],0)
        self.assertEqual(missing['abs_td_mean'],'')

    def test_real_agent_update_same_with_and_without_callback(self):
        torch.set_num_threads(1)
        left=V12Agent(hidden_dim=24,seed=20,device='cpu',q_network='set_ddqn')
        right=V12Agent(hidden_dim=24,seed=20,device='cpu',q_network='set_ddqn')
        callback=TDRecorder(sample_every=1,window=5)
        right.diagnostic_callback=callback
        batch=[step(1,i,i%2==0) for i in range(8)]
        for _ in range(5):
            x=left.update(batch)
            y=right.update(batch)
            self.assertEqual(x,y)
        self.assertEqual(left.gradient_steps,right.gradient_steps)
        for k in left.online.state_dict():
            self.assertTrue(torch.equal(left.online.state_dict()[k],right.online.state_dict()[k]),k)
        for k in left.target.state_dict():
            self.assertTrue(torch.equal(left.target.state_dict()[k],right.target.state_dict()[k]),k)
        self.assertTrue(callback.completed)


class QGapTests(unittest.TestCase):
    def test_eval_never_modifies_model_or_rng(self):
        agent=V12Agent(hidden_dim=24,seed=19,device='cpu',q_network='set_ddqn')
        obs=np.zeros(24,dtype=np.float32)
        actions=np.zeros((5,24),dtype=np.float32)
        mask=np.array([True,True,False,False,False])
        state=dict(state_id='example',obs_global=obs,obs_actions=actions,obs_mask=mask,
             source_time=4.,remaining_budget=2,candidate_ids=(7,),
             episode_id=1,raw_step_index=0,case_sha256='a',scenario='shock',
             initial_budget=2,candidate_fingerprint='fingerprint')
        state_before=agent.rng.getstate()
        net_mode=agent.online.training
        weights={k:v.clone() for k,v in agent.online.state_dict().items()}
        rows=evaluate_agent_on_probes(agent,[state],batch_size=1)
        self.assertEqual(len(rows),1)
        self.assertEqual(agent.rng.getstate(),state_before)
        self.assertEqual(agent.online.training,net_mode)
        self.assertTrue(all(torch.equal(weights[k],v) for k,v in agent.online.state_dict().items()))


class PathwiseMetricsTests(unittest.TestCase):
    def test_wrong_keep_and_units(self):
        labels=[];q=[]
        for action,gain,delta in ((0,0.,0.),(1,30.,-.1),(2,-20.,-.15)):
            labels.append(dict(state_id='s',candidate_fingerprint='c',continuation_policy='greedy',
                action_index=action,delta_cost_path_pi=gain,label_kind='realized_path_fixed_continuation'))
            if action:
                q.append(dict(checkpoint_label='A0_42_EP400',state_id='s',candidate_fingerprint='c',
                    action_index=action,delta_q=delta,selected_action=0,scenario='shock',
                    initial_budget=2,remaining_budget=1))
        states,actions=compare(q,labels)
        self.assertEqual(len(states),1)
        self.assertEqual(states[0]['pathwise_wrong_keep'],1)
        self.assertEqual(states[0]['missed_gain'],30.)
        self.assertEqual(states[0]['decision_regret'],30.)
        self.assertEqual(len(actions),2)
        self.assertEqual(summarize(states)[0]['wrong_keep_count'],1)


class ValidationIntegrationContract(unittest.TestCase):
    def test_cache_off_miss_hit_identical_result_rows(self):
        """Patched API boundaries, NOT a substitute for real port evaluation."""
        from unittest.mock import patch
        from v12_protocol import validation as module
        from types import SimpleNamespace
        import os
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            case=Path(folder)/'8.txt';case.write_text('synthetic-case')
            cfg=dict(candidate_limit=4,clock='union',period=12.,reward_scale=100.,
                     validation_uncertainty=8.,updates=2,candidate_screen='positive',
                     min_gain=.1,budget_features='absolute',value_features='global')
            calls=[]
            def fake_run(instance,budget,method,agent=None,**kw):
                calls.append(method)
                return dict(wait=float({'keep':100,'greedy':90,'periodic':85,'ddqn':88}[method]),
                            spent=int(method!='keep'),runtime_seconds=1.,steps=10,
                            return_unscaled=0.,edits_by_clock=0,edits_by_eta=0,
                            last_revision_time=0.,screened_proposals=0,
                            mean_estimated_edit_gain=0.)
            def fake_paired(**kw):
                rows=[]
                for method in ('keep','greedy','ddqn','periodic'):
                    item=fake_run(None,kw['budgets'][0],method)
                    row=dict(case=str(case),scenario='shock',seed=400001,budget=2,
                             method=method,**item)
                    row.update(gain_vs_keep=100.-item['wait'],
                               gain_vs_periodic=85.-item['wait'],
                               regret_vs_keep=max(0.,item['wait']-100.))
                    rows.append(row)
                return rows,[]
            before=(module._CFG,module._CACHE_DIR,module._CACHE_CODE_HASHES)
            try:
                module._CFG=cfg;module._CACHE_CODE_HASHES={'core':'fixed'}
                module._AGENT=object()
                with patch.object(module,'read_case',return_value=([],None)),\
                     patch.object(module,'generate_scenario',return_value=object()),\
                     patch.object(module,'run_episode',side_effect=fake_run),\
                     patch.object(module,'evaluate_paired',side_effect=fake_paired):
                    module._CACHE_DIR=None
                    old=module._one((str(case),'shock',400001,2))
                    module._CACHE_DIR=folder+'/cache'
                    first=module._one((str(case),'shock',400001,2))
                    second=module._one((str(case),'shock',400001,2))
                    self.assertEqual(old['rows'],first['rows'])
                    self.assertEqual(first['rows'],second['rows'])
                    self.assertEqual((first['hit'],first['miss']),(0,1))
                    self.assertEqual((second['hit'],second['miss']),(1,0))
                    self.assertEqual(len([x for x in calls if x=='ddqn']),3)
                    cfg['min_gain']=.2
                    changed=module._one((str(case),'shock',400001,2))
                    self.assertEqual(changed['miss'],1)
            finally:
                module._CFG,module._CACHE_DIR,module._CACHE_CODE_HASHES=before


if __name__=='__main__':unittest.main()
