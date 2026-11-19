"""Standalone tests without IAMU/CUDA. Real API integration on server separately."""
from __future__ import annotations
import sys,unittest,random
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from v11_protocol.indexed_replay import IndexedReplay
from v11_protocol.rounds import protocol_rounds
from v11_protocol.build_variants import change,replace_span

@dataclass
class Transition:
    action:int
    obs_mask:np.ndarray
    episode_id:int|None=None

def tr(action,act=True):return Transition(action,np.array([True,bool(act)],dtype=bool))

class TestIndexedReplay(unittest.TestCase):
    def test_eviction_and_strata_stay_synced(self):
        b=IndexedReplay(8,7)
        for i in range(100):
            b.items.append(tr(i%2,True) if i%3 else tr(0,False))
            self.assertEqual(b.stats()['n'],min(i+1,8))
            self.assertEqual(b.stats()['actionable']+b.stats()['forced'],len(b))
            self.assertEqual(b.stats()['actionable'],sum(t.obs_mask[1:].any() for t in b.items))
        self.assertEqual(len(list(b.items)),8)
        self.assertEqual(len(b.items[:3]),3)
    def test_stratified_fraction_within_batch_and_sample_unique(self):
        b=IndexedReplay(1000,98)
        b.items.extend([tr(0,False) for _ in range(700)])
        b.items.extend([tr(0,True) for _ in range(150)])
        b.items.extend([tr(1,True) for _ in range(150)])
        batch,weights,m=b.sample_stratified(64,.25)
        self.assertEqual(m['actionable'],16)
        self.assertEqual(m['forced'],48)
        self.assertEqual(m['distinct_slots'],64)
        self.assertEqual(len({id(t) for t in batch}),64)
        self.assertIsNone(weights)
    def test_oversubscription_backfill_and_metadata(self):
        b=IndexedReplay(200,1)
        b.items.extend([tr(0,False) for _ in range(190)])
        b.items.extend([tr(1,True) for _ in range(10)])
        batch,w,m=b.sample_stratified(64,.75)
        self.assertEqual(m['actionable'],10)
        self.assertEqual(m['forced'],54)
        self.assertEqual(m['shortfall'],1)
        self.assertAlmostEqual(m['actual_rho'],10/64)
    def test_importance_weights_recover_original_mixture(self):
        b=IndexedReplay(1000,45)
        b.items.extend([tr(0,False) for _ in range(800)])
        b.items.extend([tr(1,True) for _ in range(200)])
        batch,w,m=b.sample_stratified(100,.5,importance=True)
        self.assertAlmostEqual(sum(w)/len(w),1.)
        self.assertAlmostEqual(sum(wi for t,wi in zip(batch,w) if t.obs_mask[1:].any())/len(w),.2)
    def test_importance_cannot_reweight_missing_stratum(self):
        b=IndexedReplay(200,2)
        b.items.extend([tr(0,False) for _ in range(100)])
        b.items.extend([tr(1,True) for _ in range(100)])
        with self.assertRaises(ValueError):b.sample_stratified(64,1.,importance=True)
    def test_recovery_facade_clear_extend(self):
        b=IndexedReplay(20,7);b.items.extend([tr(0,False),tr(1,True),tr(0,True)])
        old=list(b.items);b.items.clear();self.assertEqual(len(b),0)
        b.items.extend(old);self.assertEqual([x.action for x in b.items],[0,1,0])
        self.assertEqual(b.stats()['actionable'],2)
    def test_seed_stability(self):
        a,b=IndexedReplay(100,13),IndexedReplay(100,13)
        items=[tr(i%2,True) if i%5 else tr(0,False) for i in range(100)]
        for buf in (a,b):buf.items.extend(items)
        for _ in range(30):
            x,_,_=a.sample_stratified(16,.5)
            y,_,_=b.sample_stratified(16,.5)
            self.assertEqual([id(z) for z in x],[id(z) for z in y])

class TestProtocol(unittest.TestCase):
    def test_rounds_respect_validation_barrier(self):
        episodes=[type('Job',(),{'episode':i})() for i in range(1,21)]
        batches=protocol_rounds(episodes,7,10)
        self.assertEqual([[x.episode for x in rr] for rr in batches],
                         [[1,2,3,4,5,6,7],[8,9,10],[11,12,13,14,15,16,17],[18,19,20]])
    def test_builder_fail_closed(self):
        with self.assertRaises(RuntimeError):change('abc','not_present','d')
        with self.assertRaises(RuntimeError):replace_span('def foo():pass','abc','xyz','...')

if __name__=='__main__':unittest.main()
