from __future__ import annotations
import unittest
from dataclasses import dataclass
import numpy as np
from v11_protocol.indexed_replay import IndexedReplay

@dataclass
class Fake:
    action:int
    obs_mask:np.ndarray
    episode_id:int

def tr(i):
    eligible=bool(i%3)
    return Fake(1 if eligible and i%2 else 0,np.array([True, eligible]),i)

class TestSnapshotV12(unittest.TestCase):
    def test_ring_order_and_sample_sequence_across_eviction(self):
        a=IndexedReplay(17,981)
        for i in range(100):a.items.append(tr(i))
        self.assertNotEqual(a._head,0)
        b=IndexedReplay(17,981)
        b.load_state_dict(a.state_dict())
        self.assertEqual([x.episode_id for x in a.items],[x.episode_id for x in b.items])
        self.assertEqual(a.actionable.values,b.actionable.values)
        self.assertEqual(a.forced.values,b.forced.values)
        for _ in range(25):
            lhs,_,_=a.sample_stratified(10,.5)
            rhs,_,_=b.sample_stratified(10,.5)
            self.assertEqual([x.episode_id for x in lhs],[x.episode_id for x in rhs])
    def test_rejects_corrupted_pool(self):
        a=IndexedReplay(6,2)
        for i in range(6):a.items.append(tr(i))
        state=a.state_dict()
        state['actionable'].append(state['forced'][0])
        with self.assertRaises(ValueError): IndexedReplay(6,2).load_state_dict(state)

if __name__=='__main__':unittest.main()
