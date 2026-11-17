"""Incremental O(1) Replay stratum maintenance, O(batch_size) sample draw.

Storage uses stable ring slots, separate dense index pools, and an items facade
compatible with parallel_recovery_v081 (append/clear/extend/list). No full
Replay scan on learner updates. Per-batch samples are distinct; reuse across
updates is expected. If a stratum is undersized, *top up from the other stratum*
without replacement, logging effective composition; fail if total is too small.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,random

class DensePool:
    def __init__(self):self.values=[];self.loc={}
    def add(self,i):
        if i in self.loc:raise AssertionError('Duplicate replay index')
        self.loc[i]=len(self.values);self.values.append(i)
    def discard(self,i):
        pos=self.loc.pop(i)
        last=self.values.pop()
        if pos<len(self.values):self.values[pos]=last;self.loc[last]=pos
    def __len__(self):return len(self.values)

class ItemsFacade:
    def __init__(self,owner):self.owner=owner
    def append(self,x):self.owner.append_transition(x)
    def extend(self,it):
        for x in it:self.append(x)
    def clear(self):self.owner.clear()
    def __len__(self):return self.owner._size
    def __iter__(self):
        for n in range(self.owner._size):yield self[n]
    def __getitem__(self,n):
        o=self.owner
        if isinstance(n,slice):return [self[i] for i in range(*n.indices(o._size))]
        if n<0:n+=o._size
        if n<0 or n>=o._size:raise IndexError(n)
        return o._slots[(o._head+n)%o.capacity]

class IndexedReplay:
    def __init__(self,capacity,seed):
        if capacity<1:raise ValueError('Invalid replay capacity')
        self.capacity=int(capacity);self.rng=random.Random(seed)
        self._slots=[None]*capacity;self._head=0;self._size=0
        self.actionable=DensePool();self.forced=DensePool()
        self.items=ItemsFacade(self)
    def __len__(self):return self._size
    @staticmethod
    def _actionable(t):return bool(t.obs_mask[1:].any())
    def append_transition(self,t):
        if not bool(t.obs_mask[t.action]):raise ValueError('Invalid action stored in Replay')
        is_actionable=self._actionable(t)
        if not is_actionable and t.action!=0:raise ValueError('Non-Keep without legal Edit')
        if self._size==self.capacity:
            slot=self._head
            (self.actionable if self._actionable(self._slots[slot]) else self.forced).discard(slot)
            self._head=(self._head+1)%self.capacity
        else:
            slot=(self._head+self._size)%self.capacity
            self._size+=1
        self._slots[slot]=t
        (self.actionable if is_actionable else self.forced).add(slot)
    def clear(self):
        self._slots=[None]*self.capacity;self._head=0;self._size=0
        self.actionable=DensePool();self.forced=DensePool()
    def stats(self):
        return dict(n=self._size,actionable=len(self.actionable),forced=len(self.forced),
            natural_fraction=len(self.actionable)/max(1,self._size))
    def sample(self,n):
        if n>len(self):raise ValueError('Insufficient Replay samples')
        indices=self.rng.sample(range(self._size),n)
        return [self.items[i] for i in indices]
    def sample_stratified(self,n,rho,*,importance=False):
        if not math.isfinite(rho) or not 0<=rho<=1:raise ValueError('rho outside [0,1]')
        if n>len(self):raise ValueError('Insufficient Replay')
        desired=int(math.floor(n*rho+0.5))
        count_actionable=min(desired,len(self.actionable))
        count_forced=min(n-count_actionable,len(self.forced))
        if count_actionable+count_forced<n:
            count_actionable=min(len(self.actionable),n-count_forced)
        if count_actionable+count_forced<n:
            count_forced=min(len(self.forced),n-count_actionable)
        if count_actionable+count_forced!=n:raise RuntimeError('Sampling shortfall despite adequate total Replay')
        slots=self.rng.sample(self.actionable.values,count_actionable)+self.rng.sample(self.forced.values,count_forced)
        self.rng.shuffle(slots)
        actual=count_actionable/n
        if importance:
            if actual in (0.,1.):
                raise ValueError('Cannot restore original distribution with missing stratum')
            base=len(self.actionable)/len(self)
            weights=[base/actual if self._actionable(self._slots[i]) else
                     (1-base)/(1-actual) for i in slots]
        else:weights=None
        batch=[self._slots[i] for i in slots]
        meta=dict(target_rho=rho,actual_rho=actual,actionable=count_actionable,
            forced=count_forced,edited=sum(t.action!=0 for t in batch),
            distinct_slots=len(set(slots)),shortfall=int(count_actionable!=desired),
            population_actionable=len(self.actionable),population_forced=len(self.forced))
        return batch,weights,meta
