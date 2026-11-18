"""Whole-episode FIFO Replay with deterministic index ordering and exact restore."""
from __future__ import annotations
from collections import deque
import random, math
from .transitions import MacroTransition

class EpisodeReplay:
    def __init__(self, max_episodes:int=200, seed:int=0, representation:str='primitive'):
        if max_episodes<1:raise ValueError('max_episodes must be >=1')
        if representation not in ('primitive','budget_tail','decision_epoch'):
            raise ValueError('Unknown representation')
        self.max_episodes=int(max_episodes)
        self.representation=representation
        self.rng=random.Random(seed)
        self.episodes=deque()  # entries: (episode ID, tuple[MacroTransition], raw count)
        self._indices=[]       # [(episode ID, position),...] in exact insertion order
        self._items=[]
        self.actionable_indices=[]
        self.forced_indices=[]
        self.sample_counts={}
        self.evictions=0
        self.seen_episodes=0

    def __len__(self):return len(self._items)
    def _reindex(self):
        self._indices=[];self._items=[];self.actionable_indices=[];self.forced_indices=[]
        for eid,trs,_ in self.episodes:
            for j,t in enumerate(trs):
                pos=len(self._items)
                self._indices.append((eid,j));self._items.append(t)
                (self.actionable_indices if bool(t.obs_mask[1:].any()) else self.forced_indices).append(pos)
        self.sample_counts={k:v for k,v in self.sample_counts.items() if k in self._indices}

    def append_episode(self, episode_id:int,transitions,raw_steps:int):
        if raw_steps<1:raise ValueError('Empty raw episode')
        if self.episodes and episode_id<=self.episodes[-1][0]:
            raise ValueError('Episodes must be strictly ordered and unique')
        ts=tuple(transitions)
        for t in ts:
            if not isinstance(t,MacroTransition) or t.episode_id!=episode_id:
                raise ValueError('Wrong episode or transition class')
            if not bool(t.obs_mask[t.action]):raise ValueError('Stored illegal action')
            if self.representation=='decision_epoch' and not bool(t.obs_mask[1:].any()):
                raise ValueError('Decision epoch contains forced Keep state')
        self.episodes.append((int(episode_id),ts,int(raw_steps)))
        self.seen_episodes+=1
        if len(self.episodes)>self.max_episodes:
            self.episodes.popleft();self.evictions+=1
        self._reindex()

    def sample(self,batch_size:int,mode:str='uniform',rho:float=.25):
        if batch_size<1 or batch_size>len(self):raise ValueError('Insufficient Replay pool')
        if mode not in ('uniform','stratified'):raise ValueError('Unknown sampling mode')
        if mode=='stratified' and self.representation!='primitive':
            raise ValueError('Stratified sampling only allowed for primitive (A1)')
        if not 0<=rho<=1 or not math.isfinite(rho):raise ValueError('Invalid rho')
        if mode=='uniform':
            indices=self.rng.sample(range(len(self)),batch_size)
        else:
            need=int(math.floor(batch_size*rho+.5))
            na=min(need,len(self.actionable_indices))
            nf=min(batch_size-na,len(self.forced_indices))
            if na+nf<batch_size:na=min(len(self.actionable_indices),batch_size-nf)
            if na+nf<batch_size:nf=min(len(self.forced_indices),batch_size-na)
            if na+nf!=batch_size:raise AssertionError('Population accounting failed')
            indices=(self.rng.sample(self.actionable_indices,na)+
                     self.rng.sample(self.forced_indices,nf))
            self.rng.shuffle(indices)
        chosen=[self._items[i] for i in indices]
        for i in indices:
            k=self._indices[i];self.sample_counts[k]=self.sample_counts.get(k,0)+1
        return chosen,dict(sample_actionable=sum(bool(x.obs_mask[1:].any()) for x in chosen),
                           sample_size=batch_size,shortfall=(mode=='stratified' and
                           sum(bool(x.obs_mask[1:].any()) for x in chosen)!=int(math.floor(batch_size*rho+.5))),
                           unique=len(set(indices)))

    def stats(self):
        n=len(self);unique=len(self.sample_counts)
        # NumPy buffer bytes; Python allocator overhead is additional.
        buffer_bytes=sum(getattr(x,field).nbytes for x in self._items
                         for field in ('obs_global','obs_actions','obs_mask',
                                       'next_global','next_actions','next_mask'))
        return dict(n=n,actionable=len(self.actionable_indices),forced=len(self.forced_indices),
                    replay_episode_count=len(self.episodes),
                    replay_raw_coverage=sum(x[2] for x in self.episodes),
                    max_macro_length=max((x.n_raw_steps for x in self._items),default=0),
                    mean_macro_length=(sum(x.n_raw_steps for x in self._items)/n if n else 0.),
                    sample_reuse=(sum(self.sample_counts.values())/max(1,unique)),
                    replay_numpy_mib=buffer_bytes/1024**2,
                    sampled_unique=unique,evictions=self.evictions)

    def state_dict(self):
        return dict(version=1,max_episodes=self.max_episodes,representation=self.representation,
                    episodes=list(self.episodes),indices=list(self._indices),
                    actionable_indices=list(self.actionable_indices),forced_indices=list(self.forced_indices),
                    rng_state=self.rng.getstate(),sample_counts=dict(self.sample_counts),
                    evictions=self.evictions,seen_episodes=self.seen_episodes)

    def load_state_dict(self,state):
        if state['version']!=1 or state['max_episodes']!=self.max_episodes or state['representation']!=self.representation:
            raise ValueError('Replay checkpoint representation/window mismatch')
        self.episodes=deque(state['episodes'])
        self._reindex()
        if (self._indices!=state['indices'] or
            self.actionable_indices!=state['actionable_indices'] or
            self.forced_indices!=state['forced_indices']):
            raise ValueError('Replay ordered index pools differ on recovery')
        self.rng.setstate(state['rng_state'])
        self.sample_counts=dict(state['sample_counts'])
        self.evictions=int(state['evictions'])
        self.seen_episodes=int(state['seen_episodes'])
