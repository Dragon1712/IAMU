"""Candidate-conditioned Double DQN for IAMU's hard per-episode edit budget.

Inspired by van Hasselt et al. (AAAI 2016), with observable remaining budget
as an explicit state input (Carrara et al., NeurIPS 2019). This implementation
is NOT a reproduction of Carrara's Budgeted Fitted-Q or an original paper's
pretrained weights. It is a deliberately small research baseline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .rl_env import GLOBAL_DIM, ACTION_DIM, RLObservation


class CandidateQNet(nn.Module):
    """Shared Q(s,a) encoder; supports a variable count of legal proposals."""
    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.layers=nn.Sequential(
            nn.Linear(GLOBAL_DIM + ACTION_DIM,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,1))

    def forward(self, global_features, action_features):
        # global_features: [B,G], action_features: [B,A,F]
        if global_features.ndim != 2 or action_features.ndim != 3:
            raise ValueError('Expected global [B,G], candidate [B,A,F]')
        context=global_features.unsqueeze(1).expand(-1,action_features.shape[1],-1)
        return self.layers(torch.cat((context,action_features),dim=-1)).squeeze(-1)


@dataclass
class Transition:
    obs_global: np.ndarray
    obs_actions: np.ndarray
    obs_mask: np.ndarray
    action: int
    reward: float
    next_global: np.ndarray
    next_actions: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self,capacity: int,seed: int):
        if capacity <= 0:raise ValueError('capacity must be positive')
        self.items=deque(maxlen=capacity)
        self.rng=random.Random(seed)

    def __len__(self):return len(self.items)

    def append(self,ob:RLObservation,action:int,reward:float,
               nxt:RLObservation,done:bool):
        self.items.append(Transition(ob.global_features.copy(),
            ob.action_features.copy(),ob.mask.copy(),int(action),float(reward),
            nxt.global_features.copy(),nxt.action_features.copy(),
            nxt.mask.copy(),bool(done)))

    def sample(self,batch_size:int):
        return self.rng.sample(self.items,batch_size)


class DoubleDQNAgent:
    def __init__(self, hidden_dim:int=96, learning_rate:float=3e-4,
                 gamma:float=1.0, target_update_steps:int=250,
                 seed:int=0,device:str='cpu'):
        if not 0<gamma<=1:raise ValueError('gamma must be in (0,1]')
        if target_update_steps<1:raise ValueError('target_update_steps >=1')
        torch.manual_seed(seed)
        self.device=torch.device(device)
        self.online=CandidateQNet(hidden_dim).to(self.device)
        self.target=CandidateQNet(hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=learning_rate)
        self.gamma=gamma
        self.target_update_steps=target_update_steps
        self.gradient_steps=0
        self.hidden_dim=hidden_dim
        self.rng=random.Random(seed+193)

    def act(self,obs:RLObservation,epsilon:float=0.):
        legal=np.flatnonzero(obs.mask)
        if not len(legal):raise AssertionError('Keep must always be legal')
        if self.rng.random()<epsilon:
            return int(self.rng.choice(legal.tolist()))
        self.online.eval()
        with torch.no_grad():
            g=torch.from_numpy(obs.global_features).to(self.device).unsqueeze(0)
            a=torch.from_numpy(obs.action_features).to(self.device).unsqueeze(0)
            values=self.online(g,a).squeeze(0)
            mask=torch.from_numpy(obs.mask).to(self.device)
            return int(values.masked_fill(~mask,-torch.inf).argmax().item())

    def update(self,transitions:list[Transition]):
        if not transitions:raise ValueError('Empty batch')
        self.online.train()
        def batch(field,dtype):
            return torch.as_tensor(np.stack([getattr(x,field) for x in transitions]),
                                  device=self.device,dtype=dtype)
        obs_g=batch('obs_global',torch.float32)
        obs_a=batch('obs_actions',torch.float32)
        nxt_g=batch('next_global',torch.float32)
        nxt_a=batch('next_actions',torch.float32)
        nxt_mask=batch('next_mask',torch.bool)
        act=torch.as_tensor([x.action for x in transitions],
                            dtype=torch.long,device=self.device)
        rewards=torch.as_tensor([x.reward for x in transitions],
                                dtype=torch.float32,device=self.device)
        dones=torch.as_tensor([x.done for x in transitions],
                              dtype=torch.float32,device=self.device)
        predicted=self.online(obs_g,obs_a).gather(1,act[:,None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: online picks action, frozen target EVALUATES it.
            next_action=self.online(nxt_g,nxt_a).masked_fill(~nxt_mask,-torch.inf).argmax(dim=1)
            next_target=self.target(nxt_g,nxt_a).gather(1,next_action[:,None]).squeeze(1)
            bellman=rewards+self.gamma*(1-dones)*next_target
        loss=F.smooth_l1_loss(predicted,bellman)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.optimizer.step()
        self.gradient_steps+=1
        if self.gradient_steps % self.target_update_steps==0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().item())

    def checkpoint(self,meta:dict):
        return dict(online=self.online.state_dict(),meta=meta,
                    hidden_dim=self.hidden_dim,global_dim=GLOBAL_DIM,
                    action_dim=ACTION_DIM)

    @classmethod
    def from_checkpoint(cls,path:str,device:str='cpu'):
        # weights_only protects against arbitrary pickle object execution.
        item=torch.load(path,map_location='cpu',weights_only=True)
        if item.get('global_dim')!=GLOBAL_DIM or item.get('action_dim')!=ACTION_DIM:
            raise ValueError('Checkpoint features do not match environment')
        agent=cls(hidden_dim=item['hidden_dim'],device=device)
        agent.online.load_state_dict(item['online'])
        agent.target.load_state_dict(item['online'])
        return agent,item['meta']
