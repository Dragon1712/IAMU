"""V1.2 Set-aware DDQN: V1.1 architecture with per-transition bootstrap discount."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent as V11Agent
from iamu_dynamic.rl_env_v11 import GLOBAL_DIM,ACTION_DIM
from .targets import ddqn_interval_target

class V12Agent(V11Agent):
    def __init__(self,*args,representation='primitive',reward_scale=100.,**kwargs):
        if representation not in ('primitive','budget_tail','decision_epoch'):
            raise ValueError('Invalid representation')
        if reward_scale<=0 or not math.isfinite(reward_scale):
            raise ValueError('Invalid reward_scale')
        super().__init__(*args,**kwargs)
        self.representation=representation
        self.reward_scale=float(reward_scale)

    def update(self,transitions,weights=None):
        if not transitions:raise ValueError('Empty batch')
        self.online.train()
        def batch(field,dtype):
            return torch.as_tensor(np.stack([getattr(x,field) for x in transitions]),
                                  device=self.device,dtype=dtype)
        og=batch('obs_global',torch.float32)
        oa=batch('obs_actions',torch.float32)
        om=batch('obs_mask',torch.bool)
        ng=batch('next_global',torch.float32)
        na=batch('next_actions',torch.float32)
        nm=batch('next_mask',torch.bool)
        actions=torch.as_tensor([x.action for x in transitions],device=self.device,dtype=torch.long)
        r=torch.as_tensor([x.reward for x in transitions],device=self.device,dtype=torch.float32)
        d=torch.as_tensor([x.bootstrap_discount for x in transitions],device=self.device,dtype=torch.float32)
        if not bool(om.gather(1,actions[:,None]).all()) or not bool(nm[:,0].all()):
            raise ValueError('Illegal Replay action/mask')
        if not bool(torch.isfinite(d).all()) or not bool(((d>=0)&(d<=1)).all()):
            raise ValueError('Invalid per-sample bootstrap discount')
        if any(bool(x.done)!=(x.bootstrap_discount==0.) for x in transitions):
            raise ValueError('Terminal flag and bootstrap discount disagree')
        if any(x.n_raw_steps<1 or abs(x.bootstrap_discount-(self.gamma**x.n_raw_steps))>1e-6
               for x in transitions if not x.done):
            raise ValueError('Discount length or gamma mismatches checkpoint')
        chosen=self.online(og,oa,om).gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            scores=self.online(ng,na,nm).masked_fill(~nm,-torch.inf).clone()
            scores[:,0]+=self.bootstrap_margin
            best=scores.argmax(dim=1)
            target=self.target(ng,na,nm).gather(1,best[:,None]).squeeze(1)
            y=ddqn_interval_target(r,d,target)
        losses=F.smooth_l1_loss(chosen,y,reduction='none')
        if weights is None:loss=losses.mean()
        else:
            w=torch.as_tensor(weights,device=self.device,dtype=torch.float32)
            if w.shape!=losses.shape or not bool(torch.isfinite(w).all()) or not bool((w>=0).all()):
                raise ValueError('Invalid sample weights')
            loss=(w*losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.optimizer.step()
        self.gradient_steps+=1
        if self.gradient_steps%self.target_update_steps==0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def checkpoint(self,meta=None):
        d=super().checkpoint(meta or {})
        d.update(protocol_version=12,representation=self.representation,
                 gamma=float(self.gamma),reward_scale=self.reward_scale,
                 feature_version='V1.1-global-absolute',
                 target_update_steps=self.target_update_steps)
        return d

    @classmethod
    def from_checkpoint(cls,path,device='cpu'):
        # Model-only checkpoints may load untrusted data: weights_only=True.
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d.get('protocol_version')!=12 or d.get('global_dim')!=GLOBAL_DIM or d.get('action_dim')!=ACTION_DIM:
            raise ValueError('Not a compatible V1.2 model checkpoint')
        if d.get('feature_version')!='V1.1-global-absolute':
            raise ValueError('Feature version mismatch')
        arch=d.get('q_network')
        if arch!='set_ddqn' or d.get('network_architecture_version')!=2:
            raise ValueError('V1.2 A0-A3 requires Set-aware DDQN architecture')
        agent=cls(hidden_dim=d['hidden_dim'],device=device,q_network=arch,
                  gamma=d['gamma'],representation=d['representation'],
                  reward_scale=d['reward_scale'],target_update_steps=d.get('target_update_steps',250),
                  execution_margin=d['execution_margin'],bootstrap_margin=d['bootstrap_margin'])
        agent.online.load_state_dict(d['online'],strict=True)
        agent.target.load_state_dict(d['online'],strict=True)
        return agent,d.get('meta',{})
