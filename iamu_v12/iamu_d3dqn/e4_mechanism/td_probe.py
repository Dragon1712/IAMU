"""Double-DQN TD residuals, with EXACT training margin and action masks.

Residual delta = r + gamma * target_Q(s',argmax(online_Q+Keep_margin))
                    - online_Q(s,a).
All computation is no-grad and does not mutate optimizer/target states.
"""
from __future__ import annotations
import math
from statistics import mean


def _summ(x):
    if not x:return dict(n=0,mean_delta=None,mae=None,rmse=None,p90_abs=None,
                          mean_pred=None,mean_target=None,positive_frac=None)
    ds=[z['delta'] for z in x];absd=sorted(abs(d) for d in ds)
    return dict(n=len(x),mean_delta=mean(ds),
        mae=mean(absd),rmse=math.sqrt(mean(d*d for d in ds)),
        p90_abs=absd[min(len(absd)-1,math.ceil(.9*len(absd))-1)],
        mean_pred=mean(z['pred'] for z in x),
        mean_target=mean(z['target'] for z in x),
        positive_frac=sum(d>0 for d in ds)/len(ds))


def td_rows(learner, transitions, *, batch_size=128, target_override=None):
    """Return per-transition residual and explanatory attrs on selected Replay.

    target_override can be an immutable target-net snapshot to distinguish
    Bellman residual drift caused by target-network updates.
    """
    import numpy as np
    import torch
    x=list(transitions)
    if not x:return []
    target=target_override if target_override is not None else learner.target
    online_train=learner.online.training
    target_train=target.training
    learner.online.eval();target.eval()
    rows=[]
    with torch.no_grad():
        for pos in range(0,len(x),batch_size):
            sub=x[pos:pos+batch_size]
            def stack(key,dtype):
                a=np.stack([getattr(t,key) for t in sub])
                return torch.as_tensor(a,device=learner.device,dtype=dtype)
            g=stack('obs_global',torch.float32)
            a=stack('obs_actions',torch.float32)
            mask=stack('obs_mask',torch.bool)
            ng=stack('next_global',torch.float32)
            na=stack('next_actions',torch.float32)
            nm=stack('next_mask',torch.bool)
            selected=torch.tensor([t.action for t in sub],device=learner.device,dtype=torch.long)
            rewards=torch.tensor([t.reward for t in sub],device=learner.device,dtype=torch.float32)
            done=torch.tensor([t.done for t in sub],device=learner.device,dtype=torch.float32)
            if not bool(mask[:,0].all()) or not bool(nm[:,0].all()) or not bool(mask.gather(1,selected[:,None]).all()):
                raise AssertionError('Bad Replay action mask')
            prediction=learner.online(g,a,mask).gather(1,selected[:,None]).squeeze(1)
            nxt=learner.online(ng,na,nm).masked_fill(~nm,-torch.inf).clone()
            nxt[:,0]+=learner.keep_margin
            greedy=nxt.argmax(dim=1)
            target_q=target(ng,na,nm).gather(1,greedy[:,None]).squeeze(1)
            bellman=rewards+learner.gamma*(1-done)*target_q
            delta=bellman-prediction
            for i,tr in enumerate(sub):
                rows.append(dict(action=tr.action,kind=('edit' if tr.action else
                                  'choice_keep' if bool(tr.obs_mask[1:].any()) else 'forced_keep'),
                    budget=round(float(tr.obs_global[19])*10),
                    done=int(tr.done),pred=float(prediction[i]),
                    target=float(bellman[i]),delta=float(delta[i]),
                    reward=float(rewards[i]),next_selected=int(greedy[i])))
    learner.online.train(online_train);target.train(target_train)
    return rows


def td_summary(rows,*,name='current'):
    subgroups={
      'all':rows,
      'edit':[x for x in rows if x['kind']=='edit'],
      'choice_keep':[x for x in rows if x['kind']=='choice_keep'],
      'forced_keep':[x for x in rows if x['kind']=='forced_keep'],
      'b2':[x for x in rows if x['budget']==2],
      'b5':[x for x in rows if x['budget']==5],
      'b10':[x for x in rows if x['budget']==10]}
    return [dict(probe=name,subset=k,**_summ(v)) for k,v in subgroups.items()]
