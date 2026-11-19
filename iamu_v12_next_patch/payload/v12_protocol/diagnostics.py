"""Non-interventional TD statistics from the actual target and chosen Q.

The agent calls TDRecorder AFTER the existing optimizer/target update; this
recorder only copies detached scalar vectors to CPU. No new forward/backward,
RNG draw, sampler call or network mutation is permitted.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

CATEGORIES=('forced_keep','choice_keep','edit_last_budget','edit_other')
LENGTH_BINS=('1','2-4','5-16','17-64','65+')
FIELDS=('gradient_window_end','group','macro_length_bin','n','q_taken_mean',
        'target_mean','td_error_mean','abs_td_mean','abs_td_sd','abs_td_p50',
        'abs_td_p90','abs_td_p99','huber_linear_fraction',
        'grad_norm_preclip_mean','target_sync_count','sampled_batches')


def category(transition):
    if transition.action==0:
        return 'choice_keep' if bool(transition.obs_mask[1:].any()) else 'forced_keep'
    return 'edit_last_budget' if int(transition.source_budget)==1 else 'edit_other'


def length_bin(n):
    if n<1:raise ValueError('Non-positive macro interval')
    if n==1:return '1'
    if n<=4:return '2-4'
    if n<=16:return '5-16'
    if n<=64:return '17-64'
    return '65+'


class TDRecorder:
    def __init__(self,sample_every=25,window=250):
        if sample_every<1 or window<1 or window%sample_every:
            raise ValueError('window must divide sample_every')
        self.sample_every=sample_every
        self.window=window
        self.samples=defaultdict(list)
        self.grad_norms=[]
        self.target_syncs=0
        self.sampled_batches=0
        self.completed=[]

    def __call__(self,*,step,transitions,q_taken,target,td_error,
                 grad_norm_preclip,target_sync):
        """Detached CPU tensors (or NumPy arrays) are required by contract."""
        if step % self.sample_every:return
        q=np.asarray(q_taken).reshape(-1)
        y=np.asarray(target).reshape(-1)
        td=np.asarray(td_error).reshape(-1)
        if len(transitions)!=len(q) or len(y)!=len(q) or len(td)!=len(q):
            raise AssertionError('TD callback sample shape mismatch')
        if not (np.isfinite(q).all() and np.isfinite(y).all() and np.isfinite(td).all()):
            raise ValueError('Non-finite TD diagnostics')
        for t,pred,targ,err in zip(transitions,q,y,td):
            k=(category(t),length_bin(int(t.n_raw_steps)))
            self.samples[k].append((float(pred),float(targ),float(err)))
        self.grad_norms.append(float(grad_norm_preclip))
        self.target_syncs+=int(bool(target_sync))
        self.sampled_batches+=1
        if step%self.window==0:self.flush(step)

    def flush(self,step):
        # If the last sampled window ended mid-window, flush its partial sample
        # with its actual end step, but never turn missing samples into zeros.
        from statistics import mean
        all_rows=[]
        for group in CATEGORIES:
            for length in LENGTH_BINS:
                sample=self.samples.get((group,length),[])
                n=len(sample)
                if n:
                    array=np.asarray(sample,dtype=np.float64)
                    err=np.abs(array[:,2])
                    stats=dict(q_taken_mean=float(array[:,0].mean()),
                         target_mean=float(array[:,1].mean()),
                         td_error_mean=float(array[:,2].mean()),
                         abs_td_mean=float(err.mean()),abs_td_sd=float(err.std()),
                         abs_td_p50=float(np.quantile(err,.5)),
                         abs_td_p90=float(np.quantile(err,.9)),
                         abs_td_p99=float(np.quantile(err,.99)),
                         huber_linear_fraction=float(np.mean(err>1.)))
                else:stats={k:'' for k in ('q_taken_mean','target_mean','td_error_mean',
                      'abs_td_mean','abs_td_sd','abs_td_p50','abs_td_p90',
                      'abs_td_p99','huber_linear_fraction')}
                all_rows.append(dict(gradient_window_end=step,group=group,
                    macro_length_bin=length,n=n,**stats,
                    grad_norm_preclip_mean=(mean(self.grad_norms) if self.grad_norms else ''),
                    target_sync_count=self.target_syncs,sampled_batches=self.sampled_batches))
        self.completed.extend(all_rows)
        self.samples.clear();self.grad_norms.clear()
        self.target_syncs=0;self.sampled_batches=0

    def finish(self,step):
        if self.sampled_batches:self.flush(step)
        return self.completed

    def write(self,path,step):
        self.finish(step)
        dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            writer.writeheader();writer.writerows(self.completed)
        return len(self.completed)
