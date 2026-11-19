#!/usr/bin/env python3
"""One immutable EP200 Double-DQN bootstrap reference for later checkpoints.

A fixed selector AND a fixed evaluator from the reference checkpoint produce
labels once. This is a diagnostic baseline, NOT optimal or realized returns.
Full Replay pickles must be locally trusted. No gradients, no env futures.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v11_protocol.train_v11 import sha

def make_labels(agent,examples,device='cpu',batch_size=128):
    import torch,numpy as np
    result=[]
    agent.online.eval();agent.target.eval()
    with torch.no_grad():
        for k in range(0,len(examples),batch_size):
            batch=examples[k:k+batch_size]
            def stack(key,dtype):
                return torch.as_tensor(np.stack([getattr(t,key) for t in batch]),device=agent.device,dtype=dtype)
            ng=stack('next_global',torch.float32)
            na=stack('next_actions',torch.float32)
            nm=stack('next_mask',torch.bool)
            reward=torch.tensor([x.reward for x in batch],device=agent.device)
            done=torch.tensor([x.done for x in batch],device=agent.device,dtype=torch.float32)
            pred=agent.online(ng,na,nm).masked_fill(~nm,-torch.inf).clone()
            pred[:,0]+=agent.bootstrap_margin
            action=pred.argmax(dim=1)
            val=agent.target(ng,na,nm).gather(1,action[:,None]).squeeze(1)
            labels=(reward+agent.gamma*(1-done)*val).cpu().tolist()
            result+=labels
    return result

def predicted_chosen(agent,examples,batch_size=128):
    import numpy as np,torch
    x=[];agent.online.eval()
    with torch.no_grad():
        for k in range(0,len(examples),batch_size):
            b=examples[k:k+batch_size]
            og=torch.as_tensor(np.stack([t.obs_global for t in b]),device=agent.device)
            oa=torch.as_tensor(np.stack([t.obs_actions for t in b]),device=agent.device)
            om=torch.as_tensor(np.stack([t.obs_mask for t in b]),device=agent.device,dtype=torch.bool)
            selected=torch.tensor([t.action for t in b],dtype=torch.long,device=agent.device)
            x.extend(agent.online(og,oa,om).gather(1,selected[:,None]).squeeze(1).cpu().tolist())
    return x

def main(argv=None):
    import numpy as np,torch
    from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent
    from iamu_dynamic.rl_experiment_v11 import write_csv
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork',required=True,help='Trusted training_state.pt from NEW v11 run at EP200')
    p.add_argument('--reference-checkpoint',required=True)
    p.add_argument('--compare',action='append',required=True,help='Repeat: checkpoints evaluated against SAME labels')
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--seed',type=int,default=2387)
    p.add_argument('--trust-local-training-state',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if not a.trust_local_training_state:p.error('Must explicitly trust the local pickle training state')
    if a.samples<1:p.error('Samples must be positive')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use a new output directory')
    fork=torch.load(a.fork,map_location='cpu',weights_only=False)
    # Local, trusted file only; the reference weights must EXACTLY match fork online.
    ref,_=DoubleDQNAgent.from_checkpoint(a.reference_checkpoint,'cpu')
    for name,values in ref.online.state_dict().items():
        if not torch.equal(values,fork['online'][name]):
            raise AssertionError('Reference checkpoint does not match fork online network')
    ref.target.load_state_dict(fork['target'])
    rng=random.Random(a.seed)
    replay=fork['replay_items']
    indices=rng.sample(range(len(replay)),min(a.samples,len(replay)))
    examples=[replay[i] for i in indices]
    labels=make_labels(ref,examples)
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in a.compare:
        agent,_=DoubleDQNAgent.from_checkpoint(name,'cpu')
        pred=predicted_chosen(agent,examples)
        for i,(t,y,predicted) in enumerate(zip(examples,labels,pred)):
            rows.append(dict(model=Path(name).name,checkpoint_sha256=sha(name),
                index=indices[i],kind=('edit' if t.action else
                      'choice_keep' if t.obs_mask[1:].any() else 'forced_keep'),
                fixed_label=y,predicted=predicted,residual=y-predicted,
                abs_error=abs(y-predicted)))
    write_csv(out/'fixed_label_errors.csv',rows)
    (out/'manifest.json').write_text(json.dumps(dict(fork_sha256=sha(a.fork),
        selector_evaluator_sha256=sha(a.reference_checkpoint),reference_bootstrap_margin=ref.bootstrap_margin,
        sample_indices=indices,config=vars(a),truth_warning='Fixed target is bootstrap reference, not counterfactual return or Q*'),indent=2))
    print(f'[DONE] examples={len(examples)}, model comparisons={len(a.compare)} => {out}')

if __name__=='__main__':main()
