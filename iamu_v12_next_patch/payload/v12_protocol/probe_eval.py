"""Evaluate any existing V1.2 model on EXACTLY the same actionable states.

Model-only checkpoints reconstruct the ONLINE network. Their historical target
network and optimizer are NOT recoverable and no historical TD error is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .baseline_cache import file_sha
from .probe_bank import load_probe_bank


def evaluate_agent_on_probes(agent,states, batch_size=32):
    if batch_size<1:raise ValueError('batch_size must be >=1')
    online=agent.online
    original_mode=online.training
    rng_state=agent.rng.getstate()
    results=[]
    try:
        online.eval()
        with torch.no_grad():
            for k in range(0,len(states),batch_size):
                chunk=states[k:k+batch_size]
                g=torch.as_tensor(np.stack([s['obs_global'] for s in chunk]),
                                  dtype=torch.float32,device=agent.device)
                a=torch.as_tensor(np.stack([s['obs_actions'] for s in chunk]),
                                  dtype=torch.float32,device=agent.device)
                m=torch.as_tensor(np.stack([s['obs_mask'] for s in chunk]),
                                  dtype=torch.bool,device=agent.device)
                q=online(g,a,m).masked_fill(~m,-torch.inf).cpu().numpy()
                for state,values in zip(chunk,q):
                    valid=np.flatnonzero(state['obs_mask'])
                    edits=valid[valid>0]
                    if not len(edits) or 0 not in valid:raise ValueError('Non-actionable probe')
                    # Actual act() resolves ties/keep margin, consuming its own RNG.
                    # We restore the RNG exactly, so this diagnostic is side-effect free.
                    from iamu_dynamic.rl_env_v11 import RLObservation
                    obs=RLObservation(state['obs_global'],state['obs_actions'],state['obs_mask'],
                                      state['source_time'],'eta',state['remaining_budget'],
                                      tuple(state['candidate_ids']))
                    actual=int(agent.act(obs,epsilon=0.))
                    keep=float(values[0]);best=int(edits[np.argmax(values[edits])])
                    for idx in edits:
                        idx=int(idx)
                        results.append(dict(state_id=state['state_id'],
                           episode_id=state['episode_id'],raw_step_index=state['raw_step_index'],
                           case_sha256=state['case_sha256'],scenario=state['scenario'],
                           initial_budget=state['initial_budget'],remaining_budget=state['remaining_budget'],
                           candidate_fingerprint=state['candidate_fingerprint'],
                           action_index=idx,candidate_id=int(state['candidate_ids'][idx-1]),
                           q_keep=keep,q_edit=float(values[idx]),delta_q=float(values[idx]-keep),
                           best_edit=best,selected_action=actual,selected_is_edit=int(actual>0),
                           execution_margin=float(agent.execution_margin)))
    finally:
        agent.rng.setstate(rng_state)
        online.train(original_mode)
    return results


def evaluate_checkpoints(bank_path,checkpoints,out_dir,*,data=None,device='cpu',batch_size=32):
    from .agent import V12Agent
    from iamu_dynamic.rl_experiment_v11 import write_csv
    bank=load_probe_bank(bank_path,data)
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    collected=[];missing=[];checkpoints_meta=[]
    for item in checkpoints:
        if '=' not in item:raise ValueError('Use GROUP_SEED_EP=/path/to/checkpoint.pt')
        label,filename=item.split('=',1)
        path=Path(filename)
        if not path.is_file():
            missing.append(dict(label=label,path=filename,reason='missing_model_checkpoint'))
            continue
        agent,meta=V12Agent.from_checkpoint(str(path),device=device)
        rows=evaluate_agent_on_probes(agent,bank['states'],batch_size=batch_size)
        digest=file_sha(path)
        for row in rows:
            row.update(checkpoint_label=label,checkpoint_sha256=digest)
        checkpoints_meta.append(dict(label=label,path=str(path),sha256=digest,
                                     n_actions=len(rows),online_only=True))
        collected.extend(rows)
    if collected:write_csv(out/'probe_qgap.csv',collected)
    else:(out/'probe_qgap.csv').write_text('state_id,action_index,delta_q,checkpoint_label\n',encoding='utf-8')
    (out/'missing_checkpoints.csv').write_text('label,path,reason\n'+''.join(
             f'{x["label"]},{x["path"]},{x["reason"]}\n' for x in missing),encoding='utf-8')
    manifest=dict(version=1,probe_sha256=file_sha(bank_path),n_states=len(bank['states']),
                  checkpoints=checkpoints_meta,missing=missing,
                  target_reconstructed=False,optimizer_reconstructed=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    if missing:
        (out/'partial.json').write_text(json.dumps(dict(rows=len(collected),
            missing=len(missing),reason='missing checkpoint(s)')))
    else:
        (out/'complete.json').write_text(json.dumps(dict(rows=len(collected),missing=0)))
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--data')
    p.add_argument('--checkpoint',nargs='+',required=True,
                   help='e.g. A0_42_EP100=/path/to/validation_snapshot.pt')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    print(json.dumps(evaluate_checkpoints(a.bank,a.checkpoint,a.out,
              data=a.data,device=a.device,batch_size=a.batch_size),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
