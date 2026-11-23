"""Fixed-data R1 rollout scorer, R2 uniform A2+gap, R3 budget-one A2+gap.

Not complete iterative approximate policy iteration, and NOT online training.
"""
from __future__ import annotations
import argparse,csv,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from v13_diagnostics.common import load_trusted,write_csv,atomic_json
from .fixed_replay import tensor_digest,shared_probe_gap,read_reference
from .utils import boundary_provenance


def read_csv(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def budget_group(b):return '1' if int(b)==1 else '2' if int(b)==2 else '3+'

def select_supervision(rows,n,seed,max_cost_mismatch=.10):
    if n<3:raise ValueError('Too few labels')
    rng=random.Random(seed)
    pools={b:[] for b in ('1','2','3+')}
    for r in rows:pools[budget_group(r['source_budget'])].append(r)
    for pool in pools.values():pool.sort(key=lambda r:(r['state_id'],int(r['action_slot'])))
    if len(pools['1'])<n:raise ValueError('Insufficient budget-one labels')
    boundary=rng.sample(pools['1'],n)
    cost_boundary=sum(float(r['allocated_sim_steps']) for r in boundary)
    counts=[n//3]*3
    for i in range(n%3):counts[i]+=1
    quota=dict(zip(('1','2','3+'),counts))
    if any(len(pools[b])<quota[b] for b in pools):raise ValueError('Insufficient uniform-budget labels')
    best=None;best_diff=float('inf')
    for _ in range(512):
        sampled=[r for b in ('1','2','3+') for r in rng.sample(pools[b],quota[b])]
        diff=abs(sum(float(x['allocated_sim_steps']) for x in sampled)-cost_boundary)
        if diff<best_diff:best_diff=diff;best=sampled
    relative=best_diff/max(1.,cost_boundary)
    if relative>max_cost_mismatch:raise ValueError(f'Label cost mismatch {relative:.1%}')
    return dict(R1=best,R2=best,R3=boundary),dict(label_count=n,
        R1_R2_allocated_sim_steps=sum(float(x['allocated_sim_steps']) for x in best),
        R3_allocated_sim_steps=cost_boundary,relative_sim_cost_gap=relative,
        unique_R2_states=len({r['state_id'] for r in best}),
        unique_R3_states=len({r['state_id'] for r in boundary}))


def q_gap(model,observations,edit_indices):
    dev=model.device
    g=torch.as_tensor(np.stack([x.global_features for x in observations]),dtype=torch.float32,device=dev)
    acts=torch.as_tensor(np.stack([x.action_features for x in observations]),dtype=torch.float32,device=dev)
    mask=torch.as_tensor(np.stack([x.mask for x in observations]),dtype=torch.bool,device=dev)
    index=torch.as_tensor(edit_indices,dtype=torch.long,device=dev)
    if not bool(mask.gather(1,index[:,None]).all()):raise AssertionError('Illegal label Edit')
    q=model.online(g,acts,mask)
    return q.gather(1,index[:,None]).squeeze(1)-q[:,0]


def td_loss(model,batch):
    dev=model.device
    def arr(name,dtype):return torch.as_tensor(np.stack([getattr(x,name) for x in batch]),dtype=dtype,device=dev)
    og=arr('obs_global',torch.float32);oa=arr('obs_actions',torch.float32);om=arr('obs_mask',torch.bool)
    ng=arr('next_global',torch.float32);na=arr('next_actions',torch.float32);nm=arr('next_mask',torch.bool)
    actions=torch.as_tensor([x.action for x in batch],dtype=torch.long,device=dev)
    rewards=torch.as_tensor([x.reward for x in batch],dtype=torch.float32,device=dev)
    discounts=torch.as_tensor([x.bootstrap_discount for x in batch],dtype=torch.float32,device=dev)
    selected=model.online(og,oa,om).gather(1,actions[:,None]).squeeze(1)
    with torch.no_grad():
        scores=model.online(ng,na,nm).masked_fill(~nm,-torch.inf).clone()
        scores[:,0]+=model.bootstrap_margin
        next_action=scores.argmax(dim=1)
        target_q=model.target(ng,na,nm).gather(1,next_action[:,None]).squeeze(1)
        target=rewards+discounts*target_q
    return F.smooth_l1_loss(selected,target)


def train(a):
    from v12_protocol.agent import V12Agent
    from v12_protocol.replay import EpisodeReplay
    labels=read_csv(a.labels)
    training=load_trusted(a.train_probes);diagnostic=load_trusted(a.diagnostic_probes)
    tr={(x['external_seed'],x['case']) for x in training['probes']}
    dv={(x['external_seed'],x['case']) for x in diagnostic['probes']}
    if tr&dv:raise ValueError('Training vs diagnostic overlap')
    obs={x['state_id']:x['snapshot']._observation() for x in training['probes']}
    if not {x['state_id'] for x in labels}<=set(obs):raise ValueError('Missing label snapshots')
    source=load_trusted(a.a2_data)
    if source.get('representation')!='budget_tail' or source.get('gamma')!=1.:
        raise ValueError('Wrong A2 training source')
    reference_rows=read_csv(a.diagnostic_reference_csv)
    if {x.get('continuation') for x in reference_rows}!={'a0'}:
        raise ValueError('Rollout supervision and probe reference must both use frozen A0 continuation')
    reference=read_reference(a.diagnostic_reference_csv)
    groups,cost=select_supervision(labels,a.label_count,a.seed,a.max_sim_cost_gap)
    dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    curves=[];probes=[];init_digests=[]
    for name in ('R1','R2','R3'):
        model=V12Agent(hidden_dim=96,learning_rate=a.learning_rate,gamma=1.,
            target_update_steps=a.target_update,seed=a.seed,device=a.device,
            q_network='set_ddqn',execution_margin=0.,bootstrap_margin=0.,
            representation='budget_tail',reward_scale=100.)
        init_digests.append(tensor_digest(model.online.state_dict()))
        replay=EpisodeReplay(max_episodes=len(source['episodes']),seed=a.seed+99,representation='budget_tail')
        for ep in source['episodes']:
            replay.append_episode(ep['episode_id'],ep['transitions'],ep['compression_audit']['raw_steps'])
        pool=groups[name];pool_rng=random.Random(a.seed+7);begin=time.perf_counter();labels_seen=0
        def report(step,l_td='',l_gap=''):
            probe_values=shared_probe_gap(model,diagnostic,reference)
            for x in probe_values:
                ref=x['reference_gap_mean'];prediction=x['predicted_gap']
                probes.append(dict(group=name,step=step,**x,
                    zero_gap_predictor_abs_error=abs(ref),
                    binary_pair_reference_regret=max(0.,ref) if prediction<=0 else max(0.,-ref)))
            curves.append(dict(group=name,step=step,td_loss=l_td,gap_loss=l_gap,
                n_labels=len(pool),unique_label_states=len({x['state_id'] for x in pool}),
                label_samples_seen=labels_seen,td_samples_seen=sum(replay.sample_counts.values()),
                target_syncs=step//a.target_update,
                amortized_sim_steps_budget=cost['R3_allocated_sim_steps'] if name=='R3'
                        else cost['R1_R2_allocated_sim_steps'],
                seconds=time.perf_counter()-begin))
            if step in (0,a.gradient_updates):
                torch.save(model.checkpoint(dict(protocol='v13-rollout-one-iteration',group=name,
                   seed=a.seed,gradient_steps=step,label_cost=cost)),
                   dest/f'{name}_{"init" if step==0 else "full"}.pt')
        report(0)
        for step in range(1,a.gradient_updates+1):
            model.online.train()
            sampled=[pool[pool_rng.randrange(len(pool))] for _ in range(a.gap_batch_size)]
            truth=torch.as_tensor([float(x['paired_return_gap']) for x in sampled],
                                  dtype=torch.float32,device=model.device)
            predicted=q_gap(model,[obs[x['state_id']] for x in sampled],
                            [int(x['action_slot']) for x in sampled])
            gap_loss=F.smooth_l1_loss(predicted,truth)
            if name=='R1':
                td=torch.zeros_like(gap_loss);total=gap_loss
            else:
                batch,_=replay.sample(a.td_batch_size,'uniform')
                td=td_loss(model,batch);total=td+a.gap_weight*gap_loss
            model.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.online.parameters(),10.)
            model.optimizer.step();model.gradient_steps+=1
            if model.gradient_steps%a.target_update==0:model.target.load_state_dict(model.online.state_dict())
            labels_seen+=len(sampled)
            if step%a.diagnostic_every==0 or step==a.gradient_updates:
                report(step,float(td.detach()),float(gap_loss.detach()))
    if len(set(init_digests))!=1:raise AssertionError('Inconsistent network initialization')
    write_csv(dest/'rollout_train_curve.csv',curves)
    write_csv(dest/'rollout_gap_metrics.csv',probes)
    atomic_json(dest/'manifest.json',boundary_provenance(a,inputs=[a.labels,a.train_probes,
         a.diagnostic_probes,a.diagnostic_reference_csv,a.a2_data])|
         dict(initial_online_sha256=init_digests[0],label_cost=cost,
              note='Fixed-data one-step rollout improvement, not online policy iteration'))
    print('[rollout] R1/R2/R3 fixed-data training complete')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('labels','train-probes','diagnostic-probes','diagnostic-reference-csv','a2-data','out'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--label-count',type=int,default=30)
    p.add_argument('--max-sim-cost-gap',type=float,default=.10)
    p.add_argument('--gradient-updates',type=int,default=4000)
    p.add_argument('--diagnostic-every',type=int,default=250)
    p.add_argument('--td-batch-size',type=int,default=64)
    p.add_argument('--gap-batch-size',type=int,default=32)
    p.add_argument('--gap-weight',type=float,default=.1)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--target-update',type=int,default=250)
    p.add_argument('--seed',type=int,default=43);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv)
    if a.gradient_updates<1 or a.diagnostic_every<1 or a.td_batch_size<1 or a.gap_batch_size<1:
        p.error('Invalid training hyperparameters')
    train(a)
if __name__=='__main__':main()
