"""Training-time behavior, Q-gap and TD-error telemetry; CPU/GPU safe.

No extra learner gradients and NO sampling from ReplayBuffer.rng here: its
random sequence controls training. Diagnostics draw via separate local RNGs,
and only read the current model and transitions.
"""
from __future__ import annotations
import csv,hashlib,json,os,random,shutil
from pathlib import Path
from .mechanism_core import replay_stats,deterministic_probe
from .td_probe import td_rows,td_summary


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()


def append_csv(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:
            header=next(csv.reader(f))
        if list(row)!=header:raise ValueError(f'Changed CSV fields: {path}')
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:writer.writeheader()
        writer.writerow(row)


def eval_qgap(learner,bank,*,batch=128):
    import torch
    import numpy as np
    data=bank
    legal=[];all_gap=[];best_gap=[];selected=[];rawkeep=[];rawedit=[];budgets=[]
    was_training=learner.online.training;learner.online.eval()
    with torch.no_grad():
        for start in range(0,len(data['g']),batch):
            end=min(len(data['g']),start+batch)
            tensor=lambda v,dtype:torch.as_tensor(data[v][start:end],device=learner.device,dtype=dtype)
            masks=tensor('mask',torch.bool)
            q=learner.online(tensor('g',torch.float32),tensor('actions',torch.float32),masks)
            keep=q[:,0]
            scores=q[:,1:].masked_fill(~masks[:,1:],-torch.inf)
            maxima,idx=scores.max(dim=1)
            gap=maxima-keep
            legalMask=masks[:,1:].any(dim=1)
            vals=(q-keep[:,None])[:,1:]
            legal.extend(legalMask.cpu().tolist())
            best_gap.extend(gap[legalMask].cpu().tolist())
            all_gap.extend(vals[masks[:,1:]].cpu().tolist())
            selected.extend((gap[legalMask]>learner.keep_margin).cpu().tolist())
            budgets.extend((data['g'][start:end,19][legalMask.cpu().numpy()]*10).round().astype(int).tolist())
            rawkeep.extend(keep[legalMask].cpu().tolist())
            rawedit.extend(maxima[legalMask].cpu().tolist())
    learner.online.train(was_training)
    if not best_gap:raise ValueError('Probe contains no legal Edit')
    n=len(best_gap)
    out=dict(probe_states=len(data['g']),actionable_states=n,edits_considered=len(all_gap),
        mean_best_q_gap=sum(best_gap)/n,mean_all_edit_q_gap=sum(all_gap)/len(all_gap),
        fraction_keep_with_legal=1-sum(selected)/n,
        fraction_gap_nonpositive=sum(x<=0 for x in best_gap)/n,
        fraction_gap_above_margin=sum(selected)/n,
        mean_q_keep=sum(rawkeep)/n,
        mean_q_best_edit=sum(rawedit)/n,
        fraction_positive_keep_q=sum(x>0 for x in rawkeep)/n,
        fraction_best_gap_near_margin=sum(abs(x-learner.keep_margin)<=.05 for x in best_gap)/n)
    for b in (2,5,10):
        these=[j for j,b0 in enumerate(budgets) if b0==b]
        out[f'b{b}_probe_n']=len(these)
        out[f'b{b}_mean_best_q_gap']=sum(best_gap[j] for j in these)/len(these) if these else None
        out[f'b{b}_fraction_keep']=sum(best_gap[j]<=learner.keep_margin for j in these)/len(these) if these else None
    return out


class MechanismMonitor:
    def __init__(self,args,out,learner,replay):
        import numpy as np
        self.out=Path(out);self.args=args
        self.dir=self.out/'e5_mechanism';self.dir.mkdir(parents=True,exist_ok=True)
        p=Path(args.e5_probe_bank)
        if not p.is_file():raise FileNotFoundError(f'Public probe bank required: {p}')
        meta=p.with_suffix('.metadata.json')
        if not meta.exists():raise FileNotFoundError(meta)
        config=json.loads(meta.read_text(encoding='utf-8'))
        if config.get('bank_sha256')!=sha(p):raise ValueError('Probe bank hash mismatch')
        self.bank={}
        with np.load(p,allow_pickle=False) as data:
            for field in ('g','actions','mask'):
                self.bank[field]=data[field].copy()
        if self.bank['mask'].shape[1]!=args.candidate_limit+1:
            raise ValueError('Probe mask incompatible with candidate_limit')
        self.probe_digest=sha(p)
        self.fixed_td_probe=None
        self.td_draw_rng=random.Random(77831)
        self.fork=int(args.e5_fork_episode)
        self.cfg_path=self.dir/'monitor.json'
        if self.cfg_path.exists():
            old=json.loads(self.cfg_path.read_text(encoding='utf-8'))
            if old['probe_sha256']!=self.probe_digest or old['fork']!=self.fork:
                raise ValueError('Resumed monitor parameters changed')
        else:
            self.cfg_path.write_text(json.dumps(dict(probe_sha256=self.probe_digest,fork=self.fork,
                every=args.e5_monitor_every,td_sample=args.e5_td_sample,
                source='same fixed PUBLIC state bank, read-only training metrics',
                note='training_checkpoint carries Replay data; no future oracle used'),indent=2))
        # On exact resume at a post-fork stage, use serialized TD probe (trusted local only).
        self.fixed_td_probe_path=self.dir/f'td_probe_ep{self.fork:04d}.pt'
        if self.fixed_td_probe_path.exists():
            import torch
            self.fixed_td_probe=torch.load(self.fixed_td_probe_path,map_location='cpu',weights_only=False)

    def reconcile(self, completed):
        """Roll back telemetry after a FULL-state resume to avoid false future rows."""
        completed=int(completed)
        for f in ('qgap_trace.csv','replay_trace.csv','td_trace.csv'):
            path=self.dir/f
            if not path.exists():continue
            with path.open(encoding='utf-8',newline='') as fh:
                rows=list(csv.DictReader(fh))
            keep=[r for r in rows if int(r['episode'])<=completed]
            # Guard replay of an already-logged checkpoint.
            if len(keep)!=len(rows):
                temp=path.with_suffix('.csv.tmp')
                with temp.open('w',encoding='utf-8',newline='') as fh:
                    wr=csv.DictWriter(fh,fieldnames=list(rows[0]))
                    wr.writeheader();wr.writerows(keep)
                temp.replace(path)
        if completed<self.fork:
            # Historical probe from a non-atomic next-phase callback is stale.
            if self.fixed_td_probe_path.exists():self.fixed_td_probe_path.unlink()
            self.fixed_td_probe=None
            state=self.dir/f'full_state_ep{self.fork:04d}.pt'
            if state.exists():raise RuntimeError('Fork archive exists but restored training state predates fork')
        elif completed>self.fork:
            state=self.dir/f'full_state_ep{self.fork:04d}.pt'
            if not state.exists():
                raise RuntimeError('Cannot recover missing EP200 fork from post-200 training_state.pt')

    def record(self,completed,learner,replay,train_rows):
        if completed%self.args.e5_monitor_every:return
        import numpy as np
        episode=int(completed)
        row=dict(episode=episode,gradient_steps=learner.gradient_steps,
                 **eval_qgap(learner,self.bank))
        append_csv(self.dir/'qgap_trace.csv',row)
        rep=dict(episode=episode,gradient_steps=learner.gradient_steps,
                 **replay_stats(replay.items))
        recent=train_rows[-self.args.e5_monitor_every:]
        for scenario in ('shock','repeated','small'):
            rep[f'recent_{scenario}_fraction']=sum(x['scenario']==scenario for x in recent)/len(recent) if recent else None
        for b in (2,5,10):
            rep[f'recent_b{b}_fraction']=sum(int(x['budget'])==b for x in recent)/len(recent) if recent else None
        rep.update(recent_mean_actor_spent=(sum(x['spent'] for x in recent)/len(recent)) if recent else None,
                   recent_mean_actor_wait=(sum(x['weighted_wait'] for x in recent)/len(recent)) if recent else None,
                   recent_mean_staleness=(sum(x['policy_staleness_steps'] for x in recent)/len(recent)) if recent else None,
                   recent_mean_epsilon=(sum(x['epsilon'] for x in recent)/len(recent)) if recent else None)
        append_csv(self.dir/'replay_trace.csv',rep)
        if episode==self.fork and self.fixed_td_probe is None:
            self.fixed_td_probe=deterministic_probe(replay.items,512,seed=777+int(self.args.seed))
            import torch
            tmp=self.fixed_td_probe_path.with_suffix('.pt.tmp')
            torch.save(self.fixed_td_probe,tmp)
            tmp.replace(self.fixed_td_probe_path)
        # Explicitly use a different RNG, NEVER replay.rng.
        current=deterministic_probe(replay.items,self.args.e5_td_sample,seed=21000+episode)
        for name,probe in [('current',current),('fork_fixed',self.fixed_td_probe or [])]:
            if not probe:continue
            for stats in td_summary(td_rows(learner,probe),name=name):
                append_csv(self.dir/'td_trace.csv',dict(episode=episode,
                    gradient_steps=learner.gradient_steps,**stats))
        print(f'[E5] ep={episode} gradients={learner.gradient_steps} '
              f'Qgap={row["mean_best_q_gap"]:.4f} '
              f'keepProbe={row["fraction_keep_with_legal"]:.3f} '
              f'replayKeepActionable={rep["keep_fraction_actionable"]:.3f}',flush=True)

    def archive_fork(self,completed,validations,state_path,replay,learner):
        if completed!=self.fork:return
        validation=validations.get(f'val_{self.fork:06d}',{})
        if validation.get('status')!='done':return
        dest=self.dir/f'full_state_ep{self.fork:04d}.pt'
        if dest.exists():
            if not dest.with_suffix('.metadata.json').exists():
                raise RuntimeError('Fork exists without provenance metadata')
            return
        if not self.fixed_td_probe_path.exists():
            raise RuntimeError('Frozen TD probe missing at fork boundary')
        # Copy AFTER persisted validation-complete training state, preserving
        # optimizer/target/replay/RNG and the exact 200-episode barrier.
        tmp=dest.with_suffix('.pt.tmp')
        shutil.copyfile(state_path,tmp)
        tmp.replace(dest)
        manifest=dict(source_out=str(self.out),full_state_sha256=sha(dest),
            completed=completed,gradient_steps=learner.gradient_steps,
            replay_size=len(replay),probe_sha256=self.probe_digest,
            td_probe_sha256=sha(self.fixed_td_probe_path),
            validation_sha256=validation.get('snapshot_sha256'),
            validation_id=validation.get('id'),status=validation.get('status'))
        dest.with_suffix('.metadata.json').write_text(json.dumps(manifest,indent=2))
        print(f'[E5-FORK] Archived verified EP{completed} full-state sha={manifest["full_state_sha256"]}',flush=True)
