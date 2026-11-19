"""E4-E pure replay statistics and balanced sampling.

Keeps forced-Keep transitions (no legal edits) separate from choice Keep;
otherwise a high overall Keep fraction is an invalid collapse diagnostic.
"""
from __future__ import annotations

import hashlib
import math
import random
from statistics import mean


def has_legal_edit(tr):
    return bool(tr.obs_mask[1:].any())


def replay_stats(transitions):
    ts=list(transitions)
    n=len(ts)
    if not n:
        return {'n':0}
    legal=[t for t in ts if has_legal_edit(t)]
    edits=[t for t in ts if t.action!=0]
    choices=[t for t in legal if t.action==0]
    forced=[t for t in ts if not has_legal_edit(t)]
    if any(t.action!=0 for t in forced):
        raise AssertionError('Replay has Edit without any legal Edit')
    if any(not bool(t.obs_mask[t.action]) for t in ts):
        raise AssertionError('Replay includes an illegal selected action')
    out={'n':n, 'n_legal_choice':len(legal), 'n_forced_keep':len(forced),
         'n_keep_with_legal_edit':len(choices), 'n_edit':len(edits),
         'keep_fraction_all':(n-len(edits))/n,
         'forced_keep_fraction':len(forced)/n,
         'keep_fraction_actionable':len(choices)/max(1,len(legal)),
         'edit_fraction_actionable':len(edits)/max(1,len(legal)),
         'mean_reward':mean(float(t.reward) for t in ts),
         'mean_abs_reward':mean(abs(float(t.reward)) for t in ts),
         'terminal_fraction':sum(bool(t.done) for t in ts)/n}
    # V0.9 absolute budget features: initial B/10 in g[19], remaining B/10 in g[20].
    # These are not authoritative training-case labels: quantized observable state only.
    for budget in (2,5,10):
        subset=[t for t in ts if abs(float(t.obs_global[19])-budget/10)<1e-5]
        actionable=[t for t in subset if has_legal_edit(t)]
        out[f'b{budget}_n']=len(subset)
        out[f'b{budget}_legal_n']=len(actionable)
        out[f'b{budget}_keep_actionable_fraction']=(
            sum(t.action==0 for t in actionable)/len(actionable) if actionable else None)
    # Newly collected vs oldest replay quintile, to reveal distribution drift.
    m=max(1,n//5)
    for name,sub in [('oldest20',ts[:m]),('newest20',ts[-m:])]:
        eligible=[t for t in sub if has_legal_edit(t)]
        out[f'{name}_actionable_n']=len(eligible)
        out[f'{name}_keep_actionable_fraction']=(
            sum(t.action==0 for t in eligible)/len(eligible) if eligible else None)
    return out


def balanced_batch(replay_items,batch_size, rng):
    """Keep/Edit 1:1 among states with legal Edit; NOT original uniform replay.

    Oversamples Edit by replacement when rare. Caller must record actual sample
    distribution and effective number of distinct transitions (not iid!).
    """
    items=list(replay_items)
    if batch_size<=1 or not items:
        raise ValueError('Need batch_size>=2 and nonempty replay')
    edits=[i for i,t in enumerate(items) if t.action!=0 and has_legal_edit(t)]
    keeps=[i for i,t in enumerate(items) if t.action==0 and has_legal_edit(t)]
    if not edits or not keeps:
        raise ValueError('Cannot balance Replay without both actionable Keep and Edit')
    ne=batch_size//2
    nk=batch_size-ne
    # With replacement only when pool smaller than target; explicit support.
    e=(rng.sample(edits,ne) if len(edits)>=ne else rng.choices(edits,k=ne))
    k=(rng.sample(keeps,nk) if len(keeps)>=nk else rng.choices(keeps,k=nk))
    idx=e+k;rng.shuffle(idx)
    return [items[i] for i in idx],len(set(idx))


def deterministic_probe(transitions, k=512,seed=779):
    items=list(transitions)
    if not items:return []
    rng=random.Random(seed)
    edit=[t for t in items if t.action!=0 and has_legal_edit(t)]
    keep=[t for t in items if t.action==0 and has_legal_edit(t)]
    other=[t for t in items if not has_legal_edit(t)]
    selected=[]
    for pool,target in [(edit,k//3),(keep,k//3),(other,k-2*(k//3))]:
        selected.extend(rng.sample(pool,min(len(pool),target)))
    if len(selected)<min(len(items),k):
        used={id(x) for x in selected}
        candidates=[x for x in items if id(x) not in used]
        selected.extend(rng.sample(candidates,min(k-len(selected),len(candidates))))
    rng.shuffle(selected)
    return selected


def fraction_reason(keeps,edits):
    if keeps<0 or edits<0:raise ValueError('Invalid counts')
    return keeps/max(1,keeps+edits)


def assert_immutable_replay(expected, current):
    if len(expected)!=len(current):raise AssertionError('Frozen Replay size changed')
    # The model must never mutate buffers; validate object values/shape, not only count.
    for i,(a,b) in enumerate(zip(expected,current)):
        if a is not b and (a.action!=b.action or a.reward!=b.reward):
            raise AssertionError(f'Frozen Replay changed at {i}')


def replay_fingerprint(transitions):
    """Content hash across all model-visible fields, not only replay length."""
    import numpy as np
    import struct
    h=hashlib.sha256()
    for tr in transitions:
        h.update(struct.pack('<if?',int(tr.action),float(tr.reward),bool(tr.done)))
        for v in ('obs_global','obs_actions','obs_mask',
                  'next_global','next_actions','next_mask'):
            arr=np.ascontiguousarray(getattr(tr,v))
            h.update(str(arr.shape).encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def actionable_batch(replay_items,batch_size,rng):
    """Uniform over actionable states, natural Keep/Edit mix; excludes forced Keep.

    This controls for the fact that balanced_batch BOTH discards forced Keep
    and changes choice Keep/Edit weights.
    """
    items=list(replay_items)
    eligible=[i for i,t in enumerate(items) if has_legal_edit(t)]
    if len(eligible)<batch_size:
        raise ValueError('Not enough actionable transitions for without-replacement sample')
    ix=rng.sample(eligible,batch_size)
    return [items[i] for i in ix],len(ix)
