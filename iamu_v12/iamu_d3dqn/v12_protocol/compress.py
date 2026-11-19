"""Lossless *training-return* segmentation of a complete V1.1 event trajectory.

Physical events are never skipped; raw updates are unchanged. The ledger verifies
return preservation even when gamma < 1. Prefix rewards are excluded from learned
state-action targets but included in the episode total.
"""
from __future__ import annotations
import math
import numpy as np
from .transitions import MacroTransition, RawStep, PrefixLedger, CompressionResult, copied_array

MODES=('primitive','budget_tail','decision_epoch')


def is_decision_state(step: RawStep) -> bool:
    return bool(np.asarray(step.transition.obs_mask, dtype=bool)[1:].any())


def validate_raw_episode(raw: list[RawStep]) -> None:
    if not raw:
        raise ValueError('Episode must contain at least a terminal transition')
    ep=raw[0].episode_id
    ver=raw[0].policy_version
    for i,s in enumerate(raw):
        t=s.transition
        if s.episode_id!=ep or s.policy_version!=ver or (t.episode_id is not None and int(t.episode_id)!=ep):
            raise ValueError(f'Mixed episode/policy_version at {i}')
        if not all(isinstance(x,int) and x>=0 for x in (s.source_budget,s.next_budget)):
            raise ValueError(f'Invalid budget metadata at {i}')
        if s.source_budget-s.next_budget!=int(t.action!=0):
            raise ValueError(f'Budget delta does not match Edit cost at {i}')
        if s.source_event_index<0 or s.next_event_index<s.source_event_index:
            raise ValueError(f'Invalid event index at {i}')
        if not math.isfinite(s.source_time) or not math.isfinite(s.next_time) or s.next_time+1e-8<s.source_time:
            raise ValueError(f'Invalid physical time at {i}')
        if not math.isfinite(float(t.reward)):
            raise ValueError(f'Non-finite reward at {i}')
        m=np.asarray(t.obs_mask,dtype=bool)
        nm=np.asarray(t.next_mask,dtype=bool)
        if m.ndim!=1 or not len(m) or not m[0] or nm.ndim!=1 or len(nm)!=len(m) or not nm[0]:
            raise ValueError(f'Invalid mask shape at {i}')
        if t.action<0 or t.action>=len(m) or not m[t.action]:
            raise ValueError(f'Illegal action at {i}')
        if s.source_budget==0 and m[1:].any():
            raise ValueError(f'Edit remains legal after exhausted budget at {i}')
        if not m[1:].any() and t.action!=0:
            raise ValueError(f'Edit in forced-Keep state at {i}')
        if bool(t.done)!=(i==len(raw)-1):
            raise ValueError(f'Only last event can be terminal at {i}')
        for f in ('obs_global','obs_actions','next_global','next_actions'):
            if not np.asarray(getattr(t,f)).size:
                raise ValueError(f'Empty observation {f} at {i}')
        if i:
            prev=raw[i-1];pt=prev.transition
            if (prev.next_event_index!=s.source_event_index or
                not math.isclose(prev.next_time,s.source_time,abs_tol=1e-8) or
                prev.next_budget!=s.source_budget):
                raise ValueError(f'Disconnected event/budget chain at {i}')
            for x,y in (('next_global','obs_global'),('next_actions','obs_actions'),('next_mask','obs_mask')):
                if not np.array_equal(np.asarray(getattr(pt,x)),np.asarray(getattr(t,y))):
                    raise ValueError(f'Disconnected observation chain at {i} ({x} != {y})')


def anchor_indices(raw: list[RawStep], mode:str) -> list[int]:
    if mode not in MODES:raise ValueError(f'Unknown representation: {mode}')
    if mode=='primitive':return list(range(len(raw)))
    if mode=='budget_tail':return [i for i,s in enumerate(raw) if s.source_budget>0]
    return [i for i,s in enumerate(raw) if is_decision_state(s)]


def compress_episode(raw_steps: list[RawStep], mode:str, gamma:float) -> CompressionResult:
    if not (0.<gamma<=1. and math.isfinite(gamma)):
        raise ValueError('gamma must be in (0,1]')
    validate_raw_episode(raw_steps)
    starts=anchor_indices(raw_steps,mode)
    prefix_stop=starts[0] if starts else len(raw_steps)
    prefix_reward=sum((gamma**i)*raw_steps[i].transition.reward for i in range(prefix_stop))
    prefix_undiscounted=sum(raw_steps[i].transition.reward for i in range(prefix_stop))
    macro=[]
    for k,start in enumerate(starts):
        stop=starts[k+1] if k+1<len(starts) else len(raw_steps)
        if stop<=start:raise AssertionError('Non-increasing anchors')
        first=raw_steps[start];last=raw_steps[stop-1]
        if mode=='budget_tail' and any(s.source_budget>0 for s in raw_steps[start+1:stop]):
            raise AssertionError('Budget tail contains positive-budget forced step')
        if mode=='decision_epoch' and any(is_decision_state(s) for s in raw_steps[start+1:stop]):
            raise AssertionError('Decision epoch contains an additional decision')
        if mode!='primitive' and any(s.transition.action!=0 for s in raw_steps[start+1:stop]):
            raise AssertionError('Compressed interval includes a non-anchor Edit')
        total=0.
        multiplier=1.
        for s in raw_steps[start:stop]:
            total+=multiplier*float(s.transition.reward)
            multiplier*=gamma
        ft,lt=first.transition,last.transition
        macro.append(MacroTransition(
            obs_global=copied_array(ft.obs_global), obs_actions=copied_array(ft.obs_actions),
            obs_mask=copied_array(ft.obs_mask), action=int(ft.action), reward=total,
            next_global=copied_array(lt.next_global), next_actions=copied_array(lt.next_actions),
            next_mask=copied_array(lt.next_mask), done=bool(lt.done),
            bootstrap_discount=(0. if lt.done else multiplier),
            n_raw_steps=stop-start, elapsed_hours=float(last.next_time-first.source_time),
            source_budget=int(first.source_budget),next_budget=int(last.next_budget),
            episode_id=int(first.episode_id),policy_version=int(first.policy_version),
            raw_start_index=start, raw_end_index=stop-1,
            reason_flags=dict(ft.reason_flags or {})))
    observed=sum(float(s.transition.reward) for s in raw_steps)
    discounted=sum((gamma**i)*s.transition.reward for i,s in enumerate(raw_steps))
    reconstructed=prefix_reward+sum((gamma**tr.raw_start_index)*tr.reward for tr in macro)
    if not math.isclose(reconstructed,discounted,rel_tol=1e-9,abs_tol=1e-8):
        raise AssertionError(f'Return ledger mismatch: {reconstructed} != {discounted}')
    if gamma==1. and not math.isclose(prefix_undiscounted+sum(tr.reward for tr in macro),observed,rel_tol=1e-9,abs_tol=1e-8):
        raise AssertionError('Undiscounted return ledger mismatch')
    if macro and (not macro[-1].done or sum(t.done for t in macro)!=1):
        raise AssertionError('Exactly one terminal macro needed when any anchor exists')
    audit=dict(raw_steps=len(raw_steps),emitted_steps=len(macro),
               actionable_steps=sum(is_decision_state(s) for s in raw_steps),
               prefix_steps=prefix_stop,tail_raw_steps=(macro[-1].n_raw_steps if macro else 0),
               terminal_segments=sum(t.done for t in macro),
               source_episode_wait=None,source_total_reward=observed,
               raw_discounted_return=discounted,
               reconstructed_discounted_return=reconstructed,
               represented_raw_steps=sum(t.n_raw_steps for t in macro),
               representation=mode,episode_id=raw_steps[0].episode_id)
    if sum(t.n_raw_steps for t in macro)+prefix_stop!=len(raw_steps):
        raise AssertionError('Source event was omitted or double-counted')
    return CompressionResult(tuple(macro),
        PrefixLedger(prefix_stop,prefix_reward,prefix_undiscounted,starts[0] if starts else None),audit)
