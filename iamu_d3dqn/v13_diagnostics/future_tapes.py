"""Externally indexed, branch-independent possible future ETA tapes.

These are *conditional synthetic futures*, NOT actual historical truth or a
calibrated posterior. Pre-generate one tape from observable ETA/state metadata,
then apply the exact same tape to both action branches. Tapes are independent
of each branch's random-number consumption or future policy choices.

Past clock/ETA events and started vessels are immutable. Only as-yet-unarrived
ships may receive synthetic new arrivals and subsequent announcements.
"""
from __future__ import annotations
import copy
import hashlib
import random
from dataclasses import dataclass


def _seed(state_id,tape_index,base_seed):
    payload=f'{state_id}:{tape_index}:{base_seed}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')


@dataclass(frozen=True)
class FutureTape:
    state_id: str
    index: int
    seed: int
    true_etas: dict
    future_updates: tuple       # tuple[Update] in physical event order
    origin_time: float
    note: str='Synthetic conditional future; not calibrated conditional posterior'


def draw_tape(env,state_id,tape_index,base_seed=910000,uncertainty=8.,updates_per_vessel=2):
    from iamu_dynamic.core import Update
    rng=random.Random(_seed(state_id,tape_index,base_seed))
    now=float(env.event_time)
    scenario=env.scenario
    scale=uncertainty*(.25 if scenario.label=='small' else (1.8 if scenario.label=='shock' else 1.))
    truth=dict(scenario.true_etas)
    # Important: do not alter any already-arrived/started true ETA. The
    # immutable snapshot reflects historical physical arrivals and costs.
    for vessel in scenario.vessels:
        vid=vessel.id
        if vid in env.runtime.started or float(scenario.true_etas[vid])<=now:
            continue
        # This is an approximation to the conditional distribution given the
        # public ETA; not an exact Bayesian conditioning of generate_scenario.
        mean=float(env.published_eta[vid])
        sampled=max(now+0.05,mean+rng.gauss(0,scale))
        if scenario.label=='shock' and rng.random()<.25:
            sampled+=rng.uniform(scale,3*scale)
        truth[vid]=sampled
    updates=[]
    for vessel in scenario.vessels:
        vid=vessel.id
        old=float(env.published_eta[vid])
        for k in range(updates_per_vessel):
            timestamp=vessel.eta0*(k+1)/(updates_per_vessel+1)
            if timestamp<=now+1e-8 or timestamp>=truth[vid]-1e-8:
                continue
            alpha=(k+1)/(updates_per_vessel+1)
            noise=rng.gauss(0,scale*(.45 if scenario.label=='repeated' else .18))
            proposed=max(timestamp,old+alpha*(truth[vid]-old)+noise)
            if abs(proposed-old)<1e-4:proposed+=.01
            updates.append(Update(timestamp,vid,proposed,1000000+vid*(updates_per_vessel+1)+k))
            old=proposed
    updates.sort(key=lambda x:(x.time,x.update_id))
    return FutureTape(state_id,tape_index,_seed(state_id,tape_index,base_seed),truth,
                      tuple(updates),now)


def apply_tape(snapshot,tape):
    from iamu_dynamic.core import Scenario
    env=copy.deepcopy(snapshot)
    if abs(float(env.event_time)-tape.origin_time)>1e-8:
        raise ValueError('Snapshot time is inconsistent with exogenous tape')
    for vid in env.runtime.started:
        if abs(env.scenario.true_etas[vid]-tape.true_etas[vid])>1e-12:
            raise AssertionError('Tape alters already-started vessel arrival')
    n=int(env.event_index)
    prior=env.events[:n]
    if any(x[0]>env.event_time+1e-8 for x in prior):
        raise AssertionError('Invalid processed-event prefix')
    # At a shared physical timestamp, one ETA event may have been processed
    # while other ETA/clock events are still pending. Those pending events
    # MUST remain in their original ordering and cannot be discarded merely
    # because their time equals env.event_time.
    pending_same_time=[ev for ev in env.events[n:]
                       if abs(ev[0]-env.event_time)<=1e-8]
    future_clock=[ev for ev in env.events[n:]
                  if ev[3]=='clock' and ev[0]>env.event_time+1e-8]
    future_eta=[(float(e.time),0,int(e.update_id),'eta',e) for e in tape.future_updates]
    future=sorted(pending_same_time+future_eta+future_clock,
                  key=lambda x:(x[0],x[1],x[2]))
    if any(x[0]<env.event_time-1e-8 for x in future):
        raise AssertionError('Unprocessed event lies before probe time')
    env.events=list(prior)+future
    env.event_index=n
    original=env.scenario
    env.scenario=Scenario(original.vessels,original.port,dict(tape.true_etas),
                          [x[4] for x in env.events if x[3]=='eta'],original.label,
                          original.seed)
    return env


def tape_hash(tape):
    from dataclasses import asdict
    from .common import stable_digest
    return stable_digest(asdict(tape))
