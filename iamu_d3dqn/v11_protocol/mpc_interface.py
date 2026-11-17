"""Shared event/candidate interfaces, WITHOUT hidden-future deployment leakage.

This is an interface/guard for subsequent MPC implementation, NOT a new MPC
policy: the available V0.9 Scenario stores true future ETA. Blindly cloning
and stepping it in ONLINE forecast would be an illegal oracle. Callers must
provide a separately calibrated synthetic future model before rollout.
"""
from __future__ import annotations
import copy

class CommonDecisionKernel:
    def __init__(self,environment, *, mode='diagnostic'):
        if mode not in ('diagnostic','forecast'):raise ValueError(mode)
        self.env=environment;self.mode=mode
    def legal_candidates(self):
        obs=self.env._observation()
        return dict(mask=obs.mask.copy(),physical_mask=obs.physical_mask.copy(),
            heuristic_mask=obs.heuristic_mask.copy(),candidate_ids=obs.candidate_ids,
            candidate_data=copy.deepcopy(self.env._candidate_data),
            reason_flags=dict(obs.decision_flags),event_type=obs.event_type,
            event_time=obs.time,remaining_budget=obs.remaining_budget)
    def branch_first_action(self,action):
        if self.mode!='diagnostic':
            raise RuntimeError('Cannot step real Scenario into unseen future. '
                'Provide an independent synthetic future ETA/message generator first.')
        cloned=copy.deepcopy(self.env)
        obs=cloned._observation()
        if not 0<=action<len(obs.mask) or not bool(obs.mask[action]):raise ValueError('Illegal candidate')
        nxt,reward,done,info=cloned.step(action)
        return cloned,nxt,reward,done,info
