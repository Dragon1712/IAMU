"""V1.0-E3 wrapper: identical state estimators for fixed, pure, adaptive fusion.

All test policies share exactly the same public-state ETA sampling seed, same
risk estimator, same paired MPC rollouts, and the same frozen Q checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from .v10_policy import (PolicySpecification, frozen_q_values,
                         public_arrival_flags)
from .v10_risk_features import event_seed, public_eta_scenarios, evaluate_proposals
from .v10_budget_lookahead import evaluate_multistage
from .v10_e3_confidence import MODES, summarize_mpc, fused_scores


@dataclass(frozen=True)
class E3Specification:
    base: PolicySpecification
    neutral_q: float = 0.0

    def __post_init__(self):
        if self.neutral_q < 0:
            raise ValueError('neutral_q must be nonnegative')
        if self.base.mpc_samples < 2:
            raise ValueError('E3 confidence needs >=2 paired MPC scenarios')
        if self.base.mpc_samples > self.base.risk.samples:
            raise ValueError('E3 uses a prefix of common risk scenarios for MPC')


def estimates_at_state(env, obs, agent, spec: E3Specification, *, want_risk=True):
    candidates = {i: env._candidates[i-1] for i in range(1,len(obs.mask))
                  if bool(obs.mask[i])}
    q = frozen_q_values(agent, obs)
    if not candidates:
        return q, None, {}, {}, 0., 0.
    seed = event_seed(env.scenario.seed, env.event_index, env.event_type)
    arrived = public_arrival_flags(env)
    risk = {}
    start = perf_counter()
    if want_risk:
        risk, samples, _ = evaluate_proposals(
            env.runtime, env.plan, candidates, env.published_eta,
            env.event_time, arrived, spec.base.risk, seed=seed)
    else:
        samples = public_eta_scenarios(
            env.published_eta,env.event_time,arrived,spec.base.risk,seed=seed)
    risk_seconds = perf_counter()-start
    start = perf_counter()
    mpc = evaluate_multistage(
        env,candidates,samples[:spec.base.mpc_samples],spec.base.lookahead)
    mpc_seconds=perf_counter()-start
    return q,mpc,summarize_mpc(mpc,env.reward_scale),risk,risk_seconds,mpc_seconds


class E3Policy:
    def __init__(self, mode: str, agent, spec: E3Specification):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode,self.agent,self.spec=mode,agent,spec

    def choose(self,env,obs):
        if self.mode=='q_only':
            return int(self.agent.act(obs,epsilon=0.)),{'e3_mode':'q_only'}
        if env.remaining_budget<=0 or int(obs.mask.sum())<=1:
            return 0,{'e3_mode':self.mode,'forced_keep':True}
        q,mpc,stats,risk,risk_seconds,mpc_seconds=estimates_at_state(
            env,obs,self.agent,self.spec,want_risk=self.spec.base.beta_risk>0)
        # Pure MPC uses a physical improvement threshold; neural Keep margin
        # only applies to policies retaining the frozen neural Q model.
        margin=(env.min_gain/env.reward_scale if self.mode in ('pure_mpc','pure_mpc_risk')
                else float(self.agent.keep_margin))
        action,all_details=fused_scores(
            q,obs.mask,stats,mode=self.mode,
            beta_budget=self.spec.base.beta_budget,
            beta_risk=self.spec.base.beta_risk,
            risk_metrics=risk,reward_scale=env.reward_scale,
            keep_margin=margin,neutral=self.spec.neutral_q)
        best=(max(all_details,key=lambda a:(all_details[a]['score'],-a))
              if all_details else None)
        info={'e3_mode':self.mode,'best_action':best,'action':action,
              'risk_seconds':risk_seconds,'mpc_seconds':mpc_seconds,
              'n_candidates':len(all_details),
              'mpc_n':mpc.scenario_count,'mpc_h':mpc.stages}
        if best is not None:
            info.update(all_details[best])
        # Preserve all per-candidate evidence in one JSON-compatible object;
        # experiment runner writes an optional independent action-detail CSV.
        info['_all_actions']=all_details
        return int(action),info
