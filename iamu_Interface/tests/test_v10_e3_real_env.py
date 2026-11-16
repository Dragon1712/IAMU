"""Integration checks; requires the V0.9 engine snapshot, no trained checkpoint."""
import unittest
from types import SimpleNamespace
from iamu_dynamic.core import Vessel,Port,generate_scenario
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.v10_policy import choose_adjusted_action
from iamu_dynamic.v10_e3_confidence import paired_confidence,fused_scores
from experiments.eval_e3_value_calibration_v10 import make_synthetic_world,continue_to_end


class AlwaysKeep:
    def act(self,obs,epsilon=0.):return 0


class RealEnvE3Tests(unittest.TestCase):
    def setUp(self):
        vessels=[Vessel(0,5.,20.,100.,2),Vessel(1,9.,30.,110.,2),
                 Vessel(2,16.,25.,120.,2)]
        port=Port(cranes=4,quay_length=600.)
        case=generate_scenario(vessels,port,label='shock',seed=117,
                               updates_per_vessel=2)
        self.env=IAMUEventEnv(case,budget=3,candidate_limit=3,
                  decision_clock='union',candidate_screen='none')

    def test_synthetic_world_does_not_change_public_prefix_or_source(self):
        env=self.env
        obs=env.reset()
        for _ in range(30):
            if env.event_time>1 and not env.done:
                break
            obs,_,done,_=env.step(0)
            if done:self.skipTest('Case ended unusually early')
        old_eta=dict(env.published_eta)
        old_events=list(env.events)
        sampled={i:max(env.event_time+30.,env.published_eta[i]+50.) for i in env.ids}
        copied=make_synthetic_world(env,sampled,step_hours=12.,
                                   signal_strength=.5,future_signals=2)
        self.assertEqual(env.published_eta,old_eta)
        self.assertEqual(env.events,old_events)
        self.assertEqual(copied.published_eta,old_eta)
        self.assertEqual(copied.events[:copied.event_index],old_events[:env.event_index])
        for e in copied.events[copied.event_index:]:
            self.assertGreater(e[0],env.event_time)
        self.assertEqual(copied.remaining_budget,env.remaining_budget)
        wait,spent=continue_to_end(copied,0,AlwaysKeep())
        self.assertGreaterEqual(wait,0.)
        self.assertLessEqual(spent,env.budget)

    def test_fixed_fusion_parity_e2(self):
        q=[0.,.2,.05]
        mask=[True,True,True]
        costs={0:(100.,101.,99.,102.),
               1:(90.,93.,89.,91.),2:(96.,97.,95.,98.)}
        st={a:paired_confidence(costs[0],costs[a],100) for a in (1,2)}
        risk={1:SimpleNamespace(tail_cost_delta=3.),
              2:SimpleNamespace(tail_cost_delta=-2.)}
        mpc=SimpleNamespace(advantage=lambda a:sum(k-e for k,e in
                 zip(costs[0],costs[a]))/len(costs[0]))
        old,old_scores,_=choose_adjusted_action(
             q,mask,margin=.05,reward_scale=100,
             risk_metrics=risk,mpc_estimate=mpc,
             beta_risk=.25,beta_budget=.25)
        new,new_scores=fused_scores(
             q,mask,st,mode='q_fixed',
             beta_budget=.25,beta_risk=.25,risk_metrics=risk,
             reward_scale=100,keep_margin=.05)
        self.assertEqual(old,new)
        for a in (1,2):
            self.assertAlmostEqual(old_scores[a],new_scores[a]['score'])


if __name__=='__main__':unittest.main()
