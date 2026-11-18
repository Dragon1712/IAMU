"""Physical reward and budget parity in the unchanged V1.1 simulator."""
import unittest
from pathlib import Path
import math

try:
    from iamu_dynamic import read_case,generate_scenario
    from iamu_dynamic.rl_env_v11 import IAMUEventEnv
    from iamu_dynamic.rl_ddqn_v11 import Transition
    from v12_protocol.transitions import RawStep
    from v12_protocol.compress import compress_episode
except ImportError:
    IAMUEventEnv=None

@unittest.skipUnless(IAMUEventEnv is not None,'Requires original IAMU V1.1 repository')
class RealEnvironmentTests(unittest.TestCase):
    def test_single_full_episode_raw_return_and_compressed_ledger(self):
        case=Path(__file__).resolve().parents[2]/'2024LiERLDDE/case_code/1.txt'
        if not case.exists():self.skipTest('Training Case 1 not mounted')
        vessels,port=read_case(str(case))
        instance=generate_scenario(vessels,port,label='small',seed=314159,
                                   updates_per_vessel=2,uncertainty=4.)
        env=IAMUEventEnv(instance,budget=2,candidate_limit=4,
            decision_clock='union',period=12.,reward_scale=100.,
            min_gain=.1,candidate_screen='positive',
            budget_features='absolute',value_features='global')
        obs=env.reset();raw=[];total=0.
        while True:
            # Keep every event; no difference in physical execution between representations.
            act=0;before=int(env.event_index)
            nxt,r,done,info=env.step(act)
            tr=Transition(obs.global_features.copy(),obs.action_features.copy(),obs.mask.copy(),
                act,float(r),nxt.global_features.copy(),nxt.action_features.copy(),nxt.mask.copy(),
                bool(done),dict(obs.decision_flags or {}),1)
            raw.append(RawStep(tr,int(obs.remaining_budget),int(nxt.remaining_budget),
                before,int(env.event_index),float(obs.time),float(nxt.time),1,0))
            total+=r;obs=nxt
            if done:break
        self.assertTrue(math.isclose(-total*100.,info['weighted_wait'],rel_tol=1e-6))
        for mode in ('primitive','budget_tail','decision_epoch'):
            compressed=compress_episode(raw,mode,1.)
            self.assertAlmostEqual(compressed.audit['source_total_reward'],total,places=6)
            self.assertAlmostEqual(compressed.audit['reconstructed_discounted_return'],total,places=6)
            self.assertEqual(compressed.audit['represented_raw_steps']+compressed.prefix.raw_steps,len(raw))
            if mode=='decision_epoch':
                self.assertTrue(all(m.obs_mask[1:].any() for m in compressed.transitions))

if __name__=='__main__':unittest.main()
