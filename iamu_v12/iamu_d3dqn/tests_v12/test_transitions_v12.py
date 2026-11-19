"""V1.2 pure protocol regression; does not require IAMU simulator or Torch."""
import unittest
from types import SimpleNamespace
import numpy as np
from v12_protocol.transitions import RawStep
from v12_protocol.compress import compress_episode,is_decision_state
from v12_protocol.replay import EpisodeReplay


def episode(rewards, actionable, budgets, actions=None, times=None, eid=8):
    actions=actions or [0]*len(rewards)
    times=times or list(range(len(rewards)+1))
    steps=[]
    for i,r in enumerate(rewards):
        m=np.array([True,bool(actionable[i]),False],dtype=np.bool_)
        n=np.array([True,bool(actionable[i+1]) if i+1<len(rewards) else False,False],dtype=np.bool_)
        g=np.full(24,float(i),dtype=np.float32)
        g2=np.full(24,float(i+1),dtype=np.float32)
        af=np.full((3,24),float(i),dtype=np.float32)
        af2=np.full((3,24),float(i+1),dtype=np.float32)
        tr=SimpleNamespace(obs_global=g,obs_actions=af,obs_mask=m,action=int(actions[i]),
            reward=float(r),next_global=g2,next_actions=af2,next_mask=n,
            done=(i==len(rewards)-1),reason_flags={'forced_no_edit':not actionable[i]},episode_id=eid)
        steps.append(RawStep(tr,int(budgets[i]),int(budgets[i]-int(actions[i]!=0)),
            i,i+1,float(times[i]),float(times[i+1]),eid,20))
        if i+1<len(rewards):
            assert steps[-1].next_budget==budgets[i+1], (i,budgets)
    return steps

class CompressionTests(unittest.TestCase):
    def test_required_numerical_example(self):
        raw=episode([-1,-2,-3,-4,-5],[False,True,False,False,True],[2]*5)
        r=compress_episode(raw,'decision_epoch',.9)
        self.assertEqual(r.prefix.raw_steps,1)
        self.assertAlmostEqual(r.prefix.reward,-1.)
        self.assertEqual(len(r.transitions),2)
        self.assertAlmostEqual(r.transitions[0].reward,-7.94)
        self.assertAlmostEqual(r.transitions[0].bootstrap_discount,.729)
        self.assertEqual(r.transitions[0].n_raw_steps,3)
        self.assertAlmostEqual(r.transitions[1].reward,-5.)
        self.assertEqual(r.transitions[1].bootstrap_discount,0)
        self.assertAlmostEqual(r.audit['reconstructed_discounted_return'],-11.4265)
        one=compress_episode(raw,'decision_epoch',1.)
        self.assertAlmostEqual(one.audit['reconstructed_discounted_return'],-15.)
        self.assertEqual(one.transitions[0].action,0) # Deliberate Keep remains an anchor

    def test_tail_after_last_edit_is_included_exactly_once(self):
        raw=episode([-1,-2,-3,-4,-5],[False,True,True,False,False],
                    [2,2,1,0,0],actions=[0,1,1,0,0])
        a=compress_episode(raw,'budget_tail',1.)
        self.assertEqual([x.n_raw_steps for x in a.transitions],[1,1,3])
        self.assertEqual([x.reward for x in a.transitions],[-1.,-2.,-12.])
        self.assertEqual([x.source_budget for x in a.transitions],[2,2,1])
        self.assertEqual([x.next_budget for x in a.transitions],[2,1,0])
        self.assertEqual(a.transitions[-1].bootstrap_discount,0.)
        self.assertAlmostEqual(sum(x.reward for x in a.transitions),-15.)
        b=compress_episode(raw,'decision_epoch',1.)
        self.assertEqual([x.n_raw_steps for x in b.transitions],[1,3])
        self.assertEqual(b.prefix.raw_steps,1)
        self.assertAlmostEqual(sum(x.reward for x in b.transitions)+b.prefix.reward,-15.)

    def test_consecutive_decisions_have_unit_length(self):
        raw=episode([-1,-2],[True,True],[2,2])
        got=compress_episode(raw,'decision_epoch',.5)
        self.assertEqual([x.n_raw_steps for x in got.transitions],[1,1])
        self.assertEqual(got.transitions[0].bootstrap_discount,.5)

    def test_no_decision_and_zero_budget(self):
        raw=episode([-1,-2,-3],[False]*3,[0]*3)
        self.assertEqual(len(compress_episode(raw,'decision_epoch',1.).transitions),0)
        self.assertEqual(len(compress_episode(raw,'budget_tail',1.).transitions),0)
        self.assertEqual(len(compress_episode(raw,'primitive',1.).transitions),3)
        self.assertEqual(compress_episode(raw,'decision_epoch',1.).prefix.raw_steps,3)
        raw2=episode([-1,-2],[False]*2,[4]*2)
        self.assertEqual(len(compress_episode(raw2,'decision_epoch',1.).transitions),0)
        self.assertEqual(len(compress_episode(raw2,'budget_tail',1.).transitions),2)

    def test_zero_elapsed_events_still_count_raw_step_gamma(self):
        raw=episode([-1,-2,-3],[False,True,False],[1]*3,times=[0,0,0,0])
        r=compress_episode(raw,'decision_epoch',.5)
        self.assertEqual(len(r.transitions),1)
        self.assertEqual(r.transitions[0].n_raw_steps,2)
        self.assertAlmostEqual(r.transitions[0].reward,-3.5)
        self.assertEqual(r.transitions[0].elapsed_hours,0.)

    def test_original_arrays_remain_unmodified_and_are_copied(self):
        raw=episode([-1,-2],[True,True],[3,3])
        a=compress_episode(raw,'primitive',1.)
        self.assertFalse(np.shares_memory(a.transitions[0].obs_global,raw[0].transition.obs_global))
        a.transitions[0].obs_global[0]=128
        self.assertEqual(raw[0].transition.obs_global[0],0)

    def test_corrupt_budget_mask_or_chain_is_rejected(self):
        raw=episode([-1,-2],[True,False],[1,0],actions=[1,0])
        raw[0]=RawStep(raw[0].transition,0,0,0,1,0.,1.,8,20)
        with self.assertRaises(ValueError):compress_episode(raw,'primitive',1.)
        raw=episode([-1,-2],[True,False],[1,0],actions=[1,0])
        raw[1].transition.obs_global[0]=-3
        with self.assertRaisesRegex(ValueError,'Disconnected'):compress_episode(raw,'primitive',1.)
        raw=episode([-1,-2],[True,False],[1,0],actions=[1,0])
        raw[0].transition.done=True
        with self.assertRaises(ValueError):compress_episode(raw,'primitive',1.)

    def test_invalid_modes_and_gamma(self):
        raw=episode([-1],[False],[0])
        with self.assertRaises(ValueError):compress_episode(raw,'bad',1.)
        with self.assertRaises(ValueError):compress_episode(raw,'primitive',0.)

class ReplayTests(unittest.TestCase):
    def _macro(self,eid):
        return compress_episode(episode([-1,-2],[True,False],[1,0],actions=[1,0],eid=eid),
                                'decision_epoch',1.).transitions

    def test_episode_fifo_including_empty_episode(self):
        r=EpisodeReplay(2,23,'decision_epoch')
        r.append_episode(1,self._macro(1),2)
        r.append_episode(2,(),4)
        self.assertEqual(r.stats()['replay_episode_count'],2)
        self.assertEqual(r.stats()['replay_raw_coverage'],6)
        r.append_episode(3,self._macro(3),2)
        self.assertEqual([x[0] for x in r.episodes],[2,3])
        self.assertEqual(r.stats()['replay_raw_coverage'],6)
        self.assertEqual(len(r),1)

    def test_restore_preserves_sample_sequence_and_index_order(self):
        r=EpisodeReplay(3,234,'decision_epoch')
        for ep in [1,2,3]:r.append_episode(ep,self._macro(ep),2)
        r.sample(2)
        saved=r.state_dict()
        q=EpisodeReplay(3,1,'decision_epoch')
        q.load_state_dict(saved)
        a,_=r.sample(2);b,_=q.sample(2)
        self.assertEqual([x.episode_id for x in a],[x.episode_id for x in b])
        self.assertEqual(r.stats(),q.stats())
        corrupt=dict(saved,actionable_indices=[999])
        with self.assertRaises(ValueError):q.load_state_dict(corrupt)

    def test_a1_stratified_and_invalid_a3(self):
        r=EpisodeReplay(4,44,'primitive')
        for i in [1,2]:
            raw=episode([-1,-2,-3,-4],[True,False,False,True],[2]*4,eid=i)
            r.append_episode(i,compress_episode(raw,'primitive',1.).transitions,4)
        xs,st=r.sample(4,'stratified',.25)
        self.assertEqual(st['sample_actionable'],1)
        self.assertEqual(len({id(x) for x in xs}),4)
        q=EpisodeReplay(4,4,'decision_epoch')
        with self.assertRaises(ValueError):q.sample(1,'stratified')
        with self.assertRaises(ValueError):r.sample(999)

if __name__=='__main__':unittest.main()
