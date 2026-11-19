"""Audit event-driven SMDP return semantics against actual V1.2 compressor."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from v12_protocol.compress import compress_episode,validate_raw_episode
from v12_protocol.transitions import RawStep
from v13_diagnostics.audit_compression import exactly_solved_toy


def raw_example(rewards,masks,actions,budgets):
    n=len(rewards);raw=[]
    for i in range(n):
        obs=np.full((24,),i,dtype=np.float32)
        nxt=np.full((24,),i+1,dtype=np.float32)
        oa=np.full((3,24),i,dtype=np.float32)
        na=np.full((3,24),i+1,dtype=np.float32)
        m=np.array(masks[i],dtype=bool)
        nm=np.array(masks[i+1] if i+1<n else [True,False,False],dtype=bool)
        tr=SimpleNamespace(obs_global=obs,obs_actions=oa,obs_mask=m,action=int(actions[i]),
              reward=float(rewards[i]),next_global=nxt,next_actions=na,
              next_mask=nm,done=(i==n-1),episode_id=11,reason_flags={})
        raw.append(RawStep(tr,budgets[i],budgets[i+1],i,i+1,float(i),float(i+1),11,0))
    return raw


class CompressionTests(unittest.TestCase):
    def test_exact_dp_gap_gamma_one(self):
        self.assertAlmostEqual(exactly_solved_toy(1.)['q_gap'],
                               exactly_solved_toy(1.)['compressed_gap'])

    def test_exact_dp_gap_gamma_nine_tenths(self):
        self.assertAlmostEqual(exactly_solved_toy(.9)['q_gap'],
                               exactly_solved_toy(.9)['compressed_gap'])

    def test_requested_five_step_discount_example(self):
        raw=raw_example([-1,-2,-3,-4,-5],
          [[1,0,0],[1,1,0],[1,0,0],[1,0,0],[1,1,0]],
          [0,0,0,0,1],[1,1,1,1,1,0])
        out=compress_episode(raw,'decision_epoch',.9)
        self.assertEqual(out.prefix.raw_steps,1)
        self.assertAlmostEqual(out.prefix.reward,-1)
        self.assertEqual(len(out.transitions),2)
        self.assertAlmostEqual(out.transitions[0].reward,-7.94)
        self.assertAlmostEqual(out.transitions[0].bootstrap_discount,.9**3)
        self.assertAlmostEqual(out.transitions[1].reward,-5.)
        self.assertEqual(out.transitions[1].bootstrap_discount,0.)
        self.assertAlmostEqual(out.audit['reconstructed_discounted_return'],-11.4265)

    def test_budget_exhaustion_edit_keeps_remaining_tail_once(self):
        raw=raw_example([-1,-2,-3,-4,-5],
          [[1,0,0],[1,1,0],[1,1,0],[1,0,0],[1,0,0]],
          [0,0,1,0,0],[2,2,2,1,1,1])
        # Final Edit consumes budget one at i2: source budget corrected below.
        raw[2]=raw[2].__class__(raw[2].transition,1,0,2,3,2.,3.,11,0)
        raw[1]=raw[1].__class__(raw[1].transition,2,1,1,2,1.,2.,11,0)
        raw[3]=raw[3].__class__(raw[3].transition,0,0,3,4,3.,4.,11,0)
        raw[4]=raw[4].__class__(raw[4].transition,0,0,4,5,4.,5.,11,0)
        # For i1 budget goes 2->1, so it must Edit, not Keep.
        raw[1].transition.action=1
        validate_raw_episode(raw)
        out=compress_episode(raw,'budget_tail',.9)
        self.assertEqual(len(out.transitions),3)
        self.assertEqual(out.transitions[-1].raw_start_index,2)
        self.assertEqual(out.transitions[-1].n_raw_steps,3)
        self.assertAlmostEqual(out.transitions[-1].reward,-3-.9*4-.9**2*5)
        self.assertEqual(out.transitions[-1].bootstrap_discount,0)

    def test_keep_at_actionable_is_anchor(self):
        raw=raw_example([-1,-2,-3],[[1,1,0],[1,0,0],[1,1,0]],
                        [0,0,1],[1,1,1,0])
        o=compress_episode(raw,'decision_epoch',1.)
        self.assertEqual([x.raw_start_index for x in o.transitions],[0,2])
        self.assertEqual(o.transitions[0].action,0)

    def test_no_decisions_empty_but_cost_preserved(self):
        raw=raw_example([-1,-2],[[1,0,0],[1,0,0]],[0,0],[0,0,0])
        o=compress_episode(raw,'decision_epoch',1.)
        self.assertEqual(len(o.transitions),0)
        self.assertEqual(o.prefix.raw_steps,2)
        self.assertAlmostEqual(o.prefix.reward,-3)

    def test_bad_budget_is_rejected(self):
        raw=raw_example([-1,-2],[[1,1,0],[1,0,0]],[1,0],[1,1,1])
        with self.assertRaises(ValueError):compress_episode(raw,'decision_epoch',.9)

if __name__=='__main__':unittest.main()
