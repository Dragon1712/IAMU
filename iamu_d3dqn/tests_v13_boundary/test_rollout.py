import unittest
from v13_boundary.train_rollout import select_supervision
from v13_boundary.rekey_training_bank import rekey_training_bank

class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for b in (1,2,4):
            for i in range(20):
                self.rows.append(dict(source_budget=b,state_id=f'{b*1000+i}:0',
                    action_slot=1,allocated_sim_steps=100+i%2))
    def test_equal_budget_label_count(self):
        g,c=select_supervision(self.rows,12,44,.15)
        self.assertEqual(len(g['R1']),12)
        self.assertEqual(g['R1'],g['R2'])
        self.assertEqual({x['source_budget'] for x in g['R3']},{1})
        self.assertEqual({x['source_budget'] for x in g['R2']},{1,2,4})
        self.assertLess(c['relative_sim_cost_gap'],.15)
    def test_budget_cost_mismatch_rejected(self):
        for x in self.rows:
            if x['source_budget']!=1:x['allocated_sim_steps']=10000
        with self.assertRaisesRegex(ValueError,'Label cost mismatch'):
            select_supervision(self.rows,12,44,.05)
    def test_namespaced_episodes(self):
        bank={'probes':[dict(episode_id=1000001,raw_step_index=8,
                    state_id='1000001:8',external_seed=900001,snapshot={'x':1})]}
        rekey=rekey_training_bank(bank)
        self.assertEqual(rekey['probes'][0]['state_id'],'11000001:8')
        self.assertEqual(bank['probes'][0]['state_id'],'1000001:8')
        self.assertEqual(rekey['probes'][0]['external_seed'],900001)
