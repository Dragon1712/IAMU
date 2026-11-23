import unittest
from v13_boundary.fixed_replay import check_data
from v13_boundary.summarize_boundary import summarise

class FixedTests(unittest.TestCase):
    def make(self,representation,n,sha):
        return dict(representation=representation,gamma=1.,source_sha256=sha,
            episodes=[dict(episode_id=1,job=dict(episode_id=1),transitions=[None]*n)])
    def test_same_trajectory_required(self):
        self.assertEqual(check_data(self.make('primitive',8,'same'),
                                    self.make('budget_tail',2,'same')),(8,2))
    def test_reject_different_source(self):
        with self.assertRaises(ValueError):
            check_data(self.make('primitive',8,'a'),self.make('budget_tail',2,'b'))
    def test_episode_clustered_summary(self):
        rows=[]
        for sid,episode,err in [('1:0',1,10),('1:1',1,20),('2:0',2,-5)]:
            for i in range(8):
                rows.append(dict(state_id=sid,episode_id=episode,model='A2',
                   continuation='self',scenario='shock',tape_index=i,
                   keep_bootstrap_error=err,edit_bootstrap_error=0,
                   predicted_gap_error=-err,return_gap=0,target_gap=-err,
                   predicted_gap=-1,identity_residual=0,keep_interval_length=1,
                   edit_interval_length=10,keep_cost_understatement=100*err))
        s,g=summarise(rows,boots=20,seed=4)
        self.assertEqual(len(s),3)
        result=next(x for x in g if x['metric']=='keep_bootstrap_error' and x['scenario']=='ALL')
        self.assertEqual(result['n_episode_clusters'],2)
        self.assertAlmostEqual(result['mean'],25/3)
