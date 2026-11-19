import unittest
from v13_diagnostics.summarize import group_metrics


class SummaryTests(unittest.TestCase):
    def test_uncertainty_not_called_wrong(self):
        rows=[dict(evaluated_model='A3',continuation='greedy',source_budget='1',
             scenario='shock',gap_error=3.,reference_sign='uncertain',
             missed_beneficial=0,false_positive_edit=0),
             dict(evaluated_model='A3',continuation='greedy',source_budget='1',
             scenario='shock',gap_error=-2.,reference_sign='beneficial',
             missed_beneficial=1,false_positive_edit=0)]
        s=group_metrics(rows,['evaluated_model','continuation','source_budget','scenario'])[0]
        self.assertEqual(s['n_uncertain'],1)
        self.assertEqual(s['missed_beneficial'],1)
        self.assertEqual(s['beneficial_recall'],0.)
        self.assertEqual(s['gap_mae'],2.5)

if __name__=='__main__':unittest.main()
