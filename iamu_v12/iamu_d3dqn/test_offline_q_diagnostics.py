"""Run from iamu_d3dqn root: python -m unittest test_offline_q_diagnostics -v

Creates three synthetic checkpoint files; does not require trained weights.
Exercises real IAMU simulator with tiny case files and the complete offline CLI.
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iamu_dynamic.rl_ddqn import DoubleDQNAgent
import offline_q_diagnostics as diag


class OfflineQDiagnosticsTests(unittest.TestCase):
    def test_mask_and_legal_action_scoring(self):
        from iamu_dynamic.rl_env import RLObservation, GLOBAL_DIM, ACTION_DIM
        agent = DoubleDQNAgent(hidden_dim=16, seed=7, device='cpu', keep_margin=0.1,
                                q_network='d3qn')
        obs = RLObservation(np.zeros(GLOBAL_DIM, dtype=np.float32),
                            np.zeros((5, ACTION_DIM), dtype=np.float32),
                            np.array([True, False, False, False, False]),
                            time=0., event_type='eta', remaining_budget=0, candidate_ids=())
        result = diag.score_observation(agent, obs)
        self.assertEqual(result['action'], 0)
        self.assertIsNone(result['best_edit'])
        self.assertIsNone(result['gap'])
        self.assertAlmostEqual(result['q'][0], result['v'], places=5)
        obs.mask[2] = True
        obs.candidate_ids = (3,)  # synthetic explicit candidate
        result = diag.score_observation(agent, obs)
        self.assertIsNotNone(result['gap'])
        self.assertEqual(result['best_edit'], 2)
        self.assertIn(result['action'], (0, 2))

    def test_three_network_full_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # All cases have distinct content and satisfy existing case TSV format.
            c8 = root / 'case8.txt'
            c9 = root / 'case9.txt'
            c8.write_text('4\t60\t380\t3\t8\n6\t80\t400\t3\n9\t75\t360\t3\n', encoding='utf-8')
            c9.write_text('5\t62\t360\t3\t8\n7\t80\t380\t3\n10\t70\t350\t3\n', encoding='utf-8')
            common = dict(val_cases=[str(c8), str(c9)],
                          training_args=dict(
                              candidate_limit=2,period=12.,reward_scale=100.,
                              candidate_screen='positive',min_gain=.1,
                              budget_features='absolute',value_features='global',
                              updates=2,validation_uncertainty=8.,clock='union',
                              val_seeds='400001:400002',scenarios='small,shock',
                              budgets='2,5',gamma=1.,keep_margin=.1,
                          ))
            filenames={}
            for name,(arch,width) in diag.MODEL_SPECS.items():
                d = root / name
                d.mkdir()
                setting = json.loads(json.dumps(common))
                setting['training_args']['q_network'] = arch
                setting['training_args']['hidden_dim'] = width
                (d/'manifest.json').write_text(json.dumps(setting),encoding='utf-8')
                agent = DoubleDQNAgent(q_network=arch,hidden_dim=width,seed=17,
                                        device='cpu',keep_margin=.1)
                torch.save(agent.checkpoint({'selected_episode':25,
                                             'n_gradient_updates':750}),d/'best.pt')
                filenames[name]=d/'best.pt'
            out=root/'diagnosis'
            cmd=['--ddqn96',str(filenames['ddqn96']),
                 '--ddqn130',str(filenames['ddqn130']),
                 '--d3qn96',str(filenames['d3qn96']),
                 '--manifest',str(root/'d3qn96/manifest.json'),
                 '--scenarios','small', '--seeds','400001', '--budgets','2',
                 '--max-groups','1','--out',str(out)]
            self.assertEqual(diag.main(cmd),0)
            def csv_rows(name):
                with (out/name).open(encoding='utf-8',newline='') as f:
                    return list(csv.DictReader(f))
            eps=csv_rows('episode_outcomes.csv')
            self.assertEqual({e['policy'] for e in eps},
                             {'keep','greedy','periodic','ddqn96','ddqn130','d3qn96'})
            self.assertEqual(len(eps),6)
            acts=csv_rows('action_q_values.csv')
            self.assertGreater(len(acts),0)
            for a in acts:
                self.assertGreaterEqual(int(a['action']),0)
            common=csv_rows('common_state_comparison.csv')
            self.assertGreater(len(common),0)
            mc=csv_rows('onpolicy_mc_returns.csv')
            self.assertGreater(len(mc),0)
            self.assertTrue(all(np.isfinite(float(x['error_q_minus_mc'])) for x in mc))
            self.assertEqual(len({e['model'] for e in mc}),3)
            self.assertTrue((out/'performance_summary.csv').is_file())
            self.assertTrue((out/'q_calibration_summary.csv').is_file())
            manifest=json.loads((out/'diagnostic_manifest.json').read_text())
            self.assertEqual(manifest['grid_groups'],1)
            self.assertEqual(manifest['models']['d3qn96']['parameters'],23522)


if __name__ == '__main__':
    unittest.main()
