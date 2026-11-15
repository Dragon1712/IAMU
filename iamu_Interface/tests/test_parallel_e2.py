"""Pure standard-library tests for functional parallelism; no checkpoint needed.

Run: python -m unittest discover -s tests -p 'test_parallel_e2.py' -v
"""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('parallel_e2_core', SCRIPT_DIR/'parallel_e2_core.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FunctionalShardTests(unittest.TestCase):
    def test_unordered_duplicate_and_invalid_ranges(self):
        self.assertEqual(module.parse_ints('2,5:7'), [2,5,6,7])
        with self.assertRaises(ValueError):
            module.parse_ints('2,2')
        with self.assertRaises(ValueError):
            module.parse_ints('5:3')

    def test_merge_summary_equals_pooled_observations_not_shard_means(self):
        # Unequal shard size: direct averaging summary means would be wrong.
        rows=[]
        for idx,wait in enumerate([20.,30.,100.]):
            rows.append(dict(case=str(idx), scenario='shock', eta_seed='1',budget='2',
                             policy='set_ddqn',weighted_wait=str(wait),spent='1',
                             gain_vs_periodic=str(25.-wait),gain_vs_set_ddqn='0',
                             elapsed_sec='9.0'))
        summary=module.summarize_runs(rows)
        overall=next(x for x in summary if x['policy']=='set_ddqn' and x['scenario']=='ALL' and x['budget']=='ALL')
        self.assertAlmostEqual(overall['mean_wait'],50.)
        self.assertAlmostEqual(overall['cvar90_regret_vs_periodic'],75.)
        self.assertEqual(overall['n'],3)

    def test_detect_duplicate_primary_keys(self):
        r=dict(case='8',scenario='shock',eta_seed='1',budget='2',policy='set_ddqn')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            module.validate_unique([r,r],module.SHARD_PRIMARY,'run')

    def test_serial_parallel_numerically_equivalent_but_different_time(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            first=dict(case='8', scenario='shock',eta_seed='400001',budget='2',
                       policy='set_ddqn',weighted_wait='1900.125',spent='2',
                       elapsed_sec='12.0',gain_vs_periodic='2.0',
                       gain_vs_keep='13',gain_vs_set_ddqn='0',
                       decisions='9',edits_eta='1',edits_clock='1')
            second={**first,'elapsed_sec':'300.5'}
            module.write_rows(base/'serial.csv',[first])
            module.write_rows(base/'parallel.csv',[second])
            n,count=module.compare_csv(base/'serial.csv',base/'parallel.csv')
            self.assertEqual(n,1)
            self.assertGreaterEqual(count,5)
            second['weighted_wait']='1901.125'
            module.write_rows(base/'parallel.csv',[second])
            with self.assertRaisesRegex(AssertionError,'Mismatch'):
                module.compare_csv(base/'serial.csv',base/'parallel.csv')

    def test_risk_replicate_flip_frequency(self):
        rows=[]
        for rep,action in enumerate([0,1,0]):
            rows.append(dict(case='8',scenario='shock',eta_seed='400001',budget='2',
                             event_index='4',sample_count='16',replicate=str(rep),
                             decision_policy='q_risk',selected_action=str(action),
                             action_agreement='1',best_tail_agreement='1',best_mean_agreement='1',
                             tail_mae='1.0',tail_max_error='1.0',gain_mae='1.0',
                             harm_prob_mae='0.1',risk_eval_seconds='.05'))
        summary=module.summarize_stability(rows)
        z=next(r for r in summary if r['scenario']=='ALL')
        self.assertEqual(z['n_unique_states'],1)
        self.assertEqual(z['n_states_with_action_disagreement'],1)
        self.assertAlmostEqual(z['mean_within_state_action_flip_rate'],1/3)

    def test_source_job_grouping_is_unique(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            localcase=p/'8.txt';localcase.write_text('fixture')
            class A: pass
            a=A(); a.scenarios='shock,repeated';a.eta_seeds='400001:400002';a.budgets='2,5'
            a.checkpoint_path=p/'best.pt';a.manifest_path=p/'manifest.json'
            a.policies='set_ddqn,q_joint';a.risk_samples=16;a.mpc_samples=8;a.h=2
            a.step_hours=12.;a.lookahead_risk_weight=0.;a.beta_risk=.25;a.beta_budget=.25
            a.legacy_beta_budget=.1;a.alpha=.9;a.sigma=7.12463;a.belief='censored'
            a.risk_sizes='16,64,128';a.reference_samples=256;a.repeats=3
            a.states_per_group=2;a.stride=10;a.mpc_sizes='4,8,16';a.h_sizes='0,1,2,3'
            s=importlib.util.spec_from_file_location('run_e2_parallel',SCRIPT_DIR/'run_e2_parallel.py')
            import sys
            sys.path.insert(0,str(SCRIPT_DIR))
            import run_e2_parallel
            jobs=run_e2_parallel.build_jobs(a,[localcase],['multistage','risk_stability','mpc_samples'])
            self.assertEqual(len(jobs), 2*2*2*(1+1+3))
            self.assertEqual(len(set(j.identity for j in jobs)),len(jobs))
            self.assertEqual(sum(j.workflow=='mpc_samples' for j in jobs),24)

if __name__=='__main__':
    unittest.main()
