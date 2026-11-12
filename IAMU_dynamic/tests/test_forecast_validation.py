"""Standalone tests for probability forecasts, holdout separation and no-oracle use."""
import argparse
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from statistics import mean

from iamu_dynamic import generate_scenario, read_case
from iamu_dynamic.risk_selection import RiskConfig, sample_future_arrivals
from validate_eta_forecast import (
    ArrivalBelief, ForecastObservation, bootstrap_test, build_observations,
    parse_seeds, run, score_one, verify_disjoint,
)


class ForecastValidationTests(unittest.TestCase):
    def test_censored_exactly_represents_existing_sampler(self):
        mu, now, sigma = 11., 13., 4.
        d = ArrivalBelief('censored',sigma)
        # The actual V0.4 sampler uses a point mass at now by max() clipping.
        batch = sample_future_arrivals({5:mu},now,set(),run_seed=4,
            event_id=3,event_type='eta',config=RiskConfig(n_samples=6000,forecast_std=sigma))
        theoretical_mass = d.cdf(now,mu,now)
        observed_mass = mean(v[5] == now for v in batch)
        self.assertAlmostEqual(observed_mass,theoretical_mass,delta=.025)
        self.assertAlmostEqual(d.quantile(.5,mu,now),now)

    def test_truncated_conditions_on_unarrived(self):
        censored=ArrivalBelief('censored',4)
        truncated=ArrivalBelief('truncated',4)
        mu, now=11.,13.
        self.assertGreater(censored.cdf(now,mu,now),0)
        self.assertAlmostEqual(truncated.cdf(now,mu,now),0)
        self.assertGreater(truncated.quantile(.5,mu,now),now)
        self.assertAlmostEqual(truncated.cdf(truncated.quantile(.9,mu,now),mu,now),.9,places=7)

    def test_crps_and_nll_are_proper_in_point_prediction_sanity_case(self):
        correct=ArrivalBelief('censored',1)
        very_broad=ArrivalBelief('censored',20)
        self.assertLess(correct.crps(10,10,0,32),very_broad.crps(10,10,0,32))
        self.assertLess(correct.nll(10,10,0),very_broad.nll(10,10,0))

    def test_predictions_do_not_accept_hidden_actual_arrival(self):
        base=ForecastObservation('test','dummy','shock',1,0,'update',0,5,13,20)
        altered=replace(base,true_arrival=200)
        belief=ArrivalBelief('censored',8)
        self.assertEqual(belief.quantile(.95,base.eta,base.time),
                         belief.quantile(.95,altered.eta,altered.time))
        self.assertNotEqual(score_one(base,belief)['crps'],score_one(altered,belief)['crps'])

    def test_observation_has_only_real_public_update_events(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory)/'one.txt'
            case.write_text('10\t50\t300\t2\t4\n20\t70\t300\t2\n')
            rows=build_observations(str(case),['shock'],[51], 'test', 2,8)
            self.assertEqual(sum(x.stage=='initial' for x in rows),2)
            self.assertTrue(all(x.true_arrival>x.time for x in rows if x.stage=='update'))
            self.assertTrue(all(x.eta>=x.time-1e-9 for x in rows if x.stage=='update'))

    def test_disjoint_holdout_guard(self):
        with tempfile.TemporaryDirectory() as d:
            ps=[Path(d)/f'{i}.txt' for i in range(3)]
            for i,p in enumerate(ps):p.write_text(f'{10+i}\t50\t300\t2\t4\n')
            cfg=argparse.Namespace(train_cases=[str(ps[0])],val_cases=[str(ps[1])],
                                   test_cases=[str(ps[2])])
            verify_disjoint(cfg,[101],[201],[1001])
            with self.assertRaises(ValueError):verify_disjoint(cfg,[101],[201],[101])
            cfg.test_cases=[str(ps[1])]
            with self.assertRaises(ValueError):verify_disjoint(cfg,[101],[201],[1001])

    def test_parse_seed_ranges(self):
        self.assertEqual(parse_seeds('4,8:10'),[4,8,9,10])
        with self.assertRaises(ValueError):parse_seeds('4,4')

    def test_integration_locked_selection_ignores_test_seed_change(self):
        with tempfile.TemporaryDirectory() as root:
            files=[]
            for i in range(3):
                p=Path(root)/f'case{i}.txt'
                p.write_text(f'{10+i}\t50\t300\t2\t4\n{30+i}\t100\t280\t2\n')
                files.append(str(p))
            base=dict(train_cases=[files[0]],val_cases=[files[1]],test_cases=[files[2]],
                train_seeds='101:102',val_seeds='201:202',test_seeds='1001:1002',
                scenarios='shock,repeated',updates=2,uncertainty=8.,ood_uncertainty=None,
                sigmas='4,8',families='censored,truncated',grid=12,bootstrap=20,
                bootstrap_seed=14,no_initial=False,save_predictions=False)
            left=run(argparse.Namespace(**base,out=str(Path(root)/'run_a')))
            right=run(argparse.Namespace(**{**base,'test_seeds':'2001:2002'},
                                         out=str(Path(root)/'run_b')))
            self.assertEqual(left['selected_model'],right['selected_model'])
            self.assertEqual(left['training_fitted_sigma'],right['training_fitted_sigma'])
            self.assertTrue((Path(root)/'run_a'/'test_bootstrap_ci.csv').is_file())


if __name__=='__main__':unittest.main()
