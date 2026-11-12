"""Causal ETA-clock and offline-only diagnostic regression tests."""
from __future__ import annotations
import json
from pathlib import Path
import unittest

from iamu_dynamic import (
    Port, Scenario, Update, Vessel, demo_case, generate_scenario,
    read_case, simulate,
)

HERE = Path(__file__).resolve().parents[1]


class ExogenousNotificationTests(unittest.TestCase):
    def test_update_times_are_public_initial_eta_schedule(self):
        vessels, port = demo_case()
        for label in ('small', 'shock', 'repeated'):
            for seed in (11, 22, 33):
                s = generate_scenario(vessels, port, label, seed, 3, 8)
                for e in s.updates:
                    v = vessels[e.vessel_id]
                    expected = {v.eta0 * k / 4 for k in (1, 2, 3)}
                    self.assertTrue(any(abs(e.time-t) < 1e-10 for t in expected))
                    self.assertLess(e.time, s.true_etas[e.vessel_id])
                    self.assertGreaterEqual(e.new_eta + 1e-10, e.time)

    def test_schedule_does_not_change_when_only_hidden_arrivals_change(self):
        vessels, port = demo_case()
        # The timestamps of *surviving* public notifications are independent
        # of latent arrival times and surprise severity.
        a = generate_scenario(vessels, port, 'shock', 11, 3, 2)
        b = generate_scenario(vessels, port, 'shock', 87, 3, 20)
        for s in (a, b):
            for i, v in enumerate(vessels):
                times = [e.time for e in s.updates if e.vessel_id == i]
                candidates = {v.eta0 * k / 4 for k in (1, 2, 3)}
                self.assertTrue(all(any(abs(t-x)<1e-10 for x in candidates) for t in times))

    def test_update_count_cannot_modify_hidden_arrivals(self):
        vessels, port = demo_case()
        a = generate_scenario(vessels, port, 'shock', 22, 1, 8)
        b = generate_scenario(vessels, port, 'shock', 22, 5, 8)
        self.assertEqual(a.true_etas, b.true_etas)

    def test_no_eta_notifications_after_early_actual_arrival(self):
        vessels = [Vessel(0, 120, 20, 100, 2)]
        port = Port(2)
        for seed in range(50):
            s = generate_scenario(vessels, port, 'shock', seed, 4, 40)
            for e in s.updates:
                self.assertLess(e.time, s.true_etas[0])
        self.assertTrue(any(
            len(generate_scenario(vessels, port, 'shock', k, 4, 40).updates) < 4
            for k in range(50)
        ))

    def test_eta_input_validation(self):
        vessels, port = demo_case()
        for invalid in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                generate_scenario(vessels, port, uncertainty=invalid)


class HistoricalFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = json.loads((HERE/'fixtures/legacy_case1_shock22.json').read_text())
        v=[Vessel(**x) for x in raw['vessels']]
        cls.scenario=Scenario(
            v, Port(**raw['port']),
            {int(k):float(a) for k,a in raw['true_etas'].items()},
            [Update(**x) for x in raw['updates']],
            raw['label'], raw['seed'],
        )

    def test_identify_exact_high_impact_edit_in_old_case(self):
        r=simulate(self.scenario,'budget_aware',5,diagnostic_hindsight=True)
        self.assertAlmostEqual(r['metrics']['weighted_wait'],2954.967851,places=5)
        self.assertEqual(r['metrics']['spent'],5)
        events=r['metrics']['hindsight_audit']
        self.assertEqual([e['changed'] for e in events],[[3],[3],[1],[7],[2]])
        offender=events[3]
        self.assertAlmostEqual(offender['time'],12.03652,places=4)
        self.assertAlmostEqual(offender['estimated_gain'],4.102242,places=4)
        self.assertAlmostEqual(offender['hindsight_gain'],-313.504627,places=4)
        self.assertEqual(offender['worst_ship'],5)
        self.assertAlmostEqual(offender['worst_start_delta'],314.800612,places=4)

    def test_offline_hindsight_never_changes_action_selection(self):
        no=simulate(self.scenario,'budget_aware',5)
        diag=simulate(self.scenario,'budget_aware',5,diagnostic_hindsight=True)
        self.assertEqual(no['revisions'],diag['revisions'])
        self.assertEqual(no['realized'],diag['realized'])
        self.assertEqual(no['final_published'],diag['final_published'])
        self.assertNotIn('hindsight_audit',no['metrics'])


if __name__=='__main__':
    unittest.main()
