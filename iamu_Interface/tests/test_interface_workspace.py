"""Setup/structural tests that run even before the V0.9 engine is installed."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import setup_interface


class WorkspaceTests(unittest.TestCase):
    def test_v10_experiment_files_present(self):
        root = setup_interface.ROOT
        for relative in (
            'iamu_dynamic/v10_risk_features.py',
            'iamu_dynamic/v10_budget_lookahead.py',
            'iamu_dynamic/v10_policy.py',
            'iamu_dynamic/v10_candidate_api.py',
            'iamu_dynamic/v10_candidate_integration.py',
            'experiments/eval_risk_budget_v10.py',
            'scripts/run_smoke.sh',
        ):
            with self.subTest(relative=relative):
                self.assertTrue((root / relative).is_file())

    def test_one_time_snapshot_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            src, workspace = p/'iamu_d3dqn', p/'iamu_Interface'
            source_pkg = src/'iamu_dynamic'
            source_pkg.mkdir(parents=True)
            workspace.mkdir()
            for name in setup_interface.V09_FILES:
                (source_pkg/name).write_text(f'# {name}\n', encoding='utf-8')
            with patch.object(setup_interface, 'ROOT', workspace):
                setup_interface.main(['--source', str(src)])
                for name in setup_interface.V09_FILES:
                    self.assertEqual((workspace/'iamu_dynamic'/name).read_text(),
                                     (source_pkg/name).read_text())
                self.assertTrue((workspace/'dependency_snapshot.json').is_file())
                setup_interface.main(['--source', str(src), '--check-only'])
                # New changes to the running V0.9 source must not overwrite frozen snapshots.
                (source_pkg/'core.py').write_text('# source changed\n')
                with self.assertRaisesRegex(RuntimeError, 'Refusing to overwrite'):
                    setup_interface.main(['--source', str(src)])
                self.assertEqual((workspace/'iamu_dynamic'/'core.py').read_text(),
                                 '# core.py\n')
                setup_interface.main(['--source', str(src), '--force'])
                self.assertEqual((workspace/'iamu_dynamic'/'core.py').read_text(),
                                 '# source changed\n')


if __name__ == '__main__':
    unittest.main()
