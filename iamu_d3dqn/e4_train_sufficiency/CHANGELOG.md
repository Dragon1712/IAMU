# E4-D Keep collapse + Q-gap error diagnosis: incremental code patch

**No existing files modified.** Adds a new, independent `iamu_d3dqn/e4_diagnostics/` directory.

- `diagnostic_core.py` — pure action/Keep-margin decomposition, stratified state sampling, paired gain calibration and tail regret helpers.
- `diagnose_keep_qgap.py` — SHA-checked E4 snapshot loading; exact validation-run parity; per-event Q diagnostics; same-state Q comparisons across checkpoints; identical-realization Keep/Edit paired counterfactuals with fixed frozen-anchor vs Greedy continuations; CSV/JSON audit trail.
- `run_parallel.py` — CPU-sharded 54-group runner with logs, non-overwriting outputs, resumable matching scientific plans and merged summaries only when every shard succeeded.
- `summarize_diagnostic.py` — human-readable Markdown summary with experiment limitations and Keep-collapse mechanism table.
- `test_diagnostic_core.py`, `test_diagnostic_runner.py` — 16 pure and synthetic-main tests (including Q mask/tie/margin, checksum tampering, counterfactual pairing and report generation).
- `test_real_environment.py` — two actual original-IAMU integration tests to run on the training server; intentionally skipped when original simulator is not installed in the local artifact-building environment.
- `README_E4_DIAG_zh.md` — scientific rationale, exact commands and acceptance criteria.

**Completed here:** Python syntax check, end-to-end toy-state pipeline and 16 offline tests. **Not completed here:** original server `.pt` checkpoint loading, actual physical IAMU simulation, training or new real E4-D measurements.
