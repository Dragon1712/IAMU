# V1.1 Parallel Controlled Protocol - Code Change Ledger

Only additive files. No original source overwritten.

- `build_variants.py` transforms exact known IAMU source anchors into independent modules `rl_env_v11`, `rl_ddqn_v11`, `rl_experiment_v11`, `parallel_validation_v11`, and generated `v11_actors`. Fails closed if sources changed.
- `indexed_replay.py` ring buffer with O(1) incremental actionable/forced index pools, without-replacement mini-batch sampling and optional importance weights; original recovery `.items` façade retained.
- `train_v11.py` multi-CPU-actor fixed-policy collection round + deterministic Episode ID replay consumption + central GPU learner, round-based recovery, frozen validation checkpoints and EP200 full-state fork.
- `validation_v11.py` isolated spawn CPU worker pool, models loaded once per worker and independent scenario jobs joined in fixed order.
- `audit_candidates.py` fixed-state full proposal coverage and small hindsight audit of screened candidates.
- `replay_reasons.py` multi-label and exclusive display summary for forced Keep.
- `margin_scan.py` frozen weights, several execution margins, same new candidate mode.
- `fixed_label_probe.py` immutable EP200 reference bootstrap labels.
- `batch_inference.py` optional masked batched scoring (off by default).
- `mpc_interface.py` shared candidate-action kernel; forecast mode refuses to use hidden true episode future.
- `launch_four_gpu.py` GPU-isolated multi-run command launcher.
- `bootstrap_margin_3way.json` reproducible A/B/C margin study starter configuration.
- `tests_v11/` local logic & server-only original-simulator integration tests.

Not included in this release: a new forecasting model for online MPC, full scenario branch cache, independent held-out Case10–14 results. No numerical performance claims have been made from this patch alone.
