#!/usr/bin/env bash
# Rollout-throughput microbenchmark. Do not run concurrently with other large experiments.
# Use unchanged train/val cases and episode seeds for 1/4/6/8 worker comparisons.
set -euo pipefail
EPISODES="${EPISODES:-40}"
WORKERS_SET="${WORKERS_SET:-1 4 6 8}"
DEVICE="${DEVICE:-cpu}" # rollout-only: CPU learner is sufficient; no gradient updates
TRAIN_CASES="${TRAIN_CASES:-../2024LiERLDDE/case_code/3.txt}"
VAL_CASES="${VAL_CASES:-../2024LiERLDDE/case_code/8.txt}"
for w in ${WORKERS_SET}; do
  echo "[BENCH] workers=$w episodes=$EPISODES"
  # Expansion deliberately unquoted: pass whitespace-separated case paths.
  # shellcheck disable=SC2086
  python -u train_budgeted_ddqn_parallel.py \
    --train-cases $TRAIN_CASES \
    --val-cases $VAL_CASES \
    --workers "$w" --prefetch-per-worker 2 --sync-every 4 \
    --episodes "$EPISODES" --device "$DEVICE" \
    --scenarios shock,repeated --budgets 2,5 \
    --uncertainties 8 --candidate-limit 4 \
    --val-seeds 400001:400001 --clock union \
    --updates-per-step 0 \
    --eval-every "$EPISODES" \
    --out "results/v08_bench_workers_${w}"
done
python - <<'PY'
import json
from pathlib import Path
for root in sorted(Path('results').glob('v08_bench_workers_*')):
 p=root/'performance.json'
 if p.exists():
  x=json.loads(p.read_text())
  print(f"{root.name}: workers={x['workers_requested']}, completed={x['completed_episodes']}, "
        f"wall={x['total_wall_seconds']:.1f}s, throughput={x['avg_episodes_per_hour']:.1f} episodes/hour")
PY
