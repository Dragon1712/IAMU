#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_budget_v10.py \
  --scenarios shock,repeated,small --seeds 400001:400003 --budgets 2,5,10 \
  --verify-baseline --risk-samples 16 --mpc-samples 8 \
  --lookahead-stages 2 --lookahead-step-hours 12 \
  --beta-budget 0.25 --beta-risk 0.25 \
  --policies set_ddqn,q_risk,q_budget,q_joint,q_budget_heuristic,q_joint_heuristic \
  --device cpu --audit --out results/v10_e2_multistage_val_seed42 "$@" \
  2>&1 | tee v10_e2_multistage_val_seed42.log
