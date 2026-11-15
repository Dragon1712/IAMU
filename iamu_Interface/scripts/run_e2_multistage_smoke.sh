#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_budget_v10.py \
  --smoke --verify-baseline --risk-samples 16 --mpc-samples 4 \
  --lookahead-stages 2 --beta-budget 0.25 --beta-risk 0.25 \
  --policies set_ddqn,q_budget,q_joint,q_budget_heuristic,q_joint_heuristic,lookahead \
  --device cpu --audit --out results/v10_e2_multistage_smoke "$@" \
  2>&1 | tee v10_e2_multistage_smoke.log
