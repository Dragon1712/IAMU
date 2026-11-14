#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_budget_v10.py \
  --seeds 400001:400003 --budgets 2,5,10 \
  --scenarios shock,repeated,small \
  --risk-samples 16 --risk-alpha 0.9 \
  --beta-risk 0.25 --beta-budget 0.1 \
  --policies keep,periodic,greedy,set_ddqn,q_risk,q_budget,q_joint,expected \
  --verify-baseline --audit --device cpu \
  --out results/v10_e1_validation_seed42 "$@" \
  2>&1 | tee v10_e1_validation_seed42.log
