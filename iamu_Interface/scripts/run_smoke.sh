#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_budget_v10.py \
  --smoke --verify-baseline --risk-samples 8 --audit \
  --policies keep,periodic,greedy,set_ddqn,q_risk,q_budget,q_joint,expected \
  --device cpu --out results/v10_e1_smoke "$@" \
  2>&1 | tee v10_e1_smoke.log
