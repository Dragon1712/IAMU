#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_budget_v10.py \
  --smoke --risk-samples 8 --lookahead-stages 1 \
  --lookahead-step-hours 12 \
  --policies keep,periodic,set_ddqn,lookahead \
  --verify-baseline --audit --device cpu \
  --out results/v10_budget_lookahead_smoke "$@" \
  2>&1 | tee v10_budget_lookahead_smoke.log
