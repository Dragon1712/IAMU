#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
for M in 4 8 16; do
  python -u experiments/eval_risk_budget_v10.py \
    --smoke --verify-baseline --risk-samples 16 --mpc-samples "$M" \
    --lookahead-stages 2 --beta-budget 0.25 --beta-risk 0.25 \
    --policies set_ddqn,q_joint --device cpu --audit \
    --out "results/v10_e2_mpc_sample_M${M}_smoke" "$@" \
    2>&1 | tee "v10_e2_mpc_sample_M${M}_smoke.log"
done
