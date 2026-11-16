#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u experiments/eval_e3_fusion_v10.py \
  --cases assets/cases/8.txt assets/cases/9.txt \
  --scenarios shock,repeated,small --seeds 400001:400003 --budgets 2,5,10 \
  --modes q_only,q_risk_only,pure_mpc,pure_mpc_risk,q_fixed,q_confidence,q_agreement,q_adaptive \
  --risk-samples 16 --mpc-samples 8 --lookahead-stages 2 \
  --beta-budget 0.25 --beta-risk 0.25 \
  --verify-baseline --out results/v10_e3_fusion_seed42 "$@"
