#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_sampling_stability_v10.py \
  --cases assets/cases/8.txt --scenarios shock,repeated \
  --seeds 400001 --budgets 2,10 --max-groups 4 \
  --states-per-group 1 --stride 2 \
  --sample-sizes 16,64,128 --reference-samples 256 --repeats 3 \
  --device cpu --out results/v10_e2_risk_stability_smoke "$@" \
  2>&1 | tee v10_e2_risk_stability_smoke.log
