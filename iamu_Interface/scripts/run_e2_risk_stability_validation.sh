#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_sampling_stability_v10.py \
  --scenarios shock,repeated,small --seeds 400001:400003 --budgets 2,5,10 \
  --states-per-group 2 --stride 10 \
  --sample-sizes 16,64,128 --reference-samples 256 --repeats 3 \
  --device cpu --out results/v10_e2_risk_stability_val_seed42 "$@" \
  2>&1 | tee v10_e2_risk_stability_val_seed42.log
