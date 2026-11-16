#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u experiments/eval_e3_value_calibration_v10.py \
  --cases assets/cases/8.txt assets/cases/9.txt \
  --scenarios shock,repeated,small --seeds 400001:400003 --budgets 2,5,10 \
  --selection near_margin --states-per-group 2 --min-candidates 1 \
  --max-edit-actions 4 --risk-samples 16 --mpc-samples 8 \
  --reference-samples 64 --future-worlds 8 \
  --future-signals 2 --stages 2 --signal-strength 0.5 \
  --out results/v10_e3_calibration_seed42 "$@"
