#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_v10_e3*.py' -v
python -u experiments/eval_e3_fusion_v10.py \
  --smoke --verify-baseline --risk-samples 8 --mpc-samples 4 \
  --lookahead-stages 2 --audit \
  --out results/v10_e3_fusion_smoke "$@"
python -u experiments/eval_e3_value_calibration_v10.py \
  --smoke --selection near_margin --states-per-group 1 \
  --risk-samples 8 --mpc-samples 4 --reference-samples 16 \
  --future-worlds 2 --future-signals 2 \
  --out results/v10_e3_calibration_smoke
