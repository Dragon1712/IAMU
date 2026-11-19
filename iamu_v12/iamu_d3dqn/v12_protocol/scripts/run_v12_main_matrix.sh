#!/usr/bin/env bash
# This script STARTS 12 long jobs. Use only AFTER E0 acceptance.
# Run from fresh checkout with resources, do not auto-launch from setup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DEVICE="${DEVICE:-cuda:0}"
for SEED in 42 43 44; do
  for G in A0 A1 A2 A3; do
    case "$G" in
      A0) REPRESENTATION=primitive; SAMPLING=uniform ;;
      A1) REPRESENTATION=primitive; SAMPLING=stratified ;;
      A2) REPRESENTATION=budget_tail; SAMPLING=uniform ;;
      A3) REPRESENTATION=decision_epoch; SAMPLING=uniform ;;
    esac
    python -u -m v12_protocol.train \
      --representation "$REPRESENTATION" --sampling "$SAMPLING" \
      --train-cases ../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt \
      --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
      --episodes 400 --workers 4 --validation-workers 4 --round-size 16 \
      --eval-every 100 --scenarios shock,repeated,small --budgets 2,5,10 \
      --candidate-screen positive --candidate-limit 4 \
      --execution-margin 0 --bootstrap-margin 0 --gamma 1 \
      --batch-size 64 --replay-episodes 200 --warmup-raw-steps 512 \
      --warmup-actionable 64 --updates-per-raw-step .25 --target-update 250 \
      --seed "$SEED" --device "$DEVICE" \
      --out "results/v12_${G}_seed${SEED}_400" \
      2>&1 | tee "v12_${G}_seed${SEED}_400.log"
  done
done
