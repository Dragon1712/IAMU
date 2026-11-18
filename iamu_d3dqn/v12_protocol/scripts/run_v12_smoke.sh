#!/usr/bin/env bash
# 8-episode, 4-actor, full-simulator smoke. Explicitly reduced *SMOKE* grid.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests_v12 -p 'test_*.py' -v
for G in A0 A1 A2 A3; do
  case "$G" in
    A0) REPRESENTATION=primitive; SAMPLING=uniform ;;
    A1) REPRESENTATION=primitive; SAMPLING=stratified ;;
    A2) REPRESENTATION=budget_tail; SAMPLING=uniform ;;
    A3) REPRESENTATION=decision_epoch; SAMPLING=uniform ;;
  esac
  python -u -m v12_protocol.train \
    --representation "$REPRESENTATION" --sampling "$SAMPLING" \
    --train-cases ../2024LiERLDDE/case_code/1.txt ../2024LiERLDDE/case_code/2.txt \
    --val-cases ../2024LiERLDDE/case_code/8.txt \
    --episodes 8 --workers 4 --validation-workers 2 --round-size 4 --eval-every 8 \
    --scenarios shock --budgets 2,5 --val-seeds 400001 \
    --batch-size 4 --warmup-raw-steps 32 --warmup-actionable 4 \
    --updates-per-raw-step 0.25 --seed 42 --device cpu \
    --out "results/v12_smoke_${G}" \
    2>&1 | tee "v12_smoke_${G}.log"
done
