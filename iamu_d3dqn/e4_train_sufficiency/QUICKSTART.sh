#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s e4_diagnostics -p 'test_*.py' -v
python -u e4_diagnostics/diagnose_keep_qgap.py \
  --model results/v10_e4_b900_seed43@100 \
  --model results/v10_e4_b900_seed43@900 \
  --cases ../2024LiERLDDE/case_code/8.txt \
  --scenarios shock --eta-seeds 400001 --budgets 2 \
  --anchors early,keep --continuations anchor,greedy \
  --states-per-group 1 --out results/v10_e4_diag_smoke
python e4_diagnostics/summarize_diagnostic.py \
  --input results/v10_e4_diag_smoke
