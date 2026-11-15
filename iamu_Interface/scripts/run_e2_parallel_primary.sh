#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -u scripts/run_e2_parallel.py \
  --workflows multistage,risk_stability \
  --workers multistage=8,risk_stability=12 \
  --cases 8,9 --scenarios shock,repeated,small \
  --eta-seeds 400001:400003 --budgets 2,5,10 \
  --risk-samples 16 --mpc-samples 8 --h 2 \
  --risk-sizes 16,64,128 --reference-samples 256 \
  --repeats 3 --states-per-group 2 --stride 10 \
  --out results/v10_e2_parallel_primary "$@"
