#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -u scripts/run_e2_parallel.py \
  --workflows multistage,risk_stability \
  --workers multistage=2,risk_stability=2 \
  --cases 8 --scenarios shock --eta-seeds 400001:400002 --budgets 2 \
  --risk-samples 16 --mpc-samples 8 --h 2 \
  --risk-sizes 16,64 --reference-samples 128 \
  --repeats 2 --states-per-group 1 --stride 2 \
  --out results/v10_e2_parallel_smoke "$@"
