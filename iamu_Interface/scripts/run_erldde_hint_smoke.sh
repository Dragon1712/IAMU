#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${ERLDDE_REPO:?Set ERLDDE_REPO to original model.py directory}"
: "${ERLDDE_CHECKPOINT:?Set ERLDDE_CHECKPOINT to a compatible frozen state_dict}"
ERLDDE_PROBE="${ERLDDE_PROBE:-$ROOT/results/erldde_probe/libBACAP_env_probe.so}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u experiments/eval_risk_budget_v10.py \
  --candidate-provider erldde_hint \
  --erldde-repo "$ERLDDE_REPO" \
  --erldde-checkpoint "$ERLDDE_CHECKPOINT" \
  --erldde-probe "$ERLDDE_PROBE" \
  --smoke --risk-samples 8 \
  --policies keep,periodic,greedy,set_ddqn \
  --device cpu --out results/v10_erldde_hint_smoke "$@" \
  2>&1 | tee v10_erldde_hint_smoke.log
