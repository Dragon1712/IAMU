#!/usr/bin/env bash
# Safe wrapper: no background execution; invoke from iamu_d3dqn or elsewhere.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
python -u e4_train_sufficiency/launch_e4.py \
  --conditions a300,b900,c900,d900 \
  --seeds 42,43,44 \
  --devices "${E4_DEVICES:-cuda:0}" \
  --actors "${E4_ACTORS:-8}" --validators "${E4_VALIDATORS:-8}" \
  --validation-every 100 --recovery-every 25 \
  --prefix v10_e4 "${@}" \
  2>&1 | tee logs/v10_e4_launcher.log
