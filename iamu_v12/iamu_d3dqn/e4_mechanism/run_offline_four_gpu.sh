#!/usr/bin/env bash
# E4-E: after ONLINE reaches Ep400, execute FOUR frozen Replay forks on FOUR GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SEED="${E5_SEED:-43}"
ONLINE="${E5_ONLINE:-results/v10_e5_online_seed${SEED}}"
BANK="${E5_BANK:-results/v10_e5_probe/public_bank.npz}"
BASE="${E5_OFFLINE_PREFIX:-results/v10_e5_offline}"
FORK="$ONLINE/e5_mechanism/full_state_ep0200.pt"
TRACE="$ONLINE/train_curve.csv"
for f in "$FORK" "$TRACE" "$BANK"; do
  if [[ ! -f "$f" ]]; then echo "Missing $f" >&2; exit 2; fi
done
mkdir -p logs
pids=()
for pair in 'actionable_uniform cuda:0' 'uniform cuda:1' 'fixed_target cuda:2' 'balanced cuda:3'; do
  read -r MODE DEVICE <<< "$pair"
  OUT="${BASE}_${MODE}_seed${SEED}"
  echo "[E5] mode=$MODE device=$DEVICE out=$OUT"
  python -u e4_mechanism/offline_branch.py \
    --fork "$FORK" --online-trace "$TRACE" --probe-bank "$BANK" \
    --mode "$MODE" --device "$DEVICE" --until 400 --every 25 \
    --trust-local-training-state --out "$OUT" \
    > "logs/v10_e5_offline_${MODE}_seed${SEED}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; echo "[FAILED] pid=$pid; see logs" >&2; fi
done
if [[ "$failed" != 0 ]]; then exit 1; fi
echo '[DONE] Four offline branches completed.'
