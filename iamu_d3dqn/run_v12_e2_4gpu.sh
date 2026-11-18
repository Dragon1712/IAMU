#!/usr/bin/env bash
# IAMU V1.2 E2: 12 scientific runs, four GPUs, three sequential seed waves.
# Safe by default: PREVIEW ONLY. Run with RUN=1 bash run_v12_e2_4gpu.sh
# Resume incomplete runs: RUN=1 RESUME=1 bash run_v12_e2_4gpu.sh
set -euo pipefail

cd "${IAMU_D3DQN_ROOT:-$HOME/code/iamu/iamu_d3dqn}"
if [[ ! -f v12_protocol/train.py ]]; then
  echo 'ERROR: Run this in iamu_d3dqn with the V1.2 patch installed.' >&2
  exit 2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
RUN="${RUN:-0}"
RESUME="${RESUME:-0}"
REQUIRE_SMOKE="${REQUIRE_SMOKE:-1}"
SEEDS="${SEEDS:-42 43 44}"

# Expand exactly seven training cases and two validation cases.
train_cases=()
for n in 1 2 3 4 5 6 7; do train_cases+=("../2024LiERLDDE/case_code/${n}.txt"); done
val_cases=("../2024LiERLDDE/case_code/8.txt" "../2024LiERLDDE/case_code/9.txt")

if [[ "$RUN" == 1 ]]; then
  for path in "${train_cases[@]}" "${val_cases[@]}"; do
    if [[ ! -f "$path" ]]; then echo "ERROR: case not found: $path" >&2; exit 3; fi
  done
  if [[ "$REQUIRE_SMOKE" == 1 ]]; then
    for g in A0 A1 A2 A3; do
      if [[ ! -f "results/v12_smoke_${g}/complete.json" ]]; then
        echo "ERROR: Smoke incomplete for $g; run 'bash v12_protocol/scripts/run_v12_smoke.sh' first." >&2
        echo "       If already accepted under a different directory: REQUIRE_SMOKE=0 RUN=1 ..." >&2
        exit 4
      fi
    done
  fi
  mkdir -p v12_logs/e2_4gpu
  nvidia-smi -L
fi

run_one() {
  local group="$1" seed="$2" physical_gpu="$3"
  local representation sampling out log
  case "$group" in
    A0) representation=primitive;      sampling=uniform ;;
    A1) representation=primitive;      sampling=stratified ;;
    A2) representation=budget_tail;    sampling=uniform ;;
    A3) representation=decision_epoch; sampling=uniform ;;
    *) echo "ERROR: unknown group $group" >&2; return 2 ;;
  esac
  out="results/v12_${group}_seed${seed}_400"
  log="v12_logs/e2_4gpu/v12_${group}_seed${seed}_gpu${physical_gpu}.log"

  local -a cmd=(
    python -u -m v12_protocol.train
    --representation "$representation" --sampling "$sampling"
    --train-cases "${train_cases[@]}"
    --val-cases "${val_cases[@]}"
    --episodes 400 --workers 4 --validation-workers 2
    --round-size 16 --eval-every 100 --state-every-rounds 1
    --scenarios shock,repeated,small --budgets 2,5,10
    --train-seed-start 200000 --val-seeds 400001:400003
    --uncertainties 4,8,12 --validation-uncertainty 8 --updates 2
    --clock union --period 12 --candidate-screen positive --candidate-limit 4
    --reward-scale 100 --budget-features absolute --value-features global
    --min-gain 0.1 --greedy-exploration-prob 0
    --q-network set_ddqn --hidden-dim 96 --lr 3e-4 --gamma 1
    --execution-margin 0 --bootstrap-margin 0
    --batch-size 64 --replay-episodes 200 --max-replay-numpy-mib 1024
    --warmup-raw-steps 512 --warmup-actionable 64
    --updates-per-raw-step 0.25 --max-updates-per-episode 0
    --target-update 250 --replay-rho 0.25
    --epsilon-start 1 --epsilon-end 0.05 --epsilon-decay-episodes 400
    --seed "$seed" --device cuda:0 --out "$out"
  )

  if [[ "$RUN" != 1 ]]; then
    printf '[PREVIEW] Physical GPU %s  %s Seed %s\n' "$physical_gpu" "$group" "$seed"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$physical_gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n\n'
    return 0
  fi

  if [[ -f "$out/complete.json" ]]; then
    echo "[SKIP] Already complete: $out"
    return 0
  fi
  if [[ -d "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ "$RESUME" == 1 && -f "$out/training_state.pt" ]]; then
      cmd+=(--resume)
      echo "[RESUME] $group Seed $seed on GPU $physical_gpu"
    else
      echo "[ERROR] Nonempty incomplete output: $out. Use RESUME=1 only if training_state.pt exists." >&2
      return 5
    fi
  fi

  echo "[START] $group Seed=$seed Physical-GPU=$physical_gpu Visible-Device=cuda:0 Out=$out"
  if CUDA_VISIBLE_DEVICES="$physical_gpu" "${cmd[@]}" 2>&1 | tee "$log"; then
    if [[ ! -f "$out/complete.json" ]]; then
      echo "[ERROR] Process exited 0 but complete.json is absent: $out" >&2
      return 6
    fi
    echo "[COMPLETE] $group Seed $seed"
  else
    local status=$?
    echo "[FAILED] $group Seed $seed exit=$status log=$log" >&2
    return "$status"
  fi
}

for seed in $SEEDS; do
  case "$seed" in 42|43|44) ;; *) echo "ERROR: SEEDS must be a subset of 42 43 44" >&2; exit 8 ;; esac
  echo "========== SEED $seed: A0/GPU0 A1/GPU1 A2/GPU2 A3/GPU3 =========="
  if [[ "$RUN" != 1 ]]; then
    run_one A0 "$seed" 0
    run_one A1 "$seed" 1
    run_one A2 "$seed" 2
    run_one A3 "$seed" 3
    continue
  fi
  pids=()
  run_one A0 "$seed" 0 & pids+=("$!")
  run_one A1 "$seed" 1 & pids+=("$!")
  run_one A2 "$seed" 2 & pids+=("$!")
  run_one A3 "$seed" 3 & pids+=("$!")
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures=$((failures+1)); fi
  done
  if ((failures)); then
    echo "[STOP] $failures job(s) failed in Seed=$seed wave. Later seeds NOT started." >&2
    exit 7
  fi
  echo "========== COMPLETED SEED $seed =========="
done

if [[ "$RUN" == 1 ]]; then
  echo '[DONE] All 12 V1.2 E2 runs completed; validate 12 complete.json files.'
else
  echo '[PREVIEW ONLY] Nothing launched. To start manually: RUN=1 bash run_v12_e2_4gpu.sh'
fi
