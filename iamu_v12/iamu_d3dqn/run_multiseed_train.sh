#!/usr/bin/env bash
# Run from ~/code/iamu/iamu_d3dqn. Reproduces Seed-42 hyperparameters.
set -euo pipefail
MODE="${1:-core}"             # core: ddqn96 ddqn130 set_ddqn96; all: five networks
SEEDS="${SEEDS:-43,44}"        # Only NEW train seeds; 42 is already trained
DEVICE="${DEVICE:-cuda:0}"    # E.g. cuda:2 or cpu
DRY_RUN="${DRY_RUN:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"

case "$MODE" in
  core) MODELS=(ddqn96 ddqn130 set_ddqn96) ;;
  all)  MODELS=(ddqn96 ddqn130 d3qn96 set_ddqn96 set_d3qn96) ;;
  *) echo "Usage: [SEEDS=43,44 DEVICE=cuda:0] bash run_multiseed_train.sh [core|all]" >&2; exit 2 ;;
esac

function directory() {
  local model="$1" seed="$2"
  case "$model" in
    ddqn96)   echo "results/v082_8workers_sync1_seed${seed}_300" ;;
    ddqn130)  echo "results/v09a_ddqn130_seed${seed}_300" ;;
    d3qn96)   echo "results/v09a_d3qn_seed${seed}_300" ;;
    set_ddqn96) echo "results/v09b_set_ddqn_seed${seed}_300" ;;
    set_d3qn96) echo "results/v09c_set_d3qn_seed${seed}_300" ;;
  esac
}

if [[ ! -f train_budgeted_ddqn_parallel.py || ! -d iamu_dynamic ]]; then
  echo "ERROR: Run from the iamu_d3dqn project root" >&2; exit 2
fi
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
for seed in "${SEED_ARRAY[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Bad seed: $seed" >&2; exit 2; }
  [[ "$seed" != 42 ]] || { echo "Seed 42 is already trained; use the original best.pt and train only 43,44" >&2; exit 2; }
  for model in "${MODELS[@]}"; do
    arch=ddqn; hidden=96
    case "$model" in
      ddqn130) hidden=130 ;;
      d3qn96) arch=d3qn ;;
      set_ddqn96) arch=set_ddqn ;;
      set_d3qn96) arch=set_d3qn ;;
    esac
    out="$(directory "$model" "$seed")"
    logfile="${out}.log"
    cmd=(python -u train_budgeted_ddqn_parallel.py
      --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt
      --val-cases ../2024LiERLDDE/case_code/{8,9}.txt
      --episodes 300 --workers 8 --validation-workers 8
      --validation-progress-every 6 --state-every 5 --log-every 1
      --prefetch-per-worker 1 --sync-every 1
      --scenarios shock,repeated,small --budgets 2,5,10
      --train-seed-start 200000 --val-seeds 400001:400003
      --uncertainties 4,8,12 --validation-uncertainty 8 --updates 2
      --clock union --period 12 --candidate-limit 4 --reward-scale 100
      --candidate-screen positive --budget-features absolute --value-features global
      --min-gain 0.1 --keep-margin 0.1 --greedy-exploration-prob 0.75
      --hidden-dim "$hidden" --q-network "$arch" --lr 0.0003 --gamma 1
      --replay-capacity 50000 --batch-size 64 --warmup 512
      --updates-per-step 0.25 --max-updates-per-episode 64 --target-update 250
      --epsilon-start 1 --epsilon-end 0.05 --epsilon-decay-episodes 300
      --eval-every 25 --seed "$seed" --device "$DEVICE" --out "$out")

    if [[ "$DRY_RUN" == 1 ]]; then
      printf 'DRY RUN: '; printf '%q ' "${cmd[@]}"; echo
      continue
    fi
    if [[ -e "$out/manifest.json" ]]; then
      if python - "$out" "$seed" "$arch" "$hidden" <<'PY'
import sys,json,pathlib
p=pathlib.Path(sys.argv[1]); seed=int(sys.argv[2]); arch=sys.argv[3]; hidden=int(sys.argv[4])
try:
    perf=json.loads((p/'performance.json').read_text())
    cfg=json.loads((p/'manifest.json').read_text())['training_args']
    assert perf['completed_episodes']==300 and perf['completed_validations']==12
    assert perf['total_validated_runs']==2592
    assert cfg['seed']==seed and cfg['q_network']==arch and cfg['hidden_dim']==hidden
    assert (p/'best.pt').is_file()
except (AssertionError,KeyError,FileNotFoundError,ValueError):sys.exit(1)
PY
      then
        echo "[SKIP verified completed] $out"
        continue
      elif [[ "$RESUME_INCOMPLETE" == 1 && -f "$out/training_state.pt" ]]; then
        cmd+=(--resume)
      else
        echo "ERROR: $out exists but is not verified complete. Inspect or set RESUME_INCOMPLETE=1; do NOT overwrite." >&2
        exit 1
      fi
    fi
    mkdir -p "$(dirname "$logfile")"
    echo "[START] model=$model seed=$seed device=$DEVICE out=$out"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "${cmd[@]}" 2>&1 | tee -a "$logfile"
    echo "[FINISHED] $out"
  done
done
