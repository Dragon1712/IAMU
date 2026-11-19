#!/usr/bin/env bash
# IAMU V1.2 P0–P4 offline commands. Every research task is OPT-IN.
set -euo pipefail
SOURCE_ROOT="${IAMU_V12_SOURCE:-/home/moco/code/iamu/iamu_v12}"
NEXT_ROOT="${IAMU_V12_NEXT:-/home/moco/code/iamu/iamu_v12_next}"
CODE_ROOT="$NEXT_ROOT/iamu_d3dqn"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$CODE_ROOT/results/v12_pnext_data"
TRAIN="$OUT/train_raw_seed42.pt"
PROBE_RAW="$OUT/probe_raw_seed42.pt"
SMOKE_BANK="$OUT/probe_bank_smoke.pt"
BANK="$OUT/probe_bank.pt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cmd="${1:-help}"; shift || true
case "$cmd" in
  help|-h|--help)
    cat <<'EOF'
First prepare a separate working copy (no network):
  export SRC=/home/moco/code/iamu/iamu_v12
  export DST=/home/moco/code/iamu/iamu_v12_next
  mkdir -p "$DST/iamu_d3dqn"
  rsync -a --exclude='/results/' --exclude='*.pt' --exclude='*.log' "$SRC/iamu_d3dqn/" "$DST/iamu_d3dqn/"
  ln -s "$SRC/2024LiERLDDE" "$DST/2024LiERLDDE"

Then run separate opt-in commands:
  bash run_next_linux.sh check | apply | unit | collect-train | collect-probe
  bash run_next_linux.sh bank-smoke | bank-full | restore-smoke | restore-full
  bash run_next_linux.sh cf-smoke | cf-full | f1 A0 | f1 A1 | f1 A2 | f1 A3
  bash run_next_linux.sh checkpoints

NONE of these commands starts an online 400-episode experiment.
EOF
    ;;
  check|apply)
    python "$PKG_ROOT/apply_next.py" --repo "$CODE_ROOT" "--$cmd" ;;
  unit)
    cd "$CODE_ROOT"
    python -m unittest discover -s tests_v12 -p 'test_*.py' -v ;;
  collect-train)
    cd "$CODE_ROOT"; mkdir -p "$OUT"
    python -u -m v12_protocol.fixed_data collect \
      --cases ../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt \
      --episodes 64 --scenarios shock,repeated,small --budgets 2,5,10 \
      --uncertainties 4,8,12 --updates 2 --train-seed-start 200000 \
      --seed 42 --out "$TRAIN" ;;
  collect-probe)
    cd "$CODE_ROOT"; mkdir -p "$OUT"
    python -u -m v12_protocol.fixed_data collect \
      --cases ../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt \
      --episodes 32 --scenarios shock,repeated,small --budgets 2,5,10 \
      --uncertainties 4,8,12 --updates 2 --train-seed-start 300000 \
      --seed 42 --out "$PROBE_RAW" ;;
  bank-smoke)
    cd "$CODE_ROOT"
    python -u -m v12_protocol.probe_bank build --data "$PROBE_RAW" \
      --seed 1729 --per-stratum 1 --max-states 8 --out "$SMOKE_BANK" ;;
  bank-full)
    cd "$CODE_ROOT"
    python -u -m v12_protocol.probe_bank build --data "$PROBE_RAW" \
      --seed 1729 --per-stratum 8 --max-states 144 --out "$BANK" ;;
  restore-smoke)
    cd "$CODE_ROOT"
    python -m v12_protocol.probe_bank restore-audit --bank "$SMOKE_BANK" \
      --data "$PROBE_RAW" --limit 8 ;;
  restore-full)
    cd "$CODE_ROOT"
    python -m v12_protocol.probe_bank restore-audit --bank "$BANK" \
      --data "$PROBE_RAW" --limit 0 ;;
  cf-smoke)
    cd "$CODE_ROOT"
    python -u -m v12_protocol.counterfactual_probe --bank "$SMOKE_BANK" \
      --data "$PROBE_RAW" --continuation greedy --limit 8 \
      --out "$OUT/counterfactual_smoke" ;;
  cf-full)
    cd "$CODE_ROOT"
    python -u -m v12_protocol.counterfactual_probe --bank "$BANK" \
      --data "$PROBE_RAW" --continuation greedy --limit 0 \
      --out "$OUT/counterfactual_full" ;;
  f1)
    GROUP="${1:-}"
    if [[ ! "$GROUP" =~ ^A[0-3]$ ]]; then echo "Expected A0/A1/A2/A3" >&2; exit 2; fi
    cd "$CODE_ROOT"
    args=()
    if [[ "${F1_SKIP_VALIDATION:-0}" == 1 ]];then args+=(--skip-validation);fi
    python -u -m v12_protocol.fixed_data train \
      --data "$TRAIN" --probe-bank "$BANK" --group "$GROUP" \
      --gradient-updates 1000 --diagnostic-every 25 --probe-every 250 \
      --diagnostics --checkpoint-every 250 --batch-size 64 \
      --replay-episodes 200 --gamma 1 --seed 42 --device "${F1_DEVICE:-cpu}" \
      --val-cases ../2024LiERLDDE/case_code/{8,9}.txt \
      --validation-workers 2 --baseline-cache-dir "$OUT/baseline_cache" \
      --out "$CODE_ROOT/results/v12_pnext_F1_${GROUP}_seed42" "${args[@]}" ;;
  checkpoints)
    find "$SOURCE_ROOT/iamu_d3dqn/results" -path '*/validation_snapshots/*.pt' \
      -type f -print | sort ;;
  *)echo "Unknown subcommand: $cmd" >&2;exit 2;;
esac
