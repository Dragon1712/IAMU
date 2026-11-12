#!/usr/bin/env bash
# ERL-DDE-KDE-Fixed: one-factor-at-a-time threshold ablation launcher.
# Run from the project directory containing main_kde_fixed.py.
# Existing training runs are never stopped or modified.
set -Eeuo pipefail

cd "$(dirname "$(realpath "$0")")"
for required in main_kde_fixed.py train_fixed.py kde_fixed.py model.py env.py; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required in $PWD" >&2; exit 1; }
done
[[ -d case_code ]] || { echo "ERROR: missing case_code directory" >&2; exit 1; }

MAX_PARALLEL="${MAX_PARALLEL:-3}"
CASE_COUNT="${CASE_COUNT:-200}"
RUN_SEED="${RUN_SEED:-42}"
ABLATION_SET="${ABLATION_SET:-both}"   # r1, r2, or both
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_PARALLEL must be a positive integer' >&2; exit 1; }
[[ "$CASE_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'CASE_COUNT must be a positive integer' >&2; exit 1; }
[[ "$ABLATION_SET" == r1 || "$ABLATION_SET" == r2 || "$ABLATION_SET" == both ]] || {
  echo 'ABLATION_SET must be r1, r2, or both' >&2; exit 1;
}

LOG_DIR="$PWD/kde_ablation_logs/$RUN_TAG"
if [[ -e "$LOG_DIR" ]]; then
  echo "ERROR: log directory already exists: $LOG_DIR" >&2
  echo 'Choose another RUN_TAG to avoid overwriting results.' >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# NOTE: CPU rollouts + CPU KDE are deliberately used across all runs.
# GPU KDE was not faster overall in the user's previous tests.

configs=()
# Candidate rejection (r1) ablation: hold r2=0.9990 fixed.
if [[ "$ABLATION_SET" == r1 || "$ABLATION_SET" == both ]]; then
  configs+=(
    'r1_0980 0.9800 0.9990'
    'r1_0985 0.9850 0.9990'
    'r1_0990 0.9900 0.9990'
    'r1_0995 0.9950 0.9990'
  )
fi
# Population exclusion (r2) ablation: hold r1=0.9975 fixed.
if [[ "$ABLATION_SET" == r2 || "$ABLATION_SET" == both ]]; then
  configs+=(
    'r2_0985 0.9975 0.9850'
    'r2_0990 0.9975 0.9900'
    'r2_0995 0.9975 0.9950'
  )
fi

printf 'name\tr1\tr2\tseed\tcases\tlog\toutput_prefix\n' > "$LOG_DIR/experiments.tsv"

running=0
for entry in "${configs[@]}"; do
  read -r name r1 r2 <<< "$entry"
  prefix="KDEABL_${RUN_TAG}_${name}"
  log="$LOG_DIR/${name}.log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$r1" "$r2" "$RUN_SEED" "$CASE_COUNT" "$log" "$prefix" \
    >> "$LOG_DIR/experiments.tsv"

  (
    if python -u main_kde_fixed.py \
      --device cpu --kde-device cpu \
      --cases "$CASE_COUNT" --loops 1 \
      --populations 5 --lambda-size 80 --max-episode-length 1000 \
      --kde-r1 "$r1" --kde-r2 "$r2" \
      --kde-max-attempts 24 --kde-restart-attempts 64 \
      --full-checkpoint-every 10 --seed "$RUN_SEED" \
      --output-prefix "$prefix" > "$log" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    printf '%s\n' "$rc" > "$LOG_DIR/${name}.exit"
    if [[ "$rc" -eq 0 ]]; then
      printf '[DONE] %s exit=0\n' "$name"
    else
      printf '[FAIL] %s exit=%s; inspect %s\n' "$name" "$rc" "$log" >&2
    fi
  ) &
  pid=$!
  printf '%s\n' "$pid" > "$LOG_DIR/${name}.pid"
  printf '[START] %s pid=%s r1=%s r2=%s log=%s\n' "$name" "$pid" "$r1" "$r2" "$log"
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n || true
    running=$((running - 1))
  fi
done
wait || true
printf '\n[ALL FINISHED] Logs and exit codes: %s\n' "$LOG_DIR"
