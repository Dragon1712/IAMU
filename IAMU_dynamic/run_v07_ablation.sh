#!/usr/bin/env bash
# Four architecture-matched ablations; new result dirs, no overwrite of V0.6.
set -euo pipefail
EPISODES="${EPISODES:-300}"
EVAL_EVERY="${EVAL_EVERY:-50}"
OUT_PREFIX="${OUT_PREFIX:-results/v07_ablation}"
common=(
  --train-cases ../2024LiERLDDE/case_code/{3,4,5,6,7}.txt
  --val-cases ../2024LiERLDDE/case_code/{8,9}.txt
  --episodes "$EPISODES"
  --scenarios shock,repeated,small --budgets 2,5,10
  --uncertainties 4,8,12 --clock union --period 12
  --val-seeds 400001:400005 --seed 42
  --candidate-limit 4 --batch-size 64 --warmup 512
  --epsilon-decay-episodes "$EPISODES" --eval-every "$EVAL_EVERY"
)
python -u train_budgeted_ddqn.py "${common[@]}" \
  --budget-features legacy --value-features legacy \
  --candidate-screen none --keep-margin 0 --greedy-exploration-prob 0 \
  --out "${OUT_PREFIX}_A"
python -u train_budgeted_ddqn.py "${common[@]}" \
  --budget-features absolute --value-features legacy \
  --candidate-screen none --keep-margin 0 --greedy-exploration-prob 0 \
  --out "${OUT_PREFIX}_B"
python -u train_budgeted_ddqn.py "${common[@]}" \
  --budget-features absolute --value-features global \
  --candidate-screen none --keep-margin 0 --greedy-exploration-prob 0 \
  --out "${OUT_PREFIX}_C"
python -u train_budgeted_ddqn.py "${common[@]}" \
  --budget-features absolute --value-features global \
  --candidate-screen positive --min-gain 0.1 \
  --keep-margin 0.1 --greedy-exploration-prob 0.75 \
  --out "${OUT_PREFIX}_D"
