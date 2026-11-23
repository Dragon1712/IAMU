#!/usr/bin/env bash
# V13-E4: stage-specific, safe DRY RUN by default; no online 400-episode job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN="${RUN:-0}"; STAGE="${STAGE:-boundary}"; SEED="${SEED:-43}"
START_INDEX="${START_INDEX:-0}"; STOP_INDEX="${STOP_INDEX:-0}"
BASE="${BASE:-results/v13_p0_p3_seed${SEED}}"
OUT="${OUT:-results/v13_boundary_seed${SEED}}"
PROBES="${PROBES:-$BASE/probe/probe_bank.pt}"
A0BEST="${A0BEST:-results/v12_A0_seed${SEED}_400/best.pt}"
A2BEST="${A2BEST:-results/v12_A2_seed${SEED}_400/best.pt}"
A3BEST="${A3BEST:-results/v12_A3_seed${SEED}_400/best.pt}"
A0_EP100="${A0_EP100:-MISSING_A0_EP100.pt}"
A2_EP100="${A2_EP100:-MISSING_A2_EP100.pt}"
A3_EP100="${A3_EP100:-MISSING_A3_EP100.pt}"
A0_EP400="${A0_EP400:-results/v12_A0_seed${SEED}_400/latest.pt}"
A2_EP400="${A2_EP400:-results/v12_A2_seed${SEED}_400/latest.pt}"
A3_EP400="${A3_EP400:-results/v12_A3_seed${SEED}_400/latest.pt}"
CASES=(../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt)
case "$STAGE" in boundary|boundary_summary|fixed|label_bank|labels|rollout) ;;
  *) echo '[ERROR] Select exactly one STAGE' >&2;exit 2;;esac
if [[ "$RUN" == 1 ]]; then
  EXPECTED_HEAD='c49e8d09517da632433afe270d9a2d01649f5cb9'
  ACTUAL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
  if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" && "${ALLOW_BASELINE_DRIFT:-0}" != 1 ]]; then
    echo "[ERROR] Git HEAD $ACTUAL_HEAD differs from $EXPECTED_HEAD" >&2
    echo 'Set ALLOW_BASELINE_DRIFT=1 only after comparing code differences.' >&2
    exit 3
  fi
fi
run() {
 printf '[V13-E4]';printf ' %q' "$@";printf '\n'
 if [[ "$RUN" == 1 ]];then "$@";fi
}
check() {
 if [[ "$RUN" == 1 ]];then
  for x in "$@";do [[ -f "$x" ]] || { echo "[MISSING] $x" >&2;exit 4; };done
 fi
}
if [[ "$STAGE" == boundary ]];then
 check "$PROBES" "$A0_EP100" "$A2_EP100" "$A3_EP100" "$A0_EP400" "$A2_EP400" "$A3_EP400" "$A0BEST" "$A2BEST" "$A3BEST"
 run python -u -m v13_boundary.audit_boundary --probes "$PROBES" \
  --checkpoint "A0_EP100=$A0_EP100" "A2_EP100=$A2_EP100" "A3_EP100=$A3_EP100" \
   "A0_EP400=$A0_EP400" "A2_EP400=$A2_EP400" "A3_EP400=$A3_EP400" \
   "A0_BEST=$A0BEST" "A2_BEST=$A2BEST" "A3_BEST=$A3BEST" \
  --continuations self,greedy --tapes 8 --seed 910000 \
  --start-index "$START_INDEX" --stop-index "$STOP_INDEX" --out "$OUT/boundary"
fi
if [[ "$STAGE" == boundary_summary ]];then
 check "$OUT/boundary/paired_gap.csv"
 run python -m v13_boundary.summarize_boundary --pairs "$OUT/boundary/paired_gap.csv" --out "$OUT/boundary_summary"
fi
if [[ "$STAGE" == fixed ]];then
 check "$BASE/compression/primitive.pt" "$BASE/compression/budget_tail.pt" "$PROBES" "$BASE/counterfactual_greedy/counterfactual_pairs.csv"
 run env CUDA_VISIBLE_DEVICES="${GPU:-0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m v13_boundary.fixed_replay --primitive "$BASE/compression/primitive.pt" \
  --budget-tail "$BASE/compression/budget_tail.pt" --probes "$PROBES" \
  --reference-csv "$BASE/counterfactual_greedy/counterfactual_pairs.csv" \
  --gradient-updates 4000 --diagnostic-every 250 --seed "$SEED" --device cuda:0 --out "$OUT/fixed"
fi
if [[ "$STAGE" == label_bank ]];then
 check "$A0BEST"
 run python -u -m v13_diagnostics.build_probe_bank --cases "${CASES[@]}" \
  --a0-checkpoint "$A0BEST" --episodes 400 --start-seed 950000 \
  --seed "$SEED" --out "$OUT/label_bank_raw"
 check "$OUT/label_bank_raw/probe_bank.pt"
 run python -m v13_boundary.rekey_training_bank \
  --input "$OUT/label_bank_raw/probe_bank.pt" --output "$OUT/label_bank/probe_bank.pt"
fi
if [[ "$STAGE" == labels ]];then
 check "$OUT/label_bank/probe_bank.pt" "$PROBES" "$A0BEST"
 run python -u -m v13_boundary.collect_rollout_labels \
  --training-probes "$OUT/label_bank/probe_bank.pt" --diagnostic-probes "$PROBES" \
  --frozen-a0 "$A0BEST" --tapes 8 --seed 940000 --out "$OUT/labels"
fi
if [[ "$STAGE" == rollout ]];then
 check "$OUT/labels/rollout_labels.csv" "$OUT/label_bank/probe_bank.pt" \
  "$PROBES" "$BASE/compression/budget_tail.pt" "$BASE/counterfactual_frozen_a0/counterfactual_pairs.csv"
 run env CUDA_VISIBLE_DEVICES="${GPU:-0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m v13_boundary.train_rollout --labels "$OUT/labels/rollout_labels.csv" \
  --train-probes "$OUT/label_bank/probe_bank.pt" --diagnostic-probes "$PROBES" \
  --diagnostic-reference-csv "$BASE/counterfactual_frozen_a0/counterfactual_pairs.csv" \
  --a2-data "$BASE/compression/budget_tail.pt" --label-count 30 --max-sim-cost-gap .10 \
  --gradient-updates 4000 --gap-weight .1 --seed "$SEED" --device cuda:0 --out "$OUT/rollout"
fi
if [[ "$RUN" != 1 ]];then
 echo '[PREVIEW ONLY] Add RUN=1 STAGE=... after acceptance tests.'
 echo 'For boundary: export exact A0_EP100, A2_EP100, A3_EP100 validation snapshot paths.'
fi
