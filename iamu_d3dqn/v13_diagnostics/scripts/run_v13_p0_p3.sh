#!/usr/bin/env bash
# IAMU v13 diagnostics, git baseline 53b8c2e. NO online training.
# DRY by default; RUN=1 STAGE=collect|audit|probes|counterfactual|paired|targets|summary|all
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN="${RUN:-0}"
STAGE="${STAGE:-all}"
SEED="${SEED:-43}"
OUT="${OUT:-results/v13_p0_p3_seed${SEED}}"
A0="${A0:-results/v12_A0_seed${SEED}_400/best.pt}"
A2="${A2:-results/v12_A2_seed${SEED}_400/best.pt}"
A3="${A3:-results/v12_A3_seed${SEED}_400/best.pt}"
CASES=(../2024LiERLDDE/case_code/{1,2,3,4,5,6,7}.txt)
RAW="$OUT/collect/raw_episodes.pt"
PROBES="$OUT/probe/probe_bank.pt"

run() {
  printf '[V13]'; printf ' %q' "$@"; printf '\n'
  if [[ "$RUN" == 1 ]]; then "$@"; fi
}
wants() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

if wants collect; then
  run python -u -m v13_diagnostics.collect_raw \
    --cases "${CASES[@]}" --a0-checkpoint "$A0" \
    --episodes-per-behavior 100 --a0-epsilon 0.2 --workers 4 \
    --seed "$SEED" --external-seed-start 600000 \
    --out "$OUT/collect"
fi
if wants audit; then
  run python -m v13_diagnostics.audit_compression \
    --data "$RAW" --out "$OUT/compression"
fi
if wants probes; then
  run python -u -m v13_diagnostics.build_probe_bank \
    --cases "${CASES[@]}" --a0-checkpoint "$A0" \
    --episodes 240 --seed "$SEED" --start-seed 800000 \
    --out "$OUT/probe"
fi
if wants counterfactual; then
  run python -u -m v13_diagnostics.counterfactual \
    --probes "$PROBES" --checkpoint "A0=$A0" "A2=$A2" "A3=$A3" \
    --continuation greedy --tapes 8 --seed 910000 \
    --out "$OUT/counterfactual_greedy"
fi
if wants a0cf; then
  run python -u -m v13_diagnostics.counterfactual \
    --probes "$PROBES" --checkpoint "A0=$A0" "A2=$A2" "A3=$A3" \
    --continuation a0 --a0-continuation "$A0" \
    --tapes 8 --seed 910000 --out "$OUT/counterfactual_frozen_a0"
fi
if wants targets; then
  run python -u -m v13_diagnostics.audit_targets \
    --data "$RAW" --representation primitive \
    --checkpoint "$A0" --out "$OUT/targets_a0"
  run python -u -m v13_diagnostics.audit_targets \
    --data "$RAW" --representation decision_epoch \
    --checkpoint "$A3" --out "$OUT/targets_a3"
fi
if wants paired; then
  run python -u -m v13_diagnostics.audit_paired_targets \
    --probes "$PROBES" --checkpoint "$A3" \
    --tapes 8 --seed 910000 --out "$OUT/paired_targets_a3"
fi
if wants summary; then
  run python -u -m v13_diagnostics.summarize \
    --pairs "$OUT/counterfactual_greedy/counterfactual_pairs.csv" \
    --targets "$OUT/targets_a0/target_audit.csv" "$OUT/targets_a3/target_audit.csv" \
    --paired-targets "$OUT/paired_targets_a3/paired_target_gap.csv" \
    --out "$OUT/summary"
fi
if [[ "$RUN" != 1 ]]; then
  printf '\n[DRY RUN ONLY] Set RUN=1 STAGE=<name> to execute one diagnostic stage.\n'
fi
