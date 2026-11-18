#!/usr/bin/env bash
# IAMU V1.2 two concurrent experiments, offline checkout, separate data/state.
# Usage: bash iamu_v12_two_experiments.sh dry|run|resume-dry|resume seed43|seed42
set -Eeuo pipefail

ACTION="${1:-dry}"
GROUP="${2:-seed43}"
REPO="${IAMU_REPO:-/home/moco/code/iamu/iamu_v12}"
MOD="$REPO/iamu_d3dqn"
SOURCE="v11_protocol/uniform_replay_v12.json"
OUT_DIR="results/v12_launch_matrices"

case "$ACTION" in dry|run|resume-dry|resume) ;; *) echo "Unknown action: $ACTION" >&2; exit 2 ;; esac
case "$GROUP" in seed43|seed42) ;; *) echo "Choose seed43 or seed42" >&2; exit 2 ;; esac
[[ -f "$MOD/$SOURCE" ]] || { echo "V1.2 matrix missing: $MOD/$SOURCE" >&2; exit 2; }
[[ -f "$MOD/v11_protocol/recovery_v12.py" ]] || { echo "V1.2 patch not installed" >&2; exit 2; }
for case_id in 1 2 3 4 5 6 7 8 9; do
  [[ -f "$REPO/2024LiERLDDE/case_code/${case_id}.txt" ]] || { echo "Case missing: $REPO/2024LiERLDDE/case_code/${case_id}.txt" >&2; exit 2; }
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd "$MOD"
mkdir -p "$OUT_DIR" results/logs

if [[ "$ACTION" == resume* ]]; then
  SUFFIX=resume
else
  SUFFIX=fresh
fi
MATRIX="$OUT_DIR/${GROUP}_${SUFFIX}.json"
python - "$SOURCE" "$MATRIX" "$GROUP" "$SUFFIX" <<'PY'
import json,sys
from pathlib import Path
source, target, group, stage=sys.argv[1:]
m=json.loads(Path(source).read_text(encoding='utf-8'))
want={
 'seed43': ('uniform_seed43_v12','rho050_seed43_v12'),
 'seed42': ('uniform_seed42_v12','rho025_seed42_v12'),
}[group]
byname={j['name']:j for j in m['jobs']}
missing=set(want)-set(byname)
if missing: raise SystemExit(f'Missing jobs in V1.2 matrix: {sorted(missing)}')
m['jobs']=[byname[x] for x in want]
for job in m['jobs']:
    out=Path(job['out'])
    if stage=='fresh':
        if out.is_dir() and any(out.iterdir()):
            raise SystemExit(f'Refusing to overwrite existing output: {out}')
    elif not (out/'training_state.pt').is_file():
        raise SystemExit(f'Cannot resume without V1.2 training_state.pt: {out}')
if stage=='resume':
    m['common']['resume']=True
Path(target).write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Selected two runs:',*(j['name'] for j in m['jobs']),sep='\n  ')
print('Matrix:', target)
PY

# Existing launcher explicitly requires 4 UNIQUE device IDs, but only schedules
# the 2 jobs present in the matrix, so only GPU 0 and GPU 1 get learners.
CMD=(python -u v11_protocol/launch_four_gpu.py \
   --matrix "$MATRIX" \
   --devices cuda:0,cuda:1,cuda:2,cuda:3 \
   --actors-per-gpu 4 --validators-per-gpu 2)
case "$ACTION" in dry|resume-dry) CMD+=(--dry-run);; esac
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
