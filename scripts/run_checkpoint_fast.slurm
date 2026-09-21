#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_FILE="${JFM_CASE_TABLE:-$ROOT/cases/fast7.csv}"
cd "$ROOT"

ROW="$((SLURM_ARRAY_TASK_ID + 2))"
LINE="$(awk -F, -v row="$ROW" 'NR==row {print; exit}' "$CASE_FILE")"
[[ -n "$LINE" ]] || { echo "No case row for task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=, read -r CASE_ID MODEL KN RT SEED FIGURE SOURCE <<< "$LINE"
RT_TAG="${RT//./p}"
OUTPUT_DIR="$OUTPUT_ROOT/vram48/runs"
mkdir -p "$OUTPUT_DIR"

if [[ "$MODEL" == "HS" ]]; then
  STEM="ThermalCavity_HS_DSMC_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
else
  STEM="ThermalCavity_${MODEL}_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
fi

echo "case=$CASE_ID figure=$FIGURE model=$MODEL Kn=$KN RT=$RT seed=$SEED"
echo "particles=80000000 steps=1500000 checkpoint=1000000 sample_window=100000:1500000:2 time_blocks=14"
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

if [[ -s "$OUTPUT_DIR/${STEM}_metrics.json" && -s "$OUTPUT_DIR/${STEM}_raw.npz" ]]; then
  echo "[SKIP] Complete outputs already exist for $STEM"
  exit 0
fi

COMMON=(
  --kn "$KN" --rt "$RT" --particles 80000000
  --steps 1500000 --sample-start 100000 --sample-every 2
  --time-blocks 14 --checkpoint-steps 1000000
  --seed "$SEED" --progress-every 10000
  --require-free-gb 40.0 --output "$OUTPUT_DIR"
)

if [[ "$MODEL" == "HS" ]]; then
  mamba run --no-capture-output -n dsmc-gpu python -u \
    solver/JFM_hs_dsmc_quarter.py "${COMMON[@]}"
else
  mamba run --no-capture-output -n dsmc-gpu python -u \
    solver/JFM_bgk_shakhov_quarter.py --mode "$MODEL" "${COMMON[@]}"
fi

test -s "$OUTPUT_DIR/${STEM}_metrics.json"
test -s "$OUTPUT_DIR/${STEM}_raw.npz"
echo "[OK] $CASE_ID complete"
