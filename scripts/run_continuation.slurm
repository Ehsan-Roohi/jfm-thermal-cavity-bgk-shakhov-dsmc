#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_FILE="${JFM_CASE_TABLE:-$ROOT/cases/fast7.csv}"
TARGET_STEPS="${JFM_CONTINUE_TARGET_STEPS:?}"
TIME_BLOCKS="${JFM_CONTINUE_TIME_BLOCKS:?}"
SOURCE_STEPS="${JFM_CONTINUE_SOURCE_STEPS:-1500000}"
cd "$ROOT"

ROW="$((SLURM_ARRAY_TASK_ID + 2))"
LINE="$(awk -F, -v row="$ROW" 'NR==row {print; exit}' "$CASE_FILE")"
[[ -n "$LINE" ]] || { echo "No case row for task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=, read -r CASE_ID MODEL KN RT SEED FIGURE SOURCE <<< "$LINE"
RT_TAG="${RT//./p}"
BASE_DIR="$OUTPUT_ROOT/vram48/runs"
OUTPUT_DIR="$OUTPUT_ROOT/continuations/step${TARGET_STEPS}/runs"
mkdir -p "$OUTPUT_DIR"

if [[ "$MODEL" == "HS" ]]; then
  STEM="ThermalCavity_HS_DSMC_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
else
  STEM="ThermalCavity_${MODEL}_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
fi
RESTART_DIR="$BASE_DIR/${STEM}_restart_step${SOURCE_STEPS}"
test -s "$RESTART_DIR/manifest.json"

echo "case=$CASE_ID continuation=${SOURCE_STEPS}:${TARGET_STEPS} model=$MODEL Kn=$KN RT=$RT seed=$SEED"
echo "particles=80000000 sample_start=100000 sample_every=2 time_blocks=$TIME_BLOCKS"
echo "restart=$RESTART_DIR"
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

if [[ -s "$OUTPUT_DIR/${STEM}_metrics.json" && -s "$OUTPUT_DIR/${STEM}_raw.npz" ]]; then
  echo "[SKIP] Complete continuation already exists for $STEM"
  exit 0
fi

CHECKPOINTS=()
for ((step=SOURCE_STEPS + 500000; step<TARGET_STEPS; step+=500000)); do
  CHECKPOINTS+=("$step")
done

COMMON=(
  --kn "$KN" --rt "$RT" --particles 80000000
  --steps "$TARGET_STEPS" --sample-start 100000 --sample-every 2
  --time-blocks "$TIME_BLOCKS" --checkpoint-steps "${CHECKPOINTS[@]}"
  --restart "$RESTART_DIR" --seed "$SEED" --progress-every 10000
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
test -s "$OUTPUT_DIR/${STEM}_restart_step${TARGET_STEPS}/manifest.json"
echo "[OK] $CASE_ID continuation complete"
