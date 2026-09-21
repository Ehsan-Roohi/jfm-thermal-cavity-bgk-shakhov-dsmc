#!/bin/bash
set -euo pipefail

TARGET_STEPS="${1:-}"
[[ "$TARGET_STEPS" =~ ^[0-9]+$ ]] || {
  echo "Usage: $0 TARGET_STEPS (for example 2000000 or 3000000)" >&2
  exit 2
}
SOURCE_STEPS=1500000
if (( TARGET_STEPS <= SOURCE_STEPS || TARGET_STEPS > 3000000 )); then
  echo "TARGET_STEPS must be greater than 1500000 and at most 3000000" >&2
  exit 2
fi
if (( TARGET_STEPS % 100000 != 0 )); then
  echo "TARGET_STEPS must be divisible by 100000" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_TABLE="$ROOT/cases/fast7.csv"
TIME_BLOCKS="$(((TARGET_STEPS - 100000) / 100000))"
mkdir -p "$OUTPUT_ROOT/continuations/step${TARGET_STEPS}" "$ROOT/slurm"

for manifest in "$OUTPUT_ROOT"/vram48/runs/*_restart_step1500000/manifest.json; do
  test -s "$manifest"
done
MANIFEST_COUNT="$(find "$OUTPUT_ROOT/vram48/runs" -maxdepth 2 \
  -type f -path '*_restart_step1500000/manifest.json' | wc -l)"
[[ "$MANIFEST_COUNT" == 7 ]] || {
  echo "Expected 7 complete source restarts, found $MANIFEST_COUNT" >&2
  exit 3
}

COMMON_EXPORT="ALL,JFM_ROOT=$ROOT,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_CASE_TABLE=$CASE_TABLE,JFM_CONTINUE_SOURCE_STEPS=$SOURCE_STEPS,JFM_CONTINUE_TARGET_STEPS=$TARGET_STEPS,JFM_CONTINUE_TIME_BLOCKS=$TIME_BLOCKS"

RUN_JOB="$(sbatch --parsable \
  --job-name="jfm-cont${TARGET_STEPS}" \
  --partition=gpu --gpus=1 --constraint=vram48 \
  --cpus-per-task=4 --mem=24G --time=168:00:00 --array=0-6%7 \
  --output="$ROOT/slurm/jfm-cont${TARGET_STEPS}-%A_%a.out" \
  --error="$ROOT/slurm/jfm-cont${TARGET_STEPS}-%A_%a.err" \
  --export="$COMMON_EXPORT" scripts/run_continuation.slurm)"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB" --kill-on-invalid-dep=yes \
  --job-name="jfm-cont${TARGET_STEPS}-sum" \
  --partition=cpu --cpus-per-task=4 --mem=24G --time=08:00:00 \
  --output="$ROOT/slurm/jfm-cont${TARGET_STEPS}-summary-%j.out" \
  --error="$ROOT/slurm/jfm-cont${TARGET_STEPS}-summary-%j.err" \
  --export="$COMMON_EXPORT" scripts/summarize_continuation.slurm)"

echo "CONTINUATION_RUN_JOB=$RUN_JOB"
echo "CONTINUATION_SUMMARY_JOB=$SUMMARY_JOB"
echo "SOURCE_STEPS=$SOURCE_STEPS"
echo "TARGET_STEPS=$TARGET_STEPS"
echo "TIME_BLOCKS=$TIME_BLOCKS"
echo "Monitor: squeue -j $RUN_JOB,$SUMMARY_JOB"
