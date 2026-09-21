#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_TABLE="$ROOT/cases/fast7.csv"
mkdir -p "$OUTPUT_ROOT" "$ROOT/slurm"
bash scripts/preflight_local.sh

COMMON_EXPORT="ALL,JFM_ROOT=$ROOT,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_CASE_TABLE=$CASE_TABLE"

RUN_JOB="$(sbatch --parsable \
  --job-name=jfm-fast15 \
  --partition=gpu --gpus=1 --constraint=vram48 \
  --cpus-per-task=4 --mem=24G --time=168:00:00 --array=0-6%7 \
  --output="$ROOT/slurm/jfm-fast15-%A_%a.out" \
  --error="$ROOT/slurm/jfm-fast15-%A_%a.err" \
  --export="$COMMON_EXPORT" scripts/run_checkpoint_fast.slurm)"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB" --kill-on-invalid-dep=yes \
  --job-name=jfm-fast15-sum \
  --partition=cpu --cpus-per-task=4 --mem=24G --time=08:00:00 \
  --output="$ROOT/slurm/jfm-fast15-summary-%j.out" \
  --error="$ROOT/slurm/jfm-fast15-summary-%j.err" \
  --export="$COMMON_EXPORT" scripts/summarize_checkpoint.slurm)"

printf 'RUN_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\n' \
  "$RUN_JOB" "$SUMMARY_JOB" "$OUTPUT_ROOT" \
  > "$ROOT/LAST_SUBMISSION.env"

echo "RUN_JOB=$RUN_JOB"
echo "SUMMARY_JOB=$SUMMARY_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j $RUN_JOB,$SUMMARY_JOB"
echo "This public workflow never cancels existing jobs."
