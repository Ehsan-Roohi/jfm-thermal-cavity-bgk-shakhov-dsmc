#!/bin/bash
set -euo pipefail
ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
TARGET_STEPS="${JFM_CONTINUE_TARGET_STEPS:?}"
INPUT_ROOT="$OUTPUT_ROOT/continuations/step${TARGET_STEPS}/runs"
cd "$ROOT"
mamba run --no-capture-output -n dsmc-gpu python -u tools/summarize_single.py \
  --existing-input "$INPUT_ROOT" \
  --new-input "$INPUT_ROOT" \
  --output "$OUTPUT_ROOT/continuations/step${TARGET_STEPS}/summary" \
  --case-table cases/fast7.csv
