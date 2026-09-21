#!/bin/bash
set -euo pipefail
ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
INPUT_ROOT="$OUTPUT_ROOT/vram48/runs"
cd "$ROOT"
mamba run --no-capture-output -n dsmc-gpu python -u tools/summarize_single.py \
  --existing-input "$INPUT_ROOT" \
  --new-input "$INPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary" \
  --case-table cases/fast7.csv
