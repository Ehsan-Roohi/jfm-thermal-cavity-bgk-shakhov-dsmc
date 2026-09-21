#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = {
    "solver/JFM_hs_dsmc_quarter.py": "2c9e2f5119802b123f0335664a564085cfe77682ae4f4af4138dc31f5876b166",
    "solver/JFM_bgk_shakhov_quarter.py": "f2f97526942c53eca0af6dd9b94aca541f06951db4755a92cc8fb11e5e0e65a2",
}
for relative, expected in EXPECTED_SHA.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

with (ROOT / "cases/fast7.csv").open(newline="") as handle:
    final = list(csv.DictReader(handle))
assert len(final) == 7
assert {int(row["seed"]) for row in final} == {104729}
assert {row["source"] for row in final} == {"new_checkpoint"}
assert {(row["figure"], row["model"], row["kn"], row["rt"]) for row in final} == {
    ("Figure5", "HS", "20", "0.2"),
    ("Figure5", "BGK", "20", "0.2"),
    ("Figure5", "SHAKHOV", "20", "0.2"),
    ("Figure6", "SHAKHOV", "5", "0.5"),
    ("Figure6", "SHAKHOV", "10", "0.5"),
    ("Figure6", "SHAKHOV", "20", "0.5"),
    ("Figure2d", "HS", "30", "0.5"),
}

run = (ROOT / "scripts/run_checkpoint_fast.slurm").read_text()
submit = (ROOT / "scripts/submit_checkpoint_fast.sh").read_text()
assert "--steps 1500000 --sample-start 100000 --sample-every 2" in run
assert "--time-blocks 14 --checkpoint-steps 1000000" in run
assert "--no-capture-output" in run and "python -u" in run
assert "--constraint=vram48" in submit and "--array=0-6%7" in submit
assert "--qos=long" not in submit
assert "scancel" not in submit
assert "never cancels existing jobs" in submit

for solver_path in (
    ROOT / "solver/JFM_hs_dsmc_quarter.py",
    ROOT / "solver/JFM_bgk_shakhov_quarter.py",
):
    source = solver_path.read_text()
    assert "--checkpoint-steps" in source
    assert "def write_checkpoint" in source
    assert "def save_restart" in source
    assert "def load_restart" in source
    assert '"particle_state_is_complete": True' in source
    assert '"rng_state_is_complete": True' in source
    assert "range(self.completed_steps_start, args.steps)" in source
    assert "spatial_smoothing_applied\": False" in source

continuation = (ROOT / "scripts/run_continuation.slurm").read_text()
continuation_submit = (ROOT / "scripts/submit_continuation.sh").read_text()
assert '--restart "$RESTART_DIR"' in continuation
assert "*_restart_step1500000/manifest.json" in continuation_submit
assert "TARGET_STEPS must be greater than 1500000" in continuation_submit
assert "--constraint=vram48" in continuation_submit

spec = importlib.util.spec_from_file_location("summary", ROOT / "tools/summarize_single.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
blocks = np.array([[[1.0]], [[3.0]], [[5.0]]])
sd, se = module.block_statistics(blocks)
assert np.allclose(sd, 2.0)
assert np.allclose(se, 2.0 / np.sqrt(3.0))
print("[OK] seven-case restartable checkpoint bundle and provenance validated")
