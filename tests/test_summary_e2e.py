#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stem(row):
    kn = f"{float(row['kn']):g}"
    rt = f"{float(row['rt']):g}".replace(".", "p")
    prefix = "ThermalCavity_HS_DSMC" if row["model"] == "HS" else f"ThermalCavity_{row['model']}"
    return f"{prefix}_Kn{kn}_RT{rt}_quarter_seed{row['seed']}"


with tempfile.TemporaryDirectory(prefix="jfm-one-seed-e2e-") as temp:
    base = Path(temp)
    old, new, output = base / "old", base / "new", base / "summary"
    old.mkdir()
    new.mkdir()
    with (ROOT / "cases/fast7.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = np.linspace(-0.4, 0.4, 4)
    for case_number, row in enumerate(rows, 1):
        destination = new
        shape = (4, 4)
        ux = np.full(shape, case_number * 1e-4)
        uy = np.full(shape, -case_number * 2e-5)
        blocks = 14
        offsets = np.linspace(-1e-5, 1e-5, blocks)[:, None, None]
        filename = stem(row)
        np.savez_compressed(destination / f"{filename}_raw.npz",
            x=x, y=x, ux=ux, uy=uy, T=np.full(shape, 0.75), rho=np.ones(shape),
            ux_time_blocks=ux + offsets, uy_time_blocks=uy + offsets,
            T_time_blocks=np.full((blocks, *shape), 0.75),
            rho_time_blocks=np.ones((blocks, *shape)),
            samples_per_time_block=np.full(blocks, 100))
        metrics = {"particles": 80_000_000,
                   "steps": 1_500_000,
                   "sample_start": 100_000, "profile_samples": blocks * 100,
                   "last_block_velocity_rmse_vs_all_samples": 1e-5}
        (destination / f"{filename}_metrics.json").write_text(json.dumps(metrics))
    subprocess.run([sys.executable, str(ROOT / "tools/summarize_single.py"),
        "--existing-input", str(old), "--new-input", str(new),
        "--output", str(output), "--case-table", str(ROOT / "cases/fast7.csv")],
        check=True)
    summaries = json.loads((output / "ALL_SINGLE_HEAVY_SEED_SUMMARY.json").read_text())
    assert len(summaries) == 7
    assert {item["time_blocks"] for item in summaries} == {14}
    assert all(item["block_uncertainty_is_not_independent_seed_uncertainty"] for item in summaries)
    assert len(list(output.glob("*_RAW_UNFILTERED.dat"))) == 7
    assert len(list(output.glob("*_diagnostic.png"))) == 7
print("[OK] synthetic checkpoint-fast seven-case summary test passed")
