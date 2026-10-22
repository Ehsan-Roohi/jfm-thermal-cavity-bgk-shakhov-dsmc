#!/usr/bin/env python3
"""Check recovered source closure, byte provenance and ensemble statistics."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "provenance/unity_recovery_20261022.json").read_text())
for record in manifest["files"]:
    data = (ROOT / record["bundled_path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == record["bundled_sha256"], record["bundled_path"]
    if record["change"] == "byte-for-byte copy":
        assert digest == record["sha256"]

module = ROOT / "solver/JFM_reviewer_quarter_domain.py"
tree = ast.parse(module.read_text(encoding="utf-8"))
exports = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for node in tree.body:
    if isinstance(node, ast.Assign):
        exports.update(t.id for t in node.targets if isinstance(t, ast.Name))
drivers = [ROOT / p for p in (
    "validation/fourier/JFM_shakhov_fourier_validation.py",
    "validation/fourier/JFM_hardsphere_fourier_validation.py",
    "validation/homogeneous/JFM_shakhov_homogeneous_validation.py",
)]
for driver in drivers:
    text = driver.read_text(encoding="utf-8")
    assert 'parents[2] / "solver"' in text
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "cavity":
                assert node.attr in exports, (driver.name, node.attr)

bgk, shk, hs = [], [], []
key = "measured_conductivity_W_m_K_from_interior_slope"
hskey = "measured_conductivity_W_m_K_from_bulk_qx_and_interior_slope"
for seed in (42, 271828, 314159):
    for model, values in (("BGK", bgk), ("SHAKHOV", shk)):
        path = ROOT / f"validation/fourier/corrected_results/Fourier_{model}_Kn0.02_seed{seed}_metrics.json"
        record = json.loads(path.read_text())
        assert record["particles"] == 2_000_000 and record["seed"] == seed
        values.append(record[key])
    path = ROOT / f"validation/fourier/hardsphere_results/Fourier_HARD_SPHERE_DSMC_Kn0.02_seed{seed}_metrics.json"
    hs.append(json.loads(path.read_text())[hskey])
bgk, shk, hs = map(np.array, (bgk, shk, hs))
summary = json.loads((ROOT / "validation/fourier/hardsphere_results/Fourier_HARD_SPHERE_ensemble_Kn0.02_summary.json").read_text())
checks = {
    "conductivity_HS_mean_W_m_K": hs.mean(),
    "conductivity_HS_over_BGK_mean_of_paired_ratios": (hs/bgk).mean(),
    "conductivity_Shakhov_over_HS_mean_of_paired_ratios": (shk/hs).mean(),
}
for key, value in checks.items():
    assert np.isclose(value, summary[key], rtol=1e-13), (key, value, summary[key])
print("[OK] recovered hashes, driver dependencies and three-seed conductivity statistics")
