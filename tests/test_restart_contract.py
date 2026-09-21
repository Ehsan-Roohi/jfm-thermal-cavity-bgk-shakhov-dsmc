#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sample_total(steps, sample_start=100_000, sample_every=2):
    return 1 + (steps - 1 - sample_start) // sample_every


segments = (
    (1_500_000, 14, 700_000),
    (2_000_000, 19, 950_000),
    (3_000_000, 29, 1_450_000),
)
for steps, blocks, expected_samples in segments:
    samples = sample_total(steps)
    assert samples == expected_samples
    assert samples % blocks == 0
    assert samples // blocks == 50_000

required_particle_state = ("x", "y", "vx", "vy", "vz", "particle_rng")
hs = (ROOT / "solver/JFM_hs_dsmc_quarter.py").read_text()
relax = (ROOT / "solver/JFM_bgk_shakhov_quarter.py").read_text()
for name in required_particle_state:
    assert f'"{name}"' in hs
    assert f'"{name}"' in relax

for name in (
    "cell_rng",
    "sigma_g_majorant",
    "candidate_pairs",
    "accepted_collisions",
    "majorant_updates",
    "accumulators",
    "block_sample_counts",
):
    assert f'"{name}"' in hs

for name in (
    "selected_relaxations",
    "negative_weight_candidates",
    "above_limiter_candidates",
    "max_trial_fallbacks",
    "accumulators",
    "block_sample_counts",
):
    assert f'"{name}"' in relax

for source in (hs, relax):
    assert "mmap_mode=\"r\"" in source
    assert "os.replace(temporary, restart_dir)" in source
    assert 'manifest.get("complete") is not True' in source
    assert "old_block_samples != new_block_samples" in source

print("[OK] restart state and equal-width time-block contract validated")
