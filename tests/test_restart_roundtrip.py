#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FakeCuda:
    @staticmethod
    def jit(*decorator_args, **decorator_kwargs):
        if len(decorator_args) == 1 and callable(decorator_args[0]):
            return decorator_args[0]

        def decorate(function):
            return function

        return decorate

    @staticmethod
    def synchronize():
        return None


fake_cupy = types.ModuleType("cupy")
fake_cupy.ndarray = np.ndarray
fake_cupy.asnumpy = np.asarray


def fake_device_copy(value, dtype=None):
    """Mirror CuPy's host-to-device copy without retaining an mmap handle."""
    return np.array(value, dtype=dtype, copy=True)


fake_cupy.asarray = fake_device_copy
fake_cupy.zeros = np.zeros
for name in ("float32", "float64", "int32", "int64", "uint64"):
    setattr(fake_cupy, name, getattr(np, name))
fake_numba = types.ModuleType("numba")
fake_numba.cuda = FakeCuda()
sys.modules["cupy"] = fake_cupy
sys.modules["numba"] = fake_numba


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def particle_arrays(count):
    return {
        name: (np.arange(count, dtype=np.float32) + offset)
        for offset, name in enumerate(("x", "y", "vx", "vy", "vz"))
    }


def common_args(mode, restart, target_steps, blocks):
    return SimpleNamespace(
        mode=mode,
        kn=20.0,
        rt=0.2,
        seed=104729,
        particles=5,
        sample_start=2,
        sample_every=1,
        time_blocks=blocks,
        steps=target_steps,
        dt=1.25e-9,
        restart=str(restart) if restart else None,
    )


def exercise(module, class_name, mode):
    cls = getattr(module, class_name)
    ncells = module.N_CELLS
    with tempfile.TemporaryDirectory(prefix=f"restart-{mode.lower()}-") as temp:
        stem = Path(temp) / "case"
        source = cls.__new__(cls)
        source.args = common_args(mode, None, 6, 2)
        source.n_particles = 5
        for name, value in particle_arrays(5).items():
            setattr(source, name, value)
        source.particle_rng = np.arange(5, dtype=np.uint64) + 100
        source.accumulators = np.arange(
            2 * 5 * ncells, dtype=np.float64
        )
        source.block_sample_counts = np.array([2, 2], dtype=np.int64)
        source.wall_seconds = 12.5

        if mode == "HS":
            source.cell_rng = np.arange(ncells, dtype=np.uint64) + 200
            source.sigma_g_majorant = np.linspace(
                1.0, 2.0, ncells, dtype=np.float32
            )
            source.candidate_pairs = np.arange(ncells, dtype=np.int64)
            source.accepted_collisions = np.arange(ncells, dtype=np.int64) + 2
            source.majorant_updates = np.arange(ncells, dtype=np.int64) + 3
        else:
            source.selected_relaxations = np.arange(ncells, dtype=np.int64)
            source.negative_weight_candidates = np.arange(ncells, dtype=np.int64) + 1
            source.above_limiter_candidates = np.arange(ncells, dtype=np.int64) + 2
            source.max_trial_fallbacks = np.arange(ncells, dtype=np.int64) + 3

        restart = source.save_restart(stem, completed_steps=6, sample_index=4)
        manifest = json.loads((restart / "manifest.json").read_text())
        assert manifest["complete"] is True
        assert manifest["completed_steps"] == 6
        assert manifest["profile_samples"] == 4

        target = cls.__new__(cls)
        target.args = common_args(mode, restart, 8, 3)
        target.n_particles = 5
        target.n_sampling_times = 6
        target.block_sample_counts = np.zeros(3, dtype=np.int64)
        target.load_restart(restart)

        assert target.completed_steps_start == 6
        assert target.samples_start == 4
        assert target.previous_wall_seconds == 12.5
        assert np.array_equal(target.block_sample_counts, [2, 2, 0])
        assert target.accumulators.shape == (3 * 5 * ncells,)
        assert np.array_equal(
            target.accumulators[: source.accumulators.size],
            source.accumulators,
        )
        assert np.count_nonzero(
            target.accumulators[source.accumulators.size :]
        ) == 0
        for name in ("x", "y", "vx", "vy", "vz", "particle_rng"):
            assert np.array_equal(getattr(target, name), getattr(source, name))


hs = load_module(
    ROOT / "solver/JFM_hs_dsmc_quarter.py", "restart_test_hs"
)
relax = load_module(
    ROOT / "solver/JFM_bgk_shakhov_quarter.py", "restart_test_relax"
)
exercise(hs, "HardSphereQuarterCavity", "HS")
exercise(relax, "RelaxationQuarterCavity", "BGK")
exercise(relax, "RelaxationQuarterCavity", "SHAKHOV")
print("[OK] HS, BGK and Shakhov restart save/load round trips passed")
