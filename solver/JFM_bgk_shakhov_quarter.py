#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGK/Shakhov particle production solver for the JFM reviewer calculations.

Case represented by the defaults:
  * paper Knudsen number: Kn = 30
  * wall-temperature ratio: Tc/Th = 0.2
  * one-quarter cavity: x,y in [0,L/2]
  * specular symmetry planes: x=0 and y=0
  * physical diffuse walls: x=L/2 (cold), y=L/2 (hot)
  * 22,000,000 simulator particles

The paper's Knudsen-number definition and model relaxation frequency are used
literally:

    Kn = 1 / (sqrt(2) * n0 * pi*d^2 * L).
    nu = n * pi*d^2 * sqrt(pi*kB*T/m).

Transport, wall treatment, grid, sampling, random-number generation, precision,
and output format are identical to JFM_hs_dsmc_quarter_22m.py. Only the
collision operator differs. No spatial filtering or projection is applied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np
from numba import cuda


SOLVER_VERSION = "2026-08-07-jfm-bgk-shakhov-restartable-v2"
RESTART_FORMAT_VERSION = 1

MASS = 6.63e-26
KB = 1.380649e-23
D_REF = 4.17e-10
SIGMA = math.pi * D_REF**2

L = 1.0e-3
LQ = 0.5 * L
T_HOT = 1000.0

NC = 100
DX = LQ / NC
INV_DX = 1.0 / DX
CELL_VOL = DX * DX
N_CELLS = NC * NC

VMP_HOT = math.sqrt(2.0 * KB * T_HOT / MASS)
DT_DEFAULT = 0.15 * DX / VMP_HOT
THREADS = 128
PRANDTL = 2.0 / 3.0
SHAKHOV_WEIGHT_CLAMP = 2.0
SHAKHOV_MAX_TRIALS = 32


@cuda.jit(device=True, inline=True)
def xorshift64star(state):
    state ^= state >> np.uint64(12)
    state ^= (state << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state ^= state >> np.uint64(27)
    state = (
        state * np.uint64(2685821657736338717)
    ) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state


@cuda.jit(device=True, inline=True)
def rng_uniform(state):
    state = xorshift64star(state)
    value = np.float32(
        ((state >> np.uint64(40)) & np.uint64(0xFFFFFF))
        / np.float32(16777216.0)
    )
    if value < np.float32(1.0e-7):
        value = np.float32(1.0e-7)
    return state, value


@cuda.jit(device=True, inline=True)
def rng_normal(state):
    state, u1 = rng_uniform(state)
    state, u2 = rng_uniform(state)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return state, np.float32(radius * math.cos(angle))


@cuda.jit(device=True, inline=True)
def diffuse_velocity(state, temperature, mass, kb, nx, ny):
    """Sample a diffuse-reflection velocity directed along inward (nx,ny)."""
    thermal_sigma = math.sqrt(kb * temperature / mass)
    state, uniform = rng_uniform(state)
    normal_speed = thermal_sigma * math.sqrt(-2.0 * math.log(uniform))
    state, tangent_gaussian = rng_normal(state)
    state, z_gaussian = rng_normal(state)

    tx = -ny
    ty = nx
    vx = normal_speed * nx + thermal_sigma * tangent_gaussian * tx
    vy = normal_speed * ny + thermal_sigma * tangent_gaussian * ty
    vz = thermal_sigma * z_gaussian
    return state, np.float32(vx), np.float32(vy), np.float32(vz)


@cuda.jit(fastmath=True)
def initialize_particles(x, y, vx, vy, vz, particle_rng, lq, t0, mass, kb):
    i = cuda.grid(1)
    if i >= x.size:
        return

    state = particle_rng[i]
    state, u1 = rng_uniform(state)
    state, u2 = rng_uniform(state)
    x[i] = u1 * lq
    y[i] = u2 * lq

    thermal_sigma = math.sqrt(kb * t0 / mass)
    state, g1 = rng_normal(state)
    state, g2 = rng_normal(state)
    state, g3 = rng_normal(state)
    vx[i] = thermal_sigma * g1
    vy[i] = thermal_sigma * g2
    vz[i] = thermal_sigma * g3
    particle_rng[i] = state


@cuda.jit(fastmath=True)
def advance_event_driven(
    x,
    y,
    vx,
    vy,
    vz,
    particle_rng,
    lq,
    dt,
    t_hot,
    t_cold,
    mass,
    kb,
):
    """
    Advance through the complete time step.

    Each wall hit is handled at its time of impact and the particle then
    advances for the remaining sub-step.  This avoids the wall artefact caused
    by reflecting an already-overshot particle only once per global step.
    """
    i = cuda.grid(1)
    if i >= x.size:
        return

    px = x[i]
    py = y[i]
    ux = vx[i]
    uy = vy[i]
    uz = vz[i]
    state = particle_rng[i]
    remaining = dt
    huge = np.float32(1.0e30)
    eps = np.float32(2.0e-7) * lq

    for _ in range(8):
        if remaining <= 0.0:
            break

        tx = huge
        ty = huge
        if ux > 0.0:
            tx = (lq - px) / ux
        elif ux < 0.0:
            tx = -px / ux

        if uy > 0.0:
            ty = (lq - py) / uy
        elif uy < 0.0:
            ty = -py / uy

        t_hit = tx
        hit_x = True
        if ty < tx:
            t_hit = ty
            hit_x = False

        if t_hit < 0.0 or t_hit >= remaining or t_hit == huge:
            px += ux * remaining
            py += uy * remaining
            remaining = 0.0
            break

        px += ux * t_hit
        py += uy * t_hit
        remaining -= t_hit

        if hit_x:
            if ux < 0.0:
                # x=0: specular symmetry plane.
                px = eps
                ux = -ux
            else:
                # x=L/2: cold diffuse wall; inward normal is -x.
                px = lq - eps
                state, ux, uy, uz = diffuse_velocity(
                    state, t_cold, mass, kb, -1.0, 0.0
                )
        else:
            if uy < 0.0:
                # y=0: specular symmetry plane.
                py = eps
                uy = -uy
            else:
                # y=L/2: hot diffuse wall; inward normal is -y.
                py = lq - eps
                state, ux, uy, uz = diffuse_velocity(
                    state, t_hot, mass, kb, 0.0, -1.0
                )

    # Extremely unlikely high-speed fallback after eight hits.
    if remaining > 0.0:
        px += ux * remaining
        py += uy * remaining

    if px < 0.0:
        px = -px
        ux = -ux
    elif px >= lq:
        px = lq - eps
        state, ux, uy, uz = diffuse_velocity(
            state, t_cold, mass, kb, -1.0, 0.0
        )

    if py < 0.0:
        py = -py
        uy = -uy
    elif py >= lq:
        py = lq - eps
        state, ux, uy, uz = diffuse_velocity(
            state, t_hot, mass, kb, 0.0, -1.0
        )

    x[i] = px
    y[i] = py
    vx[i] = ux
    vy[i] = uy
    vz[i] = uz
    particle_rng[i] = state


@cuda.jit(fastmath=True)
def assign_cells(x, y, cell_id, inv_dx, nc):
    i = cuda.grid(1)
    if i >= x.size:
        return
    ix = int(x[i] * inv_dx)
    iy = int(y[i] * inv_dx)
    if ix < 0:
        ix = 0
    elif ix >= nc:
        ix = nc - 1
    if iy < 0:
        iy = 0
    elif iy >= nc:
        iy = nc - 1
    cell_id[i] = iy * nc + ix


@cuda.jit(fastmath=True)
def collide_bgk_relaxation(
    vx, vy, vz, starts, counts, particle_rng, selected_relaxations,
    dt, cell_vol, fnum, sigma_hs, mass, kb,
):
    """One-thread-per-cell stochastic BGK relaxation."""
    cid = cuda.grid(1)
    if cid >= counts.size:
        return
    count = counts[cid]
    if count < 2:
        return
    start = starts[cid]

    sum_u = 0.0
    sum_v = 0.0
    sum_w = 0.0
    sum_v2 = 0.0
    for p in range(start, start + count):
        up = float(vx[p])
        vp = float(vy[p])
        wp = float(vz[p])
        sum_u += up
        sum_v += vp
        sum_w += wp
        sum_v2 += up * up + vp * vp + wp * wp

    inv_count = 1.0 / count
    mean_u = sum_u * inv_count
    mean_v = sum_v * inv_count
    mean_w = sum_w * inv_count
    thermal_v2 = sum_v2 * inv_count - (
        mean_u * mean_u + mean_v * mean_v + mean_w * mean_w
    )
    temperature = mass * thermal_v2 / (3.0 * kb)
    if temperature < 1.0:
        temperature = 1.0

    number_density = count * fnum / cell_vol
    frequency = (
        number_density
        * sigma_hs
        * math.sqrt(math.pi * kb * temperature / mass)
    )
    probability = 1.0 - math.exp(-frequency * dt)
    probability = min(1.0, max(0.0, probability))
    thermal_sigma = math.sqrt(kb * temperature / mass)
    local_selected = 0

    for p in range(start, start + count):
        state = particle_rng[p]
        state, uniform = rng_uniform(state)
        if uniform < probability:
            state, g1 = rng_normal(state)
            state, g2 = rng_normal(state)
            state, g3 = rng_normal(state)
            vx[p] = mean_u + thermal_sigma * g1
            vy[p] = mean_v + thermal_sigma * g2
            vz[p] = mean_w + thermal_sigma * g3
            local_selected += 1
        particle_rng[p] = state

    selected_relaxations[cid] += local_selected


@cuda.jit(fastmath=True)
def collide_shakhov_relaxation(
    vx, vy, vz, starts, counts, particle_rng, selected_relaxations,
    negative_weight_candidates, above_limiter_candidates,
    max_trial_fallbacks, dt, cell_vol, fnum, sigma_hs, mass, kb,
    prandtl, weight_clamp, max_trials,
):
    """Stochastic Shakhov relaxation with positivity diagnostics."""
    cid = cuda.grid(1)
    if cid >= counts.size:
        return
    count = counts[cid]
    if count < 2:
        return
    start = starts[cid]

    sum_u = 0.0
    sum_v = 0.0
    sum_w = 0.0
    sum_v2 = 0.0
    for p in range(start, start + count):
        up = float(vx[p])
        vp = float(vy[p])
        wp = float(vz[p])
        sum_u += up
        sum_v += vp
        sum_w += wp
        sum_v2 += up * up + vp * vp + wp * wp

    inv_count = 1.0 / count
    mean_u = sum_u * inv_count
    mean_v = sum_v * inv_count
    mean_w = sum_w * inv_count
    thermal_v2 = sum_v2 * inv_count - (
        mean_u * mean_u + mean_v * mean_v + mean_w * mean_w
    )
    temperature = mass * thermal_v2 / (3.0 * kb)
    if temperature < 1.0:
        temperature = 1.0

    number_density = count * fnum / cell_vol
    pressure = number_density * kb * temperature
    frequency = (
        number_density
        * sigma_hs
        * math.sqrt(math.pi * kb * temperature / mass)
    )
    probability = 1.0 - math.exp(-frequency * dt)
    probability = min(1.0, max(0.0, probability))
    theta = kb * temperature / mass
    thermal_sigma = math.sqrt(theta)

    sum_qx = 0.0
    sum_qy = 0.0
    sum_qz = 0.0
    for p in range(start, start + count):
        cx = vx[p] - mean_u
        cy = vy[p] - mean_v
        cz = vz[p] - mean_w
        c2 = cx * cx + cy * cy + cz * cz
        sum_qx += c2 * cx
        sum_qy += c2 * cy
        sum_qz += c2 * cz
    q_scale = 0.5 * mass * number_density * inv_count
    qx = q_scale * sum_qx
    qy = q_scale * sum_qy
    qz = q_scale * sum_qz
    denominator = 5.0 * pressure * theta + 1.0e-30
    correction_scale = 1.0 - prandtl

    local_selected = 0
    local_negative = 0
    local_above = 0
    local_fallback = 0
    for p in range(start, start + count):
        state = particle_rng[p]
        state, uniform = rng_uniform(state)
        if uniform < probability:
            local_selected += 1
            accepted = False
            for _ in range(max_trials):
                state, g1 = rng_normal(state)
                state, g2 = rng_normal(state)
                state, g3 = rng_normal(state)
                new_vx = mean_u + thermal_sigma * g1
                new_vy = mean_v + thermal_sigma * g2
                new_vz = mean_w + thermal_sigma * g3
                cx = new_vx - mean_u
                cy = new_vy - mean_v
                cz = new_vz - mean_w
                c2 = cx * cx + cy * cy + cz * cz
                c_dot_q = cx * qx + cy * qy + cz * qz
                shape = c2 / (2.0 * theta + 1.0e-30) - 2.5
                weight = (
                    1.0
                    + correction_scale * (c_dot_q / denominator) * shape
                )
                if weight < 0.0:
                    local_negative += 1
                    continue
                if weight > weight_clamp:
                    local_above += 1
                    weight = weight_clamp
                state, accept_uniform = rng_uniform(state)
                if accept_uniform < weight / weight_clamp:
                    vx[p] = new_vx
                    vy[p] = new_vy
                    vz[p] = new_vz
                    accepted = True
                    break

            if not accepted:
                local_fallback += 1
                state, g1 = rng_normal(state)
                state, g2 = rng_normal(state)
                state, g3 = rng_normal(state)
                vx[p] = mean_u + thermal_sigma * g1
                vy[p] = mean_v + thermal_sigma * g2
                vz[p] = mean_w + thermal_sigma * g3
        particle_rng[p] = state

    selected_relaxations[cid] += local_selected
    negative_weight_candidates[cid] += local_negative
    above_limiter_candidates[cid] += local_above
    max_trial_fallbacks[cid] += local_fallback


@cuda.jit(fastmath=True)
def sample_sorted_cells(vx, vy, vz, starts, counts, accumulators, block_id, ncells):
    """
    Accumulate raw cell moments with one thread per sorted cell.

    This avoids tens of millions of atomic operations at each sampling time.
    """
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    count = counts[cid]
    start = starts[cid]
    sum_u = 0.0
    sum_v = 0.0
    sum_w = 0.0
    sum_v2 = 0.0
    for p in range(start, start + count):
        up = float(vx[p])
        vp = float(vy[p])
        wp = float(vz[p])
        sum_u += up
        sum_v += vp
        sum_w += wp
        sum_v2 += up * up + vp * vp + wp * wp

    offset = block_id * 5 * ncells + cid
    accumulators[offset] += count
    accumulators[offset + ncells] += sum_u
    accumulators[offset + 2 * ncells] += sum_v
    accumulators[offset + 3 * ncells] += sum_w
    accumulators[offset + 4 * ncells] += sum_v2


def reconstruct_scalar(quadrant):
    top = np.concatenate((np.fliplr(quadrant), quadrant), axis=1)
    return np.concatenate((np.flipud(top), top), axis=0)


def reconstruct_velocity(u_quadrant, v_quadrant):
    u_top = np.concatenate((-np.fliplr(u_quadrant), u_quadrant), axis=1)
    v_top = np.concatenate((np.fliplr(v_quadrant), v_quadrant), axis=1)
    u_full = np.concatenate((np.flipud(u_top), u_top), axis=0)
    v_full = np.concatenate((-np.flipud(v_top), v_top), axis=0)
    return u_full, v_full


def moments_to_fields(moment_sum, sample_count, fnum, n0):
    sn, su, sv, sw, sv2 = moment_sum
    safe = np.maximum(sn, 1.0)
    uq = (su / safe).reshape(NC, NC)
    vq = (sv / safe).reshape(NC, NC)
    wq = (sw / safe).reshape(NC, NC)
    mean_v2 = (sv2 / safe).reshape(NC, NC)
    thermal = np.maximum(mean_v2 - uq * uq - vq * vq - wq * wq, 1.0e-12)
    tq = MASS * thermal / (3.0 * KB)

    mean_particles_per_sample = sn / max(sample_count, 1)
    number_density = (
        mean_particles_per_sample * fnum / CELL_VOL
    ).reshape(NC, NC)
    rhoq = number_density / n0

    u_full, v_full = reconstruct_velocity(uq, vq)
    return (
        u_full / VMP_HOT,
        v_full / VMP_HOT,
        reconstruct_scalar(tq) / T_HOT,
        reconstruct_scalar(rhoq),
    )


def write_tecplot(path, x, y, ux, uy, temperature, density):
    with path.open("w", encoding="utf-8") as handle:
        handle.write('TITLE="JFM relaxation-model raw unfiltered field"\n')
        handle.write('VARIABLES="x","y","u_x","u_y","T","rho","Umag"\n')
        handle.write(f"ZONE I={x.size}, J={y.size}, F=POINT\n")
        for j, y_value in enumerate(y):
            for i, x_value in enumerate(x):
                speed = math.hypot(ux[j, i], uy[j, i])
                handle.write(
                    f"{x_value:.10e} {y_value:.10e} "
                    f"{ux[j, i]:.10e} {uy[j, i]:.10e} "
                    f"{temperature[j, i]:.10e} {density[j, i]:.10e} "
                    f"{speed:.10e}\n"
                )


def sample_total(steps, sample_start, sample_every):
    return 1 + (steps - 1 - sample_start) // sample_every


def memory_gib(number_bytes):
    return number_bytes / 1024.0**3


class RelaxationQuarterCavity:
    def __init__(self, args):
        self.args = args
        self.n_particles = args.particles
        self.t_cold = args.rt * T_HOT
        self.dt = args.dt
        self.n0 = 1.0 / (
            math.sqrt(2.0) * SIGMA * args.kn * L
        )
        self.fnum = self.n0 * LQ * LQ / self.n_particles
        self.n_sampling_times = sample_total(
            args.steps, args.sample_start, args.sample_every
        )
        self.block_sample_counts = np.zeros(args.time_blocks, dtype=np.int64)
        self.completed_steps_start = 0
        self.samples_start = 0
        self.previous_wall_seconds = 0.0

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        self.gpu_free_gib_at_start = memory_gib(free_bytes)
        self.gpu_total_gib = memory_gib(total_bytes)
        if self.gpu_free_gib_at_start < args.require_free_gb:
            raise RuntimeError(
                f"Only {self.gpu_free_gib_at_start:.2f} GiB GPU memory is free; "
                f"--require-free-gb={args.require_free_gb:.2f}."
            )

        if args.restart:
            self.load_restart(Path(args.restart))
        else:
            # Particle state is float32; sampled moment sums are float64.
            self.x = cp.empty(self.n_particles, dtype=cp.float32)
            self.y = cp.empty(self.n_particles, dtype=cp.float32)
            self.vx = cp.empty(self.n_particles, dtype=cp.float32)
            self.vy = cp.empty(self.n_particles, dtype=cp.float32)
            self.vz = cp.empty(self.n_particles, dtype=cp.float32)

            seed_u64 = np.uint64(args.seed)
            seed_offset = (
                seed_u64 * np.uint64(1442695040888963407)
                + np.uint64(6364136223846793005)
            )
            self.particle_rng = (
                cp.arange(self.n_particles, dtype=cp.uint64)
                * cp.uint64(2862933555777941757)
                + cp.uint64(seed_offset)
            )
            self.selected_relaxations = cp.zeros(N_CELLS, dtype=cp.int64)
            self.negative_weight_candidates = cp.zeros(
                N_CELLS, dtype=cp.int64
            )
            self.above_limiter_candidates = cp.zeros(
                N_CELLS, dtype=cp.int64
            )
            self.max_trial_fallbacks = cp.zeros(N_CELLS, dtype=cp.int64)
            self.accumulators = cp.zeros(
                args.time_blocks * 5 * N_CELLS, dtype=cp.float64
            )

            blocks_particles = (self.n_particles + THREADS - 1) // THREADS
            t0 = 0.5 * (T_HOT + self.t_cold)
            initialize_particles[blocks_particles, THREADS](
                self.x,
                self.y,
                self.vx,
                self.vy,
                self.vz,
                self.particle_rng,
                LQ,
                t0,
                MASS,
                KB,
            )
            cuda.synchronize()

        self.cell_id = cp.empty(self.n_particles, dtype=cp.int32)

    @staticmethod
    def _load_npy(path, expected_dtype, expected_shape):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(expected_dtype):
            raise ValueError(
                f"Restart dtype mismatch for {path.name}: "
                f"{array.dtype} != {np.dtype(expected_dtype)}"
            )
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError(
                f"Restart shape mismatch for {path.name}: "
                f"{array.shape} != {tuple(expected_shape)}"
            )
        return array

    def load_restart(self, restart_dir):
        args = self.args
        manifest_path = restart_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Complete restart manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("complete") is not True
            or manifest.get("restart_format_version")
            != RESTART_FORMAT_VERSION
        ):
            raise ValueError(f"Incomplete or unsupported restart: {restart_dir}")

        expected = {
            "model": args.mode,
            "seed": args.seed,
            "particles": args.particles,
            "sample_start": args.sample_start,
            "sample_every": args.sample_every,
            "quadrant_cells": NC,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Restart mismatch for {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        for key, value in (("Kn_paper", args.kn), ("RT", args.rt), ("dt", args.dt)):
            if not math.isclose(
                float(manifest.get(key)), float(value), rel_tol=1.0e-13,
                abs_tol=1.0e-18,
            ):
                raise ValueError(
                    f"Restart mismatch for {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )

        old_steps = int(manifest["completed_steps"])
        old_samples = int(manifest["profile_samples"])
        old_blocks = int(manifest["time_blocks"])
        if args.steps <= old_steps:
            raise ValueError(
                f"Continuation target {args.steps} must exceed restart step {old_steps}"
            )
        if old_samples % old_blocks or self.n_sampling_times % args.time_blocks:
            raise ValueError("Restart and continuation require equal-sized time blocks")
        old_block_samples = old_samples // old_blocks
        new_block_samples = self.n_sampling_times // args.time_blocks
        if old_block_samples != new_block_samples:
            raise ValueError(
                "Continuation time-block width does not match the restart: "
                f"{new_block_samples} != {old_block_samples} samples"
            )
        if args.time_blocks < old_blocks:
            raise ValueError("Continuation cannot discard existing time blocks")

        particle_shape = (self.n_particles,)
        self.x = cp.asarray(self._load_npy(
            restart_dir / "x.npy", np.float32, particle_shape
        ))
        self.y = cp.asarray(self._load_npy(
            restart_dir / "y.npy", np.float32, particle_shape
        ))
        self.vx = cp.asarray(self._load_npy(
            restart_dir / "vx.npy", np.float32, particle_shape
        ))
        self.vy = cp.asarray(self._load_npy(
            restart_dir / "vy.npy", np.float32, particle_shape
        ))
        self.vz = cp.asarray(self._load_npy(
            restart_dir / "vz.npy", np.float32, particle_shape
        ))
        self.particle_rng = cp.asarray(self._load_npy(
            restart_dir / "particle_rng.npy", np.uint64, particle_shape
        ))
        counter_shape = (N_CELLS,)
        self.selected_relaxations = cp.asarray(self._load_npy(
            restart_dir / "selected_relaxations.npy", np.int64, counter_shape
        ))
        self.negative_weight_candidates = cp.asarray(self._load_npy(
            restart_dir / "negative_weight_candidates.npy", np.int64,
            counter_shape,
        ))
        self.above_limiter_candidates = cp.asarray(self._load_npy(
            restart_dir / "above_limiter_candidates.npy", np.int64,
            counter_shape,
        ))
        self.max_trial_fallbacks = cp.asarray(self._load_npy(
            restart_dir / "max_trial_fallbacks.npy", np.int64, counter_shape
        ))

        old_accumulator_size = old_blocks * 5 * N_CELLS
        old_accumulators = self._load_npy(
            restart_dir / "accumulators.npy", np.float64,
            (old_accumulator_size,),
        )
        self.accumulators = cp.zeros(
            args.time_blocks * 5 * N_CELLS, dtype=cp.float64
        )
        self.accumulators[:old_accumulator_size] = cp.asarray(old_accumulators)
        old_counts = np.asarray(self._load_npy(
            restart_dir / "block_sample_counts.npy", np.int64,
            (old_blocks,),
        ))
        self.block_sample_counts[:old_blocks] = old_counts
        if int(old_counts.sum()) != old_samples:
            raise ValueError("Restart sample-count checksum failed")

        self.completed_steps_start = old_steps
        self.samples_start = old_samples
        self.previous_wall_seconds = float(
            manifest.get("cumulative_solver_wall_seconds", 0.0)
        )
        cuda.synchronize()
        print(
            f"[RESTART] loaded {restart_dir}; steps={old_steps}; "
            f"samples={old_samples}; blocks={old_blocks}->{args.time_blocks}",
            flush=True,
        )

    def save_restart(self, stem, completed_steps, sample_index):
        args = self.args
        restart_dir = Path(str(stem) + f"_restart_step{completed_steps}")
        if restart_dir.exists():
            manifest = restart_dir / "manifest.json"
            if manifest.is_file():
                print(f"[RESTART] existing complete restart: {restart_dir}", flush=True)
                return restart_dir
            raise FileExistsError(
                f"Partial restart directory already exists: {restart_dir}"
            )

        restart_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=restart_dir.name + ".partial-", dir=restart_dir.parent
        ))
        array_metadata = {}

        def save_array(name, value):
            host = cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
            np.save(temporary / f"{name}.npy", host, allow_pickle=False)
            array_metadata[name] = {
                "dtype": str(host.dtype),
                "shape": list(host.shape),
                "bytes": int(host.nbytes),
            }
            del host

        cuda.synchronize()
        for name in ("x", "y", "vx", "vy", "vz", "particle_rng"):
            save_array(name, getattr(self, name))
        for name in (
            "selected_relaxations",
            "negative_weight_candidates",
            "above_limiter_candidates",
            "max_trial_fallbacks",
            "accumulators",
        ):
            save_array(name, getattr(self, name))
        save_array("block_sample_counts", self.block_sample_counts)

        manifest = {
            "complete": True,
            "restart_format_version": RESTART_FORMAT_VERSION,
            "solver_version": SOLVER_VERSION,
            "model": args.mode,
            "Kn_paper": args.kn,
            "RT": args.rt,
            "seed": args.seed,
            "particles": args.particles,
            "quadrant_cells": NC,
            "dt": args.dt,
            "completed_steps": completed_steps,
            "sample_start": args.sample_start,
            "sample_every": args.sample_every,
            "profile_samples": sample_index,
            "time_blocks": args.time_blocks,
            "samples_per_time_block": self.block_sample_counts.tolist(),
            "cumulative_solver_wall_seconds": self.wall_seconds,
            "arrays": array_metadata,
            "particle_state_is_complete": True,
            "rng_state_is_complete": True,
            "moment_accumulators_are_complete": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, restart_dir)
        print(
            f"[RESTART] complete restart written: {restart_dir}", flush=True
        )
        return restart_dir

    def sort_by_cell(self):
        blocks_particles = (self.n_particles + THREADS - 1) // THREADS
        assign_cells[blocks_particles, THREADS](
            self.x, self.y, self.cell_id, INV_DX, NC
        )
        counts = cp.bincount(self.cell_id, minlength=N_CELLS).astype(
            cp.int32, copy=False
        )
        order = cp.argsort(self.cell_id)
        self.x = self.x[order]
        self.y = self.y[order]
        self.vx = self.vx[order]
        self.vy = self.vy[order]
        self.vz = self.vz[order]
        self.particle_rng = self.particle_rng[order]
        starts = cp.empty(N_CELLS, dtype=cp.int64)
        starts[0] = 0
        cp.cumsum(counts[:-1], dtype=cp.int64, out=starts[1:])
        return starts, counts

    def write_checkpoint(self, completed_steps, sample_index, run_start):
        """Write a read-only statistical snapshot without stopping the run."""
        args = self.args
        if sample_index <= 0:
            raise RuntimeError("Cannot write a checkpoint before sampling starts")

        cuda.synchronize()
        raw = cp.asnumpy(self.accumulators).reshape(
            args.time_blocks, 5, N_CELLS
        )
        counts = self.block_sample_counts.copy()
        used_blocks = np.flatnonzero(counts > 0)
        total_moments = raw.sum(axis=0)
        ux, uy, temperature, density = moments_to_fields(
            total_moments, sample_index, self.fnum, self.n0
        )
        block_entries = [
            moments_to_fields(
                raw[b], int(counts[b]), self.fnum, self.n0
            )
            for b in used_blocks
        ]
        block_fields = tuple(
            np.stack([entry[k] for entry in block_entries], axis=0)
            for k in range(4)
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        rt_tag = str(args.rt).replace(".", "p")
        stem = output_dir / (
            f"ThermalCavity_{args.mode}_Kn{args.kn:g}_RT{rt_tag}"
            f"_quarter_seed{args.seed}_step{completed_steps}"
        )
        nf = 2 * NC
        coordinate = -0.5 + (np.arange(nf) + 0.5) / nf
        np.savez_compressed(
            str(stem) + "_raw.npz",
            x=coordinate,
            y=coordinate,
            ux=ux,
            uy=uy,
            T=temperature,
            rho=density,
            ux_time_blocks=block_fields[0],
            uy_time_blocks=block_fields[1],
            T_time_blocks=block_fields[2],
            rho_time_blocks=block_fields[3],
            samples_per_time_block=counts[used_blocks],
            time_block_ids=used_blocks,
            seed=np.int64(args.seed),
            Kn_paper=np.float64(args.kn),
            RT=np.float64(args.rt),
            mode=np.asarray(args.mode),
            checkpoint_steps_completed=np.int64(completed_steps),
            target_steps=np.int64(args.steps),
        )
        write_tecplot(
            Path(str(stem) + "_raw.dat"),
            coordinate,
            coordinate,
            ux,
            uy,
            temperature,
            density,
        )
        checkpoint = {
            "solver_version": SOLVER_VERSION,
            "mode": args.mode,
            "Kn_paper": args.kn,
            "RT": args.rt,
            "seed": args.seed,
            "particles": args.particles,
            "steps_completed": completed_steps,
            "target_steps": args.steps,
            "sample_start": args.sample_start,
            "sample_every": args.sample_every,
            "profile_samples": sample_index,
            "time_blocks_written": int(used_blocks.size),
            "samples_per_time_block": counts[used_blocks].tolist(),
            "wall_clock_seconds": time.time() - run_start,
            "quantitative_fields_are_unfiltered": True,
            "spatial_smoothing_applied": False,
            "velocity_projection_applied": False,
        }
        Path(str(stem) + "_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[CHECKPOINT] steps={completed_steps}; samples={sample_index}; "
            f"outputs={stem}_raw.*",
            flush=True,
        )

    def run(self):
        args = self.args
        blocks_particles = (self.n_particles + THREADS - 1) // THREADS
        blocks_cells = (N_CELLS + THREADS - 1) // THREADS
        print(
            f"--- {args.mode} quarter cavity: Kn={args.kn:g}, RT={args.rt:g}, "
            f"N={self.n_particles}, seed={args.seed}, NCq={NC} ---",
            flush=True,
        )
        print(
            f"solver={SOLVER_VERSION}; dt={self.dt:.8e} s; "
            f"steps={args.steps}; sample={args.sample_start}:{args.sample_every}",
            flush=True,
        )
        print(
            f"n0={self.n0:.8e} 1/m^3; FNUM={self.fnum:.8e}; "
            f"GPU memory free/total at start="
            f"{self.gpu_free_gib_at_start:.2f}/{self.gpu_total_gib:.2f} GiB",
            flush=True,
        )

        run_start = time.time()
        sample_index = self.samples_start
        checkpoint_steps = set(args.checkpoint_steps)
        if any(step <= self.completed_steps_start for step in checkpoint_steps):
            raise ValueError(
                "Checkpoint steps in a continuation must exceed the restart step"
            )
        for step in range(self.completed_steps_start, args.steps):
            advance_event_driven[blocks_particles, THREADS](
                self.x,
                self.y,
                self.vx,
                self.vy,
                self.vz,
                self.particle_rng,
                LQ,
                self.dt,
                T_HOT,
                self.t_cold,
                MASS,
                KB,
            )
            starts, counts = self.sort_by_cell()
            if args.mode == "BGK":
                collide_bgk_relaxation[blocks_cells, THREADS](
                    self.vx, self.vy, self.vz, starts, counts,
                    self.particle_rng, self.selected_relaxations,
                    self.dt, CELL_VOL, self.fnum, np.float32(SIGMA),
                    MASS, KB,
                )
            else:
                collide_shakhov_relaxation[blocks_cells, THREADS](
                    self.vx, self.vy, self.vz, starts, counts,
                    self.particle_rng, self.selected_relaxations,
                    self.negative_weight_candidates,
                    self.above_limiter_candidates,
                    self.max_trial_fallbacks,
                    self.dt, CELL_VOL, self.fnum, np.float32(SIGMA),
                    MASS, KB, PRANDTL, SHAKHOV_WEIGHT_CLAMP,
                    SHAKHOV_MAX_TRIALS,
                )

            if (
                step >= args.sample_start
                and (step - args.sample_start) % args.sample_every == 0
            ):
                block_id = min(
                    args.time_blocks - 1,
                    sample_index * args.time_blocks // self.n_sampling_times,
                )
                sample_sorted_cells[blocks_cells, THREADS](
                    self.vx,
                    self.vy,
                    self.vz,
                    starts,
                    counts,
                    self.accumulators,
                    block_id,
                    N_CELLS,
                )
                self.block_sample_counts[block_id] += 1
                sample_index += 1

            if step % args.progress_every == 0:
                elapsed_hours = (time.time() - run_start) / 3600.0
                print(
                    f"step {step}/{args.steps}; samples={sample_index}; "
                    f"elapsed={elapsed_hours:.3f} h",
                    flush=True,
                )

            completed_steps = step + 1
            if completed_steps in checkpoint_steps:
                self.write_checkpoint(
                    completed_steps, sample_index, run_start
                )

        cuda.synchronize()
        self.segment_wall_seconds = time.time() - run_start
        self.wall_seconds = (
            self.previous_wall_seconds + self.segment_wall_seconds
        )
        if sample_index != self.n_sampling_times:
            raise RuntimeError(
                f"Expected {self.n_sampling_times} samples, obtained {sample_index}"
            )

        raw = cp.asnumpy(self.accumulators).reshape(
            args.time_blocks, 5, N_CELLS
        )
        total_moments = raw.sum(axis=0)
        self.fields = moments_to_fields(
            total_moments, sample_index, self.fnum, self.n0
        )
        block_fields = [
            moments_to_fields(
                raw[b],
                int(self.block_sample_counts[b]),
                self.fnum,
                self.n0,
            )
            for b in range(args.time_blocks)
        ]
        self.block_fields = tuple(
            np.stack([entry[k] for entry in block_fields], axis=0)
            for k in range(4)
        )

        self.diagnostics = {
            "selected_relaxations": int(
                cp.asnumpy(self.selected_relaxations).sum()
            ),
            "negative_weight_candidates": int(
                cp.asnumpy(self.negative_weight_candidates).sum()
            ),
            "above_limiter_candidates": int(
                cp.asnumpy(self.above_limiter_candidates).sum()
            ),
            "max_trial_fallbacks": int(
                cp.asnumpy(self.max_trial_fallbacks).sum()
            ),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="JFM heavy BGK/Shakhov quarter-cavity production run"
    )
    parser.add_argument(
        "--mode", choices=("BGK", "SHAKHOV"), required=True
    )
    parser.add_argument("--kn", type=float, default=30.0)
    parser.add_argument("--rt", type=float, default=0.2)
    parser.add_argument("--particles", type=int, default=22_000_000)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--sample-start", type=int, default=400_000)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-blocks", type=int, default=4)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=[],
        help="Completed step counts at which to write interim raw fields",
    )
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--require-free-gb", type=float, default=8.5)
    parser.add_argument(
        "--restart",
        default=None,
        help="Complete restart directory from an earlier segment",
    )
    parser.add_argument(
        "--no-write-restart",
        action="store_false",
        dest="write_restart",
        help="Do not write a complete continuation restart at final step",
    )
    parser.set_defaults(write_restart=True)
    parser.add_argument(
        "--output", default="results_bgk_shakhov_heavy"
    )
    return parser.parse_args()


def validate_args(args):
    if args.kn <= 0.0:
        raise ValueError("--kn must be positive")
    if not 0.0 < args.rt < 1.0:
        raise ValueError("--rt must satisfy 0 < RT < 1")
    if args.particles < 100_000:
        raise ValueError("--particles must be at least 100000")
    if args.steps <= 0 or not 0 <= args.sample_start < args.steps:
        raise ValueError("Require 0 <= sample-start < steps")
    if args.sample_every <= 0 or args.time_blocks <= 0:
        raise ValueError("sample-every and time-blocks must be positive")
    if args.time_blocks > sample_total(
        args.steps, args.sample_start, args.sample_every
    ):
        raise ValueError("More time blocks than sampling times")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if len(args.checkpoint_steps) != len(set(args.checkpoint_steps)):
        raise ValueError("--checkpoint-steps must not contain duplicates")
    if any(
        step <= args.sample_start or step >= args.steps
        for step in args.checkpoint_steps
    ):
        raise ValueError(
            "Each checkpoint step must be after sample-start and before steps"
        )
    if args.restart and not Path(args.restart).is_dir():
        raise FileNotFoundError(f"Restart directory not found: {args.restart}")


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rt_tag = str(args.rt).replace(".", "p")
    stem = output_dir / (
        f"ThermalCavity_{args.mode}_Kn{args.kn:g}_RT{rt_tag}"
        f"_quarter_seed{args.seed}"
    )

    total_start = time.time()
    solver = RelaxationQuarterCavity(args)
    solver.run()
    ux, uy, temperature, density = solver.fields
    ux_blocks, uy_blocks, t_blocks, rho_blocks = solver.block_fields

    nf = 2 * NC
    coordinate = -0.5 + (np.arange(nf) + 0.5) / nf
    np.savez_compressed(
        str(stem) + "_raw.npz",
        x=coordinate,
        y=coordinate,
        ux=ux,
        uy=uy,
        T=temperature,
        rho=density,
        ux_time_blocks=ux_blocks,
        uy_time_blocks=uy_blocks,
        T_time_blocks=t_blocks,
        rho_time_blocks=rho_blocks,
        samples_per_time_block=solver.block_sample_counts,
        seed=np.int64(args.seed),
        Kn_paper=np.float64(args.kn),
        RT=np.float64(args.rt),
        mode=np.asarray(args.mode),
    )
    write_tecplot(
        Path(str(stem) + "_raw.dat"),
        coordinate,
        coordinate,
        ux,
        uy,
        temperature,
        density,
    )

    speed = np.hypot(ux, uy)
    dx_nd = 1.0 / nf
    kinetic_energy = float(
        np.sum(density * speed * speed) * dx_nd * dx_nd
    )
    block_speed_max = [
        float(np.hypot(ux_blocks[b], uy_blocks[b]).max())
        for b in range(args.time_blocks)
    ]
    last_block_velocity_rmse = float(
        np.sqrt(
            np.mean(
                (ux_blocks[-1] - ux) ** 2
                + (uy_blocks[-1] - uy) ** 2
            )
        )
    )

    restart_path = None
    if args.write_restart:
        restart_path = solver.save_restart(
            stem, args.steps, solver.n_sampling_times
        )

    metrics = {
        "solver_version": SOLVER_VERSION,
        "mode": args.mode,
        "collision_model": (
            "stochastic BGK relaxation"
            if args.mode == "BGK"
            else "stochastic Shakhov relaxation"
        ),
        "seed": args.seed,
        "Kn_paper": args.kn,
        "RT": args.rt,
        "kn_definition": "1/(sqrt(2)*n0*pi*d^2*L)",
        "quarter_domain": True,
        "specular_planes": ["x=0", "y=0"],
        "physical_diffuse_walls": {
            "x=L/2": "cold",
            "y=L/2": "hot",
        },
        "wall_advance_scheme": (
            "event-driven time-of-impact with residual-time advance"
        ),
        "quadrant_cells": [NC, NC],
        "full_reconstructed_cells": [nf, nf],
        "particles": args.particles,
        "mean_particles_per_quadrant_cell": args.particles / N_CELLS,
        "particle_storage_dtype": "float32",
        "moment_accumulator_dtype": "float64",
        "steps": args.steps,
        "segment_start_step": solver.completed_steps_start,
        "continued_from_restart": args.restart,
        "sample_start": args.sample_start,
        "sample_every": args.sample_every,
        "profile_samples": solver.n_sampling_times,
        "time_blocks": args.time_blocks,
        "samples_per_time_block": solver.block_sample_counts.tolist(),
        "dt_seconds": args.dt,
        "n0_per_m3": solver.n0,
        "FNUM": solver.fnum,
        "max_speed_nondimensional": float(speed.max()),
        "kinetic_energy_nondimensional": kinetic_energy,
        "time_block_max_speeds_nondimensional": block_speed_max,
        "last_block_velocity_rmse_vs_all_samples": (
            last_block_velocity_rmse
        ),
        "gpu_free_GiB_at_start": solver.gpu_free_gib_at_start,
        "gpu_total_GiB": solver.gpu_total_gib,
        "solver_wall_clock_seconds": solver.wall_seconds,
        "segment_wall_clock_seconds": solver.segment_wall_seconds,
        "total_wall_clock_seconds": time.time() - total_start,
        "restart_format_version": (
            RESTART_FORMAT_VERSION if restart_path is not None else None
        ),
        "restart_directory": (
            str(restart_path) if restart_path is not None else None
        ),
        "relaxation_frequency": "n*pi*d^2*sqrt(pi*kB*T/m)",
        "prandtl_number": (
            None if args.mode == "BGK" else PRANDTL
        ),
        "shakhov_weight_limiter": (
            None if args.mode == "BGK"
            else [0.0, SHAKHOV_WEIGHT_CLAMP]
        ),
        "shakhov_max_accept_reject_trials": (
            None if args.mode == "BGK" else SHAKHOV_MAX_TRIALS
        ),
        "relaxation_diagnostics": solver.diagnostics,
        "quantitative_fields_are_unfiltered": True,
        "spatial_smoothing_applied": False,
        "velocity_projection_applied": False,
    }
    Path(str(stem) + "_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[OK] Raw unfiltered outputs: {stem}_raw.*", flush=True)


if __name__ == "__main__":
    main()
