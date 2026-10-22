# -*- coding: utf-8 -*-
"""Hard-sphere DSMC validation in planar Fourier flow.

This add-on intentionally reuses the physical constants, RNG, initialization,
sampling, and paper Knudsen-number convention of JFM_CORRECTED_V3:

    Kn = 1/(sqrt(2)*n0*pi*d^2*L).

Collisions are handled by a no-time-counter (NTC) hard-sphere kernel.  Every
accepted binary collision conserves momentum and energy to roundoff.  The code
also measures energy transfer at both diffuse walls, giving an independent
steady-state heat-flux and energy-balance audit.
"""

import argparse
import csv
import json
import math
import os
import time

import cupy as cp
import numpy as np
from numba import cuda

# Resolve the recovered shared solver when launched from any directory.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solver"))

import JFM_reviewer_quarter_domain as cavity
import JFM_shakhov_fourier_validation as common


SOLVER_VERSION = "2026-07-23-fourier-hard-sphere-dsmc-v1"


def hard_sphere_fourier_reference(t_left, t_right, length):
    """First Chapman--Enskog hard-sphere conductivity and Fourier heat flux.

    For a monatomic hard-sphere gas,

        k_HS(T) = C_HS*sqrt(T),
        C_HS = 75*pi*kB/(64*sigma)*sqrt(kB/(pi*m)),

    where sigma=pi*d^2.  Integration of q=-k(T)dT/dx gives the reference heat
    flux used here.  This is a near-continuum, no-temperature-jump reference.
    """
    c_hs = (
        75.0
        * math.pi
        * cavity.KB
        / (64.0 * cavity.SIGMA)
        * math.sqrt(cavity.KB / (math.pi * cavity.MASS))
    )
    q_reference = (
        2.0
        * c_hs
        / (3.0 * length)
        * (t_left**1.5 - t_right**1.5)
    )
    return c_hs, q_reference


@cuda.jit(fastmath=True)
def move_fourier_with_wall_audit(
    x,
    vx,
    vy,
    vz,
    seeds,
    length,
    dt,
    t_left,
    t_right,
    mass,
    kb,
    collect_wall_flux,
    wall_energy_into_gas,
    wall_hit_count,
):
    """Event-driven diffuse-wall motion with residual-time advancement.

    ``wall_energy_into_gas[0]`` and ``[1]`` accumulate the simulated-particle
    energy changes at the hot and cold walls, respectively.  Multiplication by
    FNUM and division by the sampled physical time converts these to W/m^2 for
    the unit cross-sectional area used by the one-dimensional simulation.
    """
    i = cuda.grid(1)
    if i >= x.size:
        return

    state = seeds[i]
    xp = x[i]
    ax = vx[i]
    ay = vy[i]
    az = vz[i]
    remaining = dt
    eps = 1.0e-12 * length

    for _hit in range(8):
        if remaining <= 0.0:
            break

        thit = remaining + 1.0
        side = 0
        if ax < 0.0:
            thit = -xp / ax
            side = -1
        elif ax > 0.0:
            thit = (length - xp) / ax
            side = 1

        if side == 0 or thit < 0.0 or thit >= remaining:
            xp += ax * remaining
            remaining = 0.0
            break

        xp += ax * thit
        remaining -= thit
        e_before = 0.5 * mass * (ax * ax + ay * ay + az * az)

        if side < 0:
            xp = eps
            state, ax, ay, az = cavity.sample_diffuse_velocity(
                state, t_left, mass, kb, 1.0, 0.0
            )
            wall_index = 0
        else:
            xp = length - eps
            state, ax, ay, az = cavity.sample_diffuse_velocity(
                state, t_right, mass, kb, -1.0, 0.0
            )
            wall_index = 1

        if collect_wall_flux != 0:
            e_after = 0.5 * mass * (ax * ax + ay * ay + az * az)
            cuda.atomic.add(
                wall_energy_into_gas, wall_index, e_after - e_before
            )
            cuda.atomic.add(wall_hit_count, wall_index, 1.0)

    if remaining > 0.0:
        xp += ax * remaining

    # This branch protects against an extraordinarily energetic tail exhausting
    # the bounded event loop; it should never be used with the production CFL.
    if xp < 0.0:
        xp = eps
    elif xp >= length:
        xp = length - eps

    x[i] = xp
    vx[i] = ax
    vy[i] = ay
    vz[i] = az
    seeds[i] = state


@cuda.jit(fastmath=True)
def collision_hard_sphere_ntc_cellwise(
    vx,
    vy,
    vz,
    starts,
    counts,
    sigma_g_majorant,
    dt,
    volume,
    fnum,
    cell_rng_states,
    sigma_hs,
    candidate_pair_count,
    accepted_collision_count,
    majorant_update_count,
):
    """One-thread-per-cell NTC collision kernel for hard spheres.

    Candidate-pair count:

        0.5*N*(N-1)*FNUM*(sigma*g)_max*dt/V.

    The two selected particles are guaranteed to be distinct.  The post-
    collision relative direction is isotropic, while its magnitude is
    unchanged, so each accepted pair conserves momentum and energy.
    """
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    cnt = counts[cid]
    if cnt < 2:
        return

    st = starts[cid]
    state = cell_rng_states[cid]
    sig_g_max = sigma_g_majorant[cid]

    base = (
        0.5
        * cnt
        * (cnt - 1)
        * fnum
        * sig_g_max
        * dt
        / volume
    )
    n_pairs = int(base)
    state, uround = cavity.rng_uniform_float(state)
    if uround < base - n_pairs:
        n_pairs += 1

    accepted = 0
    updates = 0

    for _pair in range(n_pairs):
        state, u1 = cavity.rng_uniform_float(state)
        state, u2 = cavity.rng_uniform_float(state)

        local_1 = int(u1 * cnt)
        if local_1 >= cnt:
            local_1 = cnt - 1

        # Draw from 0..cnt-2, then skip local_1.  This avoids null pairs.
        local_2 = int(u2 * (cnt - 1))
        if local_2 >= local_1:
            local_2 += 1

        p1 = st + local_1
        p2 = st + local_2

        gx = vx[p1] - vx[p2]
        gy = vy[p1] - vy[p2]
        gz = vz[p1] - vz[p2]
        g = math.sqrt(gx * gx + gy * gy + gz * gz)
        sig_g = sigma_hs * g

        if sig_g > sig_g_max:
            sig_g_max = sig_g
            updates += 1

        state, uaccept = cavity.rng_uniform_float(state)
        if uaccept * sig_g_max < sig_g:
            accepted += 1

            state, ucostheta = cavity.rng_uniform_float(state)
            cos_theta = 2.0 * ucostheta - 1.0
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

            state, uphi = cavity.rng_uniform_float(state)
            phi = 2.0 * math.pi * uphi

            vcmx = 0.5 * (vx[p1] + vx[p2])
            vcmy = 0.5 * (vy[p1] + vy[p2])
            vcmz = 0.5 * (vz[p1] + vz[p2])

            dgx = 0.5 * g * sin_theta * math.cos(phi)
            dgy = 0.5 * g * sin_theta * math.sin(phi)
            dgz = 0.5 * g * cos_theta

            vx[p1] = vcmx + dgx
            vy[p1] = vcmy + dgy
            vz[p1] = vcmz + dgz

            vx[p2] = vcmx - dgx
            vy[p2] = vcmy - dgy
            vz[p2] = vcmz - dgz

    sigma_g_majorant[cid] = sig_g_max
    cell_rng_states[cid] = state
    candidate_pair_count[cid] += n_pairs
    accepted_collision_count[cid] += accepted
    majorant_update_count[cid] += updates


def arguments():
    parser = argparse.ArgumentParser(
        description="Hard-sphere DSMC planar-Fourier validation"
    )
    parser.add_argument("--kn", type=float, default=0.02)
    parser.add_argument("--t-left", type=float, default=1020.0)
    parser.add_argument("--t-right", type=float, default=980.0)
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--particles", type=int, default=2_000_000)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--sample-start", type=int, default=100_000)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--majorant-factor",
        type=float,
        default=6.0,
        help="Initial g_max divided by sqrt(kB*T_left/m)",
    )
    parser.add_argument("--output", default="fourier_hardsphere")
    return parser.parse_args()


def main():
    args = arguments()
    if args.t_left <= args.t_right:
        raise ValueError("Require T_left > T_right")
    if args.kn <= 0.0 or args.cells < 20 or args.particles < 2:
        raise ValueError("Require Kn>0, at least 20 cells, and particles>=2")
    if not 0 <= args.sample_start < args.steps:
        raise ValueError("sample-start must be in [0,steps)")
    if args.sample_every < 1 or args.majorant_factor <= 0.0:
        raise ValueError("sample-every and majorant-factor must be positive")

    mass = cavity.MASS
    kb = cavity.KB
    sigma = cavity.SIGMA
    length = cavity.L
    nc = args.cells
    npart = args.particles
    dx = length / nc
    volume = dx  # unit cross-sectional area

    vref = math.sqrt(2.0 * kb * args.t_left / mass)
    dt = 0.15 * dx / vref
    n0 = 1.0 / (math.sqrt(2.0) * sigma * args.kn * length)
    fnum = n0 * length / npart

    threads = cavity.THREADS
    particle_blocks = (npart + threads - 1) // threads
    cell_blocks = (nc + threads - 1) // threads

    particle_seeds = (
        cp.arange(npart, dtype=cp.uint64)
        * cp.uint64(2862933555777941757)
        + cp.uint64(args.seed)
    )
    cell_seed_offset = (int(args.seed) ^ 0xD1B54A32D192ED03) & (
        (1 << 64) - 1
    )
    cell_rng_states = (
        cp.arange(nc, dtype=cp.uint64)
        * cp.uint64(2862933555777941757)
        + cp.uint64(cell_seed_offset)
    )

    x = cp.zeros(npart, dtype=cp.float64)
    vx = cp.zeros(npart, dtype=cp.float64)
    vy = cp.zeros_like(vx)
    vz = cp.zeros_like(vx)
    common.init_fourier[particle_blocks, threads](
        x,
        vx,
        vy,
        vz,
        particle_seeds,
        length,
        0.5 * (args.t_left + args.t_right),
        mass,
        kb,
    )

    accumulators = [cp.zeros(nc, dtype=cp.float64) for _ in range(9)]
    wall_energy_into_gas = cp.zeros(2, dtype=cp.float64)
    # Float64 is used because many particle threads update these two entries
    # atomically; the largest production count remains exactly representable.
    wall_hit_count = cp.zeros(2, dtype=cp.float64)

    initial_g_majorant = (
        args.majorant_factor * math.sqrt(kb * args.t_left / mass)
    )
    sigma_g_majorant = cp.full(
        nc, sigma * initial_g_majorant, dtype=cp.float64
    )
    candidate_pair_count = cp.zeros(nc, dtype=cp.int64)
    accepted_collision_count = cp.zeros(nc, dtype=cp.int64)
    majorant_update_count = cp.zeros(nc, dtype=cp.int64)

    print(
        f"Hard-sphere DSMC Fourier: Kn={args.kn:g}, seed={args.seed}, "
        f"N={npart}, cells={nc}"
    )
    print(
        f"dt={dt:.6e} s, steps={args.steps}, sample-start="
        f"{args.sample_start}, sample-every={args.sample_every}"
    )
    print(
        f"n0={n0:.6e} 1/m^3, FNUM={fnum:.6e}, "
        f"initial g-majorant={initial_g_majorant:.6e} m/s"
    )

    start_time = time.time()
    number_of_profile_samples = 0

    for step in range(args.steps):
        collect_wall_flux = 1 if step >= args.sample_start else 0
        move_fourier_with_wall_audit[particle_blocks, threads](
            x,
            vx,
            vy,
            vz,
            particle_seeds,
            length,
            dt,
            args.t_left,
            args.t_right,
            mass,
            kb,
            collect_wall_flux,
            wall_energy_into_gas,
            wall_hit_count,
        )

        cell = cp.clip((x / dx).astype(cp.int32), 0, nc - 1)
        order = cp.argsort(cell)
        x = x[order]
        vx = vx[order]
        vy = vy[order]
        vz = vz[order]
        particle_seeds = particle_seeds[order]

        sorted_cell = cell[order]
        counts = cp.bincount(sorted_cell, minlength=nc).astype(cp.int32)
        starts = cp.zeros_like(counts)
        cp.cumsum(counts[:-1], out=starts[1:])

        collision_hard_sphere_ntc_cellwise[cell_blocks, threads](
            vx,
            vy,
            vz,
            starts,
            counts,
            sigma_g_majorant,
            dt,
            volume,
            fnum,
            cell_rng_states,
            sigma,
            candidate_pair_count,
            accepted_collision_count,
            majorant_update_count,
        )

        if (
            step >= args.sample_start
            and (step - args.sample_start) % args.sample_every == 0
        ):
            common.sample_fourier[particle_blocks, threads](
                x,
                vx,
                vy,
                vz,
                *accumulators,
                length,
                nc,
            )
            number_of_profile_samples += 1

        if step % 10_000 == 0:
            print(f"HARD_SPHERE_DSMC: step {step}/{args.steps}", flush=True)

    (
        sn,
        sx,
        sy,
        sz,
        sv2,
        sv2x,
        sxx,
        syx,
        szx,
    ) = [cp.asnumpy(value) for value in accumulators]
    sn = np.maximum(sn, 1.0)

    ux = sx / sn
    uy = sy / sn
    uz = sz / sn
    v2 = sv2 / sn

    c2cx = (
        sv2x / sn
        - ux * v2
        - 2.0
        * (
            ux * (sxx / sn)
            + uy * (syx / sn)
            + uz * (szx / sn)
        )
        + 2.0 * ux * (ux * ux + uy * uy + uz * uz)
    )
    number_density = (
        sn / max(number_of_profile_samples, 1) * fnum / volume
    )
    qx = 0.5 * mass * number_density * c2cx
    temperature = mass / (3.0 * kb) * np.maximum(
        v2 - (ux * ux + uy * uy + uz * uz), 0.0
    )

    x_over_l = (np.arange(nc) + 0.5) / nc
    interior = (x_over_l >= 0.2) & (x_over_l <= 0.8)

    qx_mean = float(np.mean(qx[interior]))
    qx_spatial_std = float(np.std(qx[interior], ddof=1))
    qx_spatial_cv = qx_spatial_std / max(abs(qx_mean), 1.0e-300)
    qx_slope = float(np.polyfit(x_over_l[interior], qx[interior], 1)[0])
    qx_relative_change_across_interior = (
        qx_slope * 0.6 / max(abs(qx_mean), 1.0e-300)
    )

    temperature_slope = float(
        np.polyfit(x_over_l[interior], temperature[interior], 1)[0]
    )

    c_hs, continuum_qx = hard_sphere_fourier_reference(
        args.t_left, args.t_right, length
    )
    continuum_temperature = (
        args.t_left**1.5
        + (args.t_right**1.5 - args.t_left**1.5) * x_over_l
    ) ** (2.0 / 3.0)

    interior_temperature_mean = float(np.mean(temperature[interior]))
    reference_conductivity = c_hs * math.sqrt(interior_temperature_mean)
    measured_conductivity = -qx_mean * length / temperature_slope
    conductivity_relative_error = (
        abs(measured_conductivity - reference_conductivity)
        / reference_conductivity
    )
    temperature_rmse = float(
        np.sqrt(
            np.mean(
                (
                    temperature[interior]
                    - continuum_temperature[interior]
                )
                ** 2
            )
        )
    )

    sampled_wall_time = (args.steps - args.sample_start) * dt
    wall_energy = cp.asnumpy(wall_energy_into_gas)
    wall_hits = cp.asnumpy(wall_hit_count)
    wall_q_into_gas = wall_energy * fnum / sampled_wall_time
    q_hot_wall = float(wall_q_into_gas[0])
    q_cold_wall = float(wall_q_into_gas[1])
    q_wall_balanced = 0.5 * (q_hot_wall - q_cold_wall)
    wall_energy_imbalance = abs(q_hot_wall + q_cold_wall) / max(
        abs(q_hot_wall) + abs(q_cold_wall), 1.0e-300
    )

    candidate_pairs = int(cp.asnumpy(candidate_pair_count).sum())
    accepted_collisions = int(
        cp.asnumpy(accepted_collision_count).sum()
    )
    majorant_updates = int(cp.asnumpy(majorant_update_count).sum())
    final_sigma_g = cp.asnumpy(sigma_g_majorant)

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.join(
        args.output,
        f"Fourier_HARD_SPHERE_DSMC_Kn{args.kn:g}_seed{args.seed}",
    )

    with open(stem + "_profile.csv", "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "x_over_L",
                "T_K",
                "T_continuum_K",
                "ux_mps",
                "qx_W_m2",
                "qx_continuum_W_m2",
            )
        )
        writer.writerows(
            zip(
                x_over_l,
                temperature,
                continuum_temperature,
                ux,
                qx,
                np.full_like(x_over_l, continuum_qx),
            )
        )

    metrics = {
        "solver_version": SOLVER_VERSION,
        "mode": "HARD_SPHERE_DSMC",
        "Kn_paper": args.kn,
        "T_left_K": args.t_left,
        "T_right_K": args.t_right,
        "interior_mean_qx_W_m2": qx_mean,
        "interior_spatial_std_qx_W_m2": qx_spatial_std,
        "interior_spatial_cv_qx": qx_spatial_cv,
        "interior_qx_slope_W_m2_per_x_over_L": qx_slope,
        "interior_qx_relative_change_over_0p2_to_0p8": (
            qx_relative_change_across_interior
        ),
        "hot_wall_energy_flux_into_gas_W_m2": q_hot_wall,
        "cold_wall_energy_flux_into_gas_W_m2": q_cold_wall,
        "wall_balanced_heat_flux_W_m2": q_wall_balanced,
        "wall_energy_imbalance_relative": wall_energy_imbalance,
        "wall_hit_count_hot": int(round(float(wall_hits[0]))),
        "wall_hit_count_cold": int(round(float(wall_hits[1]))),
        "continuum_analytic_qx_W_m2": continuum_qx,
        "relative_error_bulk_qx_vs_continuum": (
            abs(qx_mean - continuum_qx) / abs(continuum_qx)
        ),
        "relative_error_wall_qx_vs_continuum": (
            abs(q_wall_balanced - continuum_qx) / abs(continuum_qx)
        ),
        "interior_temperature_slope_K_per_x_over_L": (
            temperature_slope
        ),
        "measured_conductivity_W_m_K_from_bulk_qx_and_interior_slope": (
            measured_conductivity
        ),
        "reference_hard_sphere_conductivity_W_m_K_at_interior_mean_T": (
            reference_conductivity
        ),
        "relative_error_conductivity": conductivity_relative_error,
        "interior_temperature_RMSE_vs_continuum_K": temperature_rmse,
        "interior_mean_ux_mps": float(np.mean(ux[interior])),
        "interior_rms_ux_mps": float(
            np.sqrt(np.mean(ux[interior] ** 2))
        ),
        "particles": npart,
        "cells": nc,
        "steps": args.steps,
        "seed": args.seed,
        "profile_samples": number_of_profile_samples,
        "wall_flux_sampled_steps": args.steps - args.sample_start,
        "wall_clock_seconds": time.time() - start_time,
        "dt_seconds": dt,
        "n0_per_m3": n0,
        "FNUM": fnum,
        "kn_definition": "1/(sqrt(2)*n0*pi*d^2*L)",
        "collision_model": "hard-sphere no-time-counter DSMC",
        "hard_sphere_reference": (
            "first Chapman-Enskog: "
            "k=75*pi*kB/(64*sigma)*sqrt(kB*T/(pi*m))"
        ),
        "wall_advance_scheme": (
            "event-driven time-of-impact with diffuse reflection "
            "and post-collision residual-time advance"
        ),
        "quantitative_profiles_are_unfiltered": True,
        "dsmc_diagnostics": {
            "candidate_pairs": candidate_pairs,
            "accepted_collisions": accepted_collisions,
            "acceptance_fraction": (
                accepted_collisions / max(candidate_pairs, 1)
            ),
            "majorant_updates": majorant_updates,
            "initial_g_majorant_mps": initial_g_majorant,
            "final_sigma_g_majorant_min": float(
                np.min(final_sigma_g)
            ),
            "final_sigma_g_majorant_mean": float(
                np.mean(final_sigma_g)
            ),
            "final_sigma_g_majorant_max": float(
                np.max(final_sigma_g)
            ),
        },
    }

    with open(stem + "_metrics.json", "w") as output_file:
        json.dump(metrics, output_file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[OK] Saved hard-sphere Fourier outputs: {stem}")


if __name__ == "__main__":
    main()
