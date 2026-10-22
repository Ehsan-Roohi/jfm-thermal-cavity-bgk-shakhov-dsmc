#!/usr/bin/env python3
"""Direct homogeneous heat-flux relaxation test of the production kernels.

For the stochastic relaxation probability P=1-exp(-nu*dt), the exact
single-step heat-flux factors are 1-P for BGK and 1-Pr*P for Shakhov.  This
test has no walls or spatial gradient, so it isolates the Shakhov coefficient
and its accept/reject implementation from the cavity and Fourier sampling.
"""

import argparse
import csv
import json
import math
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

# Resolve the recovered shared solver when launched from any directory.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solver"))

import JFM_reviewer_quarter_domain as cavity

SOLVER_VERSION = "2026-07-22-homogeneous-shakhov-v1"


@cuda.jit(fastmath=True)
def init_heat_flux_mixture(vx, vy, vz, seeds, t0, delta_t, drift, mass, kb):
    i = cuda.grid(1)
    if i < vx.size:
        state = seeds[i]
        if i % 2 == 0:
            temperature = t0*(1.0+delta_t)
            offset = drift
        else:
            temperature = t0*(1.0-delta_t)
            offset = -drift
        scale = math.sqrt(kb*temperature/mass)
        state, gx = cavity.rng_normal_float(state)
        state, gy = cavity.rng_normal_float(state)
        state, gz = cavity.rng_normal_float(state)
        vx[i] = offset + scale*gx
        vy[i] = scale*gy
        vz[i] = scale*gz
        seeds[i] = state


def global_heat_flux(vx, vy, vz, mass, number_density):
    ux = cp.mean(vx); uy = cp.mean(vy); uz = cp.mean(vz)
    cx = vx-ux; cy = vy-uy; cz = vz-uz
    c2 = cx*cx+cy*cy+cz*cz
    return float(cp.asnumpy(0.5*mass*number_density*cp.mean(c2*cx)))


def arguments():
    parser = argparse.ArgumentParser(
        description="homogeneous BGK/Shakhov heat-flux relaxation")
    parser.add_argument("--mode", choices=("BGK", "SHAKHOV"), required=True)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--particles-per-cell", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--nu-dt", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=1000.0)
    parser.add_argument("--temperature-asymmetry", type=float, default=0.15)
    parser.add_argument("--drift-over-sqrt-theta", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="homogeneous_validation")
    return parser.parse_args()


def main():
    a = arguments()
    if a.cells < 1 or a.particles_per_cell < 128 or a.steps < 10:
        raise ValueError("increase cells, particles-per-cell, and steps")
    if not (0.0 < a.nu_dt < 0.2):
        raise ValueError("use 0 < nu-dt < 0.2")
    m = cavity.MASS; kb = cavity.KB; sigma = cavity.SIGMA
    n0 = 1.0e20
    theta0 = kb*a.temperature/m
    drift = a.drift_over_sqrt_theta*math.sqrt(theta0)
    # The symmetric mixture has this nominal kinetic temperature.
    t_nominal = a.temperature + m*drift*drift/(3.0*kb)
    nu0 = n0*sigma*math.sqrt(math.pi*kb*t_nominal/m)
    dt = a.nu_dt/nu0
    probability = 1.0-math.exp(-nu0*dt)
    expected_factor = ((1.0-probability) if a.mode == "BGK"
                       else (1.0-cavity.PRANDTL*probability))
    expected_rate = -math.log(expected_factor)/dt

    nc = a.cells; ppc = a.particles_per_cell; npart = nc*ppc
    starts = cp.arange(nc, dtype=cp.int32)*np.int32(ppc)
    counts = cp.full(nc, ppc, dtype=cp.int32)
    # Unit cell volume and FNUM chosen so every cell has number density n0.
    volume = 1.0
    fnum = n0*volume/ppc
    seeds = (cp.arange(npart, dtype=cp.uint64)
             * cp.uint64(2862933555777941757)+cp.uint64(a.seed))
    vx = cp.empty(npart, dtype=cp.float64)
    vy = cp.empty_like(vx); vz = cp.empty_like(vx)
    threads = cavity.THREADS
    bp = (npart+threads-1)//threads; bc = (nc+threads-1)//threads
    init_heat_flux_mixture[bp, threads](
        vx, vy, vz, seeds, a.temperature, a.temperature_asymmetry,
        drift, m, kb
    )
    diag = [cp.zeros(nc, dtype=cp.int64) for _ in range(4)]
    q = [global_heat_flux(vx, vy, vz, m, n0)]
    for _ in range(a.steps):
        if a.mode == "BGK":
            cavity.collision_bgk_cellwise[bc, threads](
                vx, vy, vz, starts, counts, dt, volume, fnum, seeds,
                m, kb, sigma
            )
        else:
            cavity.collision_shakhov_cellwise[bc, threads](
                vx, vy, vz, starts, counts, dt, volume, fnum, seeds,
                m, kb, sigma, np.float32(cavity.PRANDTL),
                np.float32(cavity.SHAK_W_CLAMP), cavity.SHAK_MAX_TRIALS,
                *diag
            )
        q.append(global_heat_flux(vx, vy, vz, m, n0))

    times = np.arange(a.steps+1)*dt
    q = np.asarray(q)
    qnorm = q/q[0]
    # Fit only above the stochastic noise floor and omit the initialization.
    mask = (np.arange(q.size) >= 1) & (np.abs(qnorm) >= 0.15)
    if np.count_nonzero(mask) < 8:
        raise RuntimeError("too few points above the heat-flux noise floor")
    slope, _ = np.polyfit(times[mask], np.log(np.abs(qnorm[mask])), 1)
    measured_rate = -float(slope)
    relative_error = abs(measured_rate-expected_rate)/expected_rate

    os.makedirs(a.output, exist_ok=True)
    stem = os.path.join(a.output,
                        f"Homogeneous_{a.mode}_seed{a.seed}")
    with open(stem+"_history.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "time_s", "qx_W_m2", "qx_over_qx0",
                         "expected_qx_over_qx0"))
        for step, (tt, qq, qn) in enumerate(zip(times, q, qnorm)):
            writer.writerow((step, tt, qq, qn, expected_factor**step))
    metrics = {
        "solver_version": SOLVER_VERSION,
        "production_kernel_version": cavity.SOLVER_VERSION,
        "mode": a.mode,
        "seed": a.seed,
        "cells": nc,
        "particles_per_cell": ppc,
        "particles": npart,
        "steps": a.steps,
        "nu_dt": a.nu_dt,
        "selection_probability": probability,
        "Prandtl": cavity.PRANDTL if a.mode == "SHAKHOV" else 1.0,
        "exact_discrete_heat_flux_factor_per_step": expected_factor,
        "expected_relaxation_rate_per_s": expected_rate,
        "measured_relaxation_rate_per_s": measured_rate,
        "relative_error_relaxation_rate": relative_error,
        "pass_8_percent_rate_tolerance": relative_error <= 0.08,
        "initial_qx_W_m2": float(q[0]),
        "fit_points": int(np.count_nonzero(mask)),
    }
    if a.mode == "SHAKHOV":
        names = ("selected_relaxations", "negative_weight_candidates",
                 "above_limiter_candidates", "max_trial_fallbacks")
        metrics["shakhov_sampling_diagnostics"] = {
            name: int(cp.asnumpy(arr).sum())
            for name, arr in zip(names, diag)
        }
    with open(stem+"_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    plt.figure(figsize=(7.2, 4.8))
    plt.semilogy(times, np.abs(qnorm), label="production kernel")
    plt.semilogy(times, expected_factor**np.arange(q.size), "k--",
                 label="exact discrete target")
    plt.xlabel("time (s)"); plt.ylabel(r"$|q_x/q_x(0)|$")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(stem+"_relaxation.png", dpi=220)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
