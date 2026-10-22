# -*- coding: utf-8 -*-
"""Independent BGK/Shakhov validation requested by Referee 1.

Planar Fourier flow between stationary diffuse walls is simulated with a
small temperature difference.  In the near-continuum limit and with the same
relaxation frequency, Shakhov (Pr=2/3) must conduct approximately 1/Pr=1.5
times the BGK heat flux.  Run this file once with --mode BGK and once with
--mode SHAKHOV, then compare interior_mean_qx.

The Kn definition and relaxation frequency are exactly those used by
JFM_reviewer_quarter_domain.py and by the paper.
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

SOLVER_VERSION = "2026-07-22-fourier-shakhov-v2"


def continuum_fourier_reference(t_left, t_right, length, mode):
    """Hard-sphere BGK/Shakhov Fourier-law reference for this code's nu.

    With nu=n*sigma*sqrt(pi*k_B*T/m), the BGK conductivity is
    k_BGK(T)=C*sqrt(T), C=5*k_B/(2*sigma)*sqrt(k_B/(pi*m)).
    Shakhov multiplies it by 1/Pr.
    """
    c_bgk = (5.0*cavity.KB/(2.0*cavity.SIGMA)
             * math.sqrt(cavity.KB/(math.pi*cavity.MASS)))
    pr_model = 1.0 if mode == "BGK" else cavity.PRANDTL
    q_reference = (2.0*c_bgk/(3.0*length*pr_model)
                   * (t_left**1.5 - t_right**1.5))
    return c_bgk, pr_model, q_reference


@cuda.jit(fastmath=True)
def init_fourier(x, vx, vy, vz, seeds, length, temperature, mass, kb):
    i = cuda.grid(1)
    if i < x.size:
        state = seeds[i]
        state, u = cavity.rng_uniform_float(state)
        x[i] = u * length
        s = math.sqrt(kb * temperature / mass)
        state, gx = cavity.rng_normal_float(state)
        state, gy = cavity.rng_normal_float(state)
        state, gz = cavity.rng_normal_float(state)
        vx[i] = s * gx
        vy[i] = s * gy
        vz[i] = s * gz
        seeds[i] = state


@cuda.jit(fastmath=True)
def move_fourier(x, vx, vy, vz, seeds, length, dt,
                 t_left, t_right, mass, kb):
    """Advance to each diffuse-wall hit, then use the new velocity for the
    residual time.  Mirroring a diffuse hit as though it were specular biases
    the first-cell velocity and heat flux, especially in a low-signal run.
    """
    i = cuda.grid(1)
    if i < x.size:
        state = seeds[i]
        xp = x[i]
        ax = vx[i]
        ay = vy[i]
        az = vz[i]
        remaining = dt
        eps = 1.0e-12 * length

        # The CFL used here makes more than one hit extraordinarily unlikely,
        # but the bounded loop also handles pathological high-speed particles.
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
            if side < 0:
                xp = eps
                state, ax, ay, az = cavity.sample_diffuse_velocity(
                    state, t_left, mass, kb, 1.0, 0.0
                )
            else:
                xp = length - eps
                state, ax, ay, az = cavity.sample_diffuse_velocity(
                    state, t_right, mass, kb, -1.0, 0.0
                )

        if remaining > 0.0:
            xp += ax * remaining
        if xp < 0.0:
            xp = eps
        elif xp >= length:
            xp = length - eps

        x[i] = xp
        vx[i], vy[i], vz[i] = ax, ay, az
        seeds[i] = state


@cuda.jit(fastmath=True)
def sample_fourier(x, vx, vy, vz, sn, sx, sy, sz, sv2, sv2x,
                   sxx, syx, szx, length, nc):
    i = cuda.grid(1)
    if i < x.size:
        ix = int(x[i] / (length / nc))
        if 0 <= ix < nc:
            ax = float(vx[i]); ay = float(vy[i]); az = float(vz[i])
            v2 = ax*ax + ay*ay + az*az
            cuda.atomic.add(sn, ix, 1.0)
            cuda.atomic.add(sx, ix, ax)
            cuda.atomic.add(sy, ix, ay)
            cuda.atomic.add(sz, ix, az)
            cuda.atomic.add(sv2, ix, v2)
            cuda.atomic.add(sv2x, ix, v2*ax)
            cuda.atomic.add(sxx, ix, ax*ax)
            cuda.atomic.add(syx, ix, ay*ax)
            cuda.atomic.add(szx, ix, az*ax)


def arguments():
    p = argparse.ArgumentParser(description="BGK/Shakhov planar-Fourier validation")
    p.add_argument("--mode", choices=("BGK", "SHAKHOV"), required=True)
    p.add_argument("--kn", type=float, default=0.02)
    p.add_argument("--t-left", type=float, default=1020.0)
    p.add_argument("--t-right", type=float, default=980.0)
    p.add_argument("--cells", type=int, default=200)
    p.add_argument("--particles", type=int, default=1000000)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--sample-start", type=int, default=50000)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="fourier_validation")
    return p.parse_args()


def main():
    a = arguments()
    if a.t_left <= a.t_right or a.kn <= 0 or a.cells < 20:
        raise ValueError("Require T_left>T_right, Kn>0, and at least 20 cells")
    if not 0 <= a.sample_start < a.steps:
        raise ValueError("sample-start must be in [0,steps)")

    m = cavity.MASS; kb = cavity.KB; sigma = cavity.SIGMA; length = cavity.L
    nc = a.cells; npart = a.particles; dx = length/nc; vol = dx
    thot = a.t_left
    vref = math.sqrt(2.0*kb*thot/m)
    dt = 0.15*dx/vref
    n0 = 1.0/(math.sqrt(2.0)*sigma*a.kn*length)
    fnum = n0*length/npart
    threads = cavity.THREADS
    bp = (npart+threads-1)//threads; bc = (nc+threads-1)//threads

    seeds = (cp.arange(npart, dtype=cp.uint64)*cp.uint64(2862933555777941757)
             + cp.uint64(a.seed))
    x = cp.zeros(npart, cp.float64)
    vx = cp.zeros(npart, cp.float64); vy = cp.zeros_like(vx); vz = cp.zeros_like(vx)
    init_fourier[bp, threads](x, vx, vy, vz, seeds, length,
                              0.5*(a.t_left+a.t_right), m, kb)

    accum = [cp.zeros(nc, cp.float64) for _ in range(9)]
    shakh_diag = [cp.zeros(nc, cp.int64) for _ in range(4)]
    t0 = time.time(); nsamples = 0
    for step in range(a.steps):
        move_fourier[bp, threads](x, vx, vy, vz, seeds, length, dt,
                                  a.t_left, a.t_right, m, kb)
        cell = cp.clip((x/dx).astype(cp.int32), 0, nc-1)
        order = cp.argsort(cell)
        x=x[order]; vx=vx[order]; vy=vy[order]; vz=vz[order]; seeds=seeds[order]
        cs = cell[order]
        counts = cp.bincount(cs, minlength=nc).astype(cp.int32)
        starts = cp.zeros_like(counts); cp.cumsum(counts[:-1], out=starts[1:])

        if a.mode == "BGK":
            cavity.collision_bgk_cellwise[bc, threads](
                vx, vy, vz, starts, counts, dt, vol, fnum, seeds, m, kb, sigma
            )
        else:
            cavity.collision_shakhov_cellwise[bc, threads](
                vx, vy, vz, starts, counts, dt, vol, fnum, seeds, m, kb,
                sigma, np.float32(cavity.PRANDTL),
                np.float32(cavity.SHAK_W_CLAMP), cavity.SHAK_MAX_TRIALS,
                *shakh_diag
            )

        if step >= a.sample_start and (step-a.sample_start) % a.sample_every == 0:
            sample_fourier[bp, threads](x, vx, vy, vz, *accum, length, nc)
            nsamples += 1
        if step % 10000 == 0:
            print(f"{a.mode}: step {step}/{a.steps}", flush=True)

    sn,sx,sy,sz,sv2,sv2x,sxx,syx,szx = [cp.asnumpy(v) for v in accum]
    sn = np.maximum(sn, 1.0)
    ux=sx/sn; uy=sy/sn; uz=sz/sn; v2=sv2/sn
    # Exact raw-moment identity for <|v-u|^2 (vx-ux)>.
    c2cx = (sv2x/sn - ux*v2
            - 2.0*(ux*(sxx/sn) + uy*(syx/sn) + uz*(szx/sn))
            + 2.0*ux*(ux*ux + uy*uy + uz*uz))
    n = (sn/max(nsamples,1))*fnum/vol
    qx = 0.5*m*n*c2cx
    temp = m/(3.0*kb)*np.maximum(v2-(ux*ux+uy*uy+uz*uz), 0.0)
    xc = (np.arange(nc)+0.5)/nc
    interior = (xc >= 0.2) & (xc <= 0.8)
    qmean = float(np.mean(qx[interior]))
    qstd = float(np.std(qx[interior], ddof=1))
    qcv = qstd/max(abs(qmean), 1.0e-300)

    c_bgk, pr_model, q_reference = continuum_fourier_reference(
        a.t_left, a.t_right, length, a.mode
    )
    t_reference = (a.t_left**1.5
                   + (a.t_right**1.5-a.t_left**1.5)*xc)**(2.0/3.0)
    slope_k_per_x_over_l = float(np.polyfit(xc[interior], temp[interior], 1)[0])
    conductivity_measured = -qmean*length/slope_k_per_x_over_l
    t_mean_interior = float(np.mean(temp[interior]))
    conductivity_reference_at_tmean = (c_bgk*math.sqrt(t_mean_interior)
                                       / pr_model)
    t_rmse = float(np.sqrt(np.mean((temp[interior]
                                   - t_reference[interior])**2)))

    os.makedirs(a.output, exist_ok=True)
    stem = os.path.join(a.output,
                        f"Fourier_{a.mode}_Kn{a.kn:g}_seed{a.seed}")
    with open(stem+"_profile.csv", "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(("x_over_L", "T_K", "T_continuum_K", "ux_mps",
                    "qx_W_m2", "qx_continuum_W_m2"))
        w.writerows(zip(xc, temp, t_reference, ux, qx,
                        np.full_like(xc, q_reference)))
    metrics = {
        "solver_version": SOLVER_VERSION,
        "mode": a.mode, "Kn_paper": a.kn,
        "T_left_K": a.t_left, "T_right_K": a.t_right,
        "interior_mean_qx_W_m2": qmean,
        "interior_spatial_std_qx_W_m2": qstd,
        "interior_spatial_cv_qx": qcv,
        "continuum_analytic_qx_W_m2": q_reference,
        "relative_error_qx_vs_continuum": abs(qmean-q_reference)/abs(q_reference),
        "interior_temperature_slope_K_per_x_over_L": slope_k_per_x_over_l,
        "measured_conductivity_W_m_K_from_interior_slope": conductivity_measured,
        "reference_conductivity_W_m_K_at_interior_mean_T": conductivity_reference_at_tmean,
        "relative_error_conductivity": abs(conductivity_measured
                                           - conductivity_reference_at_tmean)
                                           / conductivity_reference_at_tmean,
        "interior_temperature_RMSE_vs_continuum_K": t_rmse,
        "expected_q_shakhov_over_q_bgk_near_continuum": 1.5,
        "particles": npart, "cells": nc, "steps": a.steps,
        "seed": a.seed,
        "samples": nsamples, "wall_clock_seconds": time.time()-t0,
        "dt_seconds": dt, "n0_per_m3": n0, "FNUM": fnum,
        "wall_advance_scheme": "event-driven time-of-impact with post-collision residual-time advance",
        "quantitative_fields_are_unfiltered": True,
        "kn_definition": "1/(sqrt(2)*n0*pi*d^2*L)",
        "relaxation_frequency": "n*pi*d^2*sqrt(pi*kB*T/m)"
    }
    if a.mode == "SHAKHOV":
        names = ("selected_relaxations", "negative_weight_candidates",
                 "above_limiter_candidates", "max_trial_fallbacks")
        metrics["shakhov_sampling_diagnostics"] = {
            name: int(cp.asnumpy(arr).sum()) for name, arr in zip(names, shakh_diag)
        }
    with open(stem+"_metrics.json", "w") as f:
        json.dump(metrics,f,indent=2)
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
