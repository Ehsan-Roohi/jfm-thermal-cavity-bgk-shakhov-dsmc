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

import JFM_reviewer_quarter_domain as cavity


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
def move_fourier(x, vx, vy, vz, seeds, length, dt, t_left, t_right, mass, kb):
    i = cuda.grid(1)
    if i < x.size:
        state = seeds[i]
        xn = x[i] + vx[i] * dt
        if xn < 0.0:
            xn = -xn
            state, vxn, vyn, vzn = cavity.sample_diffuse_velocity(
                state, t_left, mass, kb, 1.0, 0.0
            )
            vx[i], vy[i], vz[i] = vxn, vyn, vzn
        elif xn > length:
            xn = 2.0 * length - xn
            state, vxn, vyn, vzn = cavity.sample_diffuse_velocity(
                state, t_right, mass, kb, -1.0, 0.0
            )
            vx[i], vy[i], vz[i] = vxn, vyn, vzn
        x[i] = xn
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

    seeds = cp.arange(npart, dtype=cp.uint64)*cp.uint64(2862933555777941757)+cp.uint64(42)
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

    os.makedirs(a.output, exist_ok=True)
    stem = os.path.join(a.output, f"Fourier_{a.mode}_Kn{a.kn:g}")
    with open(stem+"_profile.csv", "w", newline="") as f:
        w=csv.writer(f); w.writerow(("x_over_L","T_K","ux_mps","qx_W_m2"))
        w.writerows(zip(xc,temp,ux,qx))
    metrics = {
        "mode": a.mode, "Kn_paper": a.kn,
        "T_left_K": a.t_left, "T_right_K": a.t_right,
        "interior_mean_qx_W_m2": qmean,
        "expected_q_shakhov_over_q_bgk_near_continuum": 1.5,
        "particles": npart, "cells": nc, "steps": a.steps,
        "samples": nsamples, "wall_clock_seconds": time.time()-t0,
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
