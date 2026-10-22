# -*- coding: utf-8 -*-
"""
JFM revision production solver: one-quarter thermal cavity.

The simulated quadrant is x,y in [0,L/2].  The planes x=0 and y=0 are
specular symmetry boundaries; x=L/2 is the cold wall and y=L/2 is the hot
wall.  Full-domain fields are reconstructed using the exact cavity symmetries.

Most importantly, KN_PAPER is exactly the Knudsen number defined in the paper,

    Kn = 1 / (sqrt(2) * n0 * pi*d^2 * L).

The BGK/Shakhov relaxation frequency is calibrated directly to the paper:

    nu = n * pi*d^2 * sqrt(pi*kB*T/m),

so the same input Kn is used for HS-DSMC, BGK, and Shakhov.  There is no
hidden 16/5 conversion (e.g. 30 -> 96).

Diffuse wall hits are advanced by time of impact: after sampling the new wall
velocity, the particle moves through the residual part of the time step with
that new velocity.  This avoids the first-cell normal-velocity bias produced by
using a specular position mirror for a diffuse collision.

Each invocation runs one (model, Kn, RT) case and writes unfiltered,
nondimensional full-domain fields plus a metrics JSON file. Gaussian filtering
is used only for streamline placement in the preview PNG, never for data or
metrics.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numba import cuda
import math
import argparse
import json
import os
import time

SOLVER_VERSION = "2026-07-22-wallfix-shakhov-v3"

# =============================================================================
# 0) Plot settings
# =============================================================================
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
})

# =============================================================================
# 1) Physical / numerical configuration
# =============================================================================
MASS = 6.63e-26          # Argon (kg)
KB   = 1.380649e-23      # J/K
D_REF = 4.17e-10         # hard-sphere diameter (m)
SIGMA = math.pi * D_REF**2

L = 1.0e-3               # cavity size (m)

T_HOT  = 1000.0
T_COLD = 200.0

# One reviewer case per invocation. These defaults target the noisy Kn=30
# figure. Command-line arguments can change them without editing the file.
KN_PAPER = 30.0
MODE = "DSMC"

# 100 x 100 cells in the quadrant gives the same dx as 200 x 200 in the full
# cavity. Keeping the full-domain particle count gives four times more PPC.
NC     = 100
N_PART = 4000000

# ---- Longer run (requested)
N_STEPS    = 300000
SAMP_START = 60000
SAMP_EVERY = 2

LQ = 0.5 * L
DX = LQ / NC
CELL_VOL = DX * DX * 1.0

VTH_HOT = math.sqrt(2.0 * KB * T_HOT / MASS)
DT_BASE = 0.15 * DX / (VTH_HOT + 1e-30)  # can be used directly; dt does not depend on Kn here

THREADS = 128

# Shakhov parameters
PRANDTL = 2.0/3.0        # monatomic gas target (Shakhov fixes Pr)
SHAK_W_CLAMP = 2.0       # positivity limiter/envelope; reported in metadata
SHAK_MAX_TRIALS = 32     # fallback probability is negligible near equilibrium

# Used only for preview streamlines, not quantitative output.
# The Kn=30 velocity is much weaker than the thermal molecular speed.  A
# moderate spatial average is therefore used for streamline placement only;
# the exported fields and every quantitative metric remain unfiltered.
SMOOTH_SIGMA_UV = 6.0
SMOOTH_SIGMA_T  = 0.9
SMOOTH_SIGMA_N  = 1.2

# =============================================================================
# 2) Robust RNG on GPU (xorshift64* + Box-Muller)
# =============================================================================
@cuda.jit(device=True, inline=True)
def xorshift64star(state):
    x = state
    x ^= (x >> 12)
    x ^= (x << 25) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> 27)
    x = (x * np.uint64(2685821657736338717)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x

@cuda.jit(device=True, inline=True)
def rng_uniform_float(state):
    state = xorshift64star(state)
    u = np.float32(((state >> np.uint64(40)) & np.uint64(0xFFFFFF)) / np.float32(16777216.0))
    if u < 1e-7:
        u = 1e-7
    return state, u

@cuda.jit(device=True, inline=True)
def rng_normal_float(state):
    state, u1 = rng_uniform_float(state)
    state, u2 = rng_uniform_float(state)
    r = math.sqrt(-2.0 * math.log(u1))
    th = 2.0 * math.pi * u2
    z = r * math.cos(th)
    return state, np.float32(z)

# =============================================================================
# 3) Diffuse reflection sampling (fully diffuse with wall temperature)
# =============================================================================
@cuda.jit(device=True, inline=True)
def sample_diffuse_velocity(state, T_wall, m, kb, nx, ny):
    """
    Fully diffuse wall velocity:
      - normal component: half-range Maxwell (Rayleigh)
      - tangential components: Gaussian
    nx, ny: inward normal (±1,0) or (0,±1)
    Returns (new_state, vx, vy, vz)
    """
    s = math.sqrt(kb * T_wall / m)

    # Rayleigh for |v_n|
    state, u = rng_uniform_float(state)
    vn = s * math.sqrt(-2.0 * math.log(u))

    # tangential gaussian
    state, g1 = rng_normal_float(state)
    state, g2 = rng_normal_float(state)
    vt = s * g1
    vz = s * g2

    # tangent in-plane
    tx = -ny
    ty =  nx

    vx = vn * nx + vt * tx
    vy = vn * ny + vt * ty
    return state, np.float32(vx), np.float32(vy), np.float32(vz)

# =============================================================================
# 4) CUDA kernels
# =============================================================================
@cuda.jit(fastmath=True)
def init_particles(x, y, vx, vy, vz, seeds, Lq, T0, m, kb):
    i = cuda.grid(1)
    if i < x.size:
        state = seeds[i]

        state, u1 = rng_uniform_float(state)
        state, u2 = rng_uniform_float(state)
        x[i] = u1 * Lq
        y[i] = u2 * Lq

        s = math.sqrt(kb * T0 / m)
        state, g1 = rng_normal_float(state)
        state, g2 = rng_normal_float(state)
        state, g3 = rng_normal_float(state)
        vx[i] = s * g1
        vy[i] = s * g2
        vz[i] = s * g3

        seeds[i] = state

@cuda.jit(fastmath=True)
def move_and_quarter_walls(x, y, vx, vy, vz, seeds, Lq, dt, Thot, Tcold, m, kb):
    """Advance particles using exact wall-hit times within the time step.

    A diffuse collision changes the velocity magnitude and direction, so the
    usual specular position reflection ``x_new = 2*x_wall - x_new`` is not
    valid.  We instead advance to the first wall hit, sample the reflected
    velocity, and advance through the remaining part of the time step.  The
    short CFL time step makes more than a few wall hits extraordinarily rare;
    eight event slots provide a conservative guard for high-speed tails.
    """
    i = cuda.grid(1)
    if i < x.size:
        state = seeds[i]

        xp = x[i]
        yp = y[i]
        vxp = vx[i]
        vyp = vy[i]
        vzp = vz[i]
        remaining = dt
        tiny_t = 1.0e-14 * dt
        tiny_x = 1.0e-12 * Lq

        for _ in range(8):
            if remaining <= tiny_t:
                remaining = 0.0
                break

            tx = 1.0e30
            ty = 1.0e30
            x_side = 0
            y_side = 0

            if vxp < 0.0:
                tx = -xp / vxp
                x_side = -1       # x=0: specular symmetry plane
            elif vxp > 0.0:
                tx = (Lq - xp) / vxp
                x_side = 1        # x=Lq: cold diffuse wall

            if vyp < 0.0:
                ty = -yp / vyp
                y_side = -1       # y=0: specular symmetry plane
            elif vyp > 0.0:
                ty = (Lq - yp) / vyp
                y_side = 1        # y=Lq: hot diffuse wall

            # Roundoff may put a particle infinitesimally beyond a wall.
            if tx < 0.0:
                tx = 0.0
            if ty < 0.0:
                ty = 0.0

            if tx >= remaining and ty >= remaining:
                xp += vxp * remaining
                yp += vyp * remaining
                remaining = 0.0
                break

            if tx <= ty:
                xp += vxp * tx
                yp += vyp * tx
                remaining -= tx

                if x_side < 0:
                    xp = tiny_x
                    vxp = -vxp
                else:
                    xp = Lq - tiny_x
                    state, vxp, vyp, vzp = sample_diffuse_velocity(
                        state, Tcold, m, kb, -1.0, 0.0
                    )
            else:
                xp += vxp * ty
                yp += vyp * ty
                remaining -= ty

                if y_side < 0:
                    yp = tiny_x
                    vyp = -vyp
                else:
                    yp = Lq - tiny_x
                    state, vxp, vyp, vzp = sample_diffuse_velocity(
                        state, Thot, m, kb, 0.0, -1.0
                    )

        # Extremely energetic tails could exhaust the event slots.  Keep the
        # particle inside rather than allowing an invalid cell index; with the
        # production CFL this branch should never be reached.
        if remaining > 0.0:
            xp += vxp * remaining
            yp += vyp * remaining
            if xp < 0.0:
                xp = tiny_x
                vxp = abs(vxp)
            elif xp > Lq:
                xp = Lq - tiny_x
                vxp = -abs(vxp)
            if yp < 0.0:
                yp = tiny_x
                vyp = abs(vyp)
            elif yp > Lq:
                yp = Lq - tiny_x
                vyp = -abs(vyp)

        x[i] = xp
        y[i] = yp
        vx[i] = vxp
        vy[i] = vyp
        vz[i] = vzp
        seeds[i] = state

@cuda.jit(fastmath=True)
def collision_bgk_cellwise(vx, vy, vz, starts, counts, dt, vol, fnum, seeds, m, kb, sigma_hs):
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    cnt = counts[cid]
    if cnt < 2:
        return

    st = starts[cid]

    # local moments
    su = 0.0; sv = 0.0; sw = 0.0
    sv2 = 0.0
    for j in range(st, st + cnt):
        u = vx[j]; v = vy[j]; w = vz[j]
        su += u; sv += v; sw += w
        sv2 += (u*u + v*v + w*w)

    ubar = su / cnt
    vbar = sv / cnt
    wbar = sw / cnt

    e_kin = sv2 / cnt - (ubar*ubar + vbar*vbar + wbar*wbar)
    Tloc = (m / (3.0 * kb)) * e_kin
    if Tloc < 1.0:
        Tloc = 1.0

    # n = (cnt * fnum)/vol, p = n k T
    nloc = (cnt * fnum) / vol
    ploc = nloc * kb * Tloc

    # Exact dimensional frequency implied by Eqs. (5) and (9) of the paper
    # together with Kn = 1/(sqrt(2)*n0*sigma*L).  Do NOT replace this by
    # p/mu_HS unless the corresponding 16/5 conversion is also applied.
    nu = nloc * sigma_hs * math.sqrt(math.pi * kb * Tloc / m)

    prob = 1.0 - math.exp(-nu * dt)
    if prob < 0.0: prob = 0.0
    if prob > 1.0: prob = 1.0

    s = math.sqrt(kb * Tloc / m)

    for j in range(st, st + cnt):
        state = seeds[j]
        state, u = rng_uniform_float(state)
        if u < prob:
            state, g1 = rng_normal_float(state)
            state, g2 = rng_normal_float(state)
            state, g3 = rng_normal_float(state)
            vx[j] = ubar + s * g1
            vy[j] = vbar + s * g2
            vz[j] = wbar + s * g3
        seeds[j] = state

@cuda.jit(fastmath=True)
def collision_shakhov_cellwise(vx, vy, vz, starts, counts, dt, vol, fnum,
                               seeds, m, kb, sigma_hs, Pr, w_clamp,
                               max_trials, diag_selected, diag_negative,
                               diag_high, diag_fallback):
    """
    Shakhov particle relaxation:
      - same paper-calibrated relaxation rate as BGK
      - relax to a Maxwellian *with Shakhov heat-flux correction* using a stable accept-reject
    """
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    cnt = counts[cid]
    if cnt < 2:
        return

    st = starts[cid]

    # --- local moments (mean velocity, temperature)
    su = 0.0; sv = 0.0; sw = 0.0
    sv2 = 0.0
    for j in range(st, st + cnt):
        u = vx[j]; v = vy[j]; w = vz[j]
        su += u; sv += v; sw += w
        sv2 += (u*u + v*v + w*w)

    ubar = su / cnt
    vbar = sv / cnt
    wbar = sw / cnt

    e_kin = sv2 / cnt - (ubar*ubar + vbar*vbar + wbar*wbar)
    Tloc = (m / (3.0 * kb)) * e_kin
    if Tloc < 1.0:
        Tloc = 1.0

    # n, p
    nloc = (cnt * fnum) / vol
    ploc = nloc * kb * Tloc

    # Same paper-calibrated relaxation frequency as BGK.
    nu = nloc * sigma_hs * math.sqrt(math.pi * kb * Tloc / m)
    prob = 1.0 - math.exp(-nu * dt)
    if prob < 0.0: prob = 0.0
    if prob > 1.0: prob = 1.0

    # thermal speed parameter
    s = math.sqrt(kb * Tloc / m)
    theta = kb * Tloc / m  # RT in kinetic form

    # --- estimate heat flux vector q ~ (m/2) n <c^2 c>
    sqx = 0.0; sqy = 0.0; sqz = 0.0
    for j in range(st, st + cnt):
        cx = vx[j] - ubar
        cy = vy[j] - vbar
        cz = vz[j] - wbar
        c2 = cx*cx + cy*cy + cz*cz
        sqx += c2 * cx
        sqy += c2 * cy
        sqz += c2 * cz

    inv_cnt = 1.0 / cnt
    qx = 0.5 * m * nloc * (sqx * inv_cnt)
    qy = 0.5 * m * nloc * (sqy * inv_cnt)
    qz = 0.5 * m * nloc * (sqz * inv_cnt)

    denom = 5.0 * ploc * theta + 1e-30
    A = (1.0 - Pr)
    n_selected = 0
    n_negative = 0
    n_high = 0
    n_fallback = 0

    for j in range(st, st + cnt):
        state = seeds[j]
        state, u = rng_uniform_float(state)

        if u < prob:
            n_selected += 1
            accepted = False
            for _ in range(max_trials):
                state, g1 = rng_normal_float(state)
                state, g2 = rng_normal_float(state)
                state, g3 = rng_normal_float(state)

                vxn = ubar + s * g1
                vyn = vbar + s * g2
                vzn = wbar + s * g3

                cx = vxn - ubar
                cy = vyn - vbar
                cz = vzn - wbar
                c2 = cx*cx + cy*cy + cz*cz

                cdotq = cx*qx + cy*qy + cz*qz
                # Standard dimensional monatomic Shakhov target:
                # f_S/M = 1 + (1-Pr) (c.q)/(5 p theta)
                #                    * (c^2/theta - 5),
                # where theta=k_B*T/m and q=(m n/2)<c^2 c>.
                # The former (c^2/(2*theta)-5/2) was smaller by a
                # factor of two and therefore produced the wrong Prandtl
                # correction and an underpredicted Fourier heat flux.
                shape = (c2 / (theta + 1e-30)) - 5.0
                w = 1.0 + A * (cdotq / denom) * shape

                if w < 0.0:
                    n_negative += 1
                    continue
                if w > w_clamp:
                    n_high += 1
                    w = w_clamp

                state, ur = rng_uniform_float(state)
                if ur < (w / w_clamp):
                    vx[j] = vxn
                    vy[j] = vyn
                    vz[j] = vzn
                    accepted = True
                    break

            if not accepted:
                n_fallback += 1
                state, g1 = rng_normal_float(state)
                state, g2 = rng_normal_float(state)
                state, g3 = rng_normal_float(state)
                vx[j] = ubar + s * g1
                vy[j] = vbar + s * g2
                vz[j] = wbar + s * g3

        seeds[j] = state

    diag_selected[cid] += n_selected
    diag_negative[cid] += n_negative
    diag_high[cid] += n_high
    diag_fallback[cid] += n_fallback

@cuda.jit(fastmath=True)
def collision_dsmc_ntc_cellwise(vx, vy, vz, starts, counts, sigma_max, dt, vol, fnum, seeds, sigma_hs):
    """
    DSMC NTC, hard-sphere, cell-wise.
    sigma_max[cid] stores max(sigma*|vr|) estimate for acceptance.
    """
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    cnt = counts[cid]
    if cnt < 2:
        return

    st = starts[cid]
    state = seeds[st]  # one state per cell (take first particle state)

    sig_vr_max = sigma_max[cid]
    if sig_vr_max < 1e-20:
        sig_vr_max = np.float32(1e-20)

    base = 0.5 * cnt * (cnt - 1) * fnum * sig_vr_max * dt / vol
    n_pairs = int(base)
    state, u = rng_uniform_float(state)
    if u < (base - n_pairs):
        n_pairs += 1

    for _ in range(n_pairs):
        state, u1 = rng_uniform_float(state)
        state, u2 = rng_uniform_float(state)
        p1 = st + int(u1 * cnt)
        p2 = st + int(u2 * cnt)
        if p1 == p2:
            continue

        vrx = vx[p1] - vx[p2]
        vry = vy[p1] - vy[p2]
        vrz = vz[p1] - vz[p2]
        vr = math.sqrt(vrx*vrx + vry*vry + vrz*vrz)

        sigvr = sigma_hs * vr
        if sigvr > sig_vr_max:
            sig_vr_max = sigvr

        state, ur = rng_uniform_float(state)
        if ur * sig_vr_max < sigvr:
            state, ucos = rng_uniform_float(state)
            cos_chi = 2.0 * ucos - 1.0
            sin_chi = math.sqrt(max(0.0, 1.0 - cos_chi*cos_chi))

            state, uphi = rng_uniform_float(state)
            phi = 2.0 * math.pi * uphi

            vcmx = 0.5*(vx[p1] + vx[p2])
            vcmy = 0.5*(vy[p1] + vy[p2])
            vcmz = 0.5*(vz[p1] + vz[p2])

            dvx = 0.5 * vr * sin_chi * math.cos(phi)
            dvy = 0.5 * vr * sin_chi * math.sin(phi)
            dvz = 0.5 * vr * cos_chi

            vx[p1] = vcmx + dvx
            vy[p1] = vcmy + dvy
            vz[p1] = vcmz + dvz

            vx[p2] = vcmx - dvx
            vy[p2] = vcmy - dvy
            vz[p2] = vcmz - dvz

    sigma_max[cid] = sig_vr_max
    seeds[st] = state

@cuda.jit(fastmath=True)
def sample_grid(x, y, vx, vy, vz, s_n, s_u, s_v, s_w, s_v2, L, nc):
    i = cuda.grid(1)
    if i < x.size:
        ix = int(x[i] / (L/nc))
        iy = int(y[i] / (L/nc))
        if 0 <= ix < nc and 0 <= iy < nc:
            idx = iy*nc + ix
            cuda.atomic.add(s_n,  idx, 1.0)
            cuda.atomic.add(s_u,  idx, float(vx[i]))
            cuda.atomic.add(s_v,  idx, float(vy[i]))
            cuda.atomic.add(s_w,  idx, float(vz[i]))
            v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
            cuda.atomic.add(s_v2, idx, float(v2))

# =============================================================================
# 5) Tecplot writer (includes density)
# =============================================================================
def write_tecplot_dat(fname, X, Y, U, V, T, Nnd):
    NCy, NCx = X.shape
    with open(fname, "w") as f:
        f.write('TITLE="Thermal Cavity"\n')
        f.write('VARIABLES="x","y","u_x","u_y","T","rho","Umag"\n')
        f.write(f'ZONE I={NCx}, J={NCy}, F=POINT\n')
        for j in range(NCy):
            for i in range(NCx):
                umag = (U[j,i]**2 + V[j,i]**2)**0.5
                f.write(f"{X[j,i]:.8e} {Y[j,i]:.8e} {U[j,i]:.8e} {V[j,i]:.8e} "
                        f"{T[j,i]:.8e} {Nnd[j,i]:.8e} {umag:.8e}\n")

# =============================================================================
# 6) No-penetration helper
# =============================================================================
def reconstruct_full_scalar(q):
    """Even reflection in x and y from the top-right quadrant."""
    top = np.concatenate((np.fliplr(q), q), axis=1)
    return np.concatenate((np.flipud(top), top), axis=0)


def reconstruct_full_velocity(uq, vq):
    """Use ux(-x,y)=-ux(x,y), ux(x,-y)=ux(x,y), and vice versa for uy."""
    u_top = np.concatenate((-np.fliplr(uq), uq), axis=1)
    v_top = np.concatenate(( np.fliplr(vq), vq), axis=1)
    u = np.concatenate(( np.flipud(u_top), u_top), axis=0)
    v = np.concatenate((-np.flipud(v_top), v_top), axis=0)
    return u, v

# =============================================================================
# 7) Plot helper
# =============================================================================
def _parity_pad(a, pad, sign_x, sign_y):
    """Pad a cell-centred field using its parity across solid walls."""
    left = sign_x * np.fliplr(a[:, :pad])
    right = sign_x * np.fliplr(a[:, -pad:])
    ax = np.concatenate((left, a, right), axis=1)
    bottom = sign_y * np.flipud(ax[:pad, :])
    top = sign_y * np.flipud(ax[-pad:, :])
    return np.concatenate((bottom, ax, top), axis=0)


def smooth_velocity_for_preview(U, V, rho, sigma):
    """Density-weighted smoothing that preserves wall impermeability parity.

    Density and tangential momentum are even across a wall; normal momentum is
    odd.  Applying those ghost-cell parities before filtering prevents the
    Gaussian kernel from spreading a spurious normal component through a wall.
    """
    pad = max(1, int(math.ceil(4.0 * sigma)))
    rp = _parity_pad(rho, pad, 1.0, 1.0)
    mxp = _parity_pad(rho * U, pad, -1.0, 1.0)
    myp = _parity_pad(rho * V, pad, 1.0, -1.0)

    rs = gaussian_filter(rp, sigma, mode="nearest")
    mxs = gaussian_filter(mxp, sigma, mode="nearest")
    mys = gaussian_filter(myp, sigma, mode="nearest")
    sl = (slice(pad, -pad), slice(pad, -pad))
    rs = np.maximum(rs[sl], 1.0e-300)
    return mxs[sl] / rs, mys[sl] / rs


def add_exact_wall_nodes(U, V, T, rt):
    """Add actual wall coordinates; set only the wall-normal velocity to zero."""
    nf = U.shape[0]
    Uw = np.empty((nf + 2, nf + 2))
    Vw = np.empty_like(Uw)
    Tw = np.empty_like(Uw)

    Uw[1:-1, 1:-1] = U
    Vw[1:-1, 1:-1] = V
    Tw[1:-1, 1:-1] = T

    # Vertical cold walls: ux=0; tangential uy is extrapolated from the
    # nearest cell centre. Horizontal hot walls: uy=0; ux is extrapolated.
    Uw[1:-1, 0] = 0.0
    Uw[1:-1, -1] = 0.0
    Vw[1:-1, 0] = V[:, 0]
    Vw[1:-1, -1] = V[:, -1]
    Vw[0, 1:-1] = 0.0
    Vw[-1, 1:-1] = 0.0
    Uw[0, 1:-1] = U[0, :]
    Uw[-1, 1:-1] = U[-1, :]
    Uw[0, 0] = Uw[0, -1] = Uw[-1, 0] = Uw[-1, -1] = 0.0
    Vw[0, 0] = Vw[0, -1] = Vw[-1, 0] = Vw[-1, -1] = 0.0

    Tw[1:-1, 0] = rt
    Tw[1:-1, -1] = rt
    Tw[0, :] = 1.0
    Tw[-1, :] = 1.0
    return Uw, Vw, Tw


def plot_results(U, V, T, rho, tag, Kn_val, output_stem):
    nf = U.shape[0]
    dx_nd = 1.0 / nf
    x = -0.5 + (np.arange(nf) + 0.5) * dx_nd
    y = -0.5 + (np.arange(nf) + 0.5) * dx_nd

    # Filtering is used only to place visually stable streamlines. All .dat
    # files and quantitative metrics are based on unfiltered fields.
    U_plot, V_plot = smooth_velocity_for_preview(
        U, V, rho, SMOOTH_SIGMA_UV
    )
    T_plot = gaussian_filter(T, SMOOTH_SIGMA_T)
    U_wall, V_wall, T_wall = add_exact_wall_nodes(
        U_plot, V_plot, T_plot, T_COLD / T_HOT
    )
    x_wall = np.concatenate(([-0.5], x, [0.5]))
    y_wall = np.concatenate(([-0.5], y, [0.5]))
    X_wall, Y_wall = np.meshgrid(x_wall, y_wall)

    plt.figure(figsize=(10, 9))
    cf = plt.contourf(X_wall, Y_wall, T_wall, levels=35)
    plt.colorbar(cf, label=r"$T/T_h$")
    plt.streamplot(
        x_wall, y_wall, U_wall, V_wall,
        density=1.4, linewidth=1.2, arrowsize=1.2
    )
    plt.title(f"{tag}: Temperature + Streamlines (Kn={Kn_val:g}, Thot={T_HOT}, Tcold={T_COLD})")
    plt.xlabel(r"$x/L$")
    plt.ylabel(r"$y/L$")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_stem + "_T_with_stream.png", dpi=250)
    plt.close()

    # Centerlines
    ix = nf//2
    iy = nf//2
    Uy = U_plot[:, ix]
    Vx = V_plot[iy, :]

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    ax[0].plot(Uy, y, 'k-', lw=3)
    ax[0].set_xlabel(r"$u_x/U_{mp,h}$")
    ax[0].set_ylabel(r"$y/L$")
    ax[0].set_title(r"$u_x(y)$ near $x=0$")
    ax[0].grid(True)

    ax[1].plot(x, Vx, 'r--', lw=3)
    ax[1].set_xlabel(r"$x/L$")
    ax[1].set_ylabel(r"$u_y/U_{mp,h}$")
    ax[1].set_title(r"$u_y(x)$ near $y=0$")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_stem + "_centerlines.png", dpi=250)
    plt.close()

# =============================================================================
# 8) Solver (BGK or DSMC or SHAKHOV)
# =============================================================================
class ThermalCavityParticle:
    def __init__(self, mode="BGK", Kn_local=1.0, seed=42):
        self.mode = mode  # "BGK" or "DSMC" or "SHAKHOV"
        self.Kn = float(Kn_local)
        self.seed = int(seed)

        # mean free path lambda = Kn*L
        # n0 = 1/(sqrt(2)*sigma*lambda)
        self.n0 = 1.0 / (math.sqrt(2.0) * SIGMA * (self.Kn * L))   # [1/m^3]

        # particle weight (real molecules per sim particle)
        self.fnum = self.n0 * (LQ*LQ*1.0) / N_PART

        # dt (keep same CFL-based dt; not Kn-dependent here)
        self.dt = DT_BASE

        # per-particle RNG states (uint64)
        self.seeds = (cp.arange(N_PART, dtype=cp.uint64)
                      * cp.uint64(2862933555777941757)
                      + cp.uint64(self.seed))

        self.x  = cp.zeros(N_PART, dtype=cp.float64)
        self.y  = cp.zeros(N_PART, dtype=cp.float64)
        self.vx = cp.zeros(N_PART, dtype=cp.float64)
        self.vy = cp.zeros(N_PART, dtype=cp.float64)
        self.vz = cp.zeros(N_PART, dtype=cp.float64)

        # DSMC needs sigma_max per cell
        self.sigma_max = cp.full(NC*NC, np.float32(2.0 * SIGMA * VTH_HOT), dtype=cp.float32)

        # accumulators
        self.s_n  = cp.zeros(NC*NC, dtype=cp.float64)
        self.s_u  = cp.zeros(NC*NC, dtype=cp.float64)
        self.s_v  = cp.zeros(NC*NC, dtype=cp.float64)
        self.s_w  = cp.zeros(NC*NC, dtype=cp.float64)      # for mean W
        self.s_v2 = cp.zeros(NC*NC, dtype=cp.float64)
        self.shakh_selected = cp.zeros(NC*NC, dtype=cp.int64)
        self.shakh_negative = cp.zeros(NC*NC, dtype=cp.int64)
        self.shakh_high = cp.zeros(NC*NC, dtype=cp.int64)
        self.shakh_fallback = cp.zeros(NC*NC, dtype=cp.int64)

        T0 = 0.5*(T_HOT + T_COLD)

        blocks = (N_PART + THREADS - 1)//THREADS
        init_particles[blocks, THREADS](self.x, self.y, self.vx, self.vy, self.vz,
                                        self.seeds, LQ, T0, MASS, KB)

    def run(self):
        blocks_p = (N_PART + THREADS - 1)//THREADS
        blocks_c = (NC*NC + THREADS - 1)//THREADS

        print(f"--- JFM quarter cavity {self.mode}: paper-Kn={self.Kn:g}, RT={T_COLD/T_HOT:g}, N={N_PART}, NCq={NC} ---")
        print(f"DT={self.dt:.3e}, steps={N_STEPS}, sample from {SAMP_START} every {SAMP_EVERY}")
        print(f"n0={self.n0:.6e} [1/m^3], FNUM={self.fnum:.6e}")

        for step in range(N_STEPS):
            move_and_quarter_walls[blocks_p, THREADS](
                self.x, self.y, self.vx, self.vy, self.vz,
                self.seeds, LQ, self.dt, T_HOT, T_COLD, MASS, KB
            )

            # cell index
            cx = cp.clip((self.x / DX).astype(cp.int32), 0, NC-1)
            cy = cp.clip((self.y / DX).astype(cp.int32), 0, NC-1)
            c_idx = cy*NC + cx

            # sort by cell
            sort_map = cp.argsort(c_idx)
            self.x  = self.x[sort_map]
            self.y  = self.y[sort_map]
            self.vx = self.vx[sort_map]
            self.vy = self.vy[sort_map]
            self.vz = self.vz[sort_map]
            self.seeds = self.seeds[sort_map]

            c_sorted = c_idx[sort_map]

            counts = cp.bincount(c_sorted, minlength=NC*NC).astype(cp.int32)
            starts = cp.zeros_like(counts)
            cp.cumsum(counts[:-1], out=starts[1:])

            if self.mode.upper() == "BGK":
                collision_bgk_cellwise[blocks_c, THREADS](
                    self.vx, self.vy, self.vz,
                    starts, counts,
                    self.dt, CELL_VOL, self.fnum,
                    self.seeds, MASS, KB, SIGMA
                )
            elif self.mode.upper() == "SHAKHOV":
                collision_shakhov_cellwise[blocks_c, THREADS](
                    self.vx, self.vy, self.vz,
                    starts, counts,
                    self.dt, CELL_VOL, self.fnum,
                    self.seeds, MASS, KB, SIGMA,
                    np.float32(PRANDTL), np.float32(SHAK_W_CLAMP), SHAK_MAX_TRIALS,
                    self.shakh_selected, self.shakh_negative,
                    self.shakh_high, self.shakh_fallback
                )
            else:
                collision_dsmc_ntc_cellwise[blocks_c, THREADS](
                    self.vx, self.vy, self.vz,
                    starts, counts,
                    self.sigma_max,
                    self.dt, CELL_VOL, self.fnum,
                    self.seeds, np.float32(SIGMA)
                )

            if step >= SAMP_START and (step - SAMP_START) % SAMP_EVERY == 0:
                sample_grid[blocks_p, THREADS](
                    self.x, self.y, self.vx, self.vy, self.vz,
                    self.s_n, self.s_u, self.s_v, self.s_w, self.s_v2,
                    LQ, NC
                )

            if step % 10000 == 0:
                print(f" step {step}/{N_STEPS}")

        # finalize (to CPU)
        sn  = cp.asnumpy(self.s_n)
        su  = cp.asnumpy(self.s_u)
        sv  = cp.asnumpy(self.s_v)
        sw  = cp.asnumpy(self.s_w)
        sv2 = cp.asnumpy(self.s_v2)

        sn_safe = sn.copy()
        sn_safe[sn_safe < 1.0] = 1.0

        U = (su / sn_safe).reshape(NC, NC)
        V = (sv / sn_safe).reshape(NC, NC)
        W = (sw / sn_safe).reshape(NC, NC)

        # Density: ``sn`` accumulates particle counts at every sampling time,
        # so divide by the number of samples before converting the mean count
        # per cell to number density.
        n_samples = ((N_STEPS - 1 - SAMP_START) // SAMP_EVERY) + 1
        n_dim = ((sn / n_samples) * self.fnum / CELL_VOL).reshape(NC, NC)  # [1/m^3]
        n_nd  = n_dim / (self.n0 + 1e-30)

        # Temperature from kinetic energy (3D): <v^2> - (U^2+V^2+W^2)
        v2_mean = (sv2 / sn_safe).reshape(NC, NC)
        thermal_energy = v2_mean - (U*U + V*V + W*W)
        thermal_energy = np.maximum(thermal_energy, 1e-10)
        T = (MASS/(3.0*KB)) * thermal_energy

        # Reconstruct full-domain cell-centered fields. No artificial wall
        # values and no Gaussian filter enter the quantitative output.
        U_full, V_full = reconstruct_full_velocity(U, V)
        T_full = reconstruct_full_scalar(T)
        nnd_full = reconstruct_full_scalar(n_nd)

        diagnostics = None
        if self.mode.upper() == "SHAKHOV":
            diagnostics = {
                "selected_relaxations": int(cp.asnumpy(self.shakh_selected).sum()),
                "negative_weight_candidates": int(cp.asnumpy(self.shakh_negative).sum()),
                "above_limiter_candidates": int(cp.asnumpy(self.shakh_high).sum()),
                "max_trial_fallbacks": int(cp.asnumpy(self.shakh_fallback).sum())
            }

        return U_full, V_full, T_full, nnd_full, diagnostics

# =============================================================================
# 9) Run all + export Tecplot
# =============================================================================
def kn_tag(kn):
    # safe filename tag
    # e.g. 0.05 -> "0p05"
    s = f"{kn:g}"
    s = s.replace('.', 'p')
    return s

def parse_args():
    p = argparse.ArgumentParser(description="JFM quarter-domain cavity production run")
    p.add_argument("--mode", choices=("DSMC", "BGK", "SHAKHOV"), default=MODE)
    p.add_argument("--kn", type=float, default=KN_PAPER,
                   help="Kn exactly as defined in the paper")
    p.add_argument("--rt", type=float, default=0.2, help="Tcold/Thot")
    p.add_argument("--particles", type=int, default=N_PART)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--sample-start", type=int, default=SAMP_START)
    p.add_argument("--sample-every", type=int, default=SAMP_EVERY)
    p.add_argument("--seed", type=int, default=42,
                   help="independent RNG seed; also included in output names")
    p.add_argument("--output", default="results_kn30")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not (0.0 < args.rt < 1.0):
        raise ValueError("--rt must satisfy 0 < RT < 1")
    if args.kn <= 0.0 or args.particles < 10000 or args.steps <= 0:
        raise ValueError("positive Kn/steps and at least 10000 particles are required")

    # The kernels/classes use these production globals.
    MODE = args.mode
    KN_PAPER = args.kn
    T_COLD = args.rt * T_HOT
    N_PART = args.particles
    N_STEPS = args.steps
    SAMP_START = args.sample_start
    SAMP_EVERY = args.sample_every
    if not (0 <= SAMP_START < N_STEPS):
        raise ValueError("--sample-start must be smaller than --steps")

    os.makedirs(args.output, exist_ok=True)
    rt_tag = str(args.rt).replace(".", "p")
    tag = f"{MODE}_Kn{kn_tag(KN_PAPER)}_RT{rt_tag}_quarter_seed{args.seed}"
    stem = os.path.join(args.output, f"ThermalCavity_{tag}")

    t0 = time.time()
    solver = ThermalCavityParticle(mode=MODE, Kn_local=KN_PAPER,
                                   seed=args.seed)
    U, V, T, rho, shakh_diag = solver.run()

    # Paper nondimensionalization and true cell-center coordinates.
    U /= VTH_HOT
    V /= VTH_HOT
    T /= T_HOT
    nf = 2 * NC
    dx_nd = 1.0 / nf
    x = -0.5 + (np.arange(nf) + 0.5) * dx_nd
    y = -0.5 + (np.arange(nf) + 0.5) * dx_nd
    X, Y = np.meshgrid(x, y)

    write_tecplot_dat(stem + "_raw.dat", X, Y, U, V, T, rho)
    np.savez_compressed(stem + "_raw.npz", x=x, y=y, ux=U, uy=V, T=T, rho=rho)

    speed = np.sqrt(U*U + V*V)
    ek = float(np.sum(rho * speed**2) * dx_nd**2)
    n_samples = ((N_STEPS - 1 - SAMP_START) // SAMP_EVERY) + 1

    def density_weighted_mean(q, r):
        return float(np.sum(r * q) / np.maximum(np.sum(r), 1.0e-300))

    wall_adjacent_normal = {
        "left_ux": density_weighted_mean(U[:, 0], rho[:, 0]),
        "right_ux": density_weighted_mean(U[:, -1], rho[:, -1]),
        "bottom_uy": density_weighted_mean(V[0, :], rho[0, :]),
        "top_uy": density_weighted_mean(V[-1, :], rho[-1, :]),
        "note": "diagnostic at nearest cell centres, not the exact wall"
    }
    wall_normal_max_abs = max(
        abs(wall_adjacent_normal["left_ux"]),
        abs(wall_adjacent_normal["right_ux"]),
        abs(wall_adjacent_normal["bottom_uy"]),
        abs(wall_adjacent_normal["top_uy"]),
    )
    metrics = {
        "solver_version": SOLVER_VERSION,
        "mode": MODE,
        "Kn_paper": KN_PAPER,
        "RT": args.rt,
        "quarter_domain": True,
        "specular_planes": ["x=0", "y=0"],
        "quadrant_cells": [NC, NC],
        "full_reconstructed_cells": [nf, nf],
        "particles": N_PART,
        "seed": args.seed,
        "mean_particles_per_quadrant_cell": N_PART/(NC*NC),
        "steps": N_STEPS,
        "sample_start": SAMP_START,
        "sample_every": SAMP_EVERY,
        "number_of_samples": n_samples,
        "dt_seconds": DT_BASE,
        "n0_per_m3": solver.n0,
        "max_speed_nondimensional": float(np.max(speed)),
        "rms_speed_nondimensional": float(np.sqrt(np.mean(speed**2))),
        "kinetic_energy_nondimensional": ek,
        "wall_clock_seconds": time.time() - t0,
        "kn_definition": "1/(sqrt(2)*n0*pi*d^2*L)",
        "relaxation_frequency": "n*pi*d^2*sqrt(pi*kB*T/m) for BGK/Shakhov",
        "shakhov_weight_limiter": [0.0, SHAK_W_CLAMP] if MODE == "SHAKHOV" else None,
        "shakhov_max_accept_reject_trials": SHAK_MAX_TRIALS if MODE == "SHAKHOV" else None,
        "shakhov_sampling_diagnostics": shakh_diag,
        "quantitative_fields_are_unfiltered": True,
        "density_is_time_averaged": True,
        "preview_velocity_gaussian_sigma_cells": SMOOTH_SIGMA_UV,
        "wall_advance_scheme": "event-driven time-of-impact with post-collision residual-time advance",
        "preview_wall_treatment": "parity-aware mass-flux smoothing plus exact wall nodes with zero normal velocity",
        "wall_adjacent_density_weighted_normal_velocity": wall_adjacent_normal,
        "wall_adjacent_normal_velocity_max_abs": wall_normal_max_abs,
        "wall_diagnostic_warning_threshold": 2.0e-4,
        "wall_diagnostic_pass": wall_normal_max_abs < 2.0e-4
    }
    with open(stem + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_results(U, V, T, rho, tag, KN_PAPER, stem)
    print(json.dumps(metrics, indent=2))
    if not metrics["wall_diagnostic_pass"]:
        print("[WARNING] Wall-adjacent mean normal velocity exceeds the diagnostic threshold.")
    print(f"[OK] Saved unfiltered reviewer outputs with stem: {stem}")
