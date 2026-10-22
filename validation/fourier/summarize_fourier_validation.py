#!/usr/bin/env python3
"""Ensemble summary for corrected BGK/Shakhov planar-Fourier runs."""

import argparse
import csv
import glob
import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_profile(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def seed_from_name(path):
    match = re.search(r"_seed(-?\d+)_metrics\.json$", path)
    if not match:
        raise ValueError(f"Cannot extract seed from {path}")
    return int(match.group(1))


def discover(directory, mode, kn):
    pattern = os.path.join(directory,
                           f"Fourier_{mode}_Kn{kn:g}_seed*_metrics.json")
    return {seed_from_name(path): path for path in sorted(glob.glob(pattern))}


def mean_sd_sem(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return mean, sd, sd/math.sqrt(values.size)


def profile_ensemble(metric_paths):
    profiles = []
    for path in metric_paths:
        profiles.append(load_profile(path.replace("_metrics.json", "_profile.csv")))
    x = profiles[0]["x_over_L"]
    for profile in profiles[1:]:
        if not np.allclose(profile["x_over_L"], x):
            raise ValueError("Ensemble profiles have different grids")
    out = {"x_over_L": x}
    for key in ("T_K", "T_continuum_K", "qx_W_m2", "qx_continuum_W_m2"):
        values = np.vstack([profile[key] for profile in profiles])
        out[key+"_mean"] = np.mean(values, axis=0)
        out[key+"_sd"] = (np.std(values, axis=0, ddof=1)
                           if values.shape[0] > 1 else np.zeros_like(x))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fourier_corrected")
    parser.add_argument("--kn", type=float, default=0.02)
    args = parser.parse_args()

    bgk_files = discover(args.input, "BGK", args.kn)
    shk_files = discover(args.input, "SHAKHOV", args.kn)
    seeds = sorted(set(bgk_files) & set(shk_files))
    if not seeds:
        raise SystemExit("No paired BGK/Shakhov seeds were found")
    if set(bgk_files) != set(shk_files):
        print("WARNING: ignoring unpaired seeds", flush=True)

    bgk = [load_json(bgk_files[s]) for s in seeds]
    shk = [load_json(shk_files[s]) for s in seeds]
    q_b = np.array([item["interior_mean_qx_W_m2"] for item in bgk])
    q_s = np.array([item["interior_mean_qx_W_m2"] for item in shk])
    kb = np.array([item["measured_conductivity_W_m_K_from_interior_slope"]
                   for item in bgk])
    ks = np.array([item["measured_conductivity_W_m_K_from_interior_slope"]
                   for item in shk])
    qb_mean, qb_sd, qb_sem = mean_sd_sem(q_b)
    qs_mean, qs_sd, qs_sem = mean_sd_sem(q_s)
    kb_mean, kb_sd, kb_sem = mean_sd_sem(kb)
    ks_mean, ks_sd, ks_sem = mean_sd_sem(ks)
    q_ratio = qs_mean/qb_mean
    k_ratio = ks_mean/kb_mean
    q_ratio_sem = abs(q_ratio)*math.sqrt((qs_sem/qs_mean)**2
                                        + (qb_sem/qb_mean)**2)
    k_ratio_sem = abs(k_ratio)*math.sqrt((ks_sem/ks_mean)**2
                                        + (kb_sem/kb_mean)**2)
    target = 1.0/(2.0/3.0)
    q_ref_b = float(bgk[0]["continuum_analytic_qx_W_m2"])
    q_ref_s = float(shk[0]["continuum_analytic_qx_W_m2"])

    summary = {
        "test": "corrected ensemble planar Fourier heat-conduction validation",
        "Kn_paper": args.kn,
        "paired_seeds": seeds,
        "number_of_independent_pairs": len(seeds),
        "q_BGK_mean_W_m2": qb_mean,
        "q_BGK_sample_sd_W_m2": qb_sd,
        "q_BGK_standard_error_W_m2": qb_sem,
        "q_BGK_continuum_analytic_W_m2": q_ref_b,
        "q_BGK_relative_error_vs_analytic": abs(qb_mean-q_ref_b)/abs(q_ref_b),
        "q_Shakhov_mean_W_m2": qs_mean,
        "q_Shakhov_sample_sd_W_m2": qs_sd,
        "q_Shakhov_standard_error_W_m2": qs_sem,
        "q_Shakhov_continuum_analytic_W_m2": q_ref_s,
        "q_Shakhov_relative_error_vs_analytic": abs(qs_mean-q_ref_s)/abs(q_ref_s),
        "q_Shakhov_over_q_BGK": q_ratio,
        "q_ratio_approx_95pct_CI": [q_ratio-1.96*q_ratio_sem,
                                     q_ratio+1.96*q_ratio_sem],
        "conductivity_BGK_mean_W_m_K": kb_mean,
        "conductivity_BGK_sample_sd_W_m_K": kb_sd,
        "conductivity_Shakhov_mean_W_m_K": ks_mean,
        "conductivity_Shakhov_sample_sd_W_m_K": ks_sd,
        "conductivity_Shakhov_over_BGK": k_ratio,
        "conductivity_ratio_approx_95pct_CI": [k_ratio-1.96*k_ratio_sem,
                                                k_ratio+1.96*k_ratio_sem],
        "expected_1_over_Pr": target,
        "q_ratio_relative_deviation_from_1_over_Pr": abs(q_ratio-target)/target,
        "conductivity_ratio_relative_deviation_from_1_over_Pr": abs(k_ratio-target)/target,
        "pass_q_ratio_10_percent_tolerance": abs(q_ratio-target)/target <= 0.10,
        "pass_conductivity_ratio_10_percent_tolerance": abs(k_ratio-target)/target <= 0.10,
        "quantitative_profiles_are_unfiltered": True,
    }

    tag = f"{args.kn:g}"
    out_stem = os.path.join(args.input, f"Fourier_ensemble_Kn{tag}")
    with open(out_stem+"_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_stem+"_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            writer.writerow((key, json.dumps(value) if isinstance(value, (list, dict)) else value))

    bp = profile_ensemble([bgk_files[s] for s in seeds])
    sp = profile_ensemble([shk_files[s] for s in seeds])
    x = bp["x_over_L"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(x, bp["T_K_mean"], label="BGK")
    axes[0].plot(x, sp["T_K_mean"], "--", label="Shakhov")
    axes[0].plot(x, bp["T_continuum_K_mean"], "k:", label="continuum")
    axes[0].set(xlabel=r"$x/L$", ylabel=r"$T$ (K)")
    axes[0].legend(frameon=False)
    axes[1].plot(x, bp["qx_W_m2_mean"], label="BGK")
    axes[1].fill_between(x, bp["qx_W_m2_mean"]-bp["qx_W_m2_sd"],
                        bp["qx_W_m2_mean"]+bp["qx_W_m2_sd"], alpha=0.18)
    axes[1].plot(x, sp["qx_W_m2_mean"], label="Shakhov")
    axes[1].fill_between(x, sp["qx_W_m2_mean"]-sp["qx_W_m2_sd"],
                        sp["qx_W_m2_mean"]+sp["qx_W_m2_sd"], alpha=0.18)
    axes[1].axhline(q_ref_b, color="C0", ls=":", label="BGK analytic")
    axes[1].axhline(q_ref_s, color="C1", ls=":", label="Shakhov analytic")
    axes[1].set(xlabel=r"$x/L$", ylabel=r"$q_x$ (W m$^{-2}$)")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle(f"Corrected Fourier ensemble: qS/qB={q_ratio:.4f} (target 1.5)")
    fig.tight_layout()
    fig.savefig(out_stem+"_validation.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
