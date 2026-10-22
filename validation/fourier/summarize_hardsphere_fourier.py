# -*- coding: utf-8 -*-
"""Summarize three-seed hard-sphere Fourier validation results."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


T_CRITICAL_95_DF2 = 4.302652729911275


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fourier_hardsphere")
    parser.add_argument("--kn", type=float, default=0.02)
    parser.add_argument(
        "--bgk-shakhov-input",
        default="fourier_corrected",
        help="Existing corrected BGK/Shakhov three-seed result directory",
    )
    return parser.parse_args()


def mean_sd_se_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan"), [float("nan"), float("nan")]
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(values.size)
    if values.size == 3:
        multiplier = T_CRITICAL_95_DF2
    else:
        multiplier = 1.96
    return mean, sd, se, [
        mean - multiplier * se,
        mean + multiplier * se,
    ]


def load_json(path):
    return json.loads(Path(path).read_text())


def load_profile(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def collect_model_metrics(directory, pattern):
    metrics = {}
    for path in sorted(Path(directory).glob(pattern)):
        data = load_json(path)
        metrics[int(data["seed"])] = data
    return metrics


def paired_ratios(numerator, denominator, key):
    seeds = sorted(set(numerator) & set(denominator))
    values = [
        float(numerator[seed][key]) / float(denominator[seed][key])
        for seed in seeds
    ]
    return seeds, values


def main():
    args = arguments()
    input_dir = Path(args.input)
    stem = f"Fourier_HARD_SPHERE_DSMC_Kn{args.kn:g}_seed"

    hs_files = sorted(input_dir.glob(stem + "*_metrics.json"))
    if len(hs_files) != 3:
        raise SystemExit(
            f"Expected exactly three hard-sphere metrics files; found "
            f"{len(hs_files)} in {input_dir}"
        )

    hs = {}
    profiles = {}
    for path in hs_files:
        data = load_json(path)
        if data.get("solver_version") != (
            "2026-07-23-fourier-hard-sphere-dsmc-v1"
        ):
            raise SystemExit(f"Unexpected solver version in {path}")
        seed = int(data["seed"])
        hs[seed] = data
        profile_path = Path(str(path).replace("_metrics.json", "_profile.csv"))
        profiles[seed] = load_profile(profile_path)

    seeds = sorted(hs)
    bulk_q = [hs[seed]["interior_mean_qx_W_m2"] for seed in seeds]
    wall_q = [hs[seed]["wall_balanced_heat_flux_W_m2"] for seed in seeds]
    conductivity = [
        hs[seed][
            "measured_conductivity_W_m_K_from_bulk_qx_and_interior_slope"
        ]
        for seed in seeds
    ]
    conductivity_reference = [
        hs[seed][
            "reference_hard_sphere_conductivity_W_m_K_at_interior_mean_T"
        ]
        for seed in seeds
    ]

    bulk_q_stats = mean_sd_se_ci95(bulk_q)
    wall_q_stats = mean_sd_se_ci95(wall_q)
    conductivity_stats = mean_sd_se_ci95(conductivity)
    conductivity_reference_mean = float(np.mean(conductivity_reference))
    continuum_q = float(
        np.mean([hs[seed]["continuum_analytic_qx_W_m2"] for seed in seeds])
    )

    summary = {
        "test": "three-seed hard-sphere DSMC planar Fourier validation",
        "Kn_paper": args.kn,
        "seeds": seeds,
        "number_of_independent_runs": len(seeds),
        "q_HS_bulk_mean_W_m2": bulk_q_stats[0],
        "q_HS_bulk_sample_sd_W_m2": bulk_q_stats[1],
        "q_HS_bulk_standard_error_W_m2": bulk_q_stats[2],
        "q_HS_bulk_95pct_t_CI_W_m2": bulk_q_stats[3],
        "q_HS_wall_balanced_mean_W_m2": wall_q_stats[0],
        "q_HS_wall_balanced_sample_sd_W_m2": wall_q_stats[1],
        "q_HS_wall_balanced_standard_error_W_m2": wall_q_stats[2],
        "q_HS_wall_balanced_95pct_t_CI_W_m2": wall_q_stats[3],
        "q_HS_continuum_analytic_W_m2": continuum_q,
        "q_HS_bulk_relative_error_vs_continuum": (
            abs(bulk_q_stats[0] - continuum_q) / abs(continuum_q)
        ),
        "q_HS_wall_relative_error_vs_continuum": (
            abs(wall_q_stats[0] - continuum_q) / abs(continuum_q)
        ),
        "conductivity_HS_mean_W_m_K": conductivity_stats[0],
        "conductivity_HS_sample_sd_W_m_K": conductivity_stats[1],
        "conductivity_HS_standard_error_W_m_K": conductivity_stats[2],
        "conductivity_HS_95pct_t_CI_W_m_K": conductivity_stats[3],
        "conductivity_HS_reference_mean_W_m_K": (
            conductivity_reference_mean
        ),
        "conductivity_HS_relative_error_vs_reference": (
            abs(conductivity_stats[0] - conductivity_reference_mean)
            / conductivity_reference_mean
        ),
        "wall_energy_imbalance_relative_mean": float(
            np.mean(
                [
                    hs[seed]["wall_energy_imbalance_relative"]
                    for seed in seeds
                ]
            )
        ),
        "bulk_q_spatial_cv_mean": float(
            np.mean(
                [hs[seed]["interior_spatial_cv_qx"] for seed in seeds]
            )
        ),
        "majorant_updates_total": int(
            sum(
                hs[seed]["dsmc_diagnostics"]["majorant_updates"]
                for seed in seeds
            )
        ),
        "candidate_pairs_total": int(
            sum(
                hs[seed]["dsmc_diagnostics"]["candidate_pairs"]
                for seed in seeds
            )
        ),
        "accepted_collisions_total": int(
            sum(
                hs[seed]["dsmc_diagnostics"]["accepted_collisions"]
                for seed in seeds
            )
        ),
        "quantitative_profiles_are_unfiltered": True,
    }

    corrected_dir = Path(args.bgk_shakhov_input)
    bgk = collect_model_metrics(
        corrected_dir, f"Fourier_BGK_Kn{args.kn:g}_seed*_metrics.json"
    )
    shakhov = collect_model_metrics(
        corrected_dir, f"Fourier_SHAKHOV_Kn{args.kn:g}_seed*_metrics.json"
    )

    conductivity_key_old = (
        "measured_conductivity_W_m_K_from_interior_slope"
    )
    if set(bgk) == set(seeds) and set(shakhov) == set(seeds):
        bgk_k = [
            bgk[seed][conductivity_key_old]
            for seed in seeds
        ]
        shakhov_k = [
            shakhov[seed][conductivity_key_old]
            for seed in seeds
        ]
        bgk_q = [bgk[seed]["interior_mean_qx_W_m2"] for seed in seeds]
        shakhov_q = [
            shakhov[seed]["interior_mean_qx_W_m2"] for seed in seeds
        ]

        hs_over_bgk_k = [
            conductivity[index] / bgk_k[index]
            for index in range(len(seeds))
        ]
        shakhov_over_hs_k = [
            shakhov_k[index] / conductivity[index]
            for index in range(len(seeds))
        ]
        hs_over_bgk_q = [
            bulk_q[index] / bgk_q[index]
            for index in range(len(seeds))
        ]
        shakhov_over_hs_q = [
            shakhov_q[index] / bulk_q[index]
            for index in range(len(seeds))
        ]

        summary.update(
            {
                "comparison_with_corrected_BGK_Shakhov_available": True,
                "conductivity_BGK_mean_W_m_K": float(np.mean(bgk_k)),
                "conductivity_Shakhov_mean_W_m_K": float(
                    np.mean(shakhov_k)
                ),
                "conductivity_HS_over_BGK_mean_of_paired_ratios": (
                    float(np.mean(hs_over_bgk_k))
                ),
                "conductivity_HS_over_BGK_paired_ratios": hs_over_bgk_k,
                "conductivity_HS_over_BGK_theory": (
                    15.0 * math.pi / 32.0
                ),
                "conductivity_Shakhov_over_HS_mean_of_paired_ratios": (
                    float(np.mean(shakhov_over_hs_k))
                ),
                "conductivity_Shakhov_over_HS_paired_ratios": (
                    shakhov_over_hs_k
                ),
                "conductivity_Shakhov_over_HS_theory": (
                    16.0 / (5.0 * math.pi)
                ),
                "bulk_q_HS_over_BGK_mean_of_paired_ratios": float(
                    np.mean(hs_over_bgk_q)
                ),
                "bulk_q_Shakhov_over_HS_mean_of_paired_ratios": float(
                    np.mean(shakhov_over_hs_q)
                ),
            }
        )
    else:
        summary["comparison_with_corrected_BGK_Shakhov_available"] = False

    summary_path = input_dir / (
        f"Fourier_HARD_SPHERE_ensemble_Kn{args.kn:g}_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = input_dir / (
        f"Fourier_HARD_SPHERE_ensemble_Kn{args.kn:g}_summary.csv"
    )
    with csv_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            writer.writerow((key, json.dumps(value) if isinstance(value, (list, dict)) else value))

    x = profiles[seeds[0]]["x_over_L"]
    temperature_stack = np.vstack(
        [profiles[seed]["T_K"] for seed in seeds]
    )
    qx_stack = np.vstack(
        [profiles[seed]["qx_W_m2"] for seed in seeds]
    )
    t_mean = np.mean(temperature_stack, axis=0)
    t_sd = np.std(temperature_stack, axis=0, ddof=1)
    q_mean = np.mean(qx_stack, axis=0)
    q_sd = np.std(qx_stack, axis=0, ddof=1)
    t_reference = profiles[seeds[0]]["T_continuum_K"]

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].plot(x, t_mean, color="tab:green", lw=2.2, label="HS DSMC")
    axes[0].fill_between(
        x, t_mean - t_sd, t_mean + t_sd, color="tab:green", alpha=0.18
    )
    axes[0].plot(
        x, t_reference, color="black", ls=":", lw=2.0, label="continuum"
    )
    axes[0].set_xlabel(r"$x/L$")
    axes[0].set_ylabel(r"$T$ (K)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, q_mean, color="tab:green", lw=2.2, label="HS DSMC")
    axes[1].fill_between(
        x, q_mean - q_sd, q_mean + q_sd, color="tab:green", alpha=0.18
    )
    axes[1].axhline(
        continuum_q, color="black", ls=":", lw=2.0, label="CE reference"
    )
    axes[1].axhline(
        wall_q_stats[0],
        color="tab:red",
        ls="--",
        lw=1.8,
        label="mean wall flux",
    )
    axes[1].set_xlabel(r"$x/L$")
    axes[1].set_ylabel(r"$q_x$ (W m$^{-2}$)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        "Hard-sphere DSMC Fourier validation: "
        rf"$Kn={args.kn:g}$, three independent seeds"
    )
    figure.tight_layout()
    figure_path = input_dir / (
        f"Fourier_HARD_SPHERE_ensemble_Kn{args.kn:g}_validation.png"
    )
    figure.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"[OK] Wrote {summary_path}")
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {figure_path}")


if __name__ == "__main__":
    main()
