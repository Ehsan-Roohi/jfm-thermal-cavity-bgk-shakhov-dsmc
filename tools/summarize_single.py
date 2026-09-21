#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ("ux", "uy", "T", "rho")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-input", type=Path, required=True)
    parser.add_argument("--new-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    return parser.parse_args()


def output_stem(row):
    kn = f"{float(row['kn']):g}"
    rt = f"{float(row['rt']):g}".replace(".", "p")
    seed = int(row["seed"])
    if row["model"] == "HS":
        return f"ThermalCavity_HS_DSMC_Kn{kn}_RT{rt}_quarter_seed{seed}"
    return f"ThermalCavity_{row['model']}_Kn{kn}_RT{rt}_quarter_seed{seed}"


def interpolate_row(y, field, target=0.25):
    return np.array([np.interp(target, y, field[:, i]) for i in range(field.shape[1])])


def block_statistics(blocks):
    """Naive block SD/SE; contiguous blocks are not claimed independent."""
    if blocks.shape[0] < 2:
        raise ValueError("At least two temporal blocks are required")
    sd = np.std(blocks, axis=0, ddof=1)
    return sd, sd / math.sqrt(blocks.shape[0])


def write_dat(path, x, y, means, sds, ses):
    with path.open("w", encoding="utf-8") as handle:
        handle.write('TITLE="JFM single heavy realization; raw unfiltered mean"\n')
        handle.write(
            'VARIABLES="x","y","u_x","u_y","T","rho","Umag",'
            '"ux_block_sd","uy_block_sd","ux_naive_block_se","uy_naive_block_se"\n'
        )
        handle.write(f"ZONE I={len(x)}, J={len(y)}, F=POINT\n")
        for j, yv in enumerate(y):
            for i, xv in enumerate(x):
                ux, uy = means["ux"][j, i], means["uy"][j, i]
                values = (xv, yv, ux, uy, means["T"][j, i], means["rho"][j, i],
                          math.hypot(ux, uy), sds["ux"][j, i], sds["uy"][j, i],
                          ses["ux"][j, i], ses["uy"][j, i])
                handle.write(" ".join(f"{v:.10e}" for v in values) + "\n")


def write_profile(path, x, y, means, sds, ses):
    series = {}
    for field in ("ux", "uy", "T"):
        series[field] = interpolate_row(y, means[field])
        series[f"{field}_block_sd"] = interpolate_row(y, sds[field])
        series[f"{field}_naive_block_se"] = interpolate_row(y, ses[field])
    names = ["x_over_L", *series]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for i, xv in enumerate(x):
            row = {"x_over_L": xv}
            row.update({name: values[i] for name, values in series.items()})
            writer.writerow(row)


def diagnostic_plot(path, x, y, means, ses, title):
    xx, yy = np.meshgrid(x, y)
    ux = interpolate_row(y, means["ux"])
    ux_se = interpolate_row(y, ses["ux"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True)
    axes[0].plot(x, ux, color="tab:blue")
    axes[0].fill_between(x, ux - 2 * ux_se, ux + 2 * ux_se,
                         color="tab:blue", alpha=0.2)
    axes[0].set_title(r"$u_x$ at $y/L=0.25$")
    contour = axes[1].contourf(xx, yy, means["T"], levels=30, cmap="coolwarm")
    axes[1].streamplot(x, y, means["ux"], means["uy"], color="k",
                       density=1.15, linewidth=0.7, arrowsize=0.7)
    axes[1].set_aspect("equal")
    axes[1].set_title("Raw mean field")
    fig.colorbar(contour, ax=axes[1], label=r"$T/T_h$")
    for axis in axes:
        axis.set_xlabel(r"$x/L$")
    axes[0].set_ylabel("nondimensional velocity")
    axes[1].set_ylabel(r"$y/L$")
    fig.suptitle(title + "\nBand: naive +/-2 block SE; not independent-seed uncertainty")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(row, input_dir, output_dir):
    stem = output_stem(row)
    npz_path = input_dir / f"{stem}_raw.npz"
    metrics_path = input_dir / f"{stem}_metrics.json"
    if not npz_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing completed outputs for {stem} in {input_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        means = {field: np.asarray(data[field], dtype=np.float64) for field in FIELDS}
        blocks = {field: np.asarray(data[f"{field}_time_blocks"], dtype=np.float64)
                  for field in FIELDS}
        samples_per_block = np.asarray(data["samples_per_time_block"], dtype=np.int64)
    if any(not np.all(np.isfinite(value)) for value in [*means.values(), *blocks.values()]):
        raise ValueError(f"Non-finite result in {stem}")
    sds, ses = {}, {}
    for field in FIELDS:
        sds[field], ses[field] = block_statistics(blocks[field])

    model, kn, rt, figure = row["model"], float(row["kn"]), float(row["rt"]), row["figure"]
    tag = f"{figure}_{model}_Kn{kn:g}_RT{str(rt).replace('.', 'p')}_ONE_HEAVY_SEED"
    np.savez_compressed(output_dir / f"{tag}_RAW_UNFILTERED.npz",
        x=x, y=y, seed=np.int64(row["seed"]), samples_per_time_block=samples_per_block,
        **{f"{f}_mean": means[f] for f in FIELDS},
        **{f"{f}_time_blocks": blocks[f] for f in FIELDS},
        **{f"{f}_block_sd": sds[f] for f in FIELDS},
        **{f"{f}_naive_block_se": ses[f] for f in FIELDS})
    write_dat(output_dir / f"{tag}_RAW_UNFILTERED.dat", x, y, means, sds, ses)
    write_profile(output_dir / f"{tag}_profile_y0p25.csv", x, y, means, sds, ses)
    diagnostic_plot(output_dir / f"{tag}_diagnostic.png", x, y, means, ses,
                    f"{figure}: {model}, Kn={kn:g}, RT={rt:g}")

    velocity_rms = float(np.sqrt(np.mean(means["ux"] ** 2 + means["uy"] ** 2)))
    velocity_block_se_rms = float(np.sqrt(np.mean(ses["ux"] ** 2 + ses["uy"] ** 2)))
    block_ek = np.sum(blocks["rho"] * (blocks["ux"] ** 2 + blocks["uy"] ** 2),
                      axis=(1, 2)) * float(np.mean(np.diff(x))) * float(np.mean(np.diff(y)))
    recent = block_ek[-min(4, len(block_ek)):]
    recent_range_rel = float(np.ptp(recent) / max(abs(np.mean(recent)), np.finfo(float).tiny))
    summary = {
        "figure": figure, "model": model, "Kn_paper": kn, "RT": rt,
        "seed": int(row["seed"]), "number_of_independent_seeds": 1,
        "particles": metrics["particles"], "steps": metrics["steps"],
        "sample_start": metrics["sample_start"], "profile_samples": metrics["profile_samples"],
        "time_blocks": int(len(samples_per_block)),
        "samples_per_time_block": samples_per_block.tolist(),
        "velocity_mean_rms": velocity_rms,
        "velocity_naive_block_se_rms": velocity_block_se_rms,
        "velocity_signal_to_naive_block_se_ratio": (
            velocity_rms / velocity_block_se_rms if velocity_block_se_rms else None),
        "recent_four_block_Ek_relative_range": recent_range_rel,
        "last_block_velocity_rmse_vs_all_samples": metrics["last_block_velocity_rmse_vs_all_samples"],
        "temporal_blocks_are_contiguous": True,
        "block_uncertainty_is_not_independent_seed_uncertainty": True,
        "quantitative_fields_are_unfiltered": True,
        "spatial_smoothing_applied": False,
        "velocity_projection_applied": False,
    }
    (output_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.case_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7 or len({row["case_id"] for row in rows}) != 7:
        raise ValueError("Expected exactly seven distinct physical cases")
    summaries = []
    for row in rows:
        input_dir = args.existing_input if row["source"] == "existing_vram48" else args.new_input
        summaries.append(summarize(row, input_dir, args.output))
        print(f"[OK] summarized {row['case_id']}", flush=True)
    (args.output / "ALL_SINGLE_HEAVY_SEED_SUMMARY.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    fields = ["figure", "model", "Kn_paper", "RT", "seed", "particles", "steps",
              "profile_samples", "time_blocks", "velocity_mean_rms",
              "velocity_naive_block_se_rms", "velocity_signal_to_naive_block_se_ratio",
              "recent_four_block_Ek_relative_range", "last_block_velocity_rmse_vs_all_samples"]
    with (args.output / "ALL_SINGLE_HEAVY_SEED_SUMMARY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})
    print(f"[OK] wrote seven single-realization summaries to {args.output}", flush=True)


if __name__ == "__main__":
    main()
