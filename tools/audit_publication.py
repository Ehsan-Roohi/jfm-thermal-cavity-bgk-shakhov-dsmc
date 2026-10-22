#!/usr/bin/env python3
"""Audit current solver hashes and unchanged historical result metadata without a GPU."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOLVERS = {
    "solver/JFM_hs_dsmc_quarter.py":
        "2c9e2f5119802b123f0335664a564085cfe77682ae4f4af4138dc31f5876b166",
    "solver/JFM_bgk_shakhov_quarter.py":
        "e05ffd201443333143ceb3ce6b86c34eb30d228863fe42cacdfd1f878b32fcd1",
}


def main() -> None:
    for relative, expected in EXPECTED_SOLVERS.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"solver hash mismatch: {relative}: {actual}")

    with (ROOT / "cases/fast7.csv").open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    if len(cases) != 7:
        raise SystemExit(f"expected 7 publication cases, found {len(cases)}")

    metrics = sorted((ROOT / "results/final_metrics").glob("*_metrics.json"))
    if len(metrics) != 7:
        raise SystemExit(f"expected 7 final metric files, found {len(metrics)}")

    for path in metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "steps": payload.get("steps") == 3_000_000,
            "unfiltered": payload.get("quantitative_fields_are_unfiltered") is True,
            "no_smoothing": payload.get("spatial_smoothing_applied") is False,
            "no_projection": payload.get("velocity_projection_applied") is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise SystemExit(f"{path.name}: failed checks: {', '.join(failed)}")

    print("[OK] 2 current solver hashes, 7 cases and 7 historical metric records validated")


if __name__ == "__main__":
    main()
