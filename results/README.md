# Result artifacts

## Historical source status after the 2026-10-22 correction

The existing `summary/` and `final_metrics/` files are unchanged historical
outputs of the restartable-v2 sources pinned in `SOURCE_PROVENANCE.md`.
That BGK/Shakhov source used the half-strength Shakhov heat-flux polynomial.
Its Shakhov results are **not validated outputs of the corrected Pr=2/3 solver**.
No corrected cavity rerun is included in this repair. HS and BGK collision
formulas are not changed by the correction.

Corrected three-seed Fourier evidence recovered from Unity is separately
stored in `validation/fourier/corrected_results/` and `hardsphere_results/`.
The two older Kn0.02 files directly under `validation/fourier/` are historical
smoke artifacts, not the three-seed validation cited by the manuscript.
The homogeneous validation evidence is under `validation/homogeneous/results/`.
See `validation/README.md` for provenance and the limits of these checks.

This repository includes compact evidence only. The full 80-million-particle
restart directories and raw 3-million-step fields are multiple gigabytes per
case and remain external to GitHub.

## `summary/`

The seven profiles, summary JSON files, CSV index, and diagnostic figures were
generated from the 1.5-million-step checkpoint workflow. They contain 700,000
sampled states per case and 14 contiguous temporal blocks. The block-derived
standard errors measure within-realization temporal stability; they are not
independent-seed uncertainty estimates.

## `final_metrics/`

These JSON files come from the successful exact-restart continuations to
3,000,000 total steps. Each case preserves the original particle state, random
number state, collision state, and accumulated moments. The metrics document
completion and numerical diagnostics but do not contain the large raw fields.

All quantitative results declare:

- `quantitative_fields_are_unfiltered: true`
- `spatial_smoothing_applied: false`
- `velocity_projection_applied: false`

Use `python tools/audit_publication.py` to validate the committed case count,
final step count, filtering flags, and solver hashes.
