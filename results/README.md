# Result artifacts

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
