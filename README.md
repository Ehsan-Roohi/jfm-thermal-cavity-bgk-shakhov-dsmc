# Thermal rarefied-gas cavity: BGK, Shakhov, and DSMC

## Shakhov correction and recovered validation on 2026-10-22

The current restartable Shakhov solver uses `c2/theta - 5` with the physical
heat flux `q = (m*n/2) <c2*c>`, restoring the intended Pr=2/3 coefficient.
The old code used half that correction. Its results remain historical:
this commit does not regenerate or certify the old cavity Shakhov fields.
Old Shakhov checkpoints are rejected by the corrected solver to avoid mixing
different collision models and accumulated samples. BGK restart compatibility
is preserved.

The missing shared validation module and the corrected Fourier, hard-sphere
Fourier, and homogeneous test scripts were recovered from Unity with SHA-256
verification. See [validation instructions](validation/README.md) and
[source provenance](SOURCE_PROVENANCE.md). The old two-case Fourier smoke
evidence remains distinct from the recovered three-seed validation.

GPU-accelerated particle solvers and compact reproducibility evidence for the
accepted manuscript:

> Ben-Adva, D., Roohi, E., Manela, A.\*, “Analysis of thermally-induced
> rarefied gas flow in a square cavity: From near-free-molecular to
> near-continuum conditions,” accepted at *Journal of Fluid Mechanics*.

The repository contains corrected restartable solvers, seven historical case
definitions, Slurm launch scripts, restart-contract tests, recovered Fourier
validation, and compact historical result summaries. The exact source hashes
used by historical runs are preserved in SOURCE_PROVENANCE.md and Git history.
Multi-gigabyte particle restart
states and raw fields remain on the cluster and are intentionally excluded.

## Models

- Hard-sphere DSMC using the no-time-counter (NTC) collision procedure.
- BGK stochastic particle relaxation.
- Shakhov stochastic particle relaxation with the documented bounded
  acceptance-rejection implementation.

All three models share particle transport, diffuse wall reflection, cell
sorting, sampling, reconstruction, and output conventions. The physical
domain is a stationary thermal square cavity. Production runs use one-quarter
of the domain with specular symmetry planes and reconstruct the full
`200 x 200` field.

## Historical final reviewer cases

| Figure | Model | Kn | RT = Tc/Th | Particles | Final steps |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 | HS/DSMC | 20 | 0.2 | 80,000,000 | 3,000,000 |
| 5 | BGK | 20 | 0.2 | 80,000,000 | 3,000,000 |
| 5 | Shakhov | 20 | 0.2 | 80,000,000 | 3,000,000 |
| 6 | Shakhov | 5 | 0.5 | 80,000,000 | 3,000,000 |
| 6 | Shakhov | 10 | 0.5 | 80,000,000 | 3,000,000 |
| 6 | Shakhov | 20 | 0.5 | 80,000,000 | 3,000,000 |
| 2(d) | HS/DSMC | 30 | 0.5 | 80,000,000 | 3,000,000 |

Every case uses seed `104729`, begins sampling at step `100000`, samples every
second step, and retains float64 moment accumulators. Quantitative fields are
unfiltered: no spatial smoothing or velocity projection is applied. The case
table is [`cases/fast7.csv`](cases/fast7.csv).

## Repository map

- `solver/`: exact restartable CUDA particle solvers.
- `cases/`: seven manuscript/reviewer case definitions.
- `scripts/`: safe Slurm submission, checkpoint, continuation, and summary
  scripts. The public scripts never cancel existing jobs.
- `tests/`: provenance, restart-contract, round-trip, and synthetic summary
  tests.
- `tools/`: compact-result summarization.
- `validation/fourier/`: BGK/Shakhov planar Fourier-flow validation.
- `results/summary/`: 1.5-million-step profiles and temporal-block diagnostics.
- `results/final_metrics/`: compact metadata from the successful
  3-million-step continuations.
- [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md): exact solver lineage and
  SHA-256 values.
- [`results/README.md`](results/README.md): result interpretation and limits.

## Environment

The production implementation targets Linux with an NVIDIA GPU and CUDA. A
portable Conda specification is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate jfm-thermal-cavity
```

Run the CPU-side checks before submitting GPU work:

```bash
bash scripts/preflight_local.sh
```

## Unity/Slurm production

The bundled settings reproduce the seven heavy final-review cases and require
a GPU with at least 40 GiB free memory:

```bash
bash scripts/submit_checkpoint_fast.sh
```

The initial workflow writes a raw checkpoint at one million steps and a full
restart at 1.5 million steps. Continue the same particle and RNG realization
without repeating the initial segment:

```bash
bash scripts/submit_continuation.sh 3000000
```

Cluster partitions, feature names, memory limits, and wall-time policies vary.
Review the `sbatch` resource requests before using the scripts outside Unity.

## Result scope

The committed profiles are compact audit artifacts, not a replacement for the
full raw production fields. Temporal-block error estimates describe stability
within one heavy realization and are not independent-seed confidence
intervals. See `results/README.md` before quantitative reuse.

## Citation and attribution

Use `CITATION.cff` when citing this software and cite the associated article
when using the scientific results. Paper authorship and software provenance
are recorded separately in `AUTHORS.md`; the repository does not infer
exclusive code authorship from article authorship.

## License

This software is released under the MIT License. See [`LICENSE`](LICENSE).
