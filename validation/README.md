# Recovered validation sources and results

These files were recovered on 2026-10-22 from Unity directory
`/project/pi_roohie_umass_edu/JFM_revision_2026/JFM_CORRECTED_V3`.
The shared module is `solver/JFM_reviewer_quarter_domain.py` with version
`2026-07-22-wallfix-shakhov-v3`. It contains the corrected Shakhov polynomial,
time-of-impact walls, and normalized sampling. It is a historical validated
module; the restartable production solver is a separate implementation.
The local import path in the three driver scripts is the only relocation
change. `provenance/unity_recovery_20261022.json` records original and bundled
hashes. The shared module and result files were copied byte for byte.

## Commands from the repository root

Use the supplied Conda environment, including CuPy, Numba, SciPy and Matplotlib.
Driver help is available without launching any simulation:

```bash
python validation/fourier/JFM_shakhov_fourier_validation.py --help
python validation/fourier/JFM_hardsphere_fourier_validation.py --help
python validation/homogeneous/JFM_shakhov_homogeneous_validation.py --help
python tests/test_shakhov_moments.py
python tests/test_validation_bundle.py
```

Run GPU validation only in an allocated compute session. For Unity, discover
or allocate shared Workspace scratch using `ws_list`/`ws_allocate`, check its
expiry and capacity, and pass an output prefix under that returned path.
No jobs are submitted by this repair or by the CPU checks above. These
recovered historical validation drivers are not full-state restartable;
they must not be described as checkpoint-safe preemptible production jobs.

## Existing evidence and its scope

`fourier/corrected_results/` has three BGK and three corrected Shakhov
realizations, with their profiles and ensemble summary. `fourier/hardsphere_results/`
has three HS realizations and the paired comparison summary. Their mean
conductivity is 0.02382142006182444 W/(m K) for HS; the mean paired HS/BGK
ratio is 1.4967861436619687 and Shakhov/HS is 1.032694985142456. These agree
with the rounded quantities in the manuscript. The mean conductivity ratio
Shakhov/BGK is 1.5426914183274316, whereas the mean bulk heat-flux ratio is
1.3701775623725885. Different statistics must not be interchanged.

`homogeneous/results/` isolates heat-flux relaxation. Its Shakhov fitted rate
differs by about 6.17% from the finite-step prediction and passes the stored
8% tolerance. These are recovered July results, not newly executed GPU runs.
The CPU moment test independently evaluates the actual weight expressions
in both bundled kernels and checks normalization, momentum, energy and heat
flux. It does not establish finite-Kn cavity accuracy or replace a GPU rerun.

The old `fourier/Fourier_*Kn0.02_*` smoke files remain unchanged and are not
the corrected three-seed evidence. Historical cavity results in `results/`
also remain unchanged. See `results/README.md` before quantitative reuse.
