# Restartable solver provenance

The physical transport, wall, collision, random-number, sampling-kernel,
precision, and reconstruction implementation originates from the validated JFM
production solvers in commit `0904a6dc2f7b6aabfe1bcd6aefe3d19640fe4265`.

Original SHA-256 values:

- HS: `d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f`
- BGK/Shakhov: `c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55`

The restartable version adds only statistical snapshot and continuation I/O plus
loop offsets required to resume from a completed step. It does not change the
advance, boundary, collision, random-number, sorting, or sampling kernels.

Restart state contains:

- all particle positions and velocity components;
- per-particle RNG state;
- HS cell RNG, NTC majorant and collision counters, or relaxation-model
  diagnostic counters;
- float64 moment accumulators and per-block sampling counts;
- the completed step, physical parameters and cumulative wall time.

Arrays are written individually as uncompressed `.npy` files to avoid a
multi-gigabyte host-memory spike. `manifest.json` is written last and the
temporary directory is atomically renamed, so the loader accepts only a
complete restart. Equal-width time blocks are enforced across segments.

Restartable SHA-256 values:

- `solver/JFM_hs_dsmc_quarter.py`:
  `2c9e2f5119802b123f0335664a564085cfe77682ae4f4af4138dc31f5876b166`
- `solver/JFM_bgk_shakhov_quarter.py`:
  `f2f97526942c53eca0af6dd9b94aca541f06951db4755a92cc8fb11e5e0e65a2`
