# Molecular-crystal benchmark (rigid-body, orientation-ON) — status

Status doc for the thesis-gap experiment: validate the headline novelty (rigid-body
molecular conformers + SO(3) **orientation** flow) on **real** molecular crystals, with
`lambda_orient > 0`. Both shipped real benchmarks (mp20, carbon24) use single-atom blocks
with `lambda_orient = 0`, so orientation was previously validated only on synthetic data.

_Last updated: 2026-06-18._

## TL;DR

The data pipeline and loader work end-to-end on real CSD data. **The lattice and centroid
flow heads learn on real molecular crystals; the SO(3) orientation head does not — it sits
at its predict-zero floor across two corpus sizes (250 and 1127 structures) and after a
gauge fix.** This is a robust negative: the orientation novelty does **not** work as-is on
real data. Leading hypothesis for why: in one-shot CFM the orientation head must predict a
molecule's pose from *noised* lattice+centroids, but that packing is exactly what determines
orientation. Next lever: two-stage / cleaner conditioning (not yet built).

## What was built (all committed, reproducible)

| component | file | what it does |
|---|---|---|
| Loader | `symmc_flow/molcrystal.py` | crystal → rigid molecular blocks `(Z, local, centroid, orient)`; PBC-unwrap molecule detection (JmolNN), per-species conformer registry, Kabsch align with automorphism-min mapping, **molecule-intrinsic gauge** (`_canonical_frame`), disorder + non-rigid skip gates, tolerant `read_cif` (special-position occupancies). Same batch dict as mp20/carbon24 → **no model change**. |
| CSD export | `scripts/csd_export.py` | runs under the **CCDC bundled interpreter**; seeded random CSD sample → filter (organic, ordered, 3D, not polymeric, R≤rmax, 0<Z'≤zmax, size cap) → CIFs + shareable `manifest.csv`. |
| COD fallback | `scripts/fetch_cod_molcrystals.py` | license-free path (COD `el…&strictmax&format=lst`); same downstream. |
| Factorizer | `scripts/factorize_cifs.py` | any CIF dir → `MolCrystalDataset` + kept/skip + Z'/atoms histograms. |
| Training | `scripts/train_csd_molcrystal.py` | orientation-ON training on the real corpus; train/val split; per-head loss untrained→trained; lattice prior matched to corpus volume/atom. |

**Reproduce** (CSD CIFs are not redistributable → `data/csd_mol/` is gitignored; the export
is reproducible from seed + CSD version `601`):

```bash
# 1. export (CCDC interpreter)
"C:/Users/frank/CCDC/ccdc-software/csd-python-api/miniconda/python.exe" \
    scripts/csd_export.py --n 3500 --sample 18000 --seed 0 --out data/csd_mol
# 2. factorize (project env)
python scripts/factorize_cifs.py --cif-dir data/csd_mol/cif --cache data/csd_mol/ds.pt
# 3. train orientation-ON
python scripts/train_csd_molcrystal.py --cache data/csd_mol/ds.pt --steps 1500 --lr 3e-4
```

## Results

Corpus: 3500 CSD CIFs (14,874 scanned) → **1127 crystals / 5048 rigid blocks** (also ran a
250-structure corpus first; same conclusion).

| head | untrained | trained | verdict |
|---|---|---|---|
| lattice | 1.4 | ~0.06–0.10 | **learns** |
| centroid | 0.36 | ~0.24 | **learns** |
| **orientation** | 5.24 (predict-zero floor) | **~5.0–5.4, no downward trend** | **at the floor — does not learn** |

- Decisive diagnostic (per the predict-zero-floor lesson): E‖u_R‖² = **5.24/5.29**; trained R
  never leaves the noise band around it over 200–350 steps while lattice/centroid converge in
  <1 epoch.
- **Scaling 250 → 1127 did not change this** → not a data-sparsity problem.
- A gauge fix was necessary first (see below) but **not sufficient**.

## What was diagnosed / fixed along the way

1. **Arbitrary global gauge → molecule-intrinsic frame.** Originally a species' body frame was
   set by whichever copy was parsed first *anywhere in the dataset*, so `R_m` was "rotation
   vs a molecule in another crystal" — conditionally random, a guaranteed floor. Fixed with
   `_canonical_frame` (gyration-tensor principal axes, element-weighted 3rd-moment sign,
   right-handed). Gauge-free → all tests + kept/skip set unchanged. Necessary, not sufficient.
2. **Early-Adam NaN at lr=1e-3.** The harder intrinsic-frame targets gave larger early
   gradients; Adam's tiny initial variance estimate amplified them (NaN ~step 17, shared
   trunk). Fixed: default lr → 3e-4.
3. **CSD/CCDC sourcing gotchas** (in `tasks/lessons.md`): Access Structures is consent-gated /
   not scriptable; `ccdc.hasattr` lies (`is_polymeric` is on `molecule` not `crystal`); COD
   uses `strictmin/strictmax` not `nel`; pymatgen "no structures" on CSD CIFs = special-position
   occupancy >1, fixed by `occupancy_tolerance`.

## Next steps (ranked, none started)

1. **Two-stage / cleaner conditioning (recommended).** Generate lattice+centroid first, then
   orientation conditioned on the (near-clean) packing, so the SO(3) head sees the geometry
   that determines orientation. Directly tests the leading hypothesis. Architectural change
   to model + sampler + training.
2. **Diagnostic: condition orientation on the TRUE (un-noised) lattice+centroids.** Cheap test
   — if orientation then learns, it confirms noised conditioning is the cause and motivates #1.
3. **Honest reframe (fallback).** Report rigid-body lattice+centroid flow as the contribution
   and characterize orientation as an open problem, using these floor diagnostics as evidence.
4. **Deprioritized:** min-over-symmetry CFM orientation target — helps only the symmetric
   molecular minority (corpus is asymmetric-dominated) and is heavy plumbing.

## Artifacts (local, not in git)

- `data/csd_mol/cif/` — 3500 CSD CIFs; `data/csd_mol/ds.pt` — factorized 1127-structure cache;
  `data/csd_mol/manifest.csv` — shareable refcode manifest.
- `checkpoints/csd_molcrystal*.pt` — trained checkpoints (incl. per-head pre/post val).
