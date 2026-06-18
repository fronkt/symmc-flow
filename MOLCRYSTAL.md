# Molecular-crystal benchmark (rigid-body, orientation-ON) — status

Status doc for the thesis-gap experiment: validate the headline novelty (rigid-body
molecular conformers + SO(3) **orientation** flow) on **real** molecular crystals, with
`lambda_orient > 0`. Both shipped real benchmarks (mp20, carbon24) use single-atom blocks
with `lambda_orient = 0`, so orientation was previously validated only on synthetic data.

_Last updated: 2026-06-18._

## TL;DR

The data pipeline and loader work end-to-end on real CSD data. **The lattice and centroid
flow heads learn on real molecular crystals; the SO(3) orientation head does not — it sits
at its predict-zero floor across two corpus sizes (250 and 1127 structures), after a gauge
fix, and even when conditioned on the TRUE clean packing.** This is a robust negative: the
orientation novelty does **not** work as-is on real data.

**The leading hypothesis — that noised lattice+centroid conditioning was starving the SO(3)
head of the packing geometry that determines orientation — has now been TESTED and REJECTED.**
A clean-packing diagnostic (`scripts/diag_orient_conditioning.py`, the `cond_clean_packing`
flag) fed the field the un-noised lattice+centroids while still noising orientation. Over 800
steps on the 1127-structure corpus, lattice (1.20→0.044) and centroid (0.355→0.254) learn, but
orientation moves only 5.35→5.18 (+3.3%) — still at its ~5.2 floor. Since clean conditioning is
the absolute best case for the second stage of a two-stage model, **this rules out the
two-stage fix.** The cause is deeper: orientation is not recoverable from packing geometry in
this mostly-asymmetric, ~1-crystal-per-molecule corpus under the absolute-target CFM (even
gauge-fixed). Direction now: the honest paper-fallback framing (#3 below).

## What was built (all committed, reproducible)

| component | file | what it does |
|---|---|---|
| Loader | `symmc_flow/molcrystal.py` | crystal → rigid molecular blocks `(Z, local, centroid, orient)`; PBC-unwrap molecule detection (JmolNN), per-species conformer registry, Kabsch align with automorphism-min mapping, **molecule-intrinsic gauge** (`_canonical_frame`), disorder + non-rigid skip gates, tolerant `read_cif` (special-position occupancies). Same batch dict as mp20/carbon24 → **no model change**. |
| CSD export | `scripts/csd_export.py` | runs under the **CCDC bundled interpreter**; seeded random CSD sample → filter (organic, ordered, 3D, not polymeric, R≤rmax, 0<Z'≤zmax, size cap) → CIFs + shareable `manifest.csv`. |
| COD fallback | `scripts/fetch_cod_molcrystals.py` | license-free path (COD `el…&strictmax&format=lst`); same downstream. |
| Factorizer | `scripts/factorize_cifs.py` | any CIF dir → `MolCrystalDataset` + kept/skip + Z'/atoms histograms. |
| Training | `scripts/train_csd_molcrystal.py` | orientation-ON training on the real corpus; train/val split; per-head loss untrained→trained; lattice prior matched to corpus volume/atom. |
| Diagnostic | `scripts/diag_orient_conditioning.py` | clean-packing test (`cond_clean_packing` flag, `train._step_loss`): conditions the field on the TRUE lattice+centroid to isolate whether noised packing floors the SO(3) head. Rules out two-stage. |

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

### Clean-packing diagnostic (2026-06-18) — rules out two-stage conditioning

`scripts/diag_orient_conditioning.py` sets `TrainConfig.cond_clean_packing=True`: the field is
fed the **true** (un-noised) lattice+centroid while orientation is still noised (`z_t.orient`)
and the orient target `u_R` is unchanged (no label leakage — `z1.orient` is never shown). This
is exactly the conditional flow "orientation | true packing" — i.e. **stage 2 of a two-stage
model in its best case.** If orientation learns here, two-stage is validated; if not, two-stage
cannot help.

| head | untrained | trained (800 steps) | note |
|---|---|---|---|
| lattice | 1.204 | 0.044 | learns (ill-posed target under this flag; ignore) |
| centroid | 0.355 | 0.254 | learns (idem) |
| **orientation** | 5.352 | **5.175 (+3.3%)** | **still at floor; R train oscillates 4.69–5.90, no trend** |

**Verdict: NULL.** Orientation does not learn even from the true packing → the two-stage fix is
ruled out, and the cause is deeper than noised conditioning (absolute SO(3) target carries no
learnable signal across this asymmetric, ~1-crystal-per-molecule corpus). Checkpoint
`checkpoints/diag_orient_cleanpack.pt` (pre/post + split). Reproduce:
`python scripts/diag_orient_conditioning.py --cache data/csd_mol/ds.pt --steps 800`.

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

## Next steps (ranked)

1. **Honest reframe — now the lead.** Report rigid-body **lattice+centroid** flow as the
   contribution and characterize **orientation as a well-diagnosed open problem**, with the
   floor evidence: predict-zero floor, gauge fix necessary-not-sufficient, scaling 250→1127
   no help, and the clean-packing diagnostic ruling out two-stage. This is a publishable,
   carefully-bounded negative on the orientation sub-problem layered on positive lattice+
   centroid results.
2. **~~Two-stage / cleaner conditioning.~~ RULED OUT (2026-06-18)** by the clean-packing
   diagnostic above — orientation does not learn even on the true packing.
3. **Last technical lever before fully committing to #1: change the orientation TARGET, not the
   conditioning.** The absolute per-molecule `R_m` (even gauge-fixed) may carry no learnable
   signal across an asymmetric, ~1-crystal-per-molecule corpus. Options: (a) restrict to
   species that recur across multiple crystals so a *relative* orientation target is defined;
   (b) min-over-site-symmetry CFM target (only helps the symmetric minority — heavy plumbing,
   low expected yield). If (a) also floors, #1 is final.

## Artifacts (local, not in git)

- `data/csd_mol/cif/` — 3500 CSD CIFs; `data/csd_mol/ds.pt` — factorized 1127-structure cache;
  `data/csd_mol/manifest.csv` — shareable refcode manifest.
- `checkpoints/csd_molcrystal*.pt` — trained checkpoints (incl. per-head pre/post val).
- `checkpoints/diag_orient_cleanpack.pt` — clean-packing diagnostic checkpoint (per-head pre/post + split).
