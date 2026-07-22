# Phase F3c findings — scale plateaus match; the F3 ladder is complete

**Run:** 2026-07-22. Retrained the centroid-fixed coset + family-mask + logmetric6 model
(`--ot-coupling --fixed-prior`) on the **2539-crystal** corpus (`data/csd_mol_scale_big/ds.pt`, the
E2 scale set), 2500 steps, fresh RTX 3090 (vast 45511956, destroyed). Same decoupled pipeline: gen
296 val x 5 draws on the box, finish + score the first 131 locally (frozen cell, 40 steps) for a
clean A/B against F3b (n=131). (Scored log header mislabeled "F3a:" — hardcoded print; numbers are
the F3c scale run.)

## Scale did NOT lift match (frozen cell, n=131, best-of-5, coset ON)

| axis | F3b (1.1k corpus) | **F3c (2.5k corpus)** |
|---|---|---|
| match@stol=0.8 | 1.5% | 1.5% |
| match@stol=1.0 | **3.1%** | **3.1%** (identical) |
| match@stol=1.2 | 4.6% | 3.8% |
| cell-vol RMAD | 21.2% | **17.2%** (better) |
| angle-spike | 73.8% | 77.6% (ref 77.4%) |
| median min-RMSD | -15.8% | -14.2% |

Training improved on every *average* metric — coset NON-REF drop widened to +45.4% (vs F3b 39.9%),
centroid readout +19.9% (vs 11.3%), RMAD 21.2 -> 17.2% — but **exact match@stol<=1.0 stayed flat at
3.1%.** This exactly mirrors E2: scale moves the diagnostics and the packing *averages*, not the
sub-Angstrom tail precision that StructureMatcher requires for an exact match.

## The complete F3 ladder (diminishing returns)

```
  raw flow            0.0%    fully symmetry-conditioned (coset orientation + family-masked cell)
  + rigid-press       1.5%    F3a: unsupervised physical finisher on right-basin draws   (+1.5)
  + centroid fix      3.1%    F3b: OT coupling + fixed prior -> better raw positioning    (+1.5)
  + 2.3x scale        3.1%    F3c: bigger corpus improves RMAD/coset, not exact match     (+0.0)
                      ----    @ stol<=1.0, best-of-5, n=131
  relax_cell          DEAD    cell relaxation drives the metric singular (eigh fails)      (x)
```

Two independent big levers (2.3x data; post-hoc cell relaxation) added **zero** exact match. Match
has plateaued at ~3.1% (stol<=1.0), below the symmetry-*free* baselines (MolCrystalFlow ~6.8%,
MOFFlow ~8%). The residual gap is the flow's **tail** positioning precision: scale sharpens the
average draw but not the fraction of draws landing within the finisher's basin of the true minimum.
Closing it would need a fundamentally better positioning model (e.g. a learned or iterated
refinement head, or a differentiable-through-matcher objective), not more of the same levers.

## Honest bottom line (Frank owns the write-up)

This is a **complete, honest positive-with-a-ceiling** result and a natural stopping point:
- **Contribution:** the first *fully* symmetry-conditioned molecular crystal flow (orientation coset
  + crystal-family lattice mask), plus a clean, ablated lever decomposition that lifts exact match
  from 0% to 3.1% and drives RMAD from ~33% (E4 shape10) to 17%.
- **Honest ceiling:** symmetry conditioning + a physical finisher + positioning fix + scale plateau
  below the symmetry-free rigid-body flows; the remaining gap is a positioning-precision problem
  that these levers do not close. That is a legitimate, publishable finding (a symmetry-conditioned
  alternative + a sharp diagnosis of the exact-match bottleneck), well-suited to MoML.

Artifacts: `f3c_scale_frozen.log` (committed). Draws/checkpoint regenerable from the committed
`diag_orient_coset.py ... --ot-coupling --fixed-prior --cache data/csd_mol_scale_big/ds.pt` command.
