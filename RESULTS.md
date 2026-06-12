# Results — carbon-24 (GPU phase, RTX 5090)

First real-data benchmark of SymMC-Flow on the CDVAE **carbon-24** dataset
(~10k carbon allotropes, 1–24 atoms/cell). Each carbon atom is a single-atom
rigid block, so orientation is disabled (`lambda_orient=0`); the model learns the
lattice + fractional-coordinate flow, conditioned on space groups recovered by
pymatgen (CDVAE CIFs are stored P1).

Hardware: vast.ai RTX 5090 (32 GB), torch 2.11.0+cu128. Model: d_model=192,
6 pair-bias attention layers. 6000 steps, batch 128, ~10–12 min/run.

## Runs

| Metric | + OT coupling | + lattice reparam | + lattice-aware pair feats | Reference |
|---|---|---|---|---|
| Centroid (fractional) loss | **0.09** | 0.09 | 0.09 | — |
| Lattice loss | 0.85 | 0.04 (param space) | 0.04 | — |
| Generated mean C–C distance | 1.01 Å | 1.05 Å | 1.05 Å | 1.45 Å |
| Generated in [1.2, 1.8] Å | 14% | **17%** | 17% | 100% |
| Generated volume / atom | — | **6.57 Å³** | 6.49 Å³ | 6.30 Å³ |
| StructureMatcher match rate | — | 0% | 0% | — |
| det(L) > 0 | 100% | 100% (by construction) | 100% | — |

Lattice-reparam run (2026-06-11, new RTX 5090 box): lattice flowed in
(per-atom log-volume, det-1 shape) ∈ R^10, prior at 9 Å³/atom, network newly
conditioned on the current lattice state (it was ignored before — the lattice
field was not a function of the lattice). 6000 steps, batch 128, ~21 min.

Lattice-aware pair-features run (2026-06-11): added Cartesian minimum-image
displacement (Δfrac @ L) + distance and DiffCSP-style Fourier features of the
fractional offset to the pair-bias input (pair_dim 4 → 20; verified active).
**Null result** — every generation metric is within noise of the previous run.

## Findings

1. **Atom-position flow needs optimal-transport coupling.** With an independent
   random prior→data atom pairing the per-atom velocity target is arbitrary
   (carbon atoms are exchangeable), so the centroid loss plateaus at the marginal
   variance. Per-structure Hungarian assignment on torus distance (`flow.ot_couple`)
   halves the target magnitude (0.26 → 0.12 on a synthetic check) and drops the
   centroid loss 0.25 → 0.09.

2. **The lattice is now the bottleneck.** Despite better fractional coordinates,
   generated structures still collapse (C–C ≈ 1.0 Å vs 1.45 ref) because the
   generated cells are too small/dense (~59 Å³). Raw 3×3 lattice flow with an
   isotropic prior is a weak parametrization, and cell volume should scale with
   atom count (V ∝ N) — which the current head does not capture.

3. **Training-loss floor ≠ failure.** The lattice loss averages over t∈[0,1];
   near t=0 the noisy state cannot reveal the target lattice, so part of the loss
   is irreducible. Sampling quality (above) is the real metric.

## Findings (lattice-reparam run)

4. **The volume problem is solved.** Generated cells now sit at ~6.5 Å³/atom vs
   6.30 reference (was ~59 Å³ total regardless of atom count). Volume scaling
   with N plus lattice-state conditioning fixed the cell-size collapse.

## Findings (lattice-aware pair-features run)

5. **Lattice-aware pair features did not help — the bottleneck is not the
   features.** Adding Cartesian + Fourier pair geometry left the centroid loss
   pinned at 0.09 and generation unchanged (C–C 1.05 Å, 0% match). The centroid
   CFM loss is already at its *irreducible floor* set by the OT-coupled target's
   residual stochasticity; richer geometric features cannot lower a floor that is
   a property of the objective, not the network.

6. **The real failure is mean-field under-dispersion of the position flow.**
   Eval harness is sound (reference-vs-reference match = 100%, ref-vs-shuffled =
   4.7% from genuine duplicate polymorphs), so 0% is real. The diagnostic shows
   the *whole* generated nearest-neighbour distribution is uniformly contracted:

   | | min | median | max |
   |---|---|---|---|
   | generated NN dist | 0.80 Å | 1.11 Å | 1.44 Å |
   | reference NN dist | 1.38 Å | 1.45 Å | 1.52 Å |

   The best generated structure (1.44 Å) barely reaches the reference *minimum*
   (1.38 Å), and 9% of structures have sub-0.9 Å overlaps. This is the classic
   signature of flow matching regressing the blurry *conditional-mean* velocity
   over exchangeable, multimodal position targets: integrating the mean field
   from a maximally-spread uniform prior over-contracts every structure.

## Next steps (position objective — the actual lever)

The fix is in the coupling/prior/sampler, not more conditioning features:
- **Wrapped-normal prior centered near data** (DiffCSP/FlowMM) instead of a
  uniform fractional prior, so the field need not learn a large mean contraction.
- **Stochastic sampling** (SDE / Langevin corrector steps) to inject the
  dispersion that the deterministic mean ODE removes.
- **Sharper coupling**: cache a fixed prior per structure (or anneal the OT cost)
  so the per-state target is less multimodal across minibatches.
- Then MP-20 loader + diffusion baselines.

## Reproduce

```bash
# on a CUDA box (Blackwell needs the cu128 wheel)
pip install -r requirements.txt
pip install pymatgen pandas
# download CDVAE carbon-24 CSVs to data/raw/carbon_{train,val,test}.csv
python scripts/train_carbon24.py --steps 6000 --batch 128
```
