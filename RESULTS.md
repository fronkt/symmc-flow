# Results — carbon-24 (GPU phase, RTX 5090)

First real-data benchmark of SymMC-Flow on the CDVAE **carbon-24** dataset
(~10k carbon allotropes, 1–24 atoms/cell). Each carbon atom is a single-atom
rigid block, so orientation is disabled (`lambda_orient=0`); the model learns the
lattice + fractional-coordinate flow, conditioned on space groups recovered by
pymatgen (CDVAE CIFs are stored P1).

Hardware: vast.ai RTX 5090 (32 GB), torch 2.11.0+cu128. Model: d_model=192,
6 pair-bias attention layers. 6000 steps, batch 128, ~10–12 min/run.

## Runs

| Metric | Baseline (random pairing) | + OT coupling | Reference (real) |
|---|---|---|---|
| Centroid (fractional) loss | 0.25 (plateau) | **0.09** | — |
| Lattice loss | 0.80 | 0.85 | — |
| Generated mean C–C distance | 1.01 Å | 1.01 Å | 1.45 Å |
| Generated in [1.2, 1.8] Å | 11% | 14% | 100% |
| Generated cell volume | 59.6 Å³ | 58.9 Å³ | (larger) |
| det(L) > 0 | 100% | 100% | — |

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

## Next steps (lattice reparametrization)

- Flow the lattice in **log-volume + normalized-shape** space instead of raw 3×3.
- Scale the lattice prior by **N^(1/3)** and condition cell size on atom count.
- Add a proper SUN/match-rate evaluation (StructureMatcher against the test set)
  rather than the C–C distance proxy used here.
- Then re-run the baseline / OT / reparametrized comparison.

## Reproduce

```bash
# on a CUDA box (Blackwell needs the cu128 wheel)
pip install -r requirements.txt
pip install pymatgen pandas
# download CDVAE carbon-24 CSVs to data/raw/carbon_{train,val,test}.csv
python scripts/train_carbon24.py --steps 6000 --batch 128
```
