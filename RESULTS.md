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

## Position-flow ablation (2026-06-11) — the concentrated prior is the fix

Three levers were implemented behind config flags and run as an ablation ladder
to attribute the under-dispersion fix:

| Rung | Config | Loss floor | NN median | in [1.2,1.8] Å | overlaps <0.9 Å | match |
|---|---|---|---|---|---|---|
| — | clumped baseline | 0.130 | 1.05 Å | 17% | 11% | 0% |
| A | churn sampler only (no retrain) | — | ~1.05 Å | 11–25% | 11–17% | 0–3% |
| B | fixed_prior coupling only | **0.052** | 1.08 Å | 22% | 12% | 1.6% |
| **C1** | **wrapped-normal prior only (std 0.25)** | 0.116 | **1.31 Å** | **84%** | **2%** | **4.7%** |
| C2 | all three combined | **0.024** | 1.12 Å | 33% | 6% | 3.1% |

Reference NN median 1.45 Å, vol/atom 6.30 Å³ (C1 matches volume exactly).

**Findings:**
1. **The concentrated fractional prior is the fix (lever 2).** A wrapped-normal
   prior at cell-center (std 0.25) instead of uniform forces the field to learn
   *expansion* rather than contraction, cancelling the mean-field barycenter
   collapse: valid-bond fraction 17% → 84%, overlaps 11% → 2%, cell volume exactly
   on reference. This is the single change that breaks the clumping.
2. **Stochastic sampling does not help (lever 1).** Churn 0→1.0 on the collapsed
   model stays in the 0–3% match noise band — adding noise around an already-
   contracted mean field cannot recover the right spacing.
3. **Lowering the loss floor does not help generation (lever 3).** fixed_prior
   coupling drives the floor lowest (0.024–0.052) by making the per-structure
   target deterministic, but the model overfits those trajectories and a fresh
   test prior still collapses. *The floor was never the cause.*
4. **The combination is worse than the prior alone.** C2 (all three) underperforms
   C1: fixed_prior's overfitting plus churn's re-injected overlaps degrade the
   clean win from the prior. Attribution only visible via the ablation.

Winning config baked into the carbon-24 recipe: `--centroid-prior-std 0.25`
(default), other levers off. Checkpoint: `checkpoints/carbon24_wn.pt`.

## Tuning + scaling sweep (2026-06-12) — match rate plateaus at ~5%

With the collapse fixed, swept the remaining knobs to push match rate. Match rate
on 64 structures is noisy (±1.6%/hit); decisions below use 256-structure eval.

**centroid_prior_std** (6000 steps, 64-struct eval): 0.15→1.6%, 0.20→1.6%,
0.25→4.7%, 0.30→7.8%, 0.35→4.7%. Re-evaluated 0.25 vs 0.30 on **256** structures:
both **3.1%** — the 0.30 peak was small-sample noise. Optimum is a flat plateau
0.25–0.30 (valid-bond ~78–84%, overlaps ≤2%). Default set to 0.30.

**Sampler steps** (std 0.30 ckpt, 256-struct eval): 50→5.1%, 100→5.5%, 200→3.1%.
The deterministic ODE is converged by 50 steps; more steps don't help.

**Model scale** (std 0.30, 256-struct eval): the d_model=192 / 6-layer / 6k-step
model and a d_model=256 / 8-layer / 15k-step model (`carbon24_big.pt`, 2× size,
2.5× steps) both give ~5% match (4.7%) and ~80% valid bonds. **Scaling did not
improve match rate.**

**Conclusion:** the structural pathology is solved (valid bonds 17% → ~82%,
overlaps → ~2%, cell volume on reference). One-to-one StructureMatcher match rate
plateaus at ~5% and is NOT limited by the prior std, sampler steps, or model
capacity — every one of those was ruled out. The remaining gap is the modelling
approach itself on a hard CSP benchmark: the model conditions only on (atom count,
space group), so it generates *a* plausible carbon polymorph, rarely *the* specific
reference. (DiffCSP reports ~17% on carbon-24 with a diffusion objective + richer
coupling.) Best checkpoint: `carbon24_big.pt`.

## Next steps (architectural, not more of the same)

- Best-of-k match-rate metric (generate k per composition, match if any) — the
  standard CSP eval; our one-to-one number understates a generator that produces
  valid-but-different polymorphs.
- Diffusion baseline (DiffCSP-style score model) to compare objectives head-to-head.
- MP-20 loader + full SUN + match-rate benchmark vs CDVAE/DiffCSP.

## Reproduce

```bash
# on a CUDA box (Blackwell needs the cu128 wheel)
pip install -r requirements.txt
pip install pymatgen pandas
# download CDVAE carbon-24 CSVs to data/raw/carbon_{train,val,test}.csv
python scripts/train_carbon24.py --steps 6000 --batch 128
```
