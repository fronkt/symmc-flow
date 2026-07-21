# Packing-bottleneck root-cause findings (2026-07-20, pre-lit-review grounding)

Context: E4 showed end-to-end match@10 ~0% and cell-volume RMAD ~31-33% vs MolCrystalFlow
3.86%. Orientation is solved by the coset (103deg -> 14deg). This doc records what our own
code + persisted E4 draws say about WHY packing fails, to ground the "go further" decision.

## Architecture facts (from symmc_flow/)
- Lattice is flowed in `k=(s, vec(S)) in R^10`: `s=log(V/n)` per-atom log-volume, `S=L/V^(1/3)`
  det-1 shape (manifolds.py:164-197). Volume and shape are already decoupled -- good design.
- Space group `sg` IS globally embedded into every token (model.py:111, `sgemb`). But the
  lattice head emits a FREE 10-DOF velocity: nothing constrains `S` to the space group's
  crystal family (cubic a=b=c, monoclinic alpha=gamma=90, ...). Model must LEARN symmetry
  from the embedding rather than have it enforced.
- Coset conditioning (the paper's contribution) touches ONLY orientation, not the lattice.
- Lattice prior: `V ~ lognormal`, mean `vol_per_atom(=10) * n`, `logvol_std=0.3` (~30% spread).
- Sampler churn = 0 (deterministic RK4), so extra variance is NOT injected noise -- it's the
  learned lattice field / integration.

## THE decisive diagnostic (scratchpad/diag_volume_prior.py on e4_draws.pkl, n=131)
Reference organic crystals pack at **10.16 +/- 1.15 A^3/atom (CV ~11%)** -- nearly constant
density. So:

| predictor                              | cell-volume RMAD |
|----------------------------------------|------------------|
| TRIVIAL  V = 10 * n_atoms (no learning)| **7.6%**         |
| our model, de-novo (coset off)         | 32.9%            |
| our model, coset on                    | 30.7%            |
| MolCrystalFlow (reference)             | 3.9%             |

- Our flow is **4x WORSE than the zero-learning constant-density baseline.**
- Generated vol/atom = 8.40 +/- **3.62** (CV 43%): the flow WIDENS the distribution 3x beyond
  the reference spread (1.15) and beyond its own prior (~30%). De-novo also undershoots the
  mean (8.4 vs 10.2 = -17% bias).
- corr(ref vol/atom, gen vol/atom) r = 0.26 -- barely tracks true density.

### Root cause
The volume prior (30% spread) is ~3x too wide for an 11%-spread quantity, and the lattice flow
adds variance instead of contracting it. A 33% volume error alone makes StructureMatcher
match impossible at any reasonable stol -> a large part of the 0% match rate is THIS.

### Cheap, high-payoff fixes (to validate against lit review)
1. **Tighten the volume prior** `logvol_std 0.3 -> ~0.11` to match data spread. Nearly free
   (one hyperparam + retrain). Expected: RMAD toward the 7.6% baseline.
2. **Anchor volume on composition** (V ~ sum of atomic/molecular volume contributions;
   organics are near-deterministic). Could beat 7.6% toward MCF's 3.9%.
3. **Crystal-family constraint on the shape S** (reparameterize S to the space group's free
   lattice DOF; reconstruct locked params). Attacks SHAPE/match, orthogonal to volume.
   This is the thesis-consistent "fully symmetry-conditioned lattice" extension.

Volume (RMAD) and shape/family (match) are DISTINCT levers for the two axes MCF beats us on.

## SHAPE diagnostic (scratchpad/diag_shape_family.py, same pickle) -- confirms lever #3
Real molecular crystals are dominated by monoclinic/orthorhombic (alpha=gamma=90 or all 90):

| source            | angles within 0.5deg of 90 | mean|ang-90| | rel. length err |
|-------------------|----------------------------|--------------|-----------------|
| reference (real)  | **72.0%**                  | 3.08 deg     | --              |
| gen (coset off)   | **3.1%**                   | 10.41 deg    | 26.4 +/- 13.2%  |
| gen (coset on)    | 3.0%                       | 11.22 deg    | (same lengths)  |

The free 9-DOF shape head ERASES the 90-deg spike: 72% -> 3%. It ignores the crystal family
entirely even though `sg` is embedded. Reparameterizing `S` to the space group's free lattice
DOF (crystal-family constraint) is the thesis-consistent fix -> directly attacks match rate.

## Lit-review corroboration (subagent 1: MCF + MOFFlow methods)
MCF's 3.86% RMAD is NOT from a clever lattice flow. It comes from:
  (a) DATA-INFORMED lattice prior (Gaussian-fit lengths, uniform[60,120] angles),
  (b) post-hoc HARD-SPHERE RIGID-PRESS (BFGS overlap min; cut Genarris RMAD 59% -> 10.7%),
  (c) two-stage u-MLIP relaxation (finisher).
MCF DELIBERATELY DOWN-WEIGHTS its lattice loss 20x (lambda_L=0.1 vs lambda_F=2) -- a heavy
lattice flow fights the coordinate flow. We use lambda_lattice=1.0 (free flow). MCF predicts
clean L1 (denoised), not a velocity. We already use fractional centroids on a torus (the RIGHT
side of the MCF-vs-MOFFlow split). MOFFlow (Cartesian centroids, no post-proc) packs molecular
crystals badly (RMAD 18.8%) -- validates our torus choice.
Refs: MCF arXiv:2602.16020v3; MOFFlow arXiv:2410.17270.

Cheap no-retrain wins ranked by subagent 1: (1) rigid-press, (2) data-informed prior,
(3) inference velocity-annealing, (4) length-sorted lower-triangular lattice canonicalization.

## DECISIVE ceiling experiment (scratchpad/diag_volfix_ceiling.py) -- volume is NOT the match blocker
Isotropically rescale every generated cell (shape-preserving: fractional coords unchanged)
to a target volume, then re-score match@10:

| variant (coset ON) | RMAD  | match@10 stol0.8 / 1.0 / 1.2 |
|--------------------|-------|------------------------------|
| as-is              | 30.7% | 0.0% / 0.8% / 0.8%           |
| const-density (V=10*n, deployable) | 7.6% | 0.0% / 0.8% / 0.8% |
| ORACLE volume (V=Vref)             | 0.0% | 0.0% / 0.8% / 0.8% |

**Perfect volume changes match@10 by nothing.** The 0% match is blocked by SHAPE (crystal-
family angles 72%->3% at 90deg; 26% length error) + molecular placement, NOT volume. Volume
(RMAD) is a cheap free win but a SEPARATE axis from match rate.

## Subagent 2 (symmetry-conditioned lattice) -- the novel lever + the gap
Crystal-family lattice masking is THE established win: reparameterize L = Q*exp(S),
S = sum_i k_i B_i, k in R^6 (O(3)-invariant log-metric), then a binary family mask m(G) freezes
the constrained k-dims (cubic 1 free DOF, tetragonal 2, monoclinic 4, triclinic 6). Proven:
DiffCSP++ 51%->80% match on MP-20; SGFM (arXiv:2509.23822) 82.7% vs FlowMM 61.4% -- and SGFM
PROVED masked-k works INSIDE flow matching (no flow-vs-diffusion risk). The k-representation is
IDENTICAL to CrystalFlow's (subagent 3) whose log-volume = tr(S) is one clean coordinate -> the
same swap fixes volume AND enables the family mask (two birds).
THE GAP: family masking exists ONLY for inorganic single-atoms (DiffCSP++, SGFM, SymmCD,
NextCrystal). Every MOLECULAR learned flow (MolCrystalFlow, PackFlow, MOFFlow) is space-group-
FREE, 6-DOF unconstrained lattice, EACH naming SG conditioning as future work. The only molecular
SG-conditioned method (Genarris-3) is a random sampler, not a learned flow. => a fully
symmetry-conditioned molecular-crystal FLOW (family-masked lattice + coset orientation) is
GENUINELY UNFILLED -- exactly our thesis extended from orientation to the lattice.
Honesty caveats: the mask mechanism itself is DiffCSP++ (not novel) -- our novelty is
transporting it into molecular rigid-body flow + composing with coset orientation. And the family
mask is NECESSARY-not-SUFFICIENT: it fixes cell-shape drift (angles) but Wyckoff-centroid
projection + rigid-press likely still needed to actually recover match@10.
Refs: DiffCSP++ arXiv:2402.03992; SGFM arXiv:2509.23822; CrystalFlow arXiv:2412.11693;
NextCrystal arXiv:2602.17176; PackFlow arXiv:2602.20140; Genarris-3 DOI 10.1021/acs.jctc.5c01080.

## Subagent 3 (lattice accuracy) -- the volume fix
Molecule-only density is predictable to <2% MAE (MolXtalNet-D, arXiv:2303.10140). A post-hoc
isotropic rescale to a predicted target volume HARD-CAPS RMAD at the predictor's error (~2-5%),
zero retrain. CrystalFlow's log-Euclidean lattice (best density Wasserstein 0.169 vs DiffCSP
0.350) = the same k-repr as the family mask. Per-entry MSE under-penalizes correlated volume
scaling -> add a |log det L| term. But (see ceiling experiment) NONE of this moves match@10.

## STRATEGIC READ (for the E5-vs-PhaseF decision)
- VOLUME/RMAD: cheaply fixable (const-density 7.6% today; density-predictor <5%), zero GPU, but
  does NOT help match rate. Worth shipping into JCIM tab:bench regardless.
- MATCH RATE (the headline): blocked by cell shape + placement. Only a retrain with the crystal-
  family-masked lattice (+ informed prior + rigid-press finisher) has a shot at moving it off 0.
  This is the genuinely novel, unfilled "fully symmetry-conditioned molecular flow" contribution.
  Risk: family mask is necessary-not-sufficient; may need Wyckoff-centroid + rigid-press + more
  corpus scale to get non-trivial match. Needs GPU + Frank's call on direction.


