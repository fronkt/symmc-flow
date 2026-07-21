# Phase F3 spec — symmetry-preserving rigid-press packing finisher

## Why (from F2)
F2 built the first fully symmetry-conditioned molecular flow and both symmetry levers work
exactly as designed (coset orientation 92°→18°; family-mask cell-angle 16%→74%=ref). Yet exact
match@10 stayed 0% at stol ≤ 1.0. The residual blocker is **fine molecular positioning /
close-packing precision** — neither orientation, nor crystal family, nor volume (oracle-volume
ceiling moved match by 0). MCF/MOFFlow reach match via an *unsupervised physical rigid-press
finisher* applied to their raw draws, not via the flow. F2 removed two of the three obstacles, so a
finisher applied now polishes structures already in the right basin (right family + right
orientation) — the case F2 was pre-registered to strengthen.

## Hypothesis (pre-registered)
Applying a symmetry- and rigidity-preserving physical packing relaxation to the family-mask model's
draws moves coset-ON match@10 (stol ≤ 1.0) off 0 (CI-separated), and/or substantially reduces the
median best-of-k structural distance to the held-out reference (min-atom-RMSD after StructureMatcher
alignment). If it does not, symmetry conditioning + a local physical finisher are *together* still
insufficient at diagnostic scale → the honest-null (F4) stands, now with the finisher ruled out too.

## Design (self-contained, unsupervised)
Hook the finisher between `rk4_sample` (returns CrystalState) and `rigid_to_structure` in the eval
loop (`eval_e4:204-218`). It never sees the reference structure — only the generated draw + the
template's space-group (the same coset template that conditioned the sampler).

**DOF (per crystal):** for each asymmetric-unit (is_ref) molecule of each species group —
centroid `c0 ∈ ℝ³` (fractional) + orientation as a tangent `δ ∈ ℝ³` with `R0(δ)=so3_exp(δ)·R0_init`
(δ init 0) — plus the cell `k ∈ ℝ⁶` (logmetric6), re-`apply_family_mask(k, sg)` every closure so the
cell stays on-family. Symmetry copies are **regenerated every step** from the ASU pose:
`c_g = wrap(W_h·c0 + t_h)` (`get_ops(sg).act`), `R_g = Rcart_h·R0` (`cartesian_rotations(sg, L)`),
with per-copy op index `h` derived once from the clean template centroids (`symmetry_op_indices`,
same centroid-matching as `assign_symmetry_cosets`). Rigid geometry is preserved by construction
(atoms = `rigid_to_frac`, `local` fixed).

**Objective — intermolecular packing energy** (`symmc_flow/rigid_press.py`):
smooth Lennard-Jones over atom pairs across molecule instances + periodic images within a cutoff,
`φ(d)=ε[(R/d̃)¹²−2(R/d̃)⁶]`, `d̃=max(d,d_floor)`, contact `R=f·(r_i+r_j)` from pymatgen vdW radii
(f≈0.9). **Intramolecular same-cell pairs excluded** (rigid → fixed, would dominate). A light volume
prior `λ_v·(ln V − ln V₀)²` (V₀ from the corpus's 10 Å³/atom) anchors k0 against LJ pressure
runaway. Minimise with `torch.optim.LBFGS` (~40 iters), CPU-fine (tens–low-hundreds of atoms).

## Sub-phases
- **F3a (LOCAL, CPU, free):** implement `rigid_press.py` + tests; new `scripts/eval_f3_finisher.py`
  regenerates draws from `checkpoints/coset_fammask_s0.pt`, finishes each, scores raw-vs-finished
  PAIRED (match@10 / RMAD / orient-err / min-atom-RMSD). Smoke on ~20 crystals, then full 131.
- **F3b (GPU, only if F3a clears the gate):** fold the finisher into the sampler + optionally fix the
  under-training centroid head (OT coupling / fixed prior / centroid_prior_std — all already in the
  repo, `flow.py:50-126`), retrain + scale, re-run E4-aligned eval.

## Pre-registered F3a gate
Promote to F3b iff, on the full 131 val crystals, the finisher moves **coset-ON match@10 at
stol ≤ 1.0 off 0 with CI separation from raw**, OR drops the **median best-of-k min-atom-RMSD to
reference by ≥ 30%** (clear basin improvement). Else → F4 honest-null (finisher ruled out; symmetry
conditioning necessary-but-not-sufficient even with a physical polish at this scale).

## Non-negotiables / risks
- Finisher is UNSUPERVISED (physical energy only) — never reads the reference; scoring stays honest.
- Gated behind a flag; the committed shape10 path + F1/F2 results are untouched (minimal impact).
- Risk: 18° orient + crude centroid may sit outside the finisher's local basin → it can't recover
  (local optimiser). That is exactly what F3a measures. No goalpost-moving on the gate.
- Frank owns final prose + the F3a→F3b GPU/scale call; [[feedback_no_coauthor]]; specific-path staging.
