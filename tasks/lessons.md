# Lessons

<!-- Capture a rule after any correction. Format below. -->

## 2026-06-10 — Project kickoff
- **Rule**: Architecture must stay dataset-agnostic — keep all dataset specifics
  behind loaders in `data.py` so the synthetic harness and real MP-20/carbon
  loaders are interchangeable.
- **Context**: SymMC-Flow CPU reference impl; GPU benchmark is a separate phase.

## 2026-06-10 — Flow-matching targets must carry conditional signal
- **Mistake**: Drew synthetic orientations as uniform-random SO(3) with no link to
  the conditioning input; the orientation CFM loss was irreducible (~5) because the
  target is conditionally random — no deterministic field exists to fit.
- **Rule**: When validating a conditional generative model on synthetic data, make
  each target a (mostly) deterministic function of the conditioning input plus small
  noise. A flat plateau in one loss head usually means "unlearnable target," not a
  broken network — check data generation before debugging the model.
- **Context**: Any flow-matching / diffusion demo with a synthetic harness.

## 2026-06-10 — SO(3) log near θ=π
- **Mistake**: Generic `θ/(2 sinθ)·vee(R−Rᵀ)` log is ill-conditioned within ~0.05 rad
  of π; the (R+I)/2=aaᵀ axis trick is only exact at π and degrades away from it.
- **Rule**: Use Shepperd's quaternion method for rotation→axis-angle; it's stable for
  all angles including π. Re-orthonormalize SO(3) via SVD after long ODE integration.
- **Context**: Riemannian flow matching / any SO(3) numerics.

## 2026-06-10 — Git commits
- **Rule**: Never add `Co-Authored-By` trailers to commits in this user's repos.
- **Context**: Standing user preference.
