# Phase F3d spec — learned self-conditioning refinement (the novel positioning lever)

## Why (from F3a-c)
Exact match plateaued at 3.1% (stol<=1.0). The lever ladder (finisher +1.5, centroid-fix +1.5,
scale +0.0; relax_cell dead) shows the binding constraint is the flow's **tail positioning
precision**: the coset gives ~18 deg median orientation and the centroid head is crude, and the
physical rigid-press is a *local* polish that cannot cross that gap. Scale sharpens the average draw
but not the fraction of draws that land inside the finisher's basin of the true minimum. Closing the
gap needs a better *positioning model*, not more of the same levers.

## The lever: self-conditioning iterative refinement
Standard in protein/structure flow-matching (Chen et al. self-conditioning; AlphaFold recycling),
**unused in molecular-crystal flows**. Let the network see its own current estimate of the terminal
structure and predict a correction; apply it iteratively at inference.

- **Architecture (model.py):** add an optional self-conditioning input = a prior estimate
  `(L_hat, x_hat, R_hat)` of the terminal state, embedded and added to the per-molecule / lattice
  tokens (zeros when absent, so the shape10/no-selfcond paths are byte-identical). Gated by
  `ModelConfig.self_cond: bool = False`.
- **Training (train.py):** with prob 0.5, run a first no-grad forward from `z_t` to get
  `x1_hat = z_t + (1-t) * v` (the flow's terminal estimate), then run the real, gradient forward
  conditioned on `x1_hat` (detached). The head learns to sharpen its own estimate. No target change.
- **Inference (sampler.py):** carry the running terminal estimate as self-cond across RK4 steps
  (self-conditioning), and optionally add M **recycling** passes at t=1 (re-embed the finished
  estimate, re-predict) before the physical finisher. Each pass tightens orientation + centroid.
- Composes with coset + logmetric6 + family-mask + OT/fixed-prior (all orthogonal); everything stays
  gated so the F1-F3c results reproduce unchanged.

## Sub-phases
- **F3d-0 (FREE pre-check, no retrain):** orientation multi-start in the finisher -- for each draw,
  finish from a few small SO(3) perturbations of the ASU orientation and pool all finished variants
  into best-of-(k*R) matching. Tests cheaply whether *escaping the 18 deg basin* is what's missing
  (if a multi-start local finisher already lifts match, the flow's orientation tail is the culprit
  and a learned refinement will help; if not, the gap is elsewhere). Runs on existing F3c draws.
- **F3d-1 (build + retrain):** implement self_cond (model/train/sampler), CPU-smoke the composition,
  retrain the full-stack model (coset + family-mask + logmetric6 + OT + fixed-prior + self_cond) on
  a box, gen with N recycling passes, finish + score vs F3c's 3.1%.
- **F3d-2:** if it lifts match toward the MCF/MOFFlow band -> that is the paper's headline; else ->
  lock F4 with self-conditioning as a tried-and-reported negative (the ceiling is fundamental).

## Pre-registered gate
Promote self-conditioning to the paper's positive result iff FIN match@stol<=1.0 rises above F3c's
3.1% with CI separation (or clearly trends toward the MCF ~6.8% band). Else the honest ceiling
stands and F3d is reported as the final lever tried.

## Non-negotiables
- Everything gated (`self_cond` default False) -> shape10 + F1-F3c paths byte-identical, tests green.
- Cheapest-first: run F3d-0 (free) before the F3d-1 retrain.
- Frank owns final prose + the F3d-1 GPU/scale call; [[feedback_no_coauthor]]; specific-path staging.
