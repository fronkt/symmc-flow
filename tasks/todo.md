# Task: SymMC-Flow reference implementation

## Plan
- [x] Scaffold project (dirs, meta files, archive old todo)
- [x] Write PLAN.md design doc
- [x] `config.py` — dataclass configs
- [x] `periodic_table.py` — Z→(r,c) table + 2D embedding
- [x] `manifolds.py` — T³ / SO(3) / lattice exp·log·geodesic·prior·dist
- [x] `egnn.py` — equivariant GNN invariant encoder
- [x] `pair_bias_attention.py` — Clari-style attention + pair bias
- [x] `space_group.py` — symmetry ops + SGFM group averaging
- [x] `flow.py` — conditional paths + CFM loss
- [x] `model.py` — combined SymMCFlow module
- [x] `sampler.py` — RK4 ODE integrator (50 steps)
- [x] `data.py` — synthetic molecular-crystal dataset + real-data hooks
- [x] `train.py` — CUDA-aware training loop
- [x] tests for every module — `pytest -q` green (28 passed)
- [x] `scripts/train_demo.py`, `scripts/sample_demo.py` run on CPU
- [x] README.md + LICENSE
- [x] Init git, create GitHub repo, push
- [x] GPU phase setup: vast.ai RTX 5090, torch cu128, repo cloned, 28 tests pass on GPU
- [x] GPU phase: carbon-24 loader (pymatgen) + real training run + OT-coupling fix
- [x] Documented carbon-24 results in RESULTS.md (centroid fixed; lattice is next)
- [ ] GPU phase next: lattice reparametrization (log-vol + N^(1/3)) + SUN/match eval
- [ ] GPU phase later: MP-20 loader, full SUN benchmark vs diffusion baselines

# Task: Lattice reparametrization (log-volume + shape) + StructureMatcher eval

## Plan
- [x] `manifolds.py`: `lattice_to_param(L, n)` / `param_to_lattice(k, n)` — k = (per-atom
      log-volume, det-normalized 3x3 shape) in R^10; `prior_lattice_param` with
      configurable per-atom volume (V ∝ N built in)
- [x] `flow.py`: lattice path/velocity in param space (u_L: (B,10)); `sample_prior`
      draws the lattice prior in param space using n = mask.sum()
- [x] `model.py`: condition tokens on current lattice (k, 10-d) — before, forward()
      ignored the lattice entirely, so the field couldn't depend on lattice state;
      `head_lattice` outputs 10
- [x] `sampler.py`: integrate k, decode to 3x3 for each field evaluation
- [x] configs/train: replace `lattice_prior_scale` with `prior_vol_per_atom`
- [x] `train_carbon24.py`: StructureMatcher match-rate eval (CDVAE tolerances)
      alongside the C–C distance proxy; vol/atom reported gen vs ref
- [x] tests: param roundtrip, volume∝N scaling, prior validity; update shape asserts
- [x] full `pytest -q` green on CPU (32 passed, 1 data-dependent skip)
- [x] CPU demos verified: train_demo 76% loss drop, sample_demo valid det>0 cells
- [x] retrain on new vast.ai box (142.171.48.138:44563); RESULTS.md updated —
      vol/atom now 6.57 vs 6.30 ref (volume problem solved); C–C still 1.05 Å,
      match rate 0% → next bottleneck is the coordinate field (lattice-aware
      pair features / Fourier features)

## Review (2026-06-11)
- Decode always renormalizes det(shape)=1 and yields det(L)>0 by construction —
  the old "det>0 frac" metric is now trivially 100%.
- Found + fixed a latent bug: the network never consumed the lattice state at all
  (forward() ignored its `lattice` arg), so the lattice ODE field was not a
  function of lattice state. Added `lattice_in` (10-d param) token conditioning.
- GPU retrain DONE: volume problem solved (gen 6.5 vs ref 6.3 Å³/atom).

# Task: Fix position-flow under-dispersion (carbon-24 match rate still 0%)

## Findings (no code beyond diagnostics this round)
- [x] Lattice-aware pair features (Cartesian Δfrac@L + Fourier) — committed 99c4c98,
      verified active (pair_dim=20). NULL RESULT: centroid loss stays 0.09, C–C 1.05 Å,
      match 0%. The loss is at its OT-coupled irreducible floor — features can't help.
- [x] Eval sanity-checked: ref-vs-ref match 100%, ref-vs-shuffled 4.7% → 0% is real.
- [x] Root cause: mean-field under-dispersion. Generated NN dist median 1.11 Å (ref
      1.45), whole distribution contracted, 9% have <0.9 Å overlaps.

## Plan (all three behind config flags, run as an ablation ladder for attribution)
- [x] Lever 1 — stochastic sampler (`sampler_churn`). Implemented + tested.
- [x] Lever 2 — wrapped-normal centroid prior (`centroid_prior_std`). Implemented.
- [x] Lever 3 — fixed per-structure prior coupling (`fixed_prior` + PriorCache, idx
      plumbed into batches). Implemented + tested.
- [x] Tests: std-prior validity, churn dispersion+validity, fixed-prior determinism;
      34 pytest green on CPU + smoke through every training path.
- [x] GPU ablation ladder run (A churn / B fixed_prior / C1 prior / C2 all-three).
- [x] WINNER: lever 2 (wrapped-normal prior std=0.25). valid bonds 17%→84%, overlaps
      11%→2%, vol exactly on ref, match 0%→4.7%. Churn null; fixed_prior overfits;
      all-three worse than prior alone. Baked `--centroid-prior-std 0.25` as default.
      Best ckpt: checkpoints/carbon24_wn.pt. RESULTS.md updated.

## Review (2026-06-11, position-flow ablation)
- "Do all three" → ablation showed they are NOT additive: the concentrated prior alone
  is the fix; stochastic sampling does nothing; sharper coupling lowers the loss floor
  but overfits and even degrades the combination. Attribution was the whole value.
- Remaining: absolute match rate ~5% on a hard CSP benchmark. Next levers are
  capacity/steps/std-tuning, NOT collapse — see RESULTS "Next steps".

# Task: Push carbon-24 match rate (std sweep + sampler steps + scale)

## Plan
- [ ] Phase 1: centroid_prior_std sweep {0.15, 0.20, 0.30, 0.35} @ 6000 steps
      (0.25 already known: 4.7%). Pick best by match rate / valid-bond %.
- [ ] Phase 2: sampler-steps sweep {50,100,200} on best ckpt via --eval-only (free)
- [ ] Phase 3: final scaled run at best std+steps — longer (18k steps) + larger
      d_model (256) — and report match/SUN proxy.
- [ ] Update RESULTS.md with the sweep; keep best ckpt.

## Open after
- [ ] MP-20 loader, full SUN + match-rate benchmark vs CDVAE/DiffCSP baselines

## Review
- **Completed:** 2026-06-10
- **What worked:**
  - All 28 unit tests green. Train demo: total CFM loss drops ~80% (8.0 → 1.5)
    with all three manifold heads (lattice/centroid/orient) learning.
  - Sample demo: 50-step manifold RK4 yields valid lattices (~5 Å near-cubic),
    wrapped centroids in [0,1), orientations with det≈1, cell volumes ~100–145 Å³.
  - SO(3) log made numerically robust via Shepperd's quaternion method (stable at θ=π).
- **What changed from plan:**
  - Synthetic orientation/centroid targets were initially pure noise → orientation
    loss was irreducible. Re-tied target pose to molecular composition so the
    conditional flow is genuinely learnable (lesson captured).
- **Known limitations:**
  - CPU reference impl on synthetic data; no real MP-20/carbon/WBM training yet.
  - Space-group op table is a curated subset (P1, P-1, P2, Pm, P222, Pmm2);
    GPU phase pulls full ops from pymatgen.
  - SGFM averaging is implemented for the centroid field; lattice/orient use the
    plain field (full joint averaging is a GPU-phase extension).
