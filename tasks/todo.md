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

# Task: MP-20 loader + flow benchmark (multi-element CSP) — IN PROGRESS

## Plan
- [x] `symmc_flow/mp20.py`: MP20Dataset reads per-atom Z (carbon hardcoded C); same
      batch schema so model/training unchanged. Single-atom blocks, λ_orient=0.
- [x] `scripts/train_mp20.py`: flow objective (reuses train()) + match@k eval that
      builds StructureMatcher inputs from the REAL per-atom species (not "C").
- [x] `tests/test_mp20.py`: per-atom Z preserved, Z-aware NaCl identity match=100%,
      collate roundtrip. 3 pass on box (full suite green).
- [x] MP-20 CSVs downloaded to box (train ~27k, val/test ~9k).
- [x] train flow on MP-20 (30k steps, d_model 256/8-layer), eval match@1 + match@20.
      RESULT (256 val): match@1 26.2%, match@20 59.8%, 0% overlaps, vol on ref.
      ~7x carbon's match@1 (composition conditions strongly). match@20 beats DiffCSP's
      ~51% match@1; our match@1 ~half DiffCSP (headroom = more train/capacity, not collapse).
- [x] RESULTS.md MP-20 numbers vs CDVAE/DiffCSP. ckpt checkpoints/mp20.pt on box.

## Added RMSE metric (2026-06-13) + matcher discrepancy to resolve
- [x] match functions now also return mean matched RMSD (RMSE) — the 2nd standard CSP
      metric (DiffCSP reports match-rate AND RMSE). Wired into carbon/MP-20/diffusion,
      tests updated. MP-20 RMSE@20 = 0.1506 (DiffCSP ~0.06).
- [ ] RESOLVE NEXT SESSION: switching match test sm.fit -> get_rms_dist changed MP-20
      match@20 59.8% -> 80.5% (~20pp). Lock one matcher (get_rms_dist, break_on_match=False
      is more thorough) and re-run match@1/@20 for carbon24_big + mp20 to get canonical
      numbers. Did NOT overwrite the 59.8% headline until reconciled.
- [ ] carbon-24 RMSE eval not yet run (carbon24_big.pt) — quick eval-only next session.

# Task: Reconcile matcher + canonical match@1/@20/RMSE (2026-06-13, new box)

## Plan
- [x] New box 192.3.91.246:26436 (RTX 5090). Old box recycled; carbon24_big.pt restored
      from local copy; mp20.pt gone -> retrain.
- [x] Matcher already locked: both scripts use sm.get_rms_dist (pymatgen internally uses
      break_on_match=False -> thorough). Confirmed identical in carbon + mp20.
- [x] Added --seed to both train scripts (prior draws are the only RNG in eval).
- [x] Box setup (clone + torch cu128 + deps + carbon24/mp20 CSVs). Full test suite ran clean.
- [x] Retrain mp20.pt (30k steps, d_model 256/8-layer). Backed up to LOCAL repo (recycle-safe).
- [x] Canonical eval (seeds 0/1/2): carbon24_big + mp20, match@1 + match@20 + RMSE.
- [x] Reconcile -> overwrote RESULTS.md + README with get_rms_dist canonical numbers. Memory updated.

## Review (2026-06-13)
- **Root cause:** "matcher discrepancy" was not a bug — `StructureMatcher.fit` (break_on_match=
  True, early-exit) vs `get_rms_dist` (break_on_match=False, exhaustive). VERIFIED via DiffCSP's
  official compute_metrics.py that the CSP-standard matcher is `get_rms_dist` (match = non-None,
  ltol .3/stol .5/angle_tol 10) — exactly what our code already used. Old headlines just predated it.
- **Canonical (get_rms_dist, 256 val, seeds 0/1/2):**
  - carbon24_big: match@1 **26.7%** (26.2–27.3), match@20 **79.7%**, RMSE@20 **0.350**.
  - mp20: match@1 **41.4%** (40.6–42.2), match@20 **80.5%**, RMSE@1 ~0.20, RMSE@20 **0.145**.
- **Impact:** the "~5% carbon plateau" was a DOUBLE artifact (match@1-vs-k + non-standard fit).
  Carbon match@1 26.7% > DiffCSP ~17%. MP-20 41.4% vs DiffCSP ~51% (now apples-to-apples = ~81%,
  not the old "half"). No capability ceiling; remaining gap = train/capacity + post-hoc relaxation.
- **Shipped:** --seed on both train scripts; scripts/diag_matcher.py (fit-vs-rms reconciliation);
  RESULTS.md + README corrected; mp20.pt + carbon24_big.pt backed up locally.
- **Lesson captured:** checkpoints are not committed (too large) — scp every new ckpt to the local
  repo immediately; the old box recycle lost mp20.pt + carbon24_diff.pt (carbon24_big survived only
  via a stale local copy).
- **Deferred:** diffusion row (carbon24_diff) get_rms_dist re-eval — ckpt lost with old box;
  relative flow≫diffusion conclusion unaffected. DiffCSP's own match@20; SUN metric.

## Note — diffusion follow-up deferred (not a quick fix)
A FAIR diffusion baseline can't be a knob-turn: VE diffusion needs a uniform prior at
t=T to be sampleable (can't init near f0 without knowing f0), so the flow's "concentrate
the prior" trick has no drop-in analog. Making it fair needs a bridge/stochastic-
interpolant reformulation WITH OT coupling (≈ flow-matching-with-noise) — real research,
uncertain payoff. Chose MP-20 (concrete, standard benchmark) over it for the warm box.

# Task: DiffCSP-style diffusion baseline (objective head-to-head vs flow)

## Plan
- [x] `diffusion.py`: DiffCSP two-process diffusion REUSING the SymMCFlow net unchanged
      (VP-DDPM lattice + wrapped-normal VE fractional score). Sampler: lattice DDIM
      (x0-clamped) + fractional predictor-corrector (ancestral SMLD + Langevin).
- [x] `tests/test_diffusion.py` (6 tests): wrapped-normal score correctness, q_sample
      validity, learnable toy fit, valid samples. Full suite 42 pass.
- [x] `scripts/train_carbon24_diffusion.py`: same data + same match@k eval as the flow.
- [x] trained on RTX 5090 (same config as flow: d_model 256, 8 layers, 15k steps).
- [x] RESULTS.md flow-vs-diffusion table.

## Review (2026-06-13) — NEGATIVE RESULT, well-diagnosed
- Flow wins decisively: match@1 3.9% vs ~1.6%, match@20 35.5% vs 14.1%, valid bonds
  82% vs 26–30%, overlaps 0% vs 10–13%.
- The diffusion COORDINATE field never learned: predict-zero DSM loss 2.538 == trained
  frac loss 2.54 (diag_diffusion_floor.py). Lattice head trained fine (vol on ref).
- Root cause = same under-dispersion the flow had: carbon frac marginal ~uniform, plain
  DSM has ~no learnable coordinate signal. Flow fixed it with OT coupling + concentrated
  prior; vanilla diffusion has neither. Langevin corrector lifted match@1 0%→1.6% but
  can't create unlearned signal. Not a code bug (unit-tested; lattice side works).
- STOPPED per "objective at its floor → change approach, don't scale": a fair diffusion
  baseline needs the structural levers ported into the noising (DiffCSP's full design).

# Task: Push carbon-24 match rate (std sweep + sampler steps + scale) — DONE

## Plan
- [x] Phase 1: centroid_prior_std sweep {0.15–0.35} @ 6k steps. 64-struct peak at
      0.30 (7.8%) was small-sample noise.
- [x] Phase 2: re-eval on 256 structures — std 0.25 ≈ 0.30 (both 3.1%); sampler
      steps 50/100/200 → no gain (ODE converged by 50). Added --eval-n.
- [x] Phase 3: scaled run (d_model 256, 8 layers, 15k steps) → 4.7%, ~82% valid.
      Scaling did NOT improve match rate. Exposed model size as CLI args.
- [x] RESULTS.md updated with full sweep + scaling. Default std 0.30. Best ckpt
      carbon24_big.pt.

## Conclusion
- Collapse SOLVED (valid bonds 17%→~82%, overlaps→2%, vol on ref). Match rate
  plateaus ~5%, NOT limited by prior std / sampler steps / model capacity (all
  ruled out). Gap is the modelling approach: conditions only on (n, space group),
  so generates *a* polymorph not *the* reference.

## Open next (architectural — not more of the same)
- [x] Best-of-k match metric (standard CSP eval) — one-to-one understates us.
      `--match-k K` in train_carbon24.py: draws K gens/ref, hit if any matches.
      Implemented + unit-tested on CPU (tests/test_match_topk.py). RESULT on fresh
      RTX 5090 (carbon24_big, 15k steps, 256 val): match@1 3.9% -> match@20 35.5%
      (~9x). The ~5% "plateau" was a match@1 metric artifact, not a capability cap.
- [ ] DiffCSP-style diffusion baseline; head-to-head vs the flow objective
- [ ] MP-20 loader, full SUN + match-rate benchmark vs CDVAE/DiffCSP

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
