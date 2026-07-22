# Digital Discovery revision roadmap — execution log (2026-06-21)

Target: **Digital Discovery (RSC)**. Items from the full 5-reviewer panel
(`paper/peer_review.md`); decision Major Revision. Tracks the Tier 1–3 roadmap.

## Tier 1 (gating)
- [x] **T1.1** Scope unlearnability (marginal≠conditional; under-featurized vs in-principle).
- [x] **T1.2** R_asym **conditional probe** (`scripts/probe_rasym_conditional.py`). 3-seed:
  probe 125.2° vs constant 122.4° vs Haar 123.5° → −2.4%, no conditional signal. In SI §S3.
- [~] **T1.3** All-atom + random-prior **de-novo baseline** (`scripts/baseline_allatom_denovo.py`).
  Random-prior floor 0% confirmed; trained all-atom run **OOM'd on a leak** → re-run pending.
- [x] **T1.4** **match@1** alongside best-of-k. De-novo match@1 = 0% (base & big). Orient-isolated
  match@1 pending the criterion re-run.
- [~] **T1.5** **Species-grouped split ×3** (`scripts/diag_orient_relative_grouped.py`;
  union-find on shared species — refcode grouping needs CSD API). Smoke-passed; **OOM'd** → re-run.

## Tier 2
- [~] **T2.6** Tolerance sweep + matched RMSD (`eval_orient_matchrate.py --sweep`). First run used
  lenient `get_rms_dist`; **fixed to `fit()`** → re-run pending.
- [x] **T2.7** Scope statement (rigid/CHNO; general-position; special-position exception).
- [x] **T2.8** Classical-CSP lineage paragraph.
- [x] **T2.9** Capacity → "capacity + training" (verified).

## Tier 3
- [ ] **T3.10** Two-head PoC — optional; decide after Tier 1–2.

## Results (`paper/gpu_results/revision/`)
- probe_s0/1/2: probe never beats constant (−1.1/−3.5/−2.5%).
- denovo_base: match@1 0%, match@20 0%, comps 0.46/0.345/57.3°.
- denovo_big:  match@1 0%, match@20 0%, comps 0.41/0.351/50.3°.
- orient_sweep: lenient criterion (discarded); grouped/baseline: OOM (leak) → pending.

## Blocker — GPU memory leak
Orphaned process (PID 26140) held ~24.7 GiB after the de-novo Pool eval, OOM-ing the grouped
splits + all-atom baseline. `kill -9` was blocked by the safety classifier (shared box; task
was the github push). **Needs user authorization** to kill the orphan, then re-run grouped ×3
+ baseline + fit-based sweep. Recurrence fix: matcher `Pool` → **spawn** context (no inherited
CUDA context in forked workers).

---

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

# Task: Close MP-20 match@1 gap to DiffCSP — capacity scaling (2026-06-13)

## Diagnosis
mp20 (d_model 256/8L/30k) val loss plateaus after ~20k (0.033->0.029) while TRAIN loss keeps
dropping -> capacity-bound, not steps-bound. (Carbon was conditioning-bound; MP-20 is not.)

## Plan
- [x] Train mp20_big: d_model 384, 8 layers, egnn_hidden 128, 40k steps, batch 256, seed 0.
      (~4.4h; box was degraded/bursty — same noisy-neighbor cause as the SSH flakiness.)
- [x] Canonical eval (get_rms_dist): match@1 43.4% (41.8–44.5, seeds 0/1/2), match@20 80.9%,
      RMSE@1 ~0.19, RMSE@20 0.141. Baseline was 41.4% / 80.5% / 0.20 / 0.145.
- [x] scp mp20_big.pt local; RESULTS/README/memory updated.

## Review (2026-06-13/14) — capacity scaling = diminishing returns
- ~3× compute (1.5× width × 1.33× steps) bought ~+2 pp match@1, flat match@20. Wider model's
  val loss tracked baseline's at every step (~0.0286 vs 0.0288 floor) -> near capacity/objective
  floor, like carbon-24. Scaling will NOT close the ~8 pp gap to DiffCSP (51%). Conclusion: the
  remaining gap is the MODELLING APPROACH (OT-coupling depth, post-hoc relaxation), not size.
  Next lever is method, not scale.

# Task: SUN metric (Stable / Unique / Novel) (2026-06-13)

## Plan
- [x] symmc_flow/sun.py: validity + within-formula unique + novel (vs train). 5 tests green.
- [x] symmc_flow/stability.py: CHGNet relax + CHGNet-consistent MP-chemsys convex hull ->
      E_above_hull (one energy model for candidate + hull anchors; sidesteps GGA/GGA+U
      DFT-correction mismatch). MP key via env, not committed.
- [x] scripts/eval_sun.py: generate per val composition, U/N/validity (+ --stability for S+SUN).
- [x] U/N/validity on mp20.pt (256, seed 0): valid 92.2%, unique 100%, novel 98.0%,
      valid∩unique∩novel 90.2%.
- [x] Full SUN with stability (CHGNet+MP, 252/256 scored, ~81min): stable 58.6%, **SUN 56.6%**,
      median E_hull 0.074 eV/atom. RESULTS.md SUN section + README + memory updated, pushed.

## Review (2026-06-13) — SUN
- mp20 SUN (256, seed 0): valid 92.2%, unique 100%, novel 98.0%, stable 58.6%, **SUN 56.6%**.
- Stability = CHGNet-relative E_above_hull on a CHGNet-energy MP-chemsys hull (one energy model
  for candidate + anchors -> no GGA/GGA+U correction mismatch). Caveat: this is COMPOSITION-
  CONDITIONED SUN (eval on known val compositions), NOT unconditional DNG-SUN — not directly
  comparable to CDVAE/FlowMM. Next: unconditional SUN (sample compositions) for that comparison.
- chgnet + mp-api installed on box; MP key passed via env (never committed).
- UNCONDITIONAL SUN DONE (--from-train, sample comps from train marginal; commit pending):
  valid 93.8%, unique 100%, novel 72.7%, stable 61.3%, **SUN 35.2%** (255/256, median E_hull
  0.071). Only novelty moves vs conditional (98%->73%) = honest de-novo signal (regenerates
  known structure ~27%). Caveats (NOT comparable to CDVAE/FlowMM DNG-SUN, optimistic): comps
  from train marginal (no novel-composition test), unique~1 is small-sample, CHGNet-relative
  metastability. RESULTS.md now has a 2-column conditional-vs-unconditional table.

## Review (2026-06-13) — matcher reconciliation task
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

# Task: Journal hardening — relaxed match + strict DiffCSP head-to-head + mol-crystal (2026-06-16)

Target: npj Computational Materials (fallback Digital Discovery). Addresses the two reviewer-facing
weaknesses (apples-to-oranges DiffCSP baseline; loose matched cells / no relaxation) and the central
thesis gap (orientation off on both real benchmarks).

## P0 — done in repo (CPU), GPU runs pending
- [x] `--relax` / `--relax-steps` on train_carbon24.py + train_mp20.py: CHGNet-relaxes GENERATED
      structures (refs never relaxed) and prints a relaxed match rate + RMSE as a SEPARATE line,
      never replacing the canonical unrelaxed numbers. Reuses symmc_flow.stability.relax_structures
      (lazy chgnet import). Added `relax=`/`relax_steps=` kwargs to match_rate/match_rate_topk.
- [x] tests/test_match_topk.py: identity-relax monkeypatch test (relaxed score == unrelaxed when
      relaxation is identity; relax hook invoked). Skips if no pymatgen; no chgnet import. py_compile OK.
- [x] scripts/diffcsp_headtohead.md: runbook to run DiffCSP's released models at num_evals=20 and
      report match@1/@20 + RMSE with the same get_rms_dist matcher (no repo code; GPU-box task).
- [ ] GPU: scp carbon24_big.pt + mp20.pt to box; `pip install chgnet`; run --relax (seeds 0/1/2)
      on both; add relaxed match@1/@20 + RMSE as NEW columns in RESULTS.md (with carbon-topology caveat).
- [ ] GPU: run scripts/diffcsp_headtohead.md; replace the "~51% (match@1)"/"~17%" placeholders in
      RESULTS.md/README with DiffCSP's own match@20 from our run of the released checkpoint.

## P0-strategic — molecular-crystal benchmark (orientation ON) — SCOPED, not built
The thesis (rigid-body conformers + SO(3) orientation flow + SGFM) is validated only on synthetic
data; both real benchmarks use single-atom blocks with lambda_orient=0. Gated on data access.
- [ ] Confirm Purdue CCDC/CSD license access (Libraries / CCDC academic program).
- [ ] Choose a CSD-derived molecular-crystal benchmark set (align with MolCrystalFlow's protocol if
      available) so numbers are comparable.
- [ ] Then design symmc_flow/molcrystal.py — analog of mp20.py but multi-atom rigid blocks (PCA
      framing) with orientation ON (lambda_orient>0), exercising the SO(3) flow + SGFM on real data.

## Deferred (real research, uncertain payoff)
- [ ] Fair stochastic-interpolant diffusion baseline (OT-coupled / concentrated coordinate noising).
- [ ] Unconditional de-novo SUN with composition sampling (model can't generate compositions yet).

## Review (2026-06-17) — GPU runs DONE
- Checkpoints carbon24_big.pt + mp20.pt were LOST (not local, boxes recycled) -> RETRAINED on
  vast.ai RTX 5090 (108.240.82.27). Reproduced canonical: carbon match@20 78.9% (rec 79.7),
  mp20 match@20 82.4% (rec 80.5). Both scp'd to LOCAL repo checkpoints/ (recycle-safe this time).
- **Relaxed match@1 (mean seeds 0/1/2, --relax):** MP-20 41.8->44.7% (HELPS, RMSE 0.21->0.16);
  carbon 29.4->21.2% (HURTS - CHGNet drifts allotropes to graphite/diamond, RMSE 0.41->0.33).
  Consistent across all 3 seeds. RESULTS.md "Post-hoc relaxation" section added.
- **DiffCSP head-to-head DONE** (2x RTX 3090 box, torch-1.9 env, released ckpts, num_evals=20,
  256 test, get_rms_dist; recipe in scripts/diffcsp_headtohead.md):
  - carbon: DiffCSP m@1 18.8% (<ours 26.7), m@20 89.1% (>ours 79.7).
  - mp20:   DiffCSP m@1 48.0% (>ours 41.4), m@20 76.2% (<ours 80.5).
  - **MIXED - each wins 2 of 4 cells.** DiffCSP RMSE much tighter (0.06-0.21 vs 0.15-0.35).
    Old "match@20 beats DiffCSP match@1" framing WITHDRAWN (metric mismatch). Honest takeaway:
    competitive at ~50x fewer sampling steps, not a uniform win. RESULTS + README corrected.
- Still TODO: molecular-crystal benchmark (orientation ON) gated on CSD access (user reached out).

# Task: molcrystal.py — real rigid-body molecular-crystal loader (lambda_orient>0)

## Plan
- [x] `symmc_flow/molcrystal.py`: `MolCrystalDataset` — molecule detection via
      `StructureGraph.from_local_env_strategy(JmolNN)` + PBC-unwrap BFS over `to_jimage`;
      per-species conformer registry keyed by WL graph hash; Kabsch alignment
      (`manifolds.project_so3`) with automorphism-min mapping (VF2, element node_match);
      rigid-strictness skip gate (`conf_tol`, default 0.3 Å) recorded in `self.skipped`;
      monatomic units -> orient=I/local=0. Same batch dict as carbon24/mp20.
- [x] `rigid_to_frac` / `rigid_to_structure` — the exact inverse factorization, for
      round-trip validation and eval-time molecular StructureMatcher reconstruction
      (the single-atom `_to_structures` in train scripts can't expand molecules).
- [x] Re-export from `data.py` + `__init__.py`.
- [x] `tests/test_molcrystal.py` — 5 tests: round-trip recovers planted atoms (gauge-free,
      frac space), all copies share one conformer + recovered relative rotations match
      planted, monatomic=identity, non-rigid copy skipped, symmetric-molecule (water)
      automorphism guard. All green.
- [x] `scripts/train_molcrystal.py` — smoke train, lambda_orient=1.0, orientation tied to
      centroid (learnable field): orient loss 5.39 -> 3.38 (37% drop). First time the SO(3)
      head trains on real multi-atom rigid blocks.
- [x] full `pytest -q` green (no regression); molcrystal deprecation warning removed.

## Review
- Factorization convention locked: `cart_atom = c@L + R@local`; species share ONE
  reference conformer `local`, copies differ only by `R` — that shared conformer is what
  makes the SO(3) flow learnable (vs single-atom mp20/carbon24 where orient≡I).
- **No model change needed**: `model.encode_molecules` already runs the EGNN over A>1
  `local`; only real multi-atom local/orient had to reach it.
- CSD-independent by design: the loader takes pymatgen `Structure`s, so the full detection
  pipeline is already exercised on genuine Structure objects; CSD only swaps the *corpus*
  (a `detect_fn` hook lets the CSD Python API plug in later). Decision: reject non-rigid
  (flexible) molecules above `conf_tol` to keep the rigid-body benchmark honest.
- Lesson captured (tasks/lessons.md 2026-06-17): synthetic bond-graph tests must separate
  copies (large cell + spread centroids) or JmolNN cross-bonds them; flex (don't snap) a
  copy to exercise the conformer gate; a passing round-trip doesn't prove correctness —
  assert the shared-conformer invariant too.
- Still TODO (on CSD access): curated molecular-crystal benchmark corpus + the actual
  orientation-ON training run and match@k numbers for the paper.

## 2026-06-18 — Real CSD molecular-crystal benchmark (orientation-ON)
Full writeup: see `MOLCRYSTAL.md`.
- [x] CSD access obtained; export (`scripts/csd_export.py`, CCDC interpreter) + COD fallback.
- [x] Factorize real CIFs → `MolCrystalDataset` (`scripts/factorize_cifs.py`); tolerant `read_cif`.
- [x] Orientation-ON training on real corpus (`scripts/train_csd_molcrystal.py`).
- [x] Molecule-intrinsic gauge (`_canonical_frame`) + lr 3e-4 stability fix.
- [x] Scale corpus 250 → 1127 structures.

### Result (robust negative)
- lattice + centroid heads LEARN on real data; **SO(3) orientation head stays at its
  predict-zero floor (5.24)** across both corpus sizes and after the gauge fix.
- Not data sparsity (scaling didn't help). Leading cause: noised flow-time conditioning —
  orientation must be predicted from half-noised lattice+centroids.

### Next (not started — paused for thesis-framing discussion)
1. Two-stage / cleaner conditioning (generate lattice+centroid first, orientation after). [rec]
2. Diagnostic: condition orientation on TRUE lattice+centroids to confirm the cause.
3. Fallback: reframe paper as rigid-body lattice+centroid flow + orientation as open problem.
4. Deprioritized: min-over-symmetry orientation target (helps only symmetric minority).

# Task: Clean-packing diagnostic — does the SO(3) head floor on noised conditioning? (2026-06-18)

Ran the cheap diagnostic (#2 above) BEFORE building two-stage (#1), because a positive result
is literally two-stage's second stage and a negative result rules two-stage out — strictly
cheaper either way.

## Plan
- [x] `TrainConfig.cond_clean_packing` flag (config.py).
- [x] `train._step_loss`/`evaluate`/`train`: when set, feed the field the TRUE (z1)
      lattice+centroid instead of noised z_t; orientation stays noised, target unchanged
      (no leakage). Only the orient loss is meaningful under the flag.
- [x] `scripts/diag_orient_conditioning.py`: clean-packing run on the 1127 CSD corpus, orient
      pre→post on held-out val, CONFIRMED/NULL verdict, saves diag_orient_cleanpack.pt.
- [x] `tests/test_cond_clean_packing.py`: 2 plumbing tests (flag feeds true vs noised packing).
      Full suite green.

## Review (2026-06-18) — NULL, two-stage RULED OUT
- 800 steps, clean packing: lattice 1.20→0.044, centroid 0.355→0.254 (both learn), **orient
  5.35→5.18 (+3.3%) — still at the ~5.2 predict-zero floor; R train oscillates 4.69–5.90,
  no trend.** Conditioning on the TRUE packing is the best case for two-stage's 2nd stage, so
  two-stage cannot help → **RULED OUT.**
- Cause is deeper: the absolute per-molecule SO(3) target (even gauge-fixed) carries no
  learnable signal on this asymmetric, ~1-crystal-per-molecule corpus.
- **Decision:** lead with the honest reframe (rigid-body lattice+centroid flow + orientation
  as a well-diagnosed open problem). One technical lever left before fully committing: change
  the orientation TARGET (restrict to species recurring across crystals → relative target),
  not the conditioning. Full writeup in MOLCRYSTAL.md.

# Task: Relative-orientation target — decompose the floor (2026-06-18)

Ran the remaining lever: change the orientation TARGET (relative, not absolute), not the
conditioning. The absolute target factors R_m = rot(g_m)·R_asym (space-group part + free
asymmetric-unit orientation); re-gauge to cancel R_asym and see if the symmetry part learns.

## Plan
- [x] `molcrystal.relative_gauge_item` / `_species_groups` / `species_multiplicity`: re-gauge
      first-copy-as-reference (orient=I), others R'_m=R_m·R0^{-1}; round-trip preserved; adds
      `is_ref` mask. Test `test_relative_gauge_preserves_roundtrip_and_targets` (suite green).
- [x] `scripts/diag_orient_relative.py`: regauge + keep multi-copy crystals (1095/1127), train,
      report orient loss split ref vs NON-ref (guards trivial predict-identity), `--clean-packing`.
- [x] Ran the full 2×2 (absolute/relative × noised/clean), 800 steps each.

## Review (2026-06-18) — PARTIAL POSITIVE; floor = free asymmetric-unit orientation
- Non-reference (symmetry-determined) held-out orient loss, untrained→trained:
  - absolute: 5.37→5.18 (+3.3%, floor) both noised & clean.
  - **relative: 5.37→3.91 (+27.1%) noised; 5.37→3.94 (+26.7%) clean** — descends + generalizes.
  - Overall orient +34–36%; reference copies +56–61%.
- **Identical noised vs clean** → the relative signal rides on the space group (directly
  conditioned) + coarse centroids; robust to conditioning noise (re-confirms two-stage moot).
- **Conclusion:** the SO(3) flow DOES learn space-group-induced relative orientation; the floor
  is the FREE asymmetric-unit orientation R_asym — gauge-arbitrary, fundamentally unlearnable,
  not a broken flow/conditioning artifact. Orientation is *partially* learnable, precisely
  decomposed. → Reframe paper with this decomposition (see MOLCRYSTAL.md Next steps).

## Review (2026-06-20) — strengthening 2a/2b/2c DONE; reframe shipped
Ran the three follow-ups the partial positive needed, then reframed the docs.
- [x] **2a match metric** (`scripts/eval_orient_matchrate.py`): orientation-isolated
      (true lattice+centroid+conformer, sample only SO(3), best-of-8), rebuild via
      `rigid_to_structure`, StructureMatcher.fit. n=131 val: trained **16.8%** vs 0% floor /
      1.5% naive R=I; oracle 100%. Single deterministic draw collapses to the conditional mean
      (relatives are ≈180°), so best-of-k is the correct generative read.
- [x] **2b capacity/steps** (`diag_orient_relative.py` + `--d-model/--n-attn-layers/--egnn-layers`):
      d_model 192 / 2000 steps → non-ref 5.35→3.56 (**+33.4%**, ref +68%). Real but diminishing;
      plateaus above 0. First "big" config (256/6/5) OOM-swapped the CPU box (19 GB) → killed,
      reran modest. Lesson: this box swaps with parallel/oversized training; run sequentially.
- [x] **2c coset conditioning** (`scripts/diag_orient_coset.py`, `assign_cosets`, model
      `n_cosets`/`coset_embed`): 707-coset per-SG codebook; conditioned non-ref 5.36→**2.65
      (+50.6%)** vs `--no-coset` control 5.37→3.91 (+27.1%, reproduces baseline). → ceiling is
      **inference-limited, not representational**; residual = free R_asym + symmetric-top multimodality.
- [x] Tests: +`test_assign_cosets_*` (loader), +`test_coset_conditioning_optional_and_active`
      (model); fixed `test_cond_clean_packing` for new `coset=` kwarg. Suite **60 passed, 2 skipped**.
- [x] **Reframe (task 1)**: MOLCRYSTAL.md (TL;DR 3-way characterization, 2a/2b/2c subsections,
      components table, next steps, artifacts) + PLAN.md §1 reframe note.

---

# Phase D — Reframe as a METHOD paper (PLAN for sign-off, 2026-07-18)

**Status:** Phase B/C GPU runs DONE, gate PASSED (commit `e035b8d`, gpu_results/). This is the
rewrite PLAN only; **awaiting Frank's sign-off before editing `paper/main.tex`.**

## Venue recommendation (Claude's pick): MoML/MIT short paper (Sep 1) FIRST, then JCIM (archival)
- **Why MoML first:** the result is a focused, positive method+analysis -> ideal workshop short
  paper. MoML/MIT (Sep 1) is ALREADY committed for symmc-flow, is the exact domain audience
  (molecular ML), is non-archival (does not burn a later journal), and Sep 1 is a clean deadline.
- **Why not straight-to-journal (yet):** DD (AE Jung) and TMLR both desk-rejected the *negative*
  version on general-interest triage. A domain-workshop review judges it on molecular-ML merit,
  where SG-conditioning for rigid-body molecular-crystal flows is squarely on-topic. Land the
  workshop result, THEN extend.
- **Free second shot:** same paper co-submits to AI4Mat (NeurIPS ws, Aug 30) -- dual non-archival
  workshops OK per the committed plan.
- **Archival top-venue follow-on (Phase E):** extend to **JCIM (J. Chem. Inf. Model.)** -- strong
  molecular-informatics journal, domain review, less general-interest triage than DD. (DD round-2
  possible but already desk-rejected once; JCIM is the cleaner target.) Journal version wants the
  optional extra runs below.

## Target format
- [ ] **Confirm MoML 2026 template + page limit** (likely ~4 pp + refs, workshop LaTeX style).
- [ ] Rewrite from `paper/main.tex` (currently RSC single-column) into the MoML short format
      (keep a copy; do NOT overwrite the TMLR/RSC package under paper/submissions/).

## The reframe (diagnostic -> method)
- OLD title: "What an SO(3) orientation flow can and cannot learn in molecular-crystal SP"
- NEW title (draft): **"Deployable Symmetry-Coset Conditioning Recovers the Orientation Benefit
  of Space-Group Symmetry in Molecular-Crystal Flow Matching"** (trim as needed)
- OLD thesis (negative): orientation is partly unlearnable (free R_asym floor).
- NEW thesis (positive): the LEARNABLE part is the SG-determined relative rotation; we expose it
  as a **deployable, leak-free coset label** (the generating SG operation, available from a
  template at sampling time) and condition the orientation flow on it -> recover ~2/3 of the
  oracle-codebook benefit; **SO(3)-averaged training closes the rest**; de-novo (no-template)
  prediction is the remaining honest gap.

## Section-by-section rewrite
- [ ] **Title + abstract** -> method+result framing (numbers below).
- [ ] **Intro**: lead with the GAP -- MolCrystalFlow/MOFFlow skip SG conditioning; SG-conditioning
      proven inorganic-only (NextCrystal / WyckoffDiff / DiffCSP++ / SymmCD). State contributions:
      (1) deployable coset label; (2) conditioning recovers the orientation benefit; (3) SO(3)-avg
      reaches the ceiling; (4) honest de-novo gap.
- [ ] **Method** (expand from the diagnostic's decomposition):
      - R_m = rot(g_m) * R_asym decomposition (keep from current paper).
      - Deployable coset = generating SG op h = g_m * g_0^{-1}, recovered from centroids
        (min-image), LEAK-FREE + template-available at sampling (contrast DiffCSP++ atom templates).
      - Coset conditioning in the flow (coset_embed; threaded through sampler, C3a).
      - SO(3)-averaged FM objective (C5).
      - Packing-only coset predictor (C4).
- [ ] **Experiments / Results** (all from `e035b8d`):
      - **Table 1 (main):** held-out NON-REF orientation-loss drop, 3 seeds:
          control (no-coset)      +27.5% (29.4 / 24.4 / 28.8)
          deployable coset K=1    +41.1% (43.2 / 37.2 / 42.9)
          + SO(3)-avg K=4         +47.7% (49.2 / 43.4 / 50.5)
          leaky-codebook oracle   ~+48% (upper bound)
      - **Fig 1:** bar chart of Table 1 with the ~+48% ceiling line.
      - **Orient-isolated physical match** (best-of-8, coset template): up to 14.5%/37.4% at loose
        tol, 9.9%/27.5% at (0.3,0.5), median 41.1 deg; tol-sweep table.
      - **Predictor (C4):** 39.5% top-1 vs 10.3% majority (4x); predicted-coset reconstruction
        collapses to 64.8 deg -> de-novo gap.
      - **Templated end-to-end:** 0.0% match; error budget lattice 0.442 / centroid 0.346 /
        orient 13.4 deg -> scopes the contribution to the ORIENTATION mechanism (lattice+centroid
        are the end-to-end bottleneck, not orientation).
- [ ] **Limitations / Analysis**: de-novo predictor gap; end-to-end bottleneck = lattice+centroid;
      symmetric-top multimodality residual.
- [ ] **Conclusion + future work**: better/top-k coset predictor for template-free use; joint
      lattice-centroid improvement.
- [ ] **Related work**: add MolCrystalFlow (2602.16020), NextCrystal (2602.17176), MOFFlow,
      SO(3)-averaged FM (2507.09785), Genarris-3; keep DiffCSP++/WyckoffDiff/SymmCD.

## Figures / tables to produce
- [ ] Table 1 (main result) -- from logs.
- [ ] Fig 1 bar chart -- matplotlib from logs.
- [ ] Tol-sweep table -- from phaseB/orient_isolated_coset.log.
- [ ] (Optional) schematic of R_asym decomposition + coset conditioning (excalidraw skill; needs
      jsdelivr import fix + dangerouslyDisableSandbox per [[reference_excalidraw_skill]]).

## Claims discipline (do NOT overclaim)
- Coset helps ON THE ORIENTATION RESIDUAL; NOT yet end-to-end match.
- De-novo (no template) does NOT work yet (39.5% too noisy -> 64.8 deg).
- SO(3)-avg is the ceiling-reaching component.

## Optional extra runs (JCIM journal version, NOT the workshop)
- [ ] Unconditioned templated baseline (was skipped: run_phaseB UNCOND fallback missed
      coset_deploy_off_s0.pt at eval time). Rerun eval_templated_matchrate on an OFF ckpt
      (CPU-heavy matcher; short GPU rental or overnight CPU). Cheap; completes the templated delta.
- [ ] Stronger / calibrated coset predictor (bigger, or top-k conditioning) to shrink de-novo gap.
- [ ] End-to-end with improved lattice/centroid generation.

## Repro / data
- Numbers: `e035b8d`, gpu_results/. Checkpoints local (gitignored, 10x ~5.5MB). Zenodo v1.0.3
  DOI 10.5281/zenodo.21384130 (bump on submission).
- Author block per [[reference_author_identity]] (Frank Cai, Purdue, ORCID 0009-0003-0041-1459).

## Open decisions for Frank (sign-off)
1. Venue: MoML-first (my rec) vs straight-to-journal?
2. Title (draft above OK, or shorter)?
3. Include the excalidraw schematic figure?
4. Do the optional unconditioned-templated rerun now, or defer to the journal version?

---

# Task: Finish JCIM manuscript for publication + achemso formatting (2026-07-22)

Finalize `paper/submissions/JCIM/` to ACS J. Chem. Inf. Model. (jcisd8) publication format;
also fix MoML stale title + verify page count. LaTeX installed locally (MiKTeX 25.12).

## JCIM
- [ ] `article` -> ACS `achemso` (journal=jcisd8, manuscript=article); drop natbib/bibstyle (achemso owns it)
- [ ] Remove `\todo` macro + 6 markers; expand each to publication prose:
  - [ ] Intro opener; Methods rigid-body flow; R_asym derivation (Eq + gauge floor + Haar + probe + 2x2);
        orientation-isolated tolerance table; Limitations (from diagnostic Discussion)
- [ ] achemso author block (\affiliation/\email/\keywords); ORCID OUT of body (JCIM rule)
- [ ] Abstract <=250 words; TOC graphic (\begin{tocentry})
- [ ] "Data and Code" -> **Data and Software Availability** (JCIM-required); competing-interest Notes
- [ ] Appendices -> Supporting Information (si.tex, manuscript=suppinfo) + \begin{suppinfo} pointer
- [ ] Self-contained figures/ in package; compile clean (0 undefined refs/cites)

## MoML
- [ ] Fix stale title in comment header; compile -> real page count; trim to <=4 pp if over

## Guidelines confirmed (researcher-resources.acs.org, jcisd8, updated 2026-05-21)
- Abstract 150-250 words; TOC graphic required (Article); Data Availability Statement mandatory;
  ORCID must NOT appear in manuscript text (auto-added on accept); refs any style but complete w/ titles;
  COI: "The authors declare no competing financial interest."

## Review (2026-07-22) — DONE, compiled + verified locally (MiKTeX 25.12)
- **LaTeX installed**: winget MiKTeX.MiKTeX 25.12 (user scope), AutoInstall on; pdflatex/bibtex/latexmk
  at `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`. pdftoppm/pdftotext there too (Read-tool PDF render
  needs poppler, which the Read tool lacks — render via MiKTeX pdftoppm to PNG instead).
- **JCIM** `submissions/JCIM/main.tex`: article -> achemso (journal=jcisd8, manuscript=article). All 6
  `\todo` expanded (intro; rigid-body flow; full R_asym derivation + 2x2 Table; orientation-isolated
  tolerance table; limitations). achemso author block (\affiliation/\email/\keywords), ORCID out of body.
  Abstract 224 words (in range). TOC graphic (fig3_ladder). "Data and Software Availability" +
  competing-interest + \begin{suppinfo}. Extended methods moved to **new `si.tex`** (achemso
  manuscript=suppinfo, 5 SI sections). fig3_ladder.pdf copied into package. **Compiles CLEAN**:
  main.pdf 19 pp (ACS double-spaced), si.pdf 6 pp; 0 undefined refs/cites, 0 errors, 0 overfull>15pt,
  19/19 bibitems resolve.
- **MoML** `submissions/MoML/main.tex`: stale title comment fixed. Trimmed main text 4.4pp -> **exactly
  4 pp** (refs start top of p5; appendix p6) via prose tightening + smaller bar chart + caption/table
  spacing. Compiles clean, 6 pp total.
- **HUMAN-GATED (Frank owns)**: (a) review the prose I wrote into the JCIM derivation/limitations — his
  paper, his voice; (b) tab:bench positioning still his call; (c) Zenodo DOI bump on both; (d) swap the
  TOC graphic for a purpose-built one if desired; (e) commit/push (not done — awaiting his ok).
