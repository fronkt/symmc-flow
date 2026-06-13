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

## 2026-06-11 — Flow matching on exchangeable point sets needs OT coupling
- **Mistake**: Trained carbon-24 with independent random prior→data atom pairing;
  centroid loss plateaued and generated atoms collapsed. Atoms are exchangeable, so
  "which prior point maps to which data atom" is undefined and the target is noise.
- **Rule**: For point-set / multi-particle flow matching, couple prior and data with
  per-sample optimal transport (Hungarian on the manifold distance) before computing
  the velocity target. See `flow.ot_couple`.
- **Context**: Crystal/molecule generation, any set-valued flow/diffusion target.

## 2026-06-11 — Lattice needs volume-aware parametrization
- **Mistake**: Raw 3×3 lattice flow with isotropic prior generated cells too small
  (~59 Å³), collapsing structures even when fractional coords were good.
- **Rule**: Parametrize the lattice in log-volume + normalized-shape space and scale
  the prior/target by N^(1/3) (cell volume ∝ atom count). Judge generation by sampled
  structure quality, not the training-loss floor (which is irreducible near t=0).
- **Context**: Crystal generative models (DiffCSP/FlowMM-style).

## 2026-06-11 — Don't add features to lower a loss that's at its irreducible floor
- **Mistake**: When carbon-24 structures clumped despite a low centroid loss (0.09),
  I added lattice-aware Cartesian + Fourier pair features to give the field "real
  geometry." Null result: loss and generation unchanged. The 0.09 was the irreducible
  floor of the OT-coupled CFM target (residual target stochasticity), not a capacity
  limit — so no amount of conditioning could move it.
- **Rule**: Before adding model capacity/features to improve a metric, check whether
  the training loss is already at its data-imposed floor. If loss won't drop, the lever
  is the objective/coupling/prior/sampler, not the network. Diagnose the *generated
  distribution* (here: NN distances uniformly contracted → mean-field under-dispersion)
  before choosing the fix. Also: always sanity-check the eval (ref-vs-ref must score
  ~100%) before trusting a 0% metric.
- **Context**: Flow matching / diffusion where a loss plateaus but samples are poor.

## 2026-06-11 — PowerShell -> ssh quoting mangles inner quotes
- **Mistake**: Inline `ssh host "python3 -c '...'"` and `export VAR=...` repeatedly
  broke (unterminated string, `expot`) because PowerShell rewrites the quoting before
  ssh sees it.
- **Rule**: For any remote command with inner quotes, `$`, or heredocs, write the
  script locally and `scp` it over (strip CRs: `tr -d '\r'`), then run the file. Don't
  fight the quoting inline.
- **Context**: Driving the vast.ai box from this Windows/PowerShell session.

## 2026-06-11 — pgrep -f watcher loops match themselves
- **Mistake**: A `while pgrep -f train_carbon24.py; do sleep; done` watcher never
  exited because `pgrep -f` also matched the watcher's own command line (and sibling
  watchers), so the loop saw a "running" process forever.
- **Rule**: Don't poll a job with `pgrep -f <script>` from a wrapper whose own command
  line contains that string. Match the actual `python3` proc (`pgrep -f 'python3.*train'`),
  use a PID file, or check `kill -0 $PID`. Prefer launching detached and tailing the log.
- **Context**: Monitoring long GPU jobs over ssh.

## 2026-06-11 — Mean-field collapse is fixed by the prior, not the sampler or floor
- **Mistake/insight**: Generated carbon structures collapsed (atoms ~1.0 Å apart).
  Natural guesses — add stochastic sampling (churn) or lower the CFM loss floor via
  sharper coupling — BOTH failed in an ablation. Churn just jitters a collapsed mean;
  fixed-prior coupling drove the loss floor to 0.024 but overfit trajectories and still
  collapsed at a fresh test prior. The fix was a **concentrated wrapped-normal prior**
  (std 0.25 around cell-center) instead of uniform: it forces the deterministic field to
  learn *expansion*, cancelling the barycenter contraction (valid bonds 17%→84%).
- **Rule**: For deterministic flow matching that under-disperses multimodal/exchangeable
  targets, change what the *mean field must do* (via the prior geometry) before reaching
  for stochastic samplers or coupling tricks. A lower training-loss floor is NOT evidence
  of better samples — fixed coupling can lower loss while worsening generation. And don't
  assume stacking all fixes is best: here all-three was worse than the prior alone.
- **Context**: Flow-matching / diffusion generative models that collapse to a barycenter.

## 2026-06-12 — Small-sample match rate is noisy; confirm before acting
- **Mistake**: A centroid_prior_std sweep on 64 eval structures showed a clean peak at
  0.30 (7.8%) vs 0.25 (4.7%). Re-evaluating both on 256 structures gave 3.1% each — the
  peak was sampling noise (64 structures quantizes match rate to ±1.6% per hit, plus the
  random prior varies run-to-run).
- **Rule**: For a noisy rate metric, fix the evaluation set size large enough that the
  quantization is well below the effect you're chasing, and re-confirm a "best" config on
  the larger set before tuning further or scaling. Don't pick hyperparameters off a metric
  whose noise floor exceeds the differences between candidates.
- **Context**: Any sweep ranked by a low-count rate (match rate, hit rate, pass@k).

## 2026-06-12 — Know when to stop scaling and change approach
- **Insight**: After fixing carbon-24 collapse, match rate plateaued ~5%. Ruled it out
  as prior-std, sampler-steps, and model-capacity limited (2× size + 2.5× steps gave no
  gain). The cap was the modelling setup (weak conditioning + one-to-one match metric),
  not under-training.
- **Rule**: When a metric is flat across the cheap knobs AND a capacity/steps bump doesn't
  move it, stop burning equivalent runs — the bottleneck is architectural/metric. Report
  the negative, switch approach (here: best-of-k eval, diffusion baseline), don't scale more.
- **Context**: Diminishing-returns calls on paid GPU.

## 2026-06-12 — Evaluate generative CSP with match@k, not match@1
- **Insight**: carbon-24 match@1 sat at ~3.9–5% and looked like a hard ceiling across
  std/steps/capacity sweeps. Implementing best-of-k (draw k gens/ref, hit if any matches)
  gave match@20 = 35.5% — a ~9× lift — from the SAME checkpoint. The "plateau" was a
  metric artifact: a generator that makes valid-but-different polymorphs is penalized by a
  one-to-one metric.
- **Rule**: For one-to-many generative tasks (CSP, molecule/conformer generation), the
  one-to-one (k=1) metric measures the wrong thing. Report the standard best-of-k the field
  uses (match@20 here) before concluding a model is capacity- or objective-limited. A flat
  k=1 number with good per-sample validity is a signal to check k, not to scale.
- **Context**: Any generative model evaluated against a single reference per input.

## 2026-06-13 — Check the predict-zero floor before blaming the sampler
- **Insight**: The diffusion baseline's fractional coords collapsed (10–13% overlaps,
  match@1 ~0%). Tempting to blame the sampler (added a Langevin corrector — it lifted
  match@1 only 0%→1.6%). The decisive test was a 10-line diagnostic: compute the
  predict-zero loss E‖target‖² and compare to the trained loss. They were equal (2.538
  vs 2.54) → the score net had learned ~nothing; the objective, not the sampler, was the
  problem. Root cause: carbon fractional marginal is ~uniform, so plain DSM has almost no
  learnable coordinate signal (the same under-dispersion the FLOW only beat via OT
  coupling + a concentrated prior — diffusion's fixed forward process has neither).
- **Rule**: When generated samples are bad, first decide *training vs sampling* with a
  cheap floor check — compare the trained loss to the loss of the trivial predictor
  (predict-zero / predict-mean). If they match, the network learned nothing and no
  sampler change will help; fix the objective/target/coupling. Don't tune the sampler to
  rescue an unlearned field. (Mirror of the earlier "is the loss at its irreducible
  floor?" lesson, but here applied to detect the *opposite* failure: floor == learned-nothing.)
- **Context**: Diffusion/score models, or any regression-trained generator with poor samples.

## 2026-06-10 — Git commits
- **Rule**: Never add `Co-Authored-By` trailers to commits in this user's repos.
- **Context**: Standing user preference.
