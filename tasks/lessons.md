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

## 2026-06-10 — Git commits
- **Rule**: Never add `Co-Authored-By` trailers to commits in this user's repos.
- **Context**: Standing user preference.
