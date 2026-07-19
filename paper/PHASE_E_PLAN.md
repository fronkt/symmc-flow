# Phase E — symmc-flow → JCIM (journal-strength)

**Goal.** Survive archival peer review at JCIM (J. Chem. Inf. Model.). The MoML/AI4Mat
workshop paper (`paper/submissions/MoML/`, commit 5d3072c) is a genuine positive result but
workshop-strength. A methods-journal *peer* reviewer (not a desk screen) raises three
objections; Phase E neutralizes them, then we expand to a full paper.

## The 3 objections to neutralize
- **O1 — end-to-end.** "Only orientation-isolated works; full de-novo generation is 0%. Does
  this actually improve structure prediction?"
- **O2 — deployability.** "It needs a template; the de-novo predictor (39.5%) collapses to
  64.8°. Is it really usable without the answer?"
- **O3 — scale/benchmark.** "~1k crystals, reconstruction-only eval, no head-to-head with a
  real baseline (MolCrystalFlow)."

## Experiments (prioritized: impact / effort)

### E1 — Coset-conditioned end-to-end vs unconditioned  [O1 · HIGH impact · LOW effort · CPU-local]
The deployment claim, as a same-model ablation. In the template regime (space group + coset
supplied, as template-based CSP operates), does supplying the coset at *generation* improve the
output vs the same model with coset off?
- [x] Run `eval_templated_matchrate.py --ckpt coset_deploy_s0.pt --ablate-no-coset` (the SKIPPED
      half). **DONE locally 2026-07-18** (n=131, k=20; `gpu_results/phaseE/templated_unconditioned.log`).
- [x] **RESULT (same model, coset template ON vs OFF, n=131 k=20):** orientation error
      **13.4° (ON) vs 93.5° (OFF)** — a 7× collapse; centroid 0.346 vs 0.345 (identical — coset
      does not touch centroid); lattice-param 0.442 vs 0.501; exact match 0% both. The coset
      template makes the ORIENTATION component work end-to-end; lattice+centroid remain the
      exact-match bottleneck. → answers O1 at the component level; scopes contribution to
      orientation. **Strengthens the MoML "Scope" paragraph too (offer to fold in).**

### E2 — Scale  [O1+O3 · HIGH impact · HIGH compute → GPU/Vast]
- [ ] Retrain coset-conditioned + no-coset control on a LARGER molecular-crystal corpus
      (5k–50k via `csd_export.py`, or the xrd-clip 315k extraction pipeline). Expect higher
      absolute end-to-end + stronger orientation numbers → directly answers "modest scale."
- [ ] Re-run the gate at scale: does the +41% coset gain hold or grow?

### E3 — Template-free predictor  [O2 · MED impact · MED effort]
- [ ] Push the 39.5% predictor: bigger model / richer features (local packing, H-bond topology,
      dipole), OR **top-k marginalization** at sampling (condition on the k most-likely cosets,
      take best-of-k). CPU-evaluable for the marginalization variant.
- [ ] Goal: predicted-coset reconstruction BEATS the no-coset baseline (currently collapses
      64.8° vs 41°). Even top-3 marginalization making it viable answers O2.

### E4 — Benchmark alignment vs MolCrystalFlow  [O3 · MED effort · CPU-local eval]
- [ ] Mirror MolCrystalFlow's protocol (match@k at stol=0.8) so numbers are legible to the same
      reviewers; report ours alongside their ~6.8% match@10. Ideally on a comparable set.

### E5 — Full journal manuscript  [writing]
- [ ] Expand the 4pp workshop paper to a full JCIM paper: reuse the journal-length diagnostic
      `main-archive-2026-07-18-diagnostic.tex` as the skeleton, fold in the deployable method +
      E1–E4, full related work. JCIM/ACS format; author block per [[reference_author_identity]].

## Compute / sequencing
- **CPU-local NOW** (this machine has torch 2.11 + pymatgen): E1 baseline, E3 top-k
  marginalization eval, E4 metric alignment (eval-only).
- **GPU (Frank triggers Vast):** E2 scale training, E3 bigger-predictor training.
- **MoML fold-in:** any E1/E3 win that lands by ~late Aug goes into the MoML short paper too
  (non-archival — only helps). Hold MoML submission until ~late Aug, not now.
- **JCIM:** after MoML (Sep 1), finish E1–E5, submit the full paper.

## JCIM-ready gate
(a) conditioned end-to-end beats unconditioned or the orientation-component win is clean (E1);
(b) template-free predictor beats the no-coset baseline OR the template-regime claim is airtight
(E3); (c) numbers at a defensible scale (E2); (d) legible vs MolCrystalFlow (E4).
