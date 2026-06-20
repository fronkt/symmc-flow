# Simulated peer review — *What an SO(3) orientation flow can and cannot learn in molecular-crystal structure prediction*

Simulated double-blind review at the standard of npj Computational Materials.
Five-dimension scoring (skill rubric); recommendation and a prioritized revision
roadmap follow. This is deliberately critical — the goal is to find what a real
referee would, before submission.

## Summary of the manuscript
The paper casts molecular-crystal CSP as a rigid-body flow on
lattice × T³ × SO(3) (SymMC-Flow), trains on 1,127 CSD crystals, and shows the
lattice and centroid flows learn while the absolute per-molecule orientation
floors. It attributes the floor to a decomposition `R_m = rot(g_m)·R_asym`,
isolates the learnable relative part (+27% loss; 16.8% orientation-isolated
reconstruction; coset conditioning → +51%), and concludes the residual is a
gauge-free, unlearnable pose.

## Scores (1–10)
| Dimension | Weight | Score | Note |
|---|---|---|---|
| Originality | 20% | 8 | The decomposition and its three-way characterization are genuinely novel; no prior work interrogates the rigid-body SO(3) field on molecular crystals. |
| Methodological rigor | 25% | 6 | Strong controls (clean-packing, oracle, no-coset), but single train/val split, no error bars, and two claims need tightening (unlearnability; coset fairness). |
| Evidence sufficiency | 25% | 6 | The orientation-isolated metric assumes true lattice+centroid; no end-to-end de-novo molecular-crystal generation result, and no head-to-head with the named neighbors. |
| Argument coherence | 15% | 8 | The falsification chain is clear and well-sequenced; the decomposition ties it together. |
| Writing quality | 15% | 8 | Clean, well-structured, honest. A few unsupported statements need citations. |

**Weighted score ≈ 7.0/10. Recommendation: Major revision.** The contribution is
real and publishable, but the evidence as presented under-determines the headline
claim and two arguments are over-stated. None of the issues is fatal.

---

## Major issues (must address)

**M1. No end-to-end de-novo generation.** The 16.8% match is *orientation-isolated*:
true lattice, centroid, and conformer are supplied and only SO(3) is sampled.
This cleanly isolates orientation, which is the paper's point, but a reader cannot
tell whether the full SymMC-Flow pipeline generates valid molecular crystals at
all. At minimum, report the *joint* generation match rate (sample all three
manifolds) on the held-out set, even if low, so the orientation-isolated number
has context. Frame the isolated metric as a diagnostic on top of the joint result,
not as the only generation evidence.

**M2. "Fundamentally unlearnable" is over-claimed.** With the molecule-intrinsic
gauge fixed, `R_asym` is a *well-defined* quantity (asymmetric-unit orientation
relative to the lattice), so a skeptic can argue it might correlate weakly with
packing and is therefore not unlearnable in principle. The defensible claim is
narrower: `R_asym` is not predictable from composition and packing under the
conditioning available here, because the body-frame gauge is arbitrary and no
training signal connects it to the inputs. Soften the wording throughout
(abstract, intro, discussion) from "fundamentally unlearnable" to "not learnable
from composition and packing in this setting," and add one sentence on why
(gauge arbitrariness ⇒ no consistent target), ideally with a measurement: is the
empirical `R_asym` distribution consistent with uniform on SO(3)? If yes, report
it; that single test would convert the claim from assertion to evidence.

**M3. Coset-conditioning fairness / leakage.** The coset id is obtained by
clustering the *observed* relative rotations, i.e. it is computed from the answer.
The manuscript should state plainly that the coset experiment is an *upper-bound
/ representability* diagnostic (it answers "can the head represent rot(g_m) if told
which coset?"), and separate that from any deployment claim. If the coset is meant
to be available at sampling time, show it is derivable from space group + Wyckoff
assignment + copy index *without* the ground-truth rotation; otherwise label it an
oracle conditioning. As written, a referee will suspect the +51% is partly leakage.

**M4. Statistics.** All headline numbers (16.8%, +27%, +51%) come from a single
seed and one 964/131 split. Report mean ± s.d. over ≥3 seeds (or ≥3 splits), and
give a confidence interval on the 16.8% match rate (n=131 is small; the 95% CI is
roughly ±6%). The paper cites the "glitter" critique, which also warns that random
splits are inappropriate when polymorphs exist — address whether the split groups
polymorphs/refcode families.

## Minor issues

- **m1.** Unsupported statements need citations: "the most common organic space
  groups relate copies by two-fold rotations and inversions" (cite a CSD-statistics
  source, e.g. space-group frequency data).
- **m2.** State the CSD version used (reproducibility depends on it).
- **m3.** Table 1 mixes the absolute-orientation floor with lattice/centroid from
  the relative-corpus run; confirm all three rows are from one run/corpus or say so.
- **m4.** The abstract is dense; one sentence stating the practical takeaway
  (condition on symmetry for relative orientation, sample the free pose) would help.
- **m5.** Define "match" (StructureMatcher tolerances) in the main text, not only by
  reference to pymatgen defaults; give the ltol/stol/angle_tol values.
- **m6.** Fig 1 is a clear schematic but stylistically a draft; consider redrawing
  in TikZ for final typesetting (a `.tex` companion is suggested).
- **m7.** Report sampling wall-clock / step count for the molecular-crystal runs to
  substantiate the efficiency framing carried over from the inorganic benchmarks.

## What is already good (keep)
The control design (clean-packing, oracle, no-coset paired control), the honest
treatment of the match-rate metric, and the decomposition narrative are strengths.
Do not dilute them.

---

## Revision status (2026-06-20)

- **M1 (de-novo generation) — ADDRESSED.** Added end-to-end joint-generation match
  rate (`eval_denovo_matchrate.py`, GPU, 3 seeds): $0\%$ at match@20 (Wilson CI
  $0$--$2.8\%$, $n=131$), with a best-of-20 component breakdown (lattice $0.46$,
  centroid $0.35$, orientation $60^\circ$) locating the bottleneck. Reported
  honestly as a new Results subsection + abstract clause + Discussion limitation;
  reinforces the characterization framing.
- **M2 (over-claim) — ADDRESSED.** Softened throughout + new $R_{\text{asym}}$
  Haar-uniformity test (Fig S1: mean $122.6^\circ$ vs $126.5^\circ$, KS $D=0.04$).
- **M3 (coset fairness) — ADDRESSED.** Reframed as an upper-bound/representability
  diagnostic with the deployment-label note.
- **M4 (statistics) — ADDRESSED.** 3-seed error bars on GPU: relative
  $27.5\pm2.7\%$, coset $47.9\pm3.2\%$; Wilson CIs on match rates. The
  orientation-isolated match rate is now reported as a 3-seed mean: the headline
  $16.8\%$ (one CPU seed) becomes $13.7\pm2.0\%$ ($12.2/13.0/16.0\%$); the
  single-seed figure sat at the top of this spread. Table~1, Table~2, Fig.~2, and
  the abstract were updated, with error bars added to Fig.~2a.
- **Minors m1, m2, m4, m5, m3 — ADDRESSED** (CSD citation + version, matcher
  tolerances, abstract takeaway, Table 1 provenance, space-group citation).
- **Capacity + de-novo reproduction (2026-06-20, re-run on a free GPU) — DONE.**
  The earlier OOMs were two real bugs, now fixed: (i) `eval_orient_matchrate.py`
  lacked `@torch.no_grad()`, retaining the EGNN autograd graph ($30.5$ GB OOM
  $\rightarrow$ $3.3$ GB); (ii) the deeper $d{=}256$ model NaN'd in training, so a
  non-finite-step guard was added in `train.py` (skips the optimizer step on a
  non-finite loss/grad; no forward change, so existing checkpoints stay valid).
  Clean capacity run ($d256$, 6 attn / 5 enc, $5{,}000$ steps, lr $1\mathrm{e}{-4}$,
  1 step skipped): non-reference drop $+45.0\%$ (was $+33.4\%$); big-model de-novo
  remains $0\%$ match@20 with sharper components (lattice $0.37$, centroid $0.35$,
  orient $48^\circ$), strengthening the structural-bottleneck argument. Logs in
  `paper/gpu_results/`.
- **Outstanding (optional):** Fig 1 TikZ redraw (m6); sampling wall-clock (m7).
  C-pivot (a working generator) is not supported by the de-novo result,
  confirming the focused framing.

## Prioritized revision roadmap

### A. Text-only fixes (no new compute) — do first
1. M2: soften "unlearnable" wording everywhere; add the gauge-arbitrariness
   sentence. *(abstract, intro, discussion)*
2. M3: reframe coset as an upper-bound/representability diagnostic; state how the
   label would be obtained at deployment. *(§2.6, methods)*
3. m1, m2, m4, m5: add the missing citation, CSD version, abstract takeaway, and
   explicit matcher tolerances.
4. m3: clarify Table 1 provenance.

### B. Cheap compute (reuse existing checkpoints / CPU)
5. M4 (partial): rerun the relative and match-rate diagnostics over 3 seeds/splits;
   report mean ± s.d. and a CI on 16.8%. *(scripts already exist; CPU-feasible)*
6. M2 (evidence): test whether the empirical `R_asym` distribution is uniform on
   SO(3) (a histogram of geodesic distances to a fixed reference vs the analytic
   uniform density). *(new ~30-line analysis on cached data)*

### C. GPU experiments (the C-pivot the author reserved) — optional but strongest
7. M1: report end-to-end joint generation match rate on the molecular-crystal
   held-out set, and ideally a head-to-head against an all-atom baseline
   (PackFlow/OXtal) or MOFFlow-style rigid baseline on a common split.

> Reviewer's bottom line: the paper is a genuine contribution and is close. The
> text fixes (A) and the cheap statistics (B) would resolve most of the rigor and
> over-claiming concerns and lift the score into clear acceptance territory for a
> mechanistic study. The de-novo generation experiment (C) is what would move it
> from "characterization study" to "characterization + working generator" and
> pre-empt the most likely rejection reason.
