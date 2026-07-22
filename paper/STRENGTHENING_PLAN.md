# SymMC-Flow strengthening plan: from diagnostic to deployable symmetry-conditioned method

**Goal.** Convert the paper from a *negative/diagnostic* result (which both Digital Discovery
and TMLR desk-rejected on general-audience interest) into a *method contribution* by turning
the coset diagnostic — currently a leaky, non-deployable upper bound — into a **deployable
symmetry-conditioned rigid-body orientation generator**, and evaluating it in the way the CSP
field actually operates (template/symmetry-conditioned generation). This directly attacks the
"inference-limited ceiling" the diagnostic identified, and it fills a gap the 2026 rigid-body
molecular-crystal flow models leave open.

---

## 1. Deep-research findings: the competitive landscape (July 2026)

The subfield moved fast since the June-2026 draft. The relevant clusters:

**A. Rigid-body molecular-crystal flows (our direct neighbors).**
- **MolCrystalFlow** (arXiv 2602.16020, 2026): near-identical setup to SymMC-Flow —
  rigid bodies, lattice + centroid + SO(3) orientation on Riemannian manifolds, GNN, flow
  matching. **Explicitly does NOT impose space-group constraints** ("MolCrystalFlow and
  MOFFlow do not impose explicit space-group constraints, while Genarris-3 conditions on
  space-group symmetries"). Handles reference-frame degeneracy with only a **binary axis-flip
  state χ**, assigned *post hoc*, not pre-conditioned. No asymmetric-unit / Wyckoff reduction;
  all molecules treated independently. Trains on 11.5k (Thurlemann/CSD) and 47k (OMC25)
  structures; gets **~6.8% match@10 at stol=0.8**, ~8% after rigid-press. The *symmetry-
  conditioned* classical baseline **Genarris-3 reaches ~10% at stricter tolerance**.
- **MOFFlow** (ICLR 2025, already cited): rigid-body SE(3) flow for MOFs; framework
  connectivity constrains orientation, so it never confronts the free-pose obstruction.
- **Köhler et al.** rigid-body flows for sampling molecular-crystal structures (Boltzmann-
  generator lineage) — precedent for rigid-body position+orientation modeling.

**B. Symmetry-/Wyckoff-conditioned generation (proven — but inorganic only).**
- **NextCrystal** (2602.17176, 2026): conditions on Wyckoff positions + space-group operations
  *during sampling*. **Inorganic atoms only; no molecular orientation / rigid bodies.**
- **DiffCSP++** (cited), **WyckoffDiff**, **SymmCD** (cited), **CrystalFormer / Wyckoff
  Transformer**, symmetry-aware Bayesian flow networks (npj Comput Mater 2026): all condition
  atomic generation on space-group / Wyckoff templates. None touch rigid-body molecular pose.

**C. SO(3) pose generation & multimodality (methods we can borrow).**
- **DiffDock / SigmaDock / DiffDock-PP**: ligand/rigid-body pose as translation × SO(3) ×
  torsions; established best-of-k sampling of a multimodal SO(3) pose distribution — the same
  read our orientation-isolated metric already uses.
- **SO(3)-Averaged Flow Matching** (arXiv 2507.09785, 2025): a training objective that
  accelerates SO(3) flow convergence and improves generation — a candidate drop-in upgrade for
  our orientation head.
- **Matcha** (multi-stage Riemannian flow matching), **MotiFlow / GO-Flow** (rigid-motif SE(3)
  flows): multi-manifold decomposition is now standard; supports our factorization.

**The gap (our novelty, sharpened).**
Symmetry-operation/Wyckoff conditioning is proven for *inorganic atoms* but **never applied to
rigid-body molecular orientation**, and the *molecular* rigid-body flows (MolCrystalFlow,
MOFFlow) **deliberately skip space-group conditioning** and treat molecules independently —
exactly the regime where our decomposition proves the free pose is unlearnable and only the
symmetry-relative rotation is. So:

> **Contribution:** the first *symmetry-coset-conditioned rigid-body molecular orientation
> flow*, motivated by a decomposition that proves why unconditioned rigid-body orientation
> (à la MolCrystalFlow/MOFFlow) leaves signal on the table, and realized with information
> available at sampling time (space group + Wyckoff + copy ordering), as the inorganic
> symmetry-conditioned models already assume.

This reframes the paper as *the analysis + the missing symmetry-conditioning piece the 2026
rigid-body molecular flows need* — a method result, not a null result.

---

## 2. What is currently in the code (seams for the work)

- `symmc_flow/space_group.py`: **only 6 space groups** hand-coded (P1, P-1, P2, Pm, P222,
  Pmm2), identity fallback otherwise. Docstring already says "replace `get_ops` with full
  operations from pymatgen for the GPU benchmark phase." → **Blocker to fix first.**
- `molcrystal.assign_cosets` (molcrystal.py:379): builds the coset codebook by **greedily
  clustering the *observed* relative rotations** within each space group. This is the leaky,
  non-deployable label (derived from the target it predicts) → the paper's honest "upper bound"
  caveat. → **Replace with a symmetry-derived label.**
- `model.SymMCFlow.forward(..., coset=None)` (model.py:99): consumes a per-molecule `coset`
  via `coset_embed` (model.py:52) when `n_cosets>0`. Conditioning mechanism already exists.
- `sampler.rk4_sample` (sampler.py): **does not pass `coset`** to `model.forward` (sampler.py:44).
  → Thread `coset` through so conditioning is usable at generation time.
- `scripts/diag_orient_coset.py`, `scripts/eval_denovo_matchrate.py`,
  `scripts/eval_orient_matchrate.py`: existing training/eval harnesses to extend.
- `relative_gauge_item` (molcrystal.py:349) already gives R'_m = rot(g_m) and `is_ref`.

---

## 3. Method components to build

### C1 (prerequisite). Full space-group operations
Replace the 6-SG registry with `pymatgen.symmetry.groups.SpaceGroup(n).symmetry_ops` (fractional
`W`, `t`), plus the **Cartesian** rotation part `R_cart = L^T W L^{-T}` (or via the standard
metric transform) needed to compare against body-frame rotations. Cache per space group. This
also fixes the SGFM group-averaging (`symmetrize_field`) that currently degrades to P1 for all
but 6 groups — an independent correctness win we can report.

### C2 (core). Symmetry-derived, leak-free, deployable coset
Define the coset label as the **index of the generating space-group rotation operation**, not a
cluster of observed rotations:
- *Training label*: for each non-reference copy, assign the SG operation `k` whose Cartesian
  rotation part best matches R'_m (argmin geodesic). Label space = the finite, deterministic set
  of the space group's proper+improper operations (from C1), **not** data-derived clusters.
- *Deployability*: at sampling time the same `k` is given by the template — space group +
  Wyckoff assignment + copy ordering — exactly the input DiffCSP++/NextCrystal/WyckoffDiff
  already assume. So conditioning uses only sampling-time-available information.
- Handle improper operations (inversion/mirror) explicitly: for chiral molecules an
  inversion-related copy is the enantiomer (det −1); decide per-species whether to (a) treat as
  a distinct proper-rotation coset via the best SO(3) projection (matches current Kabsch
  behavior) or (b) flag enantiomer copies. Document the choice; report the achiral/centrosym
  breakdown.

**Why it's not circular:** using ground-truth structure to derive a *label* at training, then
supplying that label from a *template* at inference, is exactly the established Wyckoff-
conditioning protocol (DiffCSP++/WyckoffDiff/SymmCD). The old codebook was circular because the
label existed only as a function of the observed rotation and had no sampling-time source; the
SG-operation label has one.

### C3 (core). Thread coset through the sampler + template-based generation
- Add `coset=` to `rk4_sample` and pass into every `model.forward` call.
- New eval `scripts/eval_templated_matchrate.py`: the **symmetry-conditioned (template-based)
  CSP setting** — given the true space group and the per-copy operation assignment (the
  template), generate lattice + centroid + coset-conditioned orientation and match with
  StructureMatcher. Report against the **unconditioned** rigid-body model (the MolCrystalFlow/
  MOFFlow regime) on the identical corpus/split — the honest apples-to-apples delta.
- Also re-run the **orientation-isolated** metric with the deployable coset to show it realizes
  (most of) the 48% oracle-codebook upper bound using only template information.

### C4 (stretch). De-novo coset predictor (close the inference gap without a template)
A small classifier head that predicts the generating operation `k` from the (noised) packing +
space group, trained with cross-entropy against the C2 labels. Plugs into the sampler when no
template is supplied. Quantifies how much of the inference-limited ceiling is recoverable
de-novo. Even a partial closure is a real, publishable result and directly answers the
diagnostic's open question.

### C5 (stretch). SO(3)-averaged flow objective
Swap the orientation training objective for SO(3)-averaged flow matching (2507.09785) to test
whether a better SO(3) objective lifts the relative-orientation numbers. Clean ablation; low
risk (training-only change).

---

## 4. Evaluation & narrative for the strengthened paper

New/updated results the paper would carry:
1. **Deployable coset realizes the ceiling** (orientation-isolated): template-derived coset vs
   the old oracle-codebook (upper bound) vs no-coset baseline. Target: recover most of ~48%.
2. **Symmetry conditioning beats unconditioned generation** (template-based, end-to-end in the
   conditional setting): coset-conditioned vs unconditioned rigid-body model, same corpus/split
   — the headline "the missing piece helps" number, framed against MolCrystalFlow/MOFFlow's
   unconditioned design.
3. **De-novo coset predictor** [C4]: fraction of the inference gap closed without a template.
4. Retained diagnostics (decomposition, floor, gauge, leakage control) as the *why*.
5. Honest scope: end-to-end de-novo at 1k scale stays hard (packing bottleneck; MolCrystalFlow
   shows single-digit match even at 10–40× scale) — we claim the **orientation** contribution
   and the **template-based** gain, not solved de-novo CSP.

Comparison framing: we cannot out-scale MolCrystalFlow (they have 11–47k structures; we have
~1.1k), so we report **controlled deltas at fixed scale/model** (conditioned − unconditioned),
which is the scientifically valid claim and immune to the scale gap.

---

## 5. Compute plan
Modest. Per [[feedback_vast_workflow]] use a Vast.ai RTX 5090 (torch cu128/cu130). Runs are
800–5000 steps (minutes–hours each); budget a handful of GPU-hours for: full-SG-ops smoke,
deployable-coset retrain (3 seeds), templated + orientation-isolated eval, coset-predictor
train/eval, optional SO(3)-averaged ablation. Share the box politely (free-memory watcher);
never kill a co-tenant job. Back up checkpoints to `checkpoints/`.

## 6. Risks / honest caveats
- **Packing still bottlenecks end-to-end de-novo.** Mitigation: lead with orientation +
  template-based claims; optionally Wyckoff-constrain centroids (bigger lift, out of core scope).
- **Deployable coset may recover < 48%** (SG-op label is cleaner but coarser than the clustered
  codebook). That is itself an honest, informative result.
- **Improper-operation / chirality handling** needs care (C2); document and report the breakdown.
- **Concurrent work** (MolCrystalFlow etc. are arXiv 2026): position as complementary — we
  supply the symmetry-conditioning analysis+mechanism they omit; cite generously.

## 7. Venue targets after strengthening
With a deployable method + the diagnostic, viable homes open up:
- **First airing:** AI4Mat (Aug 30) + MoML (Sep 1) workshops — ideal audience, non-archival.
- **Archival:** a top ML venue (ICLR 2027 deadline ~Sep/Oct 2026, or ICML 2027 ~Jan 2027) now
  that it's method+analysis, or **npj Computational Materials** (the working-method + symmetry
  angle gives the impact story DD/TMLR found missing). Decide once results 1–2 are in.

## 8. Phased task list (decision gate after Phase B)
- **Phase A — enablement:** C1 full SG ops (+ tests); verify SGFM averaging; regenerate coset
  labels leak-free (C2). Gate: labels sane vs old codebook on the 1,095 multi-copy crystals.
- **Phase B — core result:** thread coset through sampler (C3); deployable-coset orientation-
  isolated + templated evals (3 seeds). **GATE: does template-derived coset beat no-coset and
  approach the 48% upper bound?** If yes → method confirmed, proceed. If no → report the gap
  honestly, reconsider.
- **Phase C — depth:** C4 coset predictor (de-novo gap closure); optional C5 SO(3)-averaged.
- **Phase D — write-up:** rebuild the manuscript as method+analysis; new tables/figs; related-
  work section vs MolCrystalFlow/MOFFlow/NextCrystal/Wyckoff line; pick venue.

**Nothing here is committed until Frank approves scope (core-only vs core+stretch).**
