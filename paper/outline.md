# Paper outline & evidence map (Phase 2) — FOR APPROVAL

**Working title:** *What an SO(3) orientation flow can and cannot learn in molecular-crystal
structure prediction*

**Alt title:** *Rigid-body flow matching for molecular crystals: a decomposition of the orientation
field into learnable and gauge-free parts*

**Venue/format:** npj Computational Materials · Nature numbered citations · LaTeX · ~5,400 words main
text · Framing A (focused orientation study; SymMC-Flow credited as the method).

**One-sentence thesis.** Casting molecular-crystal structure prediction as a rigid-body flow on
lattice × T³ × SO(3), the per-molecule orientation target decomposes as `R_m = rot(g_m)·R_asym`: the
flow learns the space-group-determined *relative* orientation between symmetry copies (16.8% of
held-out packings reconstructed exactly; ceiling shown to be inference-limited), while the
asymmetric-unit's *free* orientation `R_asym` is gauge-arbitrary and fundamentally unlearnable.

---

## Section plan (npj order: Intro → Results → Discussion → Methods)

### 1. Introduction (~750 w)
- CSP and its cost; generative crystal models as a fast alternative (CDVAE → DiffCSP → FlowMM).
- Molecular crystals are harder: flexible intramolecular DOF **and** rigid-body packing; two design
  axes — **all-atom** (OXtal, PackFlow) vs **rigid-body** (MOFFlow, for MOFs).
- Rigid-body factorization is attractive (low-dim, modular, ~50-step sampling) but introduces a
  per-molecule **SO(3) orientation field** whose learnability on molecular crystals is *untested*.
- **This paper:** SymMC-Flow (rigid-body flow on lattice/centroid/orientation, space-group
  conditioned). On real CSD data we find lattice+centroid learn but the orientation field floors —
  and we explain exactly why via the `R_m = rot(g_m)·R_asym` decomposition, with three independent
  lines of evidence.
- **Contributions** (bulleted): (i) rigid-body SO(3) flow for molecular crystals + intrinsic gauge;
  (ii) the decomposition + a falsification chain isolating the learnable vs gauge-free parts;
  (iii) an orientation-isolated StructureMatcher metric (16.8% vs 0%) turning loss into geometry;
  (iv) coset-conditioning evidence that the ceiling is inference-limited, not representational.

### 2. Results (~2,300 w)
- **2.1 Rigid-body factorization and the flow (setup, ~300w).** `cart = c@L + R_m·local`; the three
  manifolds; what each head predicts; space-group conditioning. → Fig 1.
- **2.2 Lattice and centroid flows learn on real molecular crystals (~300w).** Per-head held-out
  loss untrained→trained on 1127 CSD crystals. → Table 1. *(The working contribution.)*
- **2.3 The absolute orientation flow sits at a predict-zero floor (~450w).** E‖u_R‖² floor;
  robust across corpus sizes (250→1127); molecule-intrinsic gauge fix necessary but not sufficient;
  clean-packing diagnostic rules out two-stage conditioning.
- **2.4 The orientation target decomposes (~450w).** `R_m = rot(g_m)·R_asym`; relative re-gauge
  cancels `R_asym`; reference vs non-reference split. The 2×2 (absolute/relative × noised/clean):
  relative non-ref +27% and generalizes; absolute ~0%. → Table 2.
- **2.5 Relative orientation reconstructs real packings (~400w).** Orientation-isolated, best-of-k,
  StructureMatcher: trained 16.8% vs 0% floor / 1.5% naive / 100% oracle; SO(3) geodesic error
  68° vs 85°. Why best-of-k (multimodal near-antipodal target). → Table 3.
- **2.6 The ceiling is inference-limited, not representational (~400w).** Capacity+steps → +33%;
  discrete **coset-id** conditioning → +51% vs +27% control. Residual = free `R_asym` +
  symmetric-top multimodality. → Table 4.
- **(2.7 optional) Method validation on inorganic benchmarks (~150w or SI).** carbon-24 / MP-20
  match rates + DiffCSP head-to-head, framed as method validation (promotable for a C-pivot).

### 3. Discussion (~900 w)
- Interpretation: rigid-body SO(3) flow is viable, but orientation splits into a symmetry-determined
  learnable part and a gauge-free unlearnable part — a property of the *representation*, not the model.
- Why this reconciles the field: all-atom methods (OXtal/PackFlow) sidestep the gauge issue; MOFFlow
  works partly because MOF block orientations are more constrained than free molecular poses.
- Measurement: per the "glitter" critique, match-rate alone misleads; we report orientation error +
  controlled best-of-k matching.
- **Limitations:** de-novo SUN not meaningful for organics (CHGNet is inorganic-trained); CPU-scale
  training; rigid-conformer gate excludes flexible molecules; global conformer registry.
- **Future:** rigid pose + flexible refinement hybrid; coset/Wyckoff conditioning at sampling time;
  an organic MLIP to enable SUN.

### 4. Methods (~1,450 w)
- Rigid-body factorization + molecule-intrinsic gauge (`_canonical_frame`: gyration PCA + 3rd-moment
  sign + right-handed).
- Manifolds & CFM: lattice (log-vol/det-1 shape param), torus geodesic, SO(3) geodesic/exp; OT
  coupling; RK4 sampler.
- Architecture: EGNN encoder + pair-bias attention + three heads; SGFM symmetrization.
- Data: CSD two-stage export → filter → 1127 crystals / 5048 rigid blocks; reproducibility (seed +
  CSD v601; CIFs not redistributable, manifest shared).
- Diagnostics (precise definitions): predict-zero floor; clean-packing flag; relative gauge + ref
  split; coset codebook (`assign_cosets`); orientation-isolated best-of-k match metric.

### 5. Back matter (mandatory)
Data Availability · Code Availability · Author Contributions (CRediT) · Competing Interests ·
Funding · AI-use disclosure · (Limitations folded into Discussion).

---

## Figures & tables
- **Fig 1** — schematic: rigid-body factorization, the three manifolds, and the
  `R_m = rot(g_m)·R_asym` decomposition (learnable vs free). *(new, to draw)*
- **Fig 2** — bar chart: match rate by condition (oracle/identity/untrained/trained) +/- the
  capacity/coset strengthening. *(from results; `visualization_agent`)*
- **Table 1** — per-head held-out loss (lattice/centroid/orient), untrained→trained.
- **Table 2** — 2×2 non-reference orient loss (absolute/relative × noised/clean).
- **Table 3** — orientation-isolated match rate + SO(3) error (4 conditions, n=131).
- **Table 4** — strengthening: baseline / +capacity (2b) / +coset (2c) / control.

## Evidence map (claim → source)
| Claim | Evidence |
|---|---|
| Lattice/centroid learn; orient floors | `train_csd_molcrystal.py` runs; MOLCRYSTAL.md Results; floor E‖u_R‖²≈5.24 |
| Gauge fix necessary-not-sufficient | `_canonical_frame`; commit c38803a; scaled run |
| Two-stage ruled out | `diag_orient_conditioning.py`; clean-packing +3.3% |
| Relative learnable +27%, absolute ~0% | `diag_orient_relative.py`; Table 2 (2×2) |
| 16.8% exact reconstruction vs 0% | `eval_orient_matchrate.py`; Table 3 |
| Inference-limited ceiling (+33%/+51%) | `diag_orient_relative.py --d-model 192`; `diag_orient_coset.py` ±control; Table 4 |
| SUN not transferable to organics | CHGNet (inorganic MPtrj training) — Deng 2023 |
| Match-rate alone misleads | Martirossyan 2025 |
| Rigid-body SE(3) flow precedent | MOFFlow (Kim 2025); FrameDiff/FrameFlow (Yim 2023) |
| All-atom molecular CSP contrast | OXtal (Jin 2025); PackFlow (Subramanian 2026) |

## Open choices to confirm before drafting
1. Title: working vs alt (above) — preference?
2. Include §2.7 inorganic benchmarks in main text, or push to SI? (Default: brief main-text para,
   SI table — keeps the C-pivot cheap.)
3. Authorship/affiliation/funding lines for back matter (need names + grant IDs, or leave TODO).
