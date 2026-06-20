# Annotated bibliography & positioning (Phase 1)

All entries verified by web search (June 2026); full records in `references.bib`. Organized by the
role each plays in the paper.

## Generative foundations (cited in Methods/Intro)
- **Lipman et al. 2023** (Flow Matching) + **Chen & Lipman 2024** (Flow Matching on General
  Geometries / Riemannian FM) — the simulation-free CFM objective and its manifold extension; the
  basis for our lattice×T³×SO(3) flow.
- **Pooladian et al. 2023** (Multisample Flow Matching) — minibatch OT couplings; justifies our
  `ot_couple` for the exchangeable centroid/orientation target.
- **Satorras, Hoogeboom & Welling 2021** (EGNN) — the invariant conformer encoder.

## SO(3)/SE(3) generative precedent (the rigid-body analogy)
- **Leach et al. 2022** (DDPM on SO(3)) — generative modeling directly on the rotation group.
- **Yim et al. 2023** (FrameDiff, SE(3) diffusion) + **Yim et al. 2023** (FrameFlow, SE(3) flow
  matching) — rigid-frame generation for protein backbones; the protein-world precedent for
  per-body SO(3)+translation, and the "~5× fewer steps with flow" point we echo.

## Inorganic crystal generation (the lineage we extend)
- **Xie et al. 2022** (CDVAE), **Jiao et al. 2023** (DiffCSP), **Miller et al. 2024** (FlowMM) —
  the diffusion→flow lineage on lattice + fractional coordinates (single-atom blocks; no SO(3)).
- **Jiao et al. 2024** (DiffCSP++), **WyckoffDiff 2025**, **SymmCD 2025** — space-group / Wyckoff
  conditioned generation; the closest prior on *discrete symmetry conditioning*, which motivates
  and contextualizes our **coset-id** conditioning experiment (2c).

## Molecular / MOF CSP — the direct neighbors (Related Work spine)
- **Kim et al. 2025 (MOFFlow)** — *the* methodological precedent: rigid-body **SE(3) flow matching**
  (metal nodes + linkers as rigid bodies, SO(3)+translation per block, Riemannian FM). Demonstrated
  it *works* for MOFs but never analyzed *what the orientation field learns or where it fails*.
- **Jin et al. 2025 (OXtal)** — **all-atom** diffusion for organic CSP; *no* rigid-body
  factorization, *no* explicit SO(3); learns joint conformer+packing directly.
- **Subramanian et al. 2026 (PackFlow)** — flow matching on **Cartesian heavy-atom** coords +
  lattice with RL physics alignment; again *no* rigid-body SO(3).
- **Wengert et al. 2021** (data-efficient molecular CSP) + **Blind Test 7 (Hunnisett et al. 2024)**
  — classical/ML molecular CSP context and why orientation+packing is hard.

> **Positioning (the gap):** molecular-crystal generation has bifurcated into *all-atom*
> approaches (OXtal, PackFlow) that sidestep orientation entirely, while *rigid-body SE(3) flow*
> was shown only for MOFs (MOFFlow) and reported as a working method, not interrogated. **No prior
> work asks what a rigid-body SO(3) orientation flow actually learns on real molecular crystals.**
> We do, and find a clean decomposition (`R_m = rot(g_m)·R_asym`) with a learnable symmetry-relative
> part and an unlearnable gauge-free part — explaining both why rigid-body SO(3) can work and where
> it must fail.

## Evaluation tooling & potentials
- **Ong et al. 2013 (pymatgen)** — `StructureMatcher` and crystal handling; our match metric.
- **Martirossyan et al. 2025 ("All that structure matches does not glitter")** — match-rate metrics
  mislead when identical building blocks have structural variety; justifies our reporting of SO(3)
  geodesic error + a controlled, orientation-isolated best-of-k match metric rather than a headline
  rate, and our caution on the carbon-24 duplicate issue.
- **Deng et al. 2023 (CHGNet)** — universal MLIP trained on *inorganic* MP data; we cite it to
  justify why a de-novo SUN evaluation does not transfer to *organic* molecular crystals (the stated
  scope limit for not reporting SUN here).
