# SymMC-Flow — Research & Engineering Plan

**Symmetric Molecular Crystal Flow Matching with Pair-Bias Attention and Rigid-Body Conformers**

Status: scaffold + reference implementation. Last updated: 2026-06-10.

---

## 1. One-paragraph thesis

Periodic molecular-crystal structure prediction (CSP) is bottlenecked by the
huge, flexible intramolecular configuration space and by the cost of diffusion
samplers (~1000 ODE/SDE steps). **SymMC-Flow** combines three recent ideas into
one flow-matching generator:

1. **Rigid-body conformer decomposition** (MolCrystalFlow) — freeze the molecule's
   internal geometry, predict only the lattice `L`, centroid fractional position
   `x ∈ T³`, and orientation `R ∈ SO(3)` per molecule.
2. **Space-group-conditioned flow** (SGFM) — condition the vector field on a
   crystallographic space group `G ∈ {1…230}` and enforce symmetry by averaging
   the predicted field over the group action.
3. **Pair-bias attention** (Clari) — replace memory-heavy triangle attention with
   plain attention plus a learned pairwise bias, plus a 2D periodic-table
   `(row, col)` atom embedding (CrystalDiT).

**Hypothesis:** order-of-magnitude fewer sampling steps than diffusion baselines
(~50 RK4 steps) while reaching a **Stable/Unique/Novel (SUN) rate > 75%** on
MP-20 and the carbon dataset.

> **Reframe (2026-06-20, real-CSD orientation finding — see `MOLCRYSTAL.md`).** On real
> molecular crystals the rigid-body **lattice + centroid** flow is the working contribution.
> The SO(3) **orientation** target decomposes as `R_m = rot(g_m)·R_asym`: the model **learns the
> space-group-determined relative orientation** between symmetry copies (+27% non-ref CFM loss;
> **16.8%** exact StructureMatcher reconstruction of held-out multi-copy packings vs 0% floor; the
> ceiling is inference-limited — discrete coset conditioning lifts it to +51%), while the **free
> asymmetric-unit orientation `R_asym`** is gauge-arbitrary and fundamentally unlearnable. The
> paper leads with this decomposition rather than an unqualified per-molecule SO(3) claim.

---

## 2. What is in scope for this repo

This repository is a **CPU-runnable reference implementation** of the full
architecture plus a **synthetic data harness** so every component is unit-tested
and the train→sample loop runs end-to-end without a GPU or external datasets.
Real benchmarking (MP-20, carbon, WBM) is a separate GPU phase, documented in §7.

| Deliverable | State |
|---|---|
| Manifold flow-matching core (`T³`, `SO(3)`, lattice `R^{3×3}`) | reference impl + tests |
| EGNN invariant encoder | reference impl + tests |
| Pair-bias attention block | reference impl + tests |
| 2D periodic-table `(r,c)` embedding | full Z=1…118 table + tests |
| Space-group action + group averaging | reference impl (subset of ops) + tests |
| Combined `SymMCFlow` model | reference impl + tests |
| RK4 ODE sampler (configurable steps) | reference impl + tests |
| Synthetic molecular-crystal dataset | generator + tests |
| Train / sample demo scripts | runnable on CPU, CUDA-aware |

**Explicitly out of scope here (GPU phase):** full MP-20/carbon ingestion via
`pymatgen`/`mp-api`, multi-GPU training, PXRD evaluation against experiment,
and the full SUN-rate benchmark. Hooks for these are left in `data.py` / `eval`.

---

## 3. Mathematical formulation

### 3.1 State variables (per crystal)

- **Lattice** `L ∈ R^{3×3}` — Euclidean manifold, optimal-transport (straight)
  conditional path: `L_t = (1−t)L_0 + t·L_1`, target velocity `u = L_1 − L_0`.
- **Centroid fractional coords** `x ∈ T³ = (R/Z)³` — torus geodesic with wrapped
  difference `w = ((x_1 − x_0 + 0.5) mod 1) − 0.5`; path `x_t = (x_0 + t·w) mod 1`,
  target velocity `u = w`.
- **Orientation** `R ∈ SO(3)` — Riemannian geodesic via matrix exp/log:
  `R_t = R_0 · exp(t · log(R_0ᵀ R_1))`, target velocity is the body-frame tangent
  vector `u = log(R_0ᵀ R_1) ∈ so(3) ≅ R³`.

### 3.2 Conditional flow matching loss

For each manifold M with conditional path `ψ_t(z_0 | z_1)` and target field `u_t`:

```
L_CFM = E_{t~U(0,1), z_0~prior, z_1~data} [ ‖ v_θ(z_t, t, c) − u_t ‖²_M ]
```

`c` is the conditioning context (space group `G`, molecular embedding, atom
features). The three manifold losses are summed with weights `λ_L, λ_x, λ_R`.

### 3.3 Space-group equivariant field

SGFM group averaging over the symmetry operations `g = (W_g, t_g)` of `G`:

```
v_t^G(x) = (1/|G|) Σ_{g∈G} σ_g · v_t(g · x)
```

where `g · x = (W_g x + t_g) mod 1` acts on fractional coords and `σ_g = W_g`
carries the pushforward back to the tangent (so the averaged field is
`G`-equivariant). The same averaging symmetrizes the lattice field via `W_g`.

### 3.4 Sampling

Integrate `dz/dt = v_θ(z_t, t, c)` from the prior (`t=0`) to data (`t=1`) with
**RK4** in `N` steps (default 50). Torus coords are re-wrapped each step;
`SO(3)` updates use the exponential map to stay on the manifold.

---

## 4. Architecture (data flow)

```
SMILES / conformer
   │  PCA → rigid body
   ▼
[2D periodic-table (r,c) embed]  +  [EGNN invariant encoder]
   │                                   │
   └──────────────┬────────────────────┘
                  ▼
        [Pair-bias attention stack]   ← pair features (distances, bonds)
                  │
                  ▼     ⊕ time embed t,  ⊕ space-group embed G
        ┌─────────┴──────────┐
        ▼         ▼          ▼
   v_L (3×3)   v_x (T³)   v_R (so(3))     ← per-molecule vector-field heads
        └─────────┬──────────┘
                  ▼
        SGFM group averaging  →  RK4 ODE solver (50 steps)
                  ▼
        Lattice L, centroids x, orientations R  →  reconstruct crystal
```

---

## 5. Repo layout

```
symmc-flow/
  PLAN.md                  ← this file
  README.md
  requirements.txt  pyproject.toml  .gitignore
  tasks/todo.md  tasks/lessons.md
  symmc_flow/
    periodic_table.py      (r,c) table + embedding
    manifolds.py           T³ / SO(3) / lattice exp·log·geodesic·prior
    egnn.py                equivariant GNN encoder
    pair_bias_attention.py Clari-style attention
    space_group.py         symmetry ops + group averaging
    flow.py                conditional paths + CFM loss
    model.py               SymMCFlow module
    sampler.py             RK4 ODE integrator
    data.py                synthetic dataset (+ real-data hooks)
    config.py              dataclass configs
    train.py               training loop (CUDA-aware)
  tests/                   pytest unit tests per module
  scripts/train_demo.py  scripts/sample_demo.py
```

---

## 6. Verification protocol (this repo)

- `pytest -q` — every module unit-tested (manifold exp/log round-trips,
  EGNN invariance, attention shape/grad, group-averaging equivariance,
  CFM loss decreases, RK4 recovers an analytic field).
- `python scripts/train_demo.py` — synthetic loss drops over a few hundred steps.
- `python scripts/sample_demo.py` — 50-step RK4 produces valid lattices /
  wrapped fractionals / orthonormal rotations.

## 7. GPU benchmark phase (next, on vast.ai)

1. `git clone` this repo on the GPU instance, `pip install -r requirements.txt`.
2. Add real loaders in `data.py` (MP-20, carbon, WBM) via `pymatgen` + `mp-api`.
3. Train experimental + control configs:
   - **Ctrl A (diffusion):** DiffCSP++, SymmCD.
   - **Ctrl B (flow, no rigid-body / no pair-bias):** FlowMM, plain SGFM.
   - **Exp:** full SymMC-Flow.
4. Metrics: SUN rate (target >75%), match rate, RMSD, sampling-step count,
   wall-clock vs diffusion, PXRD pattern comparison.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| SO(3) flow numerical drift over many steps | exp-map updates; re-orthonormalize via SVD each step |
| Full 230-group symmetry table is large | start with a curated op subset; pull full ops from `pymatgen.symmetry` in GPU phase |
| Synthetic data ≠ real distribution | data hooks isolate the loader; architecture is dataset-agnostic |
| CFM weighting `λ` imbalance across manifolds | tune on synthetic val loss; expose in `config.py` |
| 75% SUN is ambitious | report honestly; ablate each component (rigid-body, pair-bias, SGFM) |
