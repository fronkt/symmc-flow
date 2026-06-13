# SymMC-Flow

**Symmetric Molecular Crystal Flow Matching with Pair-Bias Attention and Rigid-Body Conformers.**

A flow-matching generator for periodic molecular-crystal structure prediction that
unifies three recent ideas:

- **Rigid-body conformers** (MolCrystalFlow) — freeze intramolecular geometry,
  generate only lattice `L`, centroid `x ∈ T³`, orientation `R ∈ SO(3)` per molecule.
- **Space-group-conditioned flow** (SGFM) — condition on a space group `G ∈ {1…230}`
  and enforce symmetry by averaging the vector field over the group action.
- **Pair-bias attention** (Clari) + **2D periodic-table `(row, col)` embeddings**
  (CrystalDiT) — long-range interactions without triangle-attention memory cost.

See [`PLAN.md`](PLAN.md) for the full research plan, math, and benchmark protocol.

> **Status (2026-06-13):** CPU reference implementation **plus** real GPU benchmarks on
> the CDVAE **carbon-24** and **MP-20** datasets (RTX 5090). Full unit-test suite green;
> train→sample runs end-to-end on CPU and GPU. Key results: the lattice was reparametrized
> in (log-volume, shape) space and a **concentrated wrapped-normal fractional prior** fixes
> the atom-collapse failure mode (valid C–C bonds 17% → ~82%). Scored with the CSP-standard
> `StructureMatcher.get_rms_dist` matcher (as in DiffCSP), **carbon-24 match@1 = 26.7%
> (match@20 79.7%), above DiffCSP's ~17%** — the earlier "~5% plateau" was a matcher/metric
> artifact (`fit` + match@1), not a real ceiling. Full experiment log in
> [`RESULTS.md`](RESULTS.md). A DiffCSP-style diffusion baseline is implemented and loses
> head-to-head (objective finding, see RESULTS).

## Install

```bash
pip install -r requirements.txt   # torch, numpy, scipy, pytest
# or: pip install -e .
```

## Quick start

```bash
python -m pytest -q              # 34 unit tests
python scripts/train_demo.py     # synthetic training; loss drops ~80%
python scripts/sample_demo.py    # 50-step RK4 sampling -> valid crystals
```

The training loop (`symmc_flow/train.py`) is CUDA-aware: it auto-selects `cuda`
when available (set `TrainConfig.device`).

### Carbon-24 benchmark (GPU)

```bash
pip install pymatgen pandas scipy
# place CDVAE carbon-24 CSVs at data/raw/carbon_{train,val,test}.csv
python scripts/train_carbon24.py --steps 6000 --batch 128   # default std=0.30
```

Key flags (see [`RESULTS.md`](RESULTS.md) for the ablation behind them):
`--centroid-prior-std` (the dispersion fix; negative ⇒ uniform prior),
`--fixed-prior`, `--churn`, `--eval-n`, `--d-model/--attn-layers`, and
`--eval-only --ckpt <path>` to sample a saved checkpoint without retraining.

## Architecture

```
atoms ─2D periodic-table embed─► EGNN (invariant per-molecule emb)
      ─► molecule tokens ⊕ (centroid, orient, time t, space group G)
      ─► pair-bias attention stack  (pair features = torus distances)
      ─► heads: v_L (3×3) · v_x (T³) · v_R (so(3))
      ─► SGFM group averaging ─► RK4 ODE solver (50 steps) ─► crystal
```

| Module | Role |
|---|---|
| `periodic_table.py` | Z → (row, col, f-index) for all 118 elements + embedding |
| `manifolds.py` | SO(3) exp/log (Shepperd-stable), torus & lattice geodesics, priors |
| `egnn.py` | E(3)-invariant EGNN encoder |
| `pair_bias_attention.py` | Clari-style attention + learned pairwise bias |
| `space_group.py` | symmetry ops + SGFM group averaging (equivariant field) |
| `flow.py` | conditional flow-matching paths + joint CFM loss |
| `model.py` | combined `SymMCFlow` vector-field network |
| `sampler.py` | manifold RK4 ODE integrator |
| `data.py` | synthetic dataset + real-data loader stubs |

## Results so far (carbon-24)

| Milestone | Outcome |
|---|---|
| Lattice reparam (log-vol + shape) | cell volume fixed: 6.3 Å³/atom vs 6.3 ref (was ~59 Å³) |
| Concentrated wrapped-normal prior | **atom collapse fixed: valid C–C bonds 17% → ~82%**, overlaps → ~2% |
| Position-flow ablation | churn (no help) / fixed-coupling (overfits) / **prior (the fix)**; combo worse than prior alone |
| Tuning + scaling | match rate plateaus ~5%; not limited by prior std, sampler steps, or model size |

See [`RESULTS.md`](RESULTS.md) for the full ablation tables, diagnostics, and reproduce steps.

## Next phase

1. Best-of-`k` match-rate metric (standard CSP eval) — one-to-one understates a
   generator that produces valid-but-different polymorphs.
2. DiffCSP-style diffusion baseline for a head-to-head objective comparison.
3. MP-20 / WBM loaders (`pymatgen` + `mp-api`); report SUN rate (target > 75%),
   match rate, RMSD, and sampling-step / wall-clock vs diffusion baselines.

## License

MIT
