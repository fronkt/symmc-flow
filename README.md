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

> **Status:** CPU-runnable **reference implementation** with a synthetic data harness.
> Every component is unit-tested and the train→sample loop runs end-to-end without a
> GPU or external datasets. Real MP-20 / carbon / WBM benchmarking is the GPU phase
> (loaders stubbed in `symmc_flow/data.py`, protocol in `PLAN.md §7`).

## Install

```bash
pip install -r requirements.txt   # torch, numpy, scipy, pytest
# or: pip install -e .
```

## Quick start

```bash
python -m pytest -q              # 28 unit tests
python scripts/train_demo.py     # synthetic training; loss drops ~80%
python scripts/sample_demo.py    # 50-step RK4 sampling -> valid crystals
```

The training loop (`symmc_flow/train.py`) is CUDA-aware: it auto-selects `cuda`
when available (set `TrainConfig.device`).

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

## GPU benchmark phase

1. `git clone` on the GPU box, `pip install -r requirements.txt`.
2. Implement the real loaders in `data.py` (`pymatgen` + `mp-api`).
3. Train experimental + control configs; report SUN rate (target > 75%),
   match rate, RMSD, and sampling-step / wall-clock vs diffusion baselines.

## License

MIT
