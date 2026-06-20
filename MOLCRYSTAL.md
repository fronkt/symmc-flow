# Molecular-crystal benchmark (rigid-body, orientation-ON) — status

Status doc for the thesis-gap experiment: validate the headline novelty (rigid-body
molecular conformers + SO(3) **orientation** flow) on **real** molecular crystals, with
`lambda_orient > 0`. Both shipped real benchmarks (mp20, carbon24) use single-atom blocks
with `lambda_orient = 0`, so orientation was previously validated only on synthetic data.

_Last updated: 2026-06-20._

## TL;DR

The data pipeline and loader work end-to-end on real CSD data. **The lattice and centroid
flow heads learn on real molecular crystals. The SO(3) orientation head learns the
space-group-determined RELATIVE orientation between symmetry copies (partial positive, robust
to conditioning), but the ABSOLUTE per-molecule orientation sits at its predict-zero floor**
across two corpus sizes (250 and 1127), after a gauge fix, and even on the true clean packing.
The floor is the **free asymmetric-unit orientation** — a gauge-arbitrary degree of freedom,
not a broken flow or a conditioning artifact.

**The relative-orientation result is now characterized three ways (2026-06-20):**
1. **Loss** — non-reference (symmetry-determined) orient loss drops **+27%** (vs ~0% for the
   absolute target), generalizing to held-out val and robust to noised-vs-clean conditioning.
2. **Physical reconstruction** — isolating orientation (true lattice+centroid+conformer, sample
   only SO(3), best-of-8), the trained flow reconstructs **16.8% of held-out multi-copy crystals'
   full packing exactly** under StructureMatcher, vs **0%** for the predict-floor and **1.5%** for
   the naive "copies un-rotated" baseline (oracle ceiling 100%). The loss drop is real geometry,
   not a curve artifact.
3. **Mechanism — the ceiling is inference-limited, not representational.** Conditioning the field
   on the discrete space-group **coset id** (which general-position op generates each copy) lifts
   non-ref **+27% → +51%**; more capacity+steps gives **+27% → +33%**. So most of the ceiling is
   the difficulty of *inferring* the coset from the (noised) packing; the irreducible residual is
   the free asymmetric-unit orientation `R_asym`.

**The leading hypothesis — that noised lattice+centroid conditioning was starving the SO(3)
head of the packing geometry that determines orientation — has been TESTED and REJECTED.**
A clean-packing diagnostic (`scripts/diag_orient_conditioning.py`, the `cond_clean_packing`
flag) fed the field the un-noised lattice+centroids while still noising orientation. Over 800
steps on the 1127-structure corpus, lattice (1.20→0.044) and centroid (0.355→0.254) learn, but
orientation moves only 5.35→5.18 (+3.3%) — still at its ~5.2 floor. Since clean conditioning is
the absolute best case for the second stage of a two-stage model, **this rules out the
two-stage fix.**

**What the floor actually is — DECOMPOSED (2026-06-18).** The absolute per-molecule target
`R_m` bundles two parts: `R_m = rot(g_m) · R_asym`, where `rot(g_m)` is the space-group op that
generates copy m (learnable) and `R_asym` is the asymmetric unit's **free** orientation in the
cell (per-crystal, gauge-arbitrary → unlearnable), and `R_asym` dominates the target. The
relative-gauge diagnostic (`scripts/diag_orient_relative.py`, `relative_gauge_item`) re-gauges
each crystal so the first copy of a species is the reference (orient := I) and the rest carry
only `R'_m = R_m · R0⁻¹`, cancelling `R_asym`. **This is a partial positive:** on the 1095
multi-copy crystals, the genuinely symmetry-determined NON-reference orient loss drops
**5.37→3.94 (+27%) and generalizes to held-out val** (vs +3.3% ≈ noise for the absolute
target). So orientation is **not uniformly unlearnable** — the SO(3) flow does learn the
space-group-induced relative orientation between symmetry copies; what is unlearnable is the
free `R_asym`. Direction now: reframe (#1 below) with this precise decomposition as the
characterization.

## What was built (all committed, reproducible)

| component | file | what it does |
|---|---|---|
| Loader | `symmc_flow/molcrystal.py` | crystal → rigid molecular blocks `(Z, local, centroid, orient)`; PBC-unwrap molecule detection (JmolNN), per-species conformer registry, Kabsch align with automorphism-min mapping, **molecule-intrinsic gauge** (`_canonical_frame`), disorder + non-rigid skip gates, tolerant `read_cif` (special-position occupancies). Same batch dict as mp20/carbon24 → **no model change**. |
| CSD export | `scripts/csd_export.py` | runs under the **CCDC bundled interpreter**; seeded random CSD sample → filter (organic, ordered, 3D, not polymeric, R≤rmax, 0<Z'≤zmax, size cap) → CIFs + shareable `manifest.csv`. |
| COD fallback | `scripts/fetch_cod_molcrystals.py` | license-free path (COD `el…&strictmax&format=lst`); same downstream. |
| Factorizer | `scripts/factorize_cifs.py` | any CIF dir → `MolCrystalDataset` + kept/skip + Z'/atoms histograms. |
| Training | `scripts/train_csd_molcrystal.py` | orientation-ON training on the real corpus; train/val split; per-head loss untrained→trained; lattice prior matched to corpus volume/atom. |
| Diagnostic | `scripts/diag_orient_conditioning.py` | clean-packing test (`cond_clean_packing` flag, `train._step_loss`): conditions the field on the TRUE lattice+centroid to isolate whether noised packing floors the SO(3) head. Rules out two-stage. |
| Diagnostic | `scripts/diag_orient_relative.py` + `relative_gauge_item` | relative-orientation test: re-gauges to first-copy-as-reference (cancels the free asymmetric-unit orientation), restricts to multi-copy crystals, reports orient loss split ref vs non-reference. Shows the symmetry-determined relative orientation is partially learnable. Now also takes `--d-model/--n-attn-layers/--egnn-layers` for the capacity sweep (2b). |
| Match metric (2a) | `scripts/eval_orient_matchrate.py` | orientation-isolated StructureMatcher match rate: holds lattice+centroid+conformer at truth, samples only SO(3) (best-of-k over orientation priors), rebuilds via `rigid_to_structure`, scores fit vs oracle/identity/untrained/trained. Converts the +27% loss into a physical 16.8% reconstruction rate. |
| Coset conditioning (2c) | `scripts/diag_orient_coset.py` + `assign_cosets` (loader), optional `coset_embed` (model, `ModelConfig.n_cosets>0`) | clusters non-reference relative rotations into a per-space-group coset codebook and conditions the field on the discrete coset id; `--no-coset` paired control. Isolates inference-limited vs representational residual (non-ref +27%→+51%). |

**Reproduce** (CSD CIFs are not redistributable → `data/csd_mol/` is gitignored; the export
is reproducible from seed + CSD version `601`):

```bash
# 1. export (CCDC interpreter)
"C:/Users/frank/CCDC/ccdc-software/csd-python-api/miniconda/python.exe" \
    scripts/csd_export.py --n 3500 --sample 18000 --seed 0 --out data/csd_mol
# 2. factorize (project env)
python scripts/factorize_cifs.py --cif-dir data/csd_mol/cif --cache data/csd_mol/ds.pt
# 3. train orientation-ON
python scripts/train_csd_molcrystal.py --cache data/csd_mol/ds.pt --steps 1500 --lr 3e-4
```

## Results

Corpus: 3500 CSD CIFs (14,874 scanned) → **1127 crystals / 5048 rigid blocks** (also ran a
250-structure corpus first; same conclusion).

| head | untrained | trained | verdict |
|---|---|---|---|
| lattice | 1.4 | ~0.06–0.10 | **learns** |
| centroid | 0.36 | ~0.24 | **learns** |
| **orientation** | 5.24 (predict-zero floor) | **~5.0–5.4, no downward trend** | **at the floor — does not learn** |

- Decisive diagnostic (per the predict-zero-floor lesson): E‖u_R‖² = **5.24/5.29**; trained R
  never leaves the noise band around it over 200–350 steps while lattice/centroid converge in
  <1 epoch.
- **Scaling 250 → 1127 did not change this** → not a data-sparsity problem.
- A gauge fix was necessary first (see below) but **not sufficient**.

### Clean-packing diagnostic (2026-06-18) — rules out two-stage conditioning

`scripts/diag_orient_conditioning.py` sets `TrainConfig.cond_clean_packing=True`: the field is
fed the **true** (un-noised) lattice+centroid while orientation is still noised (`z_t.orient`)
and the orient target `u_R` is unchanged (no label leakage — `z1.orient` is never shown). This
is exactly the conditional flow "orientation | true packing" — i.e. **stage 2 of a two-stage
model in its best case.** If orientation learns here, two-stage is validated; if not, two-stage
cannot help.

| head | untrained | trained (800 steps) | note |
|---|---|---|---|
| lattice | 1.204 | 0.044 | learns (ill-posed target under this flag; ignore) |
| centroid | 0.355 | 0.254 | learns (idem) |
| **orientation** | 5.352 | **5.175 (+3.3%)** | **still at floor; R train oscillates 4.69–5.90, no trend** |

**Verdict: NULL.** Orientation does not learn even from the true packing → the two-stage fix is
ruled out, and the cause is deeper than noised conditioning. The next diagnostic identifies what
that cause is. Checkpoint `checkpoints/diag_orient_cleanpack.pt` (pre/post + split). Reproduce:
`python scripts/diag_orient_conditioning.py --cache data/csd_mol/ds.pt --steps 800`.

### Relative-orientation diagnostic (2026-06-18) — decomposes the floor (PARTIAL positive)

The absolute target factors as `R_m = rot(g_m) · R_asym`: a space-group-determined relative part
`rot(g_m)` (learnable) and the asymmetric unit's **free** orientation `R_asym` (per-crystal,
gauge-arbitrary, **unlearnable**, and dominant). `scripts/diag_orient_relative.py` re-gauges each
crystal via `relative_gauge_item` (first copy of each species → reference `orient=I`; others carry
`R'_m = R_m·R0⁻¹`, cancelling `R_asym`), keeps only the **1095** multi-copy crystals (≥2 copies of
a species; multiplicity histogram `{2:329, 3:10, 4:650, 6:12, 8:86, 12:1, 16:7}`), and splits the
orient loss into **reference** copies (target I, trivial) vs **non-reference** copies (genuinely
symmetry-determined — guards against a trivial predict-identity win).

Held-out val, non-reference orient loss (untrained → trained, 800 steps), across the full 2×2:

| target ＼ conditioning | noised (realistic) | clean packing (best case) |
|---|---|---|
| **absolute `R_m`** | 5.37 → 5.18 (**+3.3%**, floor) | 5.35 → 5.18 (**+3.3%**, floor) |
| **relative `R'_m`** | 5.37 → 3.91 (**+27.1%**) | 5.37 → 3.94 (**+26.7%**) |

**Verdict: PARTIAL positive.** Strip out `R_asym` and the SO(3) flow **learns** the
space-group-induced relative orientation between symmetry copies (~27% non-ref drop, generalizes;
overall orient +34–36%, reference copies +56–61%) — vs ~0% for the absolute target. The result is
**identical under noised and clean conditioning**, so the signal rides on the space group (directly
conditioned) + coarse centroid arrangement and is robust to conditioning noise. **Conclusion: the
orientation floor is the free asymmetric-unit orientation — a fundamental gauge-arbitrary degree
of freedom, not a broken SO(3) flow and not a conditioning artifact.** Orientation is therefore
*partially* learnable, with the unlearnable part precisely identified. Checkpoints
`checkpoints/diag_orient_relative{,_noised}.pt`. Reproduce:
`python scripts/diag_orient_relative.py --cache data/csd_mol/ds.pt --steps 800 [--clean-packing]`.

### Strengthening the partial positive (2026-06-20)

Three follow-ups turn the +27% loss-curve claim into a complete characterization.

**2a — physical match rate (`scripts/eval_orient_matchrate.py`).** To prove the loss drop is real
geometry, orientation is *isolated*: hold the true lattice + centroid + conformer fixed (the
clean-packing conditioning the diagnostic showed is equivalent), sample **only** the SO(3) field,
rebuild a `pymatgen` Structure (`rigid_to_structure`), and score `StructureMatcher.fit` vs the
ground truth. Because the symmetry-induced relative rotations are large (≈180° for the dominant
2-fold/inversion copies), a *single* deterministic ODE draw collapses to the conditional mean and
under-performs; the correct generative read is **best-of-k** over orientation priors (k=8 here).
Held-out val (n=131 multi-copy crystals):

| condition | match rate | non-ref orient err |
|---|---|---|
| oracle (true orient) | **100.0%** | 0° (validates matcher tolerances) |
| identity (naive, R=I) | 1.5% | 165° |
| untrained (predict-floor) | 0.0% | 85° (≈ random SO(3)) |
| **trained** | **16.8%** | **68°** |

**The trained relative-orientation flow reconstructs 16.8% of held-out multi-copy packings
exactly, vs 0% at the floor and 1.5% naive** — the +27% loss is genuine reconstructed geometry.
Reproduce: `python scripts/eval_orient_matchrate.py --ckpt checkpoints/diag_orient_relative_noised.pt`.

**2b — capacity + steps headroom (`diag_orient_relative.py`, new capacity flags).** Is +27% the
ceiling, or under-trained? At `d_model=192`, 5 attn layers, 2000 steps (vs 128/4/800), non-ref
improves **5.35→3.56 (+33.4%)** (ref +68%). Real but diminishing headroom; it plateaus well above
zero, i.e. scaling alone does not close the gap. Checkpoint `checkpoints/diag_orient_relative_big.pt`.
Reproduce: `python scripts/diag_orient_relative.py --steps 2000 --d-model 192 --n-attn-layers 5`.

**2c — coset conditioning isolates the bottleneck (`scripts/diag_orient_coset.py`).** For a copy
rigidly generated by a crystallographic op, its relative rotation `R'_m` *is* that op's Cartesian
rotation part, so within a space group the non-reference relatives fall into a small discrete set
of cosets. `assign_cosets` clusters them into a per-SG codebook (707 cosets at 20° tol) and the
field is conditioned on the discrete per-molecule coset id (a standard CSP-style label — not the
continuous rotation). Paired comparison (128/800, identical corpus + split):

| run | non-ref orient loss | drop |
|---|---|---|
| `--no-coset` control | 5.37 → 3.91 | +27.1% (reproduces baseline) |
| **coset-conditioned** | 5.36 → **2.65** | **+50.6%** (ref +88.9%) |

**Conditioning on the discrete coset nearly doubles the learnable signal (3.91→2.65).** So most of
the +27% ceiling is the difficulty of *inferring* which symmetry op generated each copy from the
noised packing, **not** an inability of the SO(3) head to represent the rotation. A residual
remains (2.65 ≠ 0): the irreducible part is the free asymmetric-unit orientation `R_asym` plus the
multimodality of symmetric tops. Checkpoints `checkpoints/diag_orient_coset_{on,off}.pt`. Reproduce:
`python scripts/diag_orient_coset.py --steps 800 [--no-coset]`.

## What was diagnosed / fixed along the way

1. **Arbitrary global gauge → molecule-intrinsic frame.** Originally a species' body frame was
   set by whichever copy was parsed first *anywhere in the dataset*, so `R_m` was "rotation
   vs a molecule in another crystal" — conditionally random, a guaranteed floor. Fixed with
   `_canonical_frame` (gyration-tensor principal axes, element-weighted 3rd-moment sign,
   right-handed). Gauge-free → all tests + kept/skip set unchanged. Necessary, not sufficient.
2. **Early-Adam NaN at lr=1e-3.** The harder intrinsic-frame targets gave larger early
   gradients; Adam's tiny initial variance estimate amplified them (NaN ~step 17, shared
   trunk). Fixed: default lr → 3e-4.
3. **CSD/CCDC sourcing gotchas** (in `tasks/lessons.md`): Access Structures is consent-gated /
   not scriptable; `ccdc.hasattr` lies (`is_polymeric` is on `molecule` not `crystal`); COD
   uses `strictmin/strictmax` not `nel`; pymatgen "no structures" on CSD CIFs = special-position
   occupancy >1, fixed by `occupancy_tolerance`.

## Next steps (ranked)

1. **Reframe with the decomposition — now the lead.** The paper story is precise and
   well-supported: rigid-body **lattice+centroid** flow as the working contribution, plus a
   **characterized orientation result** — the SO(3) flow *learns the space-group-determined
   relative orientation* between symmetry copies (partial positive, robust to conditioning), and
   the residual floor is the **free asymmetric-unit orientation `R_asym`**, a gauge-arbitrary
   degree of freedom that is fundamentally unlearnable from composition+packing. Evidence chain:
   predict-zero floor → gauge fix necessary-not-sufficient → scaling no help → clean-packing
   rules out two-stage → relative-gauge isolates the learnable vs free parts.
2. **~~Optional strengthening of the partial positive.~~ DONE (2026-06-20).**
   (a) ✅ match metric — `scripts/eval_orient_matchrate.py`: **16.8%** exact StructureMatcher
   reconstruction of held-out multi-copy packings (vs 0% floor / 1.5% naive; oracle 100%);
   (b) ✅ capacity/steps — non-ref **3.91→3.56 (+33%)** at d_model 192 / 2000 steps (diminishing,
   plateaus above 0);
   (c) ✅ coset (≈ site-symmetry) conditioning — non-ref **3.91→2.65 (+51%)** with a `--no-coset`
   control reproducing +27%, proving the ceiling is **inference-limited, not representational**.
3. **~~Two-stage / cleaner conditioning.~~ RULED OUT (2026-06-18)** — clean-packing diagnostic;
   orientation does not learn even on the true packing because the dominant target part is free.
4. **Deprioritized:** min-over-site-symmetry CFM target (helps only the symmetric minority).

## Artifacts (local, not in git)

- `data/csd_mol/cif/` — 3500 CSD CIFs; `data/csd_mol/ds.pt` — factorized 1127-structure cache;
  `data/csd_mol/manifest.csv` — shareable refcode manifest.
- `checkpoints/csd_molcrystal*.pt` — trained checkpoints (incl. per-head pre/post val).
- `checkpoints/diag_orient_cleanpack.pt` — clean-packing diagnostic checkpoint (per-head pre/post + split).
- `checkpoints/diag_orient_relative{,_noised}.pt` — relative-orientation diagnostic (clean / noised conditioning; ref vs non-ref split).
- `checkpoints/diag_orient_relative_big.pt` — 2b capacity sweep (d_model 192, 2000 steps; non-ref +33%).
- `checkpoints/diag_orient_coset_{on,off}.pt` — 2c coset conditioning + paired control (non-ref +51% vs +27%).
