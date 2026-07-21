# Phase F2 findings — crystal-family-masked lattice + coset orientation

**Run:** RTX 3090 (vast 45486350), 2026-07-21. 800 steps, seed 0, `logvol-std 0.11`,
corpus `data/csd_mol/ds.pt` (1127 → 1095 multi-copy, 238 deployable SG-op cosets, 964/131 split).
Two models trained back-to-back: `logmetric6 + family-mask` and a `logmetric6, NO-mask` ablation
(isolates the mask from the informed volume prior + repr swap). Eval = `eval_e4_molcrystalflow.py`
(pymatgen StructureMatcher, best-of-k=10, ltol 0.3 / angle_tol 10°, stol sweep) + cell-volume RMAD
+ the new crystal-family angle-spike (fraction of cell angles within 2° of 90°).

## Result (coset ON = symmetry template supplied at sampling)

| axis | shape10 baseline (E4) | logmetric6, **no** mask | logmetric6 **+ family-mask** | real crystals |
|---|---|---|---|---|
| **cell-angle spike** (∠ within 2° of 90°) | ~3–12% | **16.3%** | **73.6%** | **74.0%** |
| orientation err (median best-of-k) | ~14–18° | 17.8° | **18.0°** | — |
| cell-vol RMAD | ~31% | 22.7% | 24.4% | 0 |
| **match@10, stol ≤ 1.0** | 0% | 0% | **0%** | — |
| match@10, stol = 1.2 | ~0.8% | 0.76% | 0.76% | — |

(coset OFF de-novo: angle-spike 73.9% with mask vs 15.4% without; orient err 91.8°; match 0% —
the mask acts on the lattice head independent of the coset, as designed.)

Training was healthy: lattice head val loss 3.39 → 0.196 (+94.2%), total +67.0%, NON-REF orient
drop +41.2% (coset still working after the repr swap; matches E2's coset advantage).

## Pre-registered gate verdict: **NOT met → F4**

The gate (tasks/phaseF_spec.md) promotes to F3 iff **(angle-spike rises toward 72% AND match@10
at stol ≤ 1.0 moves off 0)** OR **(RMAD < ~10% with the spike restored)**.

- angle-spike → 73.6%, dead-on the reference 74.0% — **emphatically met.**
- match@10 at stol ≤ 1.0 → still **0.00%** — **not met.**
- RMAD → 24.4%, not < 10% — **not met.**

Per pre-registration this is the **F4 (honest-null)** branch. No goalpost-moving.

## What F2 actually establishes (the good kind of null)

This is the first **fully symmetry-conditioned molecular flow**: orientation is handled by the
SG-op coset (E2: 92° → 18°) and the unit-cell *shape* is now handled by the crystal-family mask
(16% → 74%, ref-matched). The mask does **exactly** what it was built to do, cleanly isolated by
the ablation — the log-metric repr + informed prior alone reach only 16%; the mask alone closes
the remaining 58 points onto the reference. This is a genuine architectural first (every prior
molecular flow — MolCrystalFlow, MOFFlow, PackFlow — is space-group-free; symmetry-conditioned
lattice masking existed only for inorganic single-atom crystals).

**And yet exact match stays ~0%.** That refutes the strong form of the F0 hypothesis (crystal
family/cell-shape is THE match blocker): fixing crystal family *perfectly* does not unlock match.
Combined with the earlier oracle-volume ceiling (rescaling to true volume moved match by 0), the
remaining bottleneck is now sharply isolated and is **neither orientation, nor crystal family, nor
volume** — it is **fine molecular positioning / close-packing precision**. Evidence: the centroid
head barely trains (val +11.3%, 0.286 → 0.254) while lattice/orient collapse; StructureMatcher at
stol ≤ 1.0 needs sub-Å fractional-coordinate agreement that crude centroid placement + residual
24% volume detuning cannot reach.

Symmetry conditioning is thus **necessary but not sufficient** for exact match without a physical
close-packing finisher — which is precisely MCF's actual lever (their 3.86% RMAD / 6.8% match comes
from a hard-sphere rigid-press + informed prior, not the flow).

## Decision surface (Frank owns the F2→F3 call)

- **F4 (what the gate says): write the honest result now.** Contribution = first fully
  symmetry-conditioned molecular flow; both symmetry levers verified to work exactly as designed
  (92°→18°, 16%→74%=ref); symmetry conditioning necessary-but-not-sufficient; the residual blocker
  is close-packing precision. Absorbs E5. Legit MoML/workshop result (architectural first + sharp
  diagnostic + honest negative).

- **F3 (one more push, needs new code + GPU): a symmetry-preserving hard-sphere rigid-press
  finisher.** F2 *strengthens* this bet — a finisher applied now polishes structures already in the
  right basin (right family + right orientation), whereas before F2 it would have polished
  wrong-family/wrong-orientation cells. Upside: could push match off 0 toward MCF's ~6.8%. Cost: a
  non-trivial symmetry-preserving BFGS relaxer to implement first, then re-run. Genuinely uncertain.

Checkpoint `coset_fammask_s0.pt` and `e4_draws.pkl` (131 refs + best-of-k draws) preserved locally
(gitignored) for a possible F3; both are cheaply regenerable from committed code (~5 min retrain).
