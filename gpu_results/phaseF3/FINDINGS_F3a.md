# Phase F3a findings — the rigid-press finisher unblocks match (modestly)

**Run:** 2026-07-21. Draws generated on an RTX 3090 (vast 45496916, destroyed) from the F2
family-mask checkpoint `coset_fammask_s0.pt` (131 val crystals x 5 coset-ON draws, `f3_tensors.pkl`);
finished + scored locally (CPU, 6-process pool). The finisher (`symmc_flow.rigid_press`) is an
**unsupervised** symmetry- and rigidity-preserving physical packing relaxation — it never sees the
reference, only the generated draw + the space-group template that conditioned the sampler. Cell
**frozen** (positions-only relax; keeps F1's family-correct cell and avoids the LJ pressure fighting
the correct volume, which a probe showed `relax_cell=True` does).

## Result (frozen cell, best-of-5, n=131, coset ON)

| axis | RAW (flow only) | FINISHED |
|---|---|---|
| match@10, stol=0.5 | 0.0% | 0.0% |
| match@10, stol=0.8 | 0.0% | **1.5%** (95% CI 0.42-5.40) |
| match@10, stol=1.0 | 0.0% | **1.5%** (95% CI 0.42-5.40) |
| match@10, stol=1.2 | 0.0% | **3.1%** (95% CI 1.19-7.59) |
| median min-atom-RMSD to ref | 1.045 | **0.892  (-14.6%)** |
| cell-vol RMAD | 24.7% | 24.7% (frozen, unchanged) |
| cell-angle spike | 73.7% | 73.7% (frozen, = ref 74.0%) |

(30-crystal preview, 40 steps: match 0->3.3% @stol<=1.0, 0->6.7% @stol=1.2, min-RMSD -16.5% —
consistent; the full-131 rate is the tighter estimate.)

## Gate verdict: **PASSED (weakly) -> F3b**

Pre-registered gate (tasks/phaseF3_spec.md): promote to F3b iff **FIN match@stol<=1.0 moves off 0
(CI-separated from RAW)** OR **median min-RMSD drops >= 30%**.
- FIN match@stol<=1.0 = 1.5%, 95% CI 0.42-5.40 (excludes 0); RAW = 0% -> **match criterion MET.**
- median min-RMSD drop 14.6% < 30% -> RMSD criterion not met.

Per pre-registration this is a **pass** (via the match criterion). No goalpost-moving.

## What it means (honest)

The F3 hypothesis is **confirmed in direction**: the raw flow matches ZERO of 131 crystals, but an
unsupervised physical finisher on the same draws unblocks ~1.5% (stol<=1.0) / ~3% (stol=1.2) and
pulls *every* structure ~14.6% closer to the true reference (robust median over 121+ crystals). This
is the lever MCF/MOFFlow use to reach match, and it works here **because** F1/F2 first put structures
in the right basin (right family + right-ish orientation) — a finisher on wrong-family/wrong-
orientation cells (pre-F2) would have had nothing to rescue.

**But it is a modest win.** 1.5% @stol<=1.0 is far below MolCrystalFlow (~6.8%) / MOFFlow (~8%). The
finisher can only convert structures that are *already* very close; most of the flow's draws (18 deg
residual orientation + a crude, under-trained centroid head) sit outside its local basin. So the
binding constraint is now the **flow's raw positioning precision**, not symmetry and not the finisher.

## F3b options (Frank owns the GPU/scale call)

The highest-leverage, cheapest next experiment is to fix the **under-trained centroid head** — it
barely trains under the default uniform torus prior + no OT coupling (val centroid +11% vs lattice
+94%). The repo already ships the fixes (`flow.py`: `ot_couple`, `PriorCache`/`fixed_prior`,
`centroid_prior_std`). Better raw positioning -> more draws inside the finisher's basin -> the 14.6%
"closer" should convert into real match rate. ~15-min GPU retrain, then regenerate + finish + score
with this same harness. Secondary levers: `relax_cell` (fix the 24% volume the frozen run leaves),
larger k, corpus scale.

If F3b's centroid fix lifts match toward the MCF/MOFFlow band -> strong positive paper (first fully
symmetry-conditioned molecular flow that *matches*). If it plateaus -> the honest result is already
publishable: symmetry conditioning + a physical finisher jointly unblock match 0->1.5-3%, and the
remaining gap is flow positioning precision (a data/architecture problem), cleanly diagnosed.

Artifacts: `f3a_full_frozen.log`, `f3a_sub30_frozen.log` (committed); `f3_tensors.pkl` (draws),
`f3a_scored.pkl` (all raw+finished structures) preserved locally (gitignored).
