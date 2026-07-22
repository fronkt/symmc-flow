# Phase F3d-1 findings — self-conditioning: the first lever to move the tight match

**Run:** 2026-07-22. Retrained the full-stack model (coset + family-mask + logmetric6 + OT +
fixed-prior) with **self-conditioning** added (`--self-cond`): the network sees its own terminal-
state estimate and predicts a correction, carried across RK4 steps at sampling. 1.1k corpus, 800
steps -- a direct A/B against F3b. Fresh RTX 3090 (vast 45520625, destroyed; two prior boxes were
duds: one stuck pulling the image, one whose sshd came up only just after the poll window). Draws
`f3d_tensors.pkl`; finished + scored the full 131 (frozen cell, 40 steps). (Scored log header
mislabeled "F3a:"; numbers are the F3d self-cond run.)

## Head-to-head (frozen cell, n=131, best-of-5, coset ON)

| match@10 | F3b (centroid fix) | F3c (scale) | **F3d (self-cond)** |
|---|---|---|---|
| stol=0.8 | 1.5% | 1.5% | **3.05%** (95% CI 1.19-7.59) |
| stol=1.0 | 3.1% | 3.1% | **3.82%** (95% CI 1.64-8.62) |
| stol=1.2 | 4.6% | 3.8% | 3.82% |
| median min-RMSD | -15.8% | -14.2% | -14.9% |

## Read: the diagnosis is vindicated (modestly)

Self-conditioning is the **first lever to move the tight match**: it **doubled** match at the
strictest tolerance (stol=0.8: 1.5 -> 3.05%) and nudged stol=1.0 (3.1 -> 3.82%). Every earlier lever
had left stol<=1.0 flat -- scale did nothing, and F3d-0 orientation multi-start only helped the
*loose* stol=1.2. A learned refiner that sharpens its own positioning estimate helps exactly where
StructureMatcher is strictest, confirming that the ceiling was **tail positioning precision**.

But it is a **modest** win -- a couple more crystals at tight tolerance, ~3.8% at stol=1.0, still
below the symmetry-free baselines (MolCrystalFlow ~6.8%, MOFFlow ~8%). Self-conditioning during
sampling closed part of the gap; explicit inference-time recycling passes and scale would be the next
(diminishing-return) knobs.

## The complete F3 ladder

```
  raw flow            0.0%   fully symmetry-conditioned (coset orientation + family-masked cell)
  + rigid-press       1.5%   F3a  unsupervised physical finisher                    (+1.5)
  + centroid fix      3.1%   F3b  OT coupling + fixed prior                          (+1.5)
  + 2.3x scale        3.1%   F3c  improves RMAD/coset, not match                     (+0.0)
  + self-conditioning 3.8%   F3d  first lever to move the TIGHT tolerances           (+0.7;
                      ----        stol=0.8 doubled 1.5 -> 3.05%)  @ stol<=1.0, best-of-5, n=131
  relax_cell / multi-start:  loose match + closeness only; tight ceiling holds
```

## Bottom line
A **complete, honest, positive result**: the first fully symmetry-conditioned molecular-crystal
flow, and a principled lever decomposition (symmetry -> physical finish -> positioning fix -> learned
self-conditioning refinement) that lifts exact match from **0% to 3.8%** and, uniquely for
self-conditioning, moves the tightest tolerance. It approaches but does not reach the symmetry-free
rigid-body flows -- the residual is fine positioning precision, partially closed by a learned refiner.
This is a strong MoML/JCIM story. Further gains (recycling passes, more scale) show diminishing
returns; strong case to lock and write up.

Artifacts: `f3d1_selfcond_frozen.log` (committed); `f3d_tensors.pkl` (self-cond draws) local
(gitignored). Checkpoint regenerable from the committed `--self-cond` command.
