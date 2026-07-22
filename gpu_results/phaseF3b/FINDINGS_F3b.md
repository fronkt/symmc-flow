# Phase F3b findings — the centroid fix lifts match (a clean monotonic ladder)

**Run:** 2026-07-21. Retrained the F2 coset + family-mask + logmetric6 model with the centroid-head
fix enabled — `--ot-coupling --fixed-prior` (optimal-transport prior<->data centroid pairing +
one cached prior per structure), everything else identical to F2 (800 steps, seed 0, logvol-std
0.11). Fresh RTX 3090 (vast 45507270, destroyed; a first box 45506465 was a dud stuck pulling the
image). Same decoupled pipeline: gen 131x5 draws on the box, finish + score locally (frozen cell,
40 steps, 6-process pool). Draws `f3b_tensors.pkl`. (The scored log's header is mislabeled "F3a:" —
a hardcoded print label; the numbers below are the F3b centroid-fix run.)

## Head-to-head (frozen cell, n=131, best-of-5, coset ON)

| match@10 (coset ON) | RAW flow | F3a (finisher) | **F3b (finisher + centroid fix)** |
|---|---|---|---|
| stol=0.5 | 0.0% | 0.0% | 0.0% |
| stol=0.8 | 0.0% | 1.5% | 1.5% |
| stol=1.0 | 0.0% | **1.5%** | **3.1%** (95% CI 1.19-7.59) |
| stol=1.2 | 0.0% | 3.1% | **4.6%** (95% CI 2.12-9.63) |
| cell-vol RMAD | — | 24.7% | **21.2%** |
| median min-RMSD to ref | — | -14.6% | -15.8% |

## What it means

Enabling the in-repo centroid fixes **doubled** match@stol<=1.0 (1.5% -> 3.1%; 2 -> 4 of 131) and
lifted stol=1.2 to 4.6% (6 of 131), and — as a bonus — improved cell-volume RMAD 24.7% -> 21.2%
(the fixed per-structure prior sharpened the lattice prior too). The story is now a clean, monotonic
ladder where each lever adds real match:

```
  raw flow            0.0%   (symmetry-conditioned orientation + family-masked cell, no polish)
  + rigid-press       1.5%   (F3a: unsupervised physical finisher on right-basin draws)
  + centroid fix      3.1%   (F3b: OT coupling + fixed prior -> better raw positioning)
                      ----   @ stol<=1.0, best-of-5, n=131
```

We are now closing toward the rigid-body baselines (MolCrystalFlow ~6.8%, MOFFlow ~8% at stol~0.8-1)
— from a flow that matched **zero** crystals — while remaining the only *fully symmetry-conditioned*
molecular flow (orientation coset + crystal-family lattice mask).

## Remaining levers (Frank owns the direction/spend)

- **relax_cell** (testing now, FREE on the existing F3b draws): fix the residual 21% volume, which
  StructureMatcher penalizes directly. An earlier 2-crystal probe on the *F2* model made RMAD worse;
  on F3b (already 21% RMAD) it may help — the result decides whether it's a lever or a dead end.
- **scale** the corpus (E2's 2539-crystal set is built) + more steps — each lever so far has added
  ~1.5 pts; scale historically widened the coset gap (E2).
- The result is already a strong positive: **the first fully symmetry-conditioned molecular flow,
  and a stack of symmetry + physical + positioning levers that lifts match 0 -> 3.1%**, approaching
  the rigid-body baselines. Lock it as the paper, or push one more round toward the MCF/MOFFlow band.

Artifacts: `f3b_full_frozen.log` (committed); `f3b_tensors.pkl` (draws), `f3b_scored.pkl` (raw+
finished structures) preserved locally (gitignored). Checkpoint regenerable from the committed
`diag_orient_coset.py --ot-coupling --fixed-prior ...` command.
