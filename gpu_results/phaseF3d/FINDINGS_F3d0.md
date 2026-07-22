# Phase F3d-0 findings — orientation multi-start: closer, but the tight-match ceiling holds

**Run:** 2026-07-22, local, on the existing F3c scale draws (`f3c_tensors.pkl`, no GPU). The finisher
was given 4 restarts per draw from small SO(3) perturbations (sigma 20 deg) of the ASU orientation,
all finished variants pooled into best-of-(5x4)=20 matching. This is the free pre-check for the
learned self-conditioning refinement lever: does *escaping the ~18 deg orientation basin* crack the
exact-match ceiling?

## Result (n=131, coset ON) vs F3c baseline

| axis | F3c (best-of-5, single finish) | F3d-0 (best-of-20, multi-start) |
|---|---|---|
| match@stol=1.0 | 3.1% | **3.1%** (flat) |
| match@stol=1.2 | 3.8% | **5.3%** |
| median min-RMSD to ref | -14.2% | **-19.3%** |
| RMAD / angle-spike | 17.2% / 77.6% | unchanged (frozen) |

## Read: partial positive, tight ceiling holds

Orientation multi-start pulls structures **meaningfully closer** (min-RMSD -14.2 -> -19.3%) and lifts
**loose** match (stol=1.2: 3.8 -> 5.3%), confirming orientation is part of the residual. But it does
**not** move the tight match (stol<=1.0 stays 3.1%) even with 4x the candidate pool. The tight-match
ceiling at ~3.1% has now held against **five** independent levers: rigid-press finisher, centroid
fix, 2.3x scale, cell relaxation (dead), and orientation multi-start.

Per the pre-registered F3d-0 logic ("if multi-start lifts match, orientation tail is the culprit and
a learned refinement will help; if not, the gap is elsewhere"), the tight-match miss says the
stol<=1.0 gap is **not purely orientation** -- it also involves centroid/fine-positioning precision
that a random-orientation + physical relaxation cannot supply. A learned self-conditioning refiner
(which corrects centroid too) is therefore a **longer shot** than the pre-check hoped, not a clear
green light.

## Free win to fold in regardless
Multi-start is an inference-time-only improvement (no retrain): it lifts stol=1.2 match to 5.3% and
min-RMSD to -19.3%. Worth reporting as the finisher's best configuration.

## Decision
The 3.1% tight-match ceiling is robust across five attacks -> strong argument to **lock** the honest
positive-with-ceiling result. Self-conditioning refinement remains buildable (Frank's named lever)
but the pre-check makes it a longer shot; his call whether to spend the build + GPU or lock now.
