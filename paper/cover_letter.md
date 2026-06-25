# Cover letter

Frank Cai
Purdue University, West Lafayette, IN, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

To the Editors, *Digital Discovery* (Editor-in-Chief: Prof. Alán Aspuru-Guzik)

Dear Editors,

I am pleased to submit the manuscript **"What an SO(3) orientation flow can and cannot
learn in molecular-crystal structure prediction"** for consideration as a Paper in
*Digital Discovery*.

Rigid-body factorization — freezing each molecule's conformer and predicting only a
lattice, a set of fractional centroids on the torus, and a per-molecule orientation on
SO(3) — is an appealing route to molecular-crystal structure prediction, because it
reduces the problem to a low-dimensional target amenable to fast flow-matching samplers.
The unexamined premise is that the orientation field is actually learnable on real
molecular crystals. This work tests that premise directly and reports what is, and is
not, learnable.

My contributions, and why they should interest the *Digital Discovery* readership:

- **A space-group-conditioned rigid-body flow (SymMC-Flow)** on
  lattice × T³ × SO(3), used as a controlled diagnostic on 1,127 crystals exported from
  the Cambridge Structural Database. The lattice and centroid flows learn normally; the
  absolute per-molecule orientation flow does not, remaining at its predict-zero floor.

- **A mechanistic explanation, not just a negative result.** I trace the failure to a
  decomposition of the orientation target into a space-group-determined relative rotation
  between symmetry copies and the asymmetric unit's free, gauge-arbitrary orientation.
  The relative part is learnable; the free part is not learnable from composition and
  packing here. Re-gauging to cancel the free part lifts the held-out non-reference
  orientation loss by 27% (versus ~0% for the absolute target) and, in an
  orientation-isolated evaluation, reconstructs 13.7% (three-seed mean) of held-out
  multi-copy packings exactly under StructureMatcher, against 0% for the predict-floor
  baseline.

- **A representability ceiling that is informative for generator design.** Conditioning
  on a discrete space-group coset identity raises the learnable signal to roughly 48%
  (three-seed mean), showing the residual ceiling is set by the difficulty of *inferring*
  the symmetry operation from packing rather than by the network's capacity to represent
  the rotation. This points to a concrete design rule: rigid-body generators should
  condition on crystal symmetry for the relative orientation.

- **Honest scope.** End-to-end joint generation, which must also sample the lattice and
  centroids, does not yet reconstruct packings exactly, so the result is framed as a
  characterization rather than a finished generator. An all-atom control on the identical
  corpus and split (the representation used by recent all-atom generators) also yields 0%
  exact reconstruction, so the joint-generation result reflects corpus scale and compute
  budget, not a defect specific to rigid-body factorization.

This is a data-driven, reproducible study of what a class of molecular-crystal generators
can and cannot learn, with the diagnostic conclusions shown to be independent of corpus
size and validated under a leakage-controlled species-grouped split. Its value lies in
mapping the representational limits of an increasingly popular modelling choice rather
than in a state-of-the-art number, which I believe fits *Digital Discovery*'s emphasis on
rigorous, reproducible methodology for materials discovery.

I confirm that this manuscript is original, has not been published previously, and is not
under consideration for publication elsewhere. It has a single author with no conflicts of
interest to declare. All source code is openly available
(https://github.com/fronkt/symmc-flow; archived at Zenodo,
https://doi.org/10.5281/zenodo.20822235), so every reported number regenerates from the
deposited artifacts.

Thank you for considering this work.

Sincerely,

Frank Cai
Purdue University
