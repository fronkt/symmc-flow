# Cover letter — Journal of Chemical Information and Modeling

Frank Cai
Purdue University, West Lafayette, IN, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

To the Editors, *Journal of Chemical Information and Modeling*

Dear Editors,

I am pleased to submit the manuscript **"Fully Symmetry-Conditioned Rigid-Body
Flow Matching for Molecular-Crystal Structure Prediction"** for consideration as
an Article in the *Journal of Chemical Information and Modeling*.

Rigid-body flow matching has become an attractive route to molecular-crystal
structure prediction: freezing each molecule's conformer reduces the problem to a
lattice, a set of fractional centroids, and a per-molecule orientation on SO(3),
a low-dimensional target that fast flow-matching samplers solve in tens of steps.
Yet the current generation of rigid-body molecular-crystal generators leaves
crystallographic symmetry unconditioned on the orientation degree of freedom, and
symmetry- or Wyckoff-conditioned generation has to date been demonstrated only for
inorganic atomic sites. This manuscript closes that gap and makes the conditioning
deployable.

The contributions, and why they fit the *JCIM* readership:

- **A mechanism, then a method.** I show that the per-molecule orientation target
  factorizes into a space-group-determined relative rotation, which a flow learns,
  and a gauge-free asymmetric-unit pose, which it cannot regress from packing. This
  explains why an unconditioned orientation flow stalls, and it identifies exactly
  what to condition on.

- **Deployable symmetry-coset conditioning.** A leak-free coset label — the
  generating space-group operation, recovered from centroids against a symmetry
  template at sampling time, never from the observed rotation — conditions the
  orientation flow. It recovers two-thirds of an oracle-codebook benefit
  (+41.1% held-out non-reference orientation loss vs. +27.5% for a paired control),
  and an SO(3)-averaged objective closes the rest (+47.7%). Supplying the coset at
  generation collapses the end-to-end median orientation error sevenfold
  (93.5° → 13.4°), and the advantage widens with data.

- **The first fully symmetry-conditioned molecular-crystal flow.** Extending the
  conditioning to the lattice — a crystal-family mask on an O(3)-invariant
  log-metric cell parametrization — composes with the orientation coset to condition
  every symmetry-constrained degree of freedom. Paired with an unsupervised,
  symmetry-preserving packing finisher and a fully ablated stack of levers, it lifts
  held-out exact match from 0% to 6.9% (strict best-of-10, StructureMatcher
  stol 1.0) — statistical parity with the symmetry-free MolCrystalFlow (~8%), from a
  flow that matched nothing — and 10.7% with orientation test-time augmentation.

- **Honest scope.** The comparison to the dedicated rigid-body flows is
  comparable-task rather than a strict head-to-head (a smaller CSD-derived corpus at
  diagnostic scale), the strict-parity figure carries a wide confidence interval,
  and template-free deployment is not yet at the template ceiling. The paper's value
  is a concrete, reproducible recipe for symmetry-conditioning rigid-body
  generators, with each lever's contribution quantified, rather than a
  state-of-the-art number.

This is a methodological contribution to chemical structure modeling: it gives
rigid-body molecular-crystal generators a concrete design rule — condition
orientation on the generating coset and the lattice on the crystal family, sample
only the free pose, and finish with a symmetry-preserving relaxation — which I
believe fits *JCIM*'s emphasis on methods and models for chemical information.

I confirm that this manuscript is original, has not been published previously, and
is not under consideration for publication elsewhere. It has a single author with
no competing financial interest to declare. The source code, the CSD refcode
manifest, and the export/factorization pipeline are openly available
(https://github.com/fronkt/symmc-flow; archived at Zenodo,
https://doi.org/10.5281/zenodo.21500734), so every reported number regenerates from
the deposited artifacts. The Cambridge Structural Database is licence-restricted;
the underlying structures are not redistributed, but the deposited refcode manifest
reproduces the corpus from a licensed CSD installation.

Thank you for considering this work.

Sincerely,

Frank Cai
Purdue University
