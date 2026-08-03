# ACS Paragon Plus — paste-ready fields (ACS Omega, direct submission)

Route is a **direct new submission**, not a transfer — see SUBMISSION_NOTES.md.
Portal: https://acs.manuscriptcentral.com/acs-omega → Author Dashboard →
Start New Submission. Manuscript type: **Research Article**.

## WARNING: the portal's AI extraction tool mangles the abstract

On the 2026-08-03 attempt, ACS's "AI extraction tool" prepopulated the Abstract
field from the DOCX and **silently dropped every equation and every number** —
SO(3), the decomposition, 41.1%, 27.5%, ~48%, 47.7%, 93.5°→13.4°, O(3), 0%,
6.9%, best-of-10, stol 1.0, ~8%, 10.7%. What remained was prose making no
quantitative claim at all. **Always paste the plain-text abstract below over
whatever the tool prepopulates, then re-read the field after saving** in case
extraction re-runs.

## File uploads

| File | Designation |
|---|---|
| `Cai_SymMCFlow_ACSOmega_manuscript.docx` **or** `symmc-flow_ACSOmega_manuscript.zip` | Manuscript File |
| `cover_letter.pdf` | Cover Letter |
| `si.pdf` | Supporting Information for Publication |
| `figures/fig_toc.png` | **TOC / Abstract Graphic — REQUIRED by ACS Omega** |

`figures/fig3_ladder.pdf` does **not** need uploading separately: Figure 1 is
embedded in the manuscript file. If uploaded anyway it is a figure file, *not*
the TOC graphic — do not let it occupy the graphic slot.

## Title

Fully Symmetry-Conditioned Rigid-Body Flow Matching for Molecular-Crystal Structure Prediction

## Abstract (plain text — 216 words, paste over the prepopulated field)

Rigid-body flow matching reduces molecular-crystal structure prediction to a lattice, fractional centroids, and per-molecule orientations on SO(3), but current rigid-body molecular-crystal generators leave crystallographic symmetry unconditioned, and symmetry-conditioned generation has been demonstrated only for inorganic atomic sites. We show that the per-molecule orientation target factorizes as R_m = rot(g_m) R_asym into a space-group-determined relative rotation, which a flow learns, and a gauge-free asymmetric-unit pose, which it cannot regress from packing. We turn this into a deployable method: a leak-free coset label - the generating space-group operation, recoverable from a symmetry template at sampling time - conditions the orientation flow. On Cambridge Structural Database crystals, deployable coset conditioning lifts the held-out non-reference orientation loss by 41.1% (versus 27.5% for a paired control), two-thirds of the way to a leaky-codebook oracle (~48%); an SO(3)-averaged objective closes the gap (47.7%). Supplying the coset at generation collapses the end-to-end orientation error sevenfold (93.5° to 13.4°), and the advantage widens with data. Extending symmetry conditioning to the lattice - a crystal-family mask on an O(3)-invariant log-metric parametrization - gives, to our knowledge, the first fully symmetry-conditioned molecular-crystal flow. Paired with an unsupervised symmetry-preserving packing finisher, a fully-ablated stack of levers lifts held-out exact match from 0% to 6.9% (strict best-of-10, StructureMatcher stol 1.0), at statistical parity with the symmetry-free MolCrystalFlow (~8%), reaching 10.7% with orientation test-time augmentation. We report the contribution of each lever.

## Keywords

crystal structure prediction; flow matching; molecular crystals; space-group symmetry; generative modeling; rigid-body

## Author

Frank Cai — corresponding and sole author
Purdue University, West Lafayette, Indiana 47907, United States
frankyc11223@gmail.com
ORCID 0009-0003-0041-1459

## Declarations (must match the manuscript text exactly)

- **Competing interests:** "The author declares no competing financial interest."
- **Funding:** none. "No external funding supported this work."
- **AI use:** disclosed in the Acknowledgments — Claude Opus 4.8 (Anthropic) used
  for language editing of the text; the author reviewed and edited all output and
  takes full responsibility for the content.
- **Data availability:** code, CSD refcode manifest, and the export/factorization
  pipeline at https://github.com/fronkt/symmc-flow, archived at Zenodo
  https://doi.org/10.5281/zenodo.21500734. CSD is licence-restricted; the
  structures themselves are not redistributed.

## Open decisions at the portal

- **Licence:** CC BY-NC-ND (default) vs CC BY (upgrade). Cost differs. Author's call.
- **APC:** US$1,935, charged on acceptance. Confirm Purdue's ACS read-and-publish
  eligibility **before** submitting — it is determined at acceptance.
- **Special issue:** none.
- **Preferred / non-preferred reviewers:** none drafted. Optional at ACS Omega.
