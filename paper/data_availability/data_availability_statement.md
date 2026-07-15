# Data Availability Statement

**Manuscript ID:** DD-ART-07-2026-000495
**Title:** What an SO(3) orientation flow can and cannot learn in molecular-crystal structure prediction
**Author:** Frank Cai, Purdue University, West Lafayette, IN, USA

## Data availability

The molecular-crystal structures analysed in this work were obtained from the
Cambridge Structural Database (CSD) and are governed by the CSD licence. The
underlying CIF structures, and the factorised tensors derived from them (lattice,
centroid, and orientation representations), are derived from licence-restricted
data and therefore cannot be redistributed under the terms of the CSD licence.
This is the sole reason these primary data are not deposited in an open repository.

To enable full reproduction by any holder of a CSD licence, the manifest of the
3,500 CSD refcodes used to construct the corpus, together with the per-entry
selection metadata (chemical formula, Z′, R-factor, atom count, and space group),
is openly available as `data/csd_mol/manifest.csv` in the archived code repository
(see Code availability). The export and factorisation pipeline that regenerates the
exact CIF corpus and the derived tensors from a licensed CSD installation is
provided as `scripts/csd_export.py` and `scripts/factorize_cifs.py` in the same
repository. A fixed random seed reproduces the identical 964/131 train/validation
split used throughout the study. No other primary experimental datasets were
generated in this work; additional data supporting the findings of this article
are provided in the Supplementary Information.

## Code availability

The SymMC-Flow implementation and the complete data-preparation, training,
diagnostic, and evaluation pipeline — including the CSD refcode manifest and the
export/factorisation scripts referenced above — are available at
https://github.com/fronkt/symmc-flow and are permanently archived at Zenodo. The
software version underlying the results reported in this article is v1.0.3
(DOI: https://doi.org/10.5281/zenodo.21384130); the concept DOI
https://doi.org/10.5281/zenodo.20822234 always resolves to the latest version.

**Formal reference (added to the reference list):**
F. Cai, *fronkt/symmc-flow: SymMC-Flow orientation flow matching and CSD data
pipeline for molecular-crystal structure prediction (v1.0.3)*, Zenodo, 2026,
DOI: 10.5281/zenodo.21384130.
