# Data Availability Statement

**Manuscript:** Symmetric Molecular Crystal Flow Matching with Pair-Bias Attention and Rigid-Body Conformers
**Author:** Frank Cai, Purdue University

The CIF structures underlying this study are drawn from the Cambridge
Structural Database (CSD) and cannot be redistributed under the CSD licence.
The manifest of refcodes and filter protocol used to select the corpus
(3,501 candidate entries; formula, Z', R-factor, atom count, and space group
per entry) is openly available at
https://github.com/fronkt/symmc-flow/blob/master/data/csd_mol/manifest.csv
and lets anyone with a licensed CSD installation regenerate the exact CIF
corpus via `scripts/csd_export.py`. The factorized dataset (lattice, centroid,
and orientation tensors derived from the CIFs) is itself derived from the
licensed structures and is therefore available from the author on reasonable
request, subject to the requester holding a CSD licence.

The SymMC-Flow implementation, the export and factorization pipeline, and the
diagnostic and evaluation scripts are available at
https://github.com/fronkt/symmc-flow and archived at
https://doi.org/10.5281/zenodo.20822235.
