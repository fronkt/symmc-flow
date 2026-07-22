# MoML 2026 submission — notes

**Venue.** Molecular Machine Learning Conference (MoML) 2026 @ MIT.
Short paper, **2–4 pages** (references + appendix do **not** count).
**Non-archival** (no proceedings, author keeps copyright), **non-anonymous**,
**simultaneous submission explicitly permitted** — so this does not conflict with
the planned JCIM journal version (Phase E). Optional Overleaf style file offered
(not mandatory). **Deadline: Sep 1 2026, 11:59 PM AOE**; decisions Sep 8.
Accepted short papers present as posters (~Oct 2026).
Source: https://www.moml.mit.edu/submit

**This package.** `main.tex` (self-contained, single-column 10pt), `references.bib`
(26 entries; the 3 new competitor refs are verified real — MolCrystalFlow
arXiv:2602.16020, NextCrystal arXiv:2602.17176, SO(3)-avg FM arXiv:2507.09785).
One results figure (pgfplots bar chart) + one table in main text; full methods,
per-seed numbers, and tolerance sweep in the (non-counting) appendix.

**The reframe.** Diagnostic → method. The old paper's own "future work" (make the
coset deployable in a sampler) is now the headline. Numbers from commit `e035b8d`
/ `gpu_results/`:
- deployable coset conditioning +41.1% vs +27.5% no-coset control (2/3 to the
  ~48% leaky-codebook ceiling);
- + SO(3)-averaged objective → +47.7% ≈ ceiling;
- packing-only predictor 39.5% top-1 (4× majority) but predicted-coset
  reconstruction collapses to 64.8° → honest template-free gap;
- end-to-end 0% (lattice 0.44 / centroid 0.35 dominate; orient 13.4°) → scopes
  the contribution to the orientation mechanism.

## TODO before submitting (human-gated)
- [ ] **Compile on Overleaf** (no local TeX here). Verify **main text ≤ 4 pages**;
      trim prose if over.
- [ ] Confirm the pgfplots bar chart (Fig. 1) renders; if pgfplots is unavailable
      in the chosen template, Table 1 already carries the same numbers — drop the
      figure.
- [ ] **Optional:** add a concept schematic (R_asym decomposition + how the coset
      label feeds the flow) IF the compiled length leaves room. Deferred here to
      avoid blowing the 4-page limit before a real page count is known.
- [ ] Final citation-accuracy pass (all keys resolve; 3 new arXiv IDs verified).
- [ ] Bump Zenodo version + DOI to match the strengthened code, update
      `cai2026symmcflow` if a new DOI is minted.
- [ ] **Frank writes/owns the final text**; this draft is for review only.

## Also submit to AI4Mat (NeurIPS workshop, Aug 30)
Same paper, dual non-archival workshop submission is fine. AI4Mat may want the
NeurIPS workshop style; reuse this content.
