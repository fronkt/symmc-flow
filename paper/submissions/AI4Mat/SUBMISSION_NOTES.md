# AI4Mat @ NeurIPS 2026 submission — notes

**Venue.** AI4Mat (AI for Accelerated Materials Design) workshop @ NeurIPS 2026.
**Non-archival** workshop paper, **~Aug 30 2026** deadline (verify on the CFP).
Simultaneous submission is fine — this co-submits with MoML (Sep 1) and does not
conflict with the JCIM journal version (Phase E/F). Part of the "submit all 3"
NeurIPS-workshops plan (symmc-flow + fe-sma + pxrd-diff); no per-author cap.

**This package.** `main.tex` + `references.bib` are a **content-identical copy of
the MoML short paper** (`paper/submissions/MoML/`), kept in a separate directory so
each venue's formatting can diverge without cross-contamination. Self-contained
(single-column 10pt article), one results table + one pgfplots bar chart in the
main text, full methods / per-seed numbers / tolerance sweep in the non-counting
appendix.

**The story (strengthened, Phase F).** Same headline as the JCIM/MoML strengthened
version — the **first fully symmetry-conditioned molecular-crystal flow**
(orientation coset × crystal-family lattice mask) + an unsupervised
symmetry-preserving finisher, whose fully-ablated lever stack lifts held-out exact
match **0% → 6.9%** (strict best-of-10, StructureMatcher stol=1.0) = statistical
parity with MolCrystalFlow (~8%, 95% CI 3.66–12.54), and **10.7%** with
orientation-TTA (3× finishing budget). Full ladder in `tab:ladder`
(`sec:ladder`); numbers from `gpu_results/FINDINGS_F_FINAL.md` +
`gpu_results/phaseF3{a..f}/`.

## TODO before submitting (human-gated)
- [ ] **Convert to the official AI4Mat/NeurIPS style file** for the 2026 edition
      (the CFP will link the template; this draft uses a neutral `article` base so
      it compiles standalone in the meantime).
- [ ] **Compile on Overleaf** (no local TeX here). Verify the page limit for the
      chosen AI4Mat track (typically ~4 pages excl. references); trim prose if over.
- [ ] Confirm the pgfplots bar chart (Fig. 1) renders under the NeurIPS template;
      Table 1 (`tab:gate`) and Table 2 (`tab:ladder`) already carry the numbers, so
      the figure can be dropped if space-constrained.
- [ ] **Optional:** include the exact-match lever-ladder figure
      (`paper/figures/fig3_ladder.pdf`) if the page budget allows — it visualizes
      the 0%→parity ladder from `tab:ladder`.
- [ ] Final citation-accuracy pass (all keys resolve; the competitor arXiv IDs are
      the same verified set as MoML).
- [ ] Sync the Zenodo DOI (`cai2026symmcflow`) if a new version is minted for the
      strengthened code.
- [ ] **Frank writes/owns the final text**; this draft is for review only.

## Keep in sync with MoML
`main.tex` here mirrors `paper/submissions/MoML/main.tex`. If the MoML text
changes, re-copy (or diff) so the two workshop versions do not drift apart before
their formatting is intentionally diverged.
