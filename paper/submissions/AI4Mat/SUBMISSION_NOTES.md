# AI4Mat @ NeurIPS 2026 submission — notes

**Venue.** AI4Mat (AI for Accelerated Materials Design) workshop @ NeurIPS 2026.
**Non-archival** workshop paper, **~Aug 30 2026** deadline (verify on the CFP).
Simultaneous submission is fine — this co-submits with MoML (Sep 1) and does not
conflict with the JCIM journal version (Phase E/F). Part of the "submit all 3"
NeurIPS-workshops plan (symmc-flow + fe-sma + pxrd-diff); no per-author cap.

**This package.** `main.tex` (NeurIPS style) + `references.bib` + `neurips_2024.sty`.
The prose/results are the **same content as the MoML short paper**
(`paper/submissions/MoML/`), reformatted into the NeurIPS style AI4Mat requires.
One results table + one pgfplots bar chart + the lever-ladder table in the main
text; full methods / per-seed numbers / tolerance sweep in the appendix (refs +
appendix in the same PDF, per AI4Mat).

**The story (strengthened, Phase F).** Same headline as the JCIM/MoML strengthened
version — the **first fully symmetry-conditioned molecular-crystal flow**
(orientation coset × crystal-family lattice mask) + an unsupervised
symmetry-preserving finisher, whose fully-ablated lever stack lifts held-out exact
match **0% → 6.9%** (strict best-of-10, StructureMatcher stol=1.0) = statistical
parity with MolCrystalFlow (~8%, 95% CI 3.66–12.54), and **10.7%** with
orientation-TTA (3× finishing budget). Full ladder in `tab:ladder`
(`sec:ladder`); numbers from `gpu_results/FINDINGS_F_FINAL.md` +
`gpu_results/phaseF3{a..f}/`.

## Status — converted to NeurIPS style + compiled locally (2026-07-22)
- [x] **Converted to the NeurIPS style** (`neurips_2024.sty`, bundled). Preamble:
      `\usepackage[preprint, nonatbib]{neurips_2024}` + numbered `natbib`
      (`unsrtnat`) + hyperref/pgfplots. Compiles clean (MiKTeX): **6 pp**, 20/20
      bibitems, 0 undefined refs/cites, 0 bad overfull. Numbered citations render.

## TODO before submitting (human-gated)
- [ ] **Swap `neurips_2024.sty` for the official AI4Mat / NeurIPS-2026 style file**
      when the CFP posts it. The layout is identical year-to-year; only the first-page
      footer text differs. `\documentclass` + author block stay the same.
- [ ] **Anonymity:** the draft uses `[preprint]` (author shown, no line numbers) to
      match the non-anonymous MoML co-submission. **If AI4Mat 2026 review is
      double-blind, remove `preprint`** → anonymized submission mode with line numbers,
      and strip the author block.
- [ ] **Page limit:** currently 6 pp (incl. refs + appendix in one PDF). AI4Mat
      recent editions accept short *and* full-length (up to ~9 pp) papers — confirm the
      2026 track limit on the CFP; the pgfplots bar chart (Fig. 1) can be dropped if
      over (Table 1 carries the same numbers).
- [ ] Sync the Zenodo DOI (`cai2026symmcflow`) if a new version is minted.
- [ ] **Frank writes/owns the final text**; this draft is for review only.

## Keep in sync with MoML
`main.tex` here mirrors `paper/submissions/MoML/main.tex`. If the MoML text
changes, re-copy (or diff) so the two workshop versions do not drift apart before
their formatting is intentionally diverged.
