# JCIM submission package — notes

**Venue.** Journal of Chemical Information and Modeling (ACS), CODEN `jcisd8`.
Archival journal Article. No page limit. Rolling submission.
Guidelines source: https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8
(last updated 2026-05-21).

**This package.**
- `main.tex` — the manuscript, ACS `achemso` class (`journal=jcisd8,manuscript=article`).
- `si.tex` — the Supporting Information (`achemso` `manuscript=suppinfo`).
- `references.bib` — 26 entries (19 cited; the 3 competitor refs are the same verified
  set as MoML: MolCrystalFlow 2602.16020, NextCrystal 2602.17176, SO(3)-avg 2507.09785).
- `figures/fig3_ladder.pdf` — the exact-match ladder figure (also the TOC graphic).

**Built + verified locally (MiKTeX 25.12), 2026-07-22.** Compiles clean with
`pdflatex → bibtex → pdflatex → pdflatex`:
- `main.pdf` 19 pp (ACS double-spaced manuscript format); `si.pdf` 6 pp.
- 0 undefined references/citations, 0 errors, 0 overfull hboxes > 15 pt, 19/19 bibitems resolve.

**Formatting compliance (against the jcisd8 guidelines).**
- achemso class = the ACS format; bibliography style is owned by achemso (do NOT add
  `\bibliographystyle` or load `natbib`).
- Abstract 224 words (guideline 150–250).
- Graphical Table of Contents entry present (`\begin{tocentry}`, renders on the last page in
  manuscript mode, ACS-standard).
- **ORCID intentionally omitted from the manuscript text** (jcisd8 rule: ORCID is added
  automatically on acceptance and must not appear in the body).
- **Data and Software Availability** statement present (mandatory for JCIM).
- Competing-interest statement present ("The author declares no competing financial interest.").
- References complete with titles (guideline requires titles).
- Extended methods (crystal-family mask, rigid-press finisher, self-conditioning, per-seed
  numbers, R_asym probe) are in the Supporting Information, per ACS structure.

## TODO before submitting (human-gated)
- [ ] **Frank reviews the expanded prose** (Introduction, Methods §orientation-decomposition
      derivation, Limitations) for voice/accuracy — this was a scaffold he owns.
- [ ] Decide whether `tab:bench` (the MolCrystalFlow comparison framing) stays as written.
- [ ] Optional: replace the TOC graphic with a purpose-built conceptual schematic
      (currently reuses `fig3_ladder.pdf`).
- [ ] Bump the Zenodo DOI (`cai2026symmcflow`) to the strengthened code before submission.
- [ ] Upload via the ACS Paragon system: `main.tex`+`references.bib`+`figures/` as the
      manuscript, `si.tex` (built to PDF) as Supporting Information; provide the TOC graphic
      and cover letter in the portal.
- [ ] Confirm the competing-interest / data-availability disclosures in the submission form
      match the manuscript statements.

## Numbers provenance
Phase B (`e035b8d`), E1 (`581723c`), E3 (`1224870`), E2 scale (`ce0e5d0`,
`gpu_results/phaseE_scale`), E4 bench (`gpu_results/phaseE4`), Phase F ladder
(`gpu_results/FINDINGS_F_FINAL.md`, `gpu_results/phaseF3{a..f}/`).
