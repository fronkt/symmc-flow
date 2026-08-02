# JCIM submission package — notes

> **SUPERSEDED — 2026-08-02.** Submitted as `ci-2026-02477r`; **rejected 28 July
> 2026 with no referee reports**, with a transfer offer to ACS Omega. The live
> package is now `paper/submissions/ACSOmega/`. This directory is kept as the
> record of what was sent to JCIM. Two defects found in it while building the
> transfer, both fixed only in the ACS Omega package:
>
> 1. **`\begin{tocentry}` had been deleted** from `main.tex` in an uncommitted
>    working-tree edit. Restored here, and `main.pdf` rebuilt (19 pp, TOC graphic
>    on the last page) so this record is not broken — but note the manuscript
>    actually submitted to JCIM was built from the version with the block present.
> 2. **`Cai_SymMCFlow_JCIM_manuscript.docx` contains `oindent extbfSupporting
>    Information`** in the Associated Content section: the `scratchpad/
>    preprocess_docx.py` rebuild step below is sed-based and ate the backslashes of
>    `\noindent` and `\textbf`. The DOCX is committed unrepaired as the historical
>    artifact. If a Word manuscript is ever needed again, use
>    `../ACSOmega/build_docx.py`, which does literal Python replacement and also
>    fixes float numbering and starred-float captions.
>
> Also unresolved here: the SI points the strict match@10 ladder at "main-text
> Table 6"; the ladder is Table 7 (Table 6 is the MolCrystalFlow comparison).
> Corrected only in the ACS Omega SI.

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
- [x] Purpose-built TOC graphic added (`figures/fig_toc.pdf`, source
      `paper/figures/fig_toc.py`): decomposition + coset/family-mask conditioning +
      the 0%→6.9%/10.7% exact-match ladder vs MolCrystalFlow. Renders in the tocentry.
- [x] Bump the Zenodo DOI (`cai2026symmcflow`) to the strengthened code before submission.
      Done: v2, DOI 10.5281/zenodo.21500734 (concept DOI 10.5281/zenodo.20822234 → latest).
      v1.0.3 (21384130) was the diagnostic-era archive; the DD/diagnostic manuscript
      (`paper/main.tex`) correctly stays on v1.0.3. MoML + AI4Mat packages still cite
      v1.0.3 and must be bumped to v2 before those (strengthened-method) submissions.
- [ ] Upload via the ACS Paragon system: `main.tex`+`references.bib`+`figures/` as the
      manuscript, `si.tex` (built to PDF) as Supporting Information; provide the TOC graphic
      and cover letter in the portal.
- [ ] Confirm the competing-interest / data-availability disclosures in the submission form
      match the manuscript statements.

## Word (DOCX) manuscript — built on the ACS template
`Cai_SymMCFlow_JCIM_manuscript.docx` is the ACS-format Word version (an alternative to
the LaTeX ZIP for the Manuscript File slot). Built via pandoc from `main.tex` +
`acs.csl` (ACS Guide 2022 style) and formatted by the `journal-submission` skill's
`apply_format.py` against `requirements.md`, then audited (`compliance_report.md`:
**21/21 PASS**) and Word-rendered to `Cai_SymMCFlow_JCIM_manuscript.pdf` (20 pp) for a
visual check. US Letter, 1-in margins, Times New Roman 12 pt, double-spaced, continuous
line numbers, centered page numbers. All `\ref`/`\eqref` resolved to hard numbers,
Tables 1–7 + Figure 1 captioned/numbered, ACS references with titles + v2 Zenodo DOI.
Rebuild: `scratchpad/preprocess_docx.py main.tex _main_docx.tex` → pandoc → apply_format
→ audit. SI stays as `si.pdf` (ACS SI is a PDF). Either the ZIP or the DOCX may be
uploaded as the Manuscript File.

## Numbers provenance
Phase B (`e035b8d`), E1 (`581723c`), E3 (`1224870`), E2 scale (`ce0e5d0`,
`gpu_results/phaseE_scale`), E4 bench (`gpu_results/phaseE4`), Phase F ladder
(`gpu_results/FINDINGS_F_FINAL.md`, `gpu_results/phaseF3{a..f}/`).
