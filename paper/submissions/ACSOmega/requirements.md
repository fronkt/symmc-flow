---
journal: "ACS Omega"
code: "acsodf"

page:
  size: letter
  width_cm: null
  height_cm: null
  margins_cm:
    top: 2.54
    bottom: 2.54
    left: 2.54
    right: 2.54

body:
  font: "Times New Roman"
  size_pt: 12
  line_spacing: 2.0
  space_before_pt: 0
  space_after_pt: 0
  alignment: left
  first_line_indent_cm: null
  preserve_fonts: []
  strip_highlights: true
  force_black: false

headings:
  font: null
  color: "000000"
  bold: true
  sizes_pt:
    1: 12
    2: 12
    3: 12

captions:
  size_pt: null

tables:
  line_spacing: null
  placement: inline

line_numbers: none
page_numbers: bottom_center
header_text: null

limits:
  abstract_words: null
  keywords_min: null
  keywords_max: null
  pages_max: null
  words_max: null
  figures_max: null
  title_words: 15

references:
  style: "ACS (numbered, superscript in text; Arabic numerals in order of first citation)"
  csl: acs.csl
  notes: "ACS Omega: 'Literature references should be numbered with Arabic numerals in the order of their first citation in the text.' ACS Guide 2022 CSL used as ground truth; titles retained (harmless where not mandated)."

sections:
  order:
    - Abstract
    - Introduction
    - Results and Discussion
    - Conclusions
    - Methods
    - References
  required:
    - Abstract
    - Data and Software Availability
    - References

figures:
  formats: [png]
  min_dpi: 300
  naming: null
  placement: inline

anonymize: false

files:
  - manuscript
  - cover_letter
  - supporting_information
  - figures
  - toc_graphic
---

## Traceability

- **journal / code acsodf** — transferred from *J. Chem. Inf. Model.* at the
  Editor's invitation. Guidelines source:
  https://researcher-resources.acs.org/publish/author_guidelines?coden=acsodf
  (checked 2026-08-02). `achemso-acsodf.cfg` v3.14 confirmed present locally.
- **sections.order: Methods AFTER Conclusions** — ACS Omega's stated preferred
  order is Introduction → Results and Discussion → Conclusions → Methods. This is
  the single structural difference from the JCIM package. The orientation
  decomposition was promoted into Results and Discussion because it reports
  held-out measurements (Table 1) rather than a procedure; the remaining
  procedural material moved to the back Methods section. Table numbering is
  unchanged from the JCIM build (the decomposition table was already Table 1).
- **limits.abstract_words: null** — ACS Omega requires an abstract for Research
  Articles but states no word limit; the manuscript abstract (224 words) is
  carried over unchanged from the JCIM build, where it satisfied the explicit
  150–250 rule. Not audited here because no limit exists to audit against.
- **limits.title_words 15** — ACS Omega recommends titles of ≤15 words. The title
  is 9 words.
- **line_numbers: none** — ACS Omega mandates no line numbers; the journal uses
  "Fast Format" for initial submission, requiring only that standard sections be
  clearly identified. Dropped deliberately (the JCIM package also carried none).
- **body Times New Roman 12 pt, double-spaced / margins 1 in** — not mandated by
  ACS Omega, which specifies no font, spacing, or margin requirements. Retained
  from the ACS Word template default as a reviewer courtesy; safe to drop.
- **TOC graphic REQUIRED** — ACS Omega: all Research Articles must include an
  Abstract (TOC) graphic. `\begin{tocentry}` is present in `main.tex` and renders
  `figures/fig_toc.pdf`. NOTE: this block had been deleted from the JCIM
  `main.tex` in an uncommitted working-tree edit and was restored for this
  package.
- **sections.required Data and Software Availability** — ACS Research Data Policy;
  ACS Omega requires a Data Availability Statement.
- **figures 300 dpi color** — ACS Omega graphics minimums are 1200 dpi line art,
  600 dpi grayscale, 300 dpi color. The single results figure is a vector PDF in
  the LaTeX package and a 300 dpi PNG in the Word build. Single-column graphics
  must fit within 240 pt (3.33 in).
- **Supporting Information as a separate file** — `si.tex` cannot use achemso's
  `manuscript=suppinfo` under the acsodf configuration, which declares
  `\acs@type@list{article}` only; passing `suppinfo` silently degrades to
  `article`. The SI therefore loads `manuscript=article` and sets the S-prefixed
  section/table/figure/equation counters and the SI title explicitly, reproducing
  suppinfo output. Verified in `si.pdf`: sections S1–S5, Table S1.

## File inventory

Uploaded via ACS Paragon Plus:
- Manuscript File — `Cai_SymMCFlow_ACSOmega_manuscript.docx` OR the LaTeX ZIP
  (`main.tex` + `references.bib` + `figures/`).
- Cover Letter — `cover_letter.pdf` (transfer letter; names the JCIM manuscript ID).
- Supporting Information for Publication — `si.pdf`.
- Table of Contents / Abstract Graphic — `figures/fig_toc.png` (or `.pdf`).

## Ambiguities

- **Open access is mandatory.** ACS Omega is fully gold OA with a required Article
  Publishing Charge (default licence CC BY-NC-ND, upgradeable to CC BY). This is
  not a formatting item but it gates submission; see SUBMISSION_NOTES.md.
- **Reviewer comments from JCIM.** If the JCIM decision carried referee reports,
  ACS transfers normally expect either a point-by-point response or an explicit
  statement that the manuscript is unchanged. The cover letter currently states
  the manuscript is unchanged apart from the structural reformat. Revise if
  reports were received.
- AI-use disclosure in Acknowledgements: carried over unchanged from the JCIM
  package (user decision 2026-07-23), scoped to language editing, per ACS's
  requirement to disclose AI use in the Acknowledgments.
