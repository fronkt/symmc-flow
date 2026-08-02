---
journal: "Journal of Chemical Information and Modeling"
code: "JCIM"

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
  abstract_words: 250
  keywords_min: null
  keywords_max: null
  pages_max: null
  words_max: null
  figures_max: null

references:
  style: "ACS (numbered, superscript in text; complete with titles)"
  csl: acs.csl
  notes: "ACS Guide 2022 style via CSL; JCIM requires references complete, including titles."

sections:
  order:
    - Abstract
    - Introduction
    - Methods
    - Results
    - Conclusions
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
---

## Traceability

- **page.size letter / margins 1 in** — ACS US-Letter manuscript format; 1-inch (2.54 cm) margins are the ACS Word template default.
- **body Times New Roman 12 pt, double-spaced** — ACS review manuscripts are prepared double-spaced in a 12-pt serif face; double spacing eases reviewer annotation.
- **line_numbers: continuous** — NOT required by JCIM (guidelines, 2026-05-21, are
  silent on line numbers; ACS uses a simplified "Fast Format" for initial submission).
  Included only as a peer-review courtesy; safe to drop. [corrected 2026-07-23]
- **page_numbers: bottom_center** — not explicitly required; included as convention.
- **limits.abstract_words 250** — JCIM guideline: abstract 150–250 words (manuscript abstract is 229).
- **references style ACS, titles required** — JCIM guideline (2026-05-21): "References can be provided in any style, but they must be complete, including titles." ACS Guide 2022 CSL used as ground truth.
- **sections.required Data and Software Availability** — JCIM applies ACS Research Data Policy Level 2: a Data Availability Statement is mandatory at submission.
- **keywords limits null** — JCIM standard Articles do not mandate a fixed keyword count; the six manuscript keywords are retained but not audited.
- **figures inline PNG, 300 dpi** — the single results figure is embedded in the review manuscript (vector source in the LaTeX package); the separate TOC graphic is uploaded in its own portal slot.

## File inventory

Uploaded via ACS Paragon Plus:
- Manuscript File — `Cai_SymMCFlow_JCIM_manuscript.docx` (this document) OR the LaTeX ZIP.
- Cover Letter — `cover_letter.pdf`.
- Supporting Information for Publication — `si.pdf`.
- Table of Contents / Abstract Graphic — `figures/fig_toc.png` (or `.pdf`).

## Ambiguities

- None outstanding. Abstract "3–4 sentences" line in the guidelines is treated as
  subordinate to the explicit 150–250-word rule (user-confirmed to keep the
  229-word paragraph abstract).
- AI-use disclosure in Acknowledgements: RESOLVED (user decision 2026-07-23) —
  added, scoped to "used Claude Opus 4.8 (Anthropic) for language editing of the
  text," per ACS's requirement to disclose AI use in the Acknowledgments.
