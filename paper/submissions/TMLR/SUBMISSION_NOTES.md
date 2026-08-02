# TMLR submission package — SymMC-Flow orientation paper

Target venue: **Transactions on Machine Learning Research (TMLR)**, submitted via OpenReview
(`https://openreview.net/group?id=TMLR`). TMLR is double-blind and reviews on
**correctness and support of claims, not novelty/impact** — the framing this package is
built for, after the Digital Discovery desk-reject on impact.

## Template & policy compliance (audited 2026-07-18)

- **Stylefiles unmodified.** `tmlr.sty`, `tmlr.bst`, and `fancyhdr.sty` are **byte-identical**
  to the canonical files in `JmlrOrg/tmlr-style-file` (verified by `diff`). No edit to the
  stylefile, font, margins, or layout — the deviation that TMLR rejects without review.
- **Preamble matches the official template**: `\documentclass[10pt]{article}` +
  `\usepackage{tmlr}` + plain `\usepackage{hyperref}`/`\usepackage{url}`. The only extra
  packages are standard content packages (`amsmath`, `amssymb`, `graphicx`, `booktabs`) that
  do not touch font/margins/layout. `microtype` and `colorlinks` were removed so nothing
  alters spacing or link styling relative to the template.
- **Anonymous** (double-blind): `\usepackage{tmlr}` without `[accepted]`; title page shows
  "Anonymous authors / Paper under double-blind review". Full-text scan of the PDF finds no
  author name, affiliation, ORCID, GitHub, or Zenodo/DOI (only the cited third-party author
  Frank Noé appears, in a reference).
- **Broader Impact Statement** included (TMLR-optional but ethics-guideline-flagged): a short,
  honest statement — diagnostic study, no released datasets, no dual-use/person-facing system,
  licensed-data usage noted.
- **Author Contributions / Acknowledgements / Funding** are intentionally **omitted** — the
  template says to add them only after acceptance and de-anonymization.
- **No page limit** applies at TMLR; 16 pp (incl. appendices) is fine. No line numbers (the
  template does not use them). Abstract is a single paragraph, as required.
- **Dual submission / self-plagiarism:** the manuscript is unpublished and not under review
  anywhere (Digital Discovery *desk-rejected* it without review). The planned AI4Mat (Aug 30)
  and MoML (Sep 1) workshops are **non-archival**, which TMLR's policy permits concurrently —
  but confirm each workshop is non-archival at submission time, and do not submit an archival
  version elsewhere while under TMLR review.

## Recommended Action Editors

After you submit, OpenReview emails you to **recommend Action Editors**. Pick 2–3 TMLR AEs
whose expertise covers geometric/equivariant deep learning and generative models for
molecules/materials (e.g. flow matching / diffusion on manifolds, SE(3)/SO(3) generative
models, crystal or molecular generative modeling). Browse the current TMLR Action Editor list
on OpenReview and match by those keywords; also enter any conflicts of interest (advisors,
recent co-authors, same institution) on the form.

## What to upload to OpenReview

TMLR wants a single self-contained anonymized PDF. **Upload `main.pdf`** — it already
contains the appendices (A–F, the former Supplementary Information) inline, so there is no
separate supplementary PDF to attach. Optionally attach the LaTeX source as supplementary
material; it is anonymized too.

Files in this folder:
- `main.tex` — TMLR-formatted, double-blind (`\usepackage{tmlr}`), appendices merged.
- `main.pdf` — compiled, 16 pp, self-contained. **This is the submission.**
- `references.bib` — 22 refs; the archived-software self-citation is removed for anonymity.
- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` — official TMLR style files (needed to recompile).
- `figures/fig2_results.pdf`, `figures/figS_rasym.pdf`.

Rebuild: `tectonic main.tex` (or `pdflatex → bibtex → pdflatex ×2`).

## OpenReview form fields

- **Title:** What an SO(3) Orientation Flow Can and Cannot Learn in Molecular-Crystal
  Structure Prediction
- **Abstract:** copy from the PDF abstract.
- **TL;DR (optional):** A rigid-body SO(3) orientation flow for molecular crystals learns
  the space-group-relative rotation between symmetry copies but provably cannot regress the
  gauge-free asymmetric-unit pose; we decompose, diagnose, and quantify the split.
- **Keywords:** crystal structure prediction; flow matching; molecular crystals; rigid-body
  factorization; SO(3); space-group symmetry; generative models; diagnostic study.
- **Anonymization:** confirmed clean — no author name, affiliation, ORCID, GitHub, or Zenodo
  in the PDF (only cited third-party authors appear, e.g. Frank Noé in a reference).
- **Conflicts / suggested Action Editor:** enter author conflicts on the form; optionally
  suggest an AE with a generative-models-for-materials / geometric-DL background.
- **Prior submission:** this work was **desk-rejected by Digital Discovery without peer
  review** (editorial impact screen). There are therefore no prior reviews to share in the
  optional "previous reviews" field. No prior *TMLR* submission.

## Camera-ready (after acceptance)

1. In `main.tex`, switch `\usepackage{tmlr}` → `\usepackage[accepted]{tmlr}`.
2. Uncomment the author block (Frank Cai, Purdue University, frankyc11223@gmail.com).
3. Restore the archived-software citation (the `cai2026symmcflow` entry from the master
   `references.bib`) and the public GitHub/Zenodo links in the Reproducibility Statement.
4. Recompile. Header flips to "Published in Transactions on Machine Learning Research".

## What changed from the Digital Discovery version

- Template: RSC single-column article + `authblk` + numbered-superscript natbib → official
  **TMLR style** (`tmlr.sty`/`tmlr.bst`), **author-year** citations.
- **Anonymized** for double-blind review (author, affiliation, ORCID, repo, DOI removed).
- Supplementary Information merged in as **Appendices A–F**; all "Supplementary Fig. S1 /
  Supplementary Information" cross-references rewired to `\ref` into the appendix.
- RSC back-matter (Data availability, Code availability, Author contributions, Conflicts of
  interest, Funding, Use of AI tools) replaced by a single **Reproducibility Statement** in
  TMLR style, plus a neutral LLM-use disclosure.
- Added an optional **Broader Impact Statement** (TMLR ethics guidelines).
- Removed the RSC keywords line from the PDF body (keywords go in the OpenReview form) and the
  unused `siunitx`, `microtype`, and `colorlinks` — nothing now alters the template layout.
- Science, tables, figures, and numbers are unchanged from the reviewed DD manuscript.
