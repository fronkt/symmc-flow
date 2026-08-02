# Compliance report — Cai_SymMCFlow_ACSOmega_manuscript.docx vs ACS Omega

| Status | Check | Detail |
|---|---|---|
| PASS | page size (section 1) | want (21.59, 27.94) cm, got (21.59, 27.94) cm |
| PASS | top margin (section 1) | want 2.54 cm, got 2.54 cm |
| PASS | bottom margin (section 1) | want 2.54 cm, got 2.54 cm |
| PASS | left margin (section 1) | want 2.54 cm, got 2.54 cm |
| PASS | right margin (section 1) | want 2.54 cm, got 2.54 cm |
| PASS | page numbers (section 1) | PAGE field in footer |
| PASS | Normal style font | want Times New Roman, got Times New Roman |
| PASS | Normal style size | want 12.0 pt, got 12.0 |
| PASS | Normal style line spacing | want 2.0, got 2.0 |
| PASS | theme-font overrides | no theme fonts in styles part |
| PASS | run-level font overrides | none found |
| PASS | run-level size overrides | none found |
| PASS | paragraph line-spacing deviations | none found |
| PASS | text highlighting | 0 highlighted runs |
| PASS | colored text (direct) | 0 colored runs (check deliberate vs leftover) |
| PASS | required section: Abstract | found |
| PASS | required section: Data and Software Availability | found |
| PASS | required section: References | found |
| PASS | section order | Abstract → Introduction → Results and Discussion → Conclusions → Methods → References |

**19 checks: 19 PASS, 0 WARN, 0 FAIL**

## 5b. Word render (Microsoft Word COM)

| Item | Result |
|---|---|
| True page count | 20 pages |
| Word count | 4,525 |
| Page-limit check | n/a — ACS Omega sets no page limit |
| Page 1 visual | Title, author + affiliation + corresponding email, Abstract heading, keywords line, double-spaced body, centered page number. OMML equations render as math, em dashes correct. |
| Figure page (p. 11) visual | Fig. 1 renders at full width, both panels legible, caption numbered and complete, no clipping. |
| LaTeX cross-check | `main.pdf` (pdflatex, achemso acsodf) also 20 pp; 19/19 bibitems resolve; 0 errors, 0 warnings, 0 overfull hboxes > 15 pt. |

## 5c. Semantic spot-check

| Item | Result |
|---|---|
| Float captions | 8/8 present and numbered: Tables 1-7 + Figure 1, matching the LaTeX numbering. |
| End-matter headings | Acknowledgments / Data and Software Availability / Associated Content (Supporting Information) / References all present as real headings. |
| Backslash integrity | No `oindent` / `extbf` / `egin{` artifacts. (The JCIM package's shipped DOCX contains `oindent extbfSupporting Information` from a sed-based build; this build uses literal Python replacement instead.) |
| TOC graphic | `\begin{tocentry}` restored in `main.tex` (it had been deleted in an uncommitted working-tree edit) and renders `fig_toc.pdf`. Excluded from the Word body, uploaded in its own Paragon Plus slot. |
| Supporting Information | `si.pdf`, 6 pp, sections S1-S5 and Table S1 verified S-prefixed under the acsodf config, which does not support `manuscript=suppinfo`. |
| SI cross-reference | SI's pointer to the strict match@10 ladder corrected from "main-text Table 6" to **Table 7** (Table 6 is the MolCrystalFlow comparison). This was wrong in the JCIM package. |
| `[DRAFTED — REVIEW]` markers | None outstanding. |
| Declarations | Competing-interest, funding, AI-use disclosure, and data-availability statements all carried over verbatim from the user-approved JCIM package; none newly authored. |

**Open items are tracked in SUBMISSION_NOTES.md — this report covers format compliance only.**
