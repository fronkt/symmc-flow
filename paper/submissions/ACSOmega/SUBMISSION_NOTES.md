# ACS Omega submission package — notes

**Venue.** ACS Omega, CODEN `acsodf`. Research Article. Rolling submission.
Transferred from *J. Chem. Inf. Model.* at the Editor's invitation.
Guidelines source: https://researcher-resources.acs.org/publish/author_guidelines?coden=acsodf
(checked 2026-08-02).

**Built + verified locally (MiKTeX, Word COM), 2026-08-02.**
- LaTeX: `pdflatex → bibtex → pdflatex → pdflatex` clean. `main.pdf` 20 pp,
  `si.pdf` 6 pp, `cover_letter.pdf` 2 pp. 0 errors, 0 warnings, 0 undefined
  refs/citations, 19/19 bibitems, 0 overfull hboxes > 15 pt.
- Word: `Cai_SymMCFlow_ACSOmega_manuscript.docx`, audited **19/19 PASS**
  (`compliance_report.md`), rendered to 20 pp / 4,525 words for a visual check.

## What changed from the JCIM package

1. **`journal=acsodf`.** `achemso-acsodf.cfg` v3.14 is present in the local
   MiKTeX tree, so the class option is valid.
2. **Methods moved after Conclusions** — ACS Omega's preferred section order is
   Introduction → Results and Discussion → Conclusions → Methods. The manuscript
   now runs in that order.
3. **The orientation decomposition moved into Results and Discussion.** It reports
   held-out measurements over three seeds (Table 1), not a procedure, and burying
   the paper's central mechanistic finding behind the Conclusions would have been
   the wrong read. The procedural material — rigid-body flow, coset construction,
   SO(3)-averaged objective, template-free predictor, family mask, finishing
   stack, data and evaluation — is what moved to the back.
   *This is the one judgment call in the reformat. Reverting it is a block move
   plus one sentence ("the coset construction given in Methods" → "below").*
4. **Table numbering is unchanged** from the JCIM build (Tables 1–7, Figure 1) —
   the decomposition table was already Table 1, so the reorder did not renumber
   anything.
5. **Cover letter rewritten as a transfer letter**, naming the JCIM manuscript ID
   and stating that the science is unchanged.

## Defects found in the JCIM package while building this one

These are in `paper/submissions/JCIM/`, not here. Fixed in the ACS Omega package;
**the JCIM package still has them.**

- **The TOC graphic was deleted.** An uncommitted working-tree edit removed the
  `\begin{tocentry}` block from `JCIM/main.tex`. A graphical TOC entry is
  mandatory at both journals — this would have been a desk return. Restored here.
- **`oindent extbfSupporting Information`** appears in the shipped
  `Cai_SymMCFlow_JCIM_manuscript.docx`. A sed-based build step ate the
  backslashes of `\noindent` and `\textbf`. `build_docx.py` here uses literal
  Python string replacement instead.
- **SI cited the wrong main-text table.** The JCIM SI points the strict match@10
  ladder result at "main-text Table 6"; Table 6 is the MolCrystalFlow comparison
  and the ladder is Table 7. Corrected in `si.tex` here.

## Supporting Information under `acsodf`

`achemso-acsodf.cfg` declares `\def\acs@type@list{article}` — unlike `jcisd8`,
which declares `{article,suppinfo}`. Passing `manuscript=suppinfo` under acsodf
does **not** error: achemso logs `Invalid manuscript type` and silently falls back
to `article`, losing the S-prefixes and the SI title. `si.tex` therefore loads
`manuscript=article` and sets the S-prefixed section/table/figure/equation
counters explicitly. Verified in `si.pdf`: sections S1–S5, Table S1.

## Rebuilding

```bash
# LaTeX (all three documents)
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex si   && bibtex si   && pdflatex si   && pdflatex si
pdflatex cover_letter

# Word manuscript
python build_docx.py                                    # main.tex -> DOCX
python ~/.claude/skills/journal-submission/scripts/apply_format.py \
    Cai_SymMCFlow_ACSOmega_manuscript.docx --spec requirements.md \
    --out Cai_SymMCFlow_ACSOmega_manuscript.docx
python ~/.claude/skills/journal-submission/scripts/audit.py \
    Cai_SymMCFlow_ACSOmega_manuscript.docx --spec requirements.md \
    --out compliance_report.md
```

`build_docx.py` handles what pandoc cannot: achemso's `tocentry`/`acknowledgement`/
`suppinfo` environments, the `\affiliation`/`\email`/`\keywords` preamble commands,
float numbering (pandoc numbers nothing and **drops starred-float captions
entirely** — Tables 5–7), and PDF→PNG figure swaps, since Word cannot display an
embedded PDF image. `main_docx.tex` is a generated intermediate; do not edit it.

## BLOCKER before submitting: the APC

**ACS Omega is fully gold open access. Publication requires an Article Publishing
Charge of US$1,935** (default licence CC BY-NC-ND, upgradeable to CC BY). There is
no non-OA route, unlike the free standard route at J. Appl. Cryst. Discounts apply
only to authors whose primary affiliation is in a World-Bank lower-middle-income
economy, which does not apply here.

This is the substantive difference between the JCIM decision and the transfer
offer, and it is not a formatting problem. Options:

1. **Purdue read-and-publish agreement.** ACS has such agreements with many US
   institutions and they normally cover the full APC. Whether it applies turns on
   whether the submitting corresponding author is recognised as Purdue-affiliated
   in the ACS Publishing Center. **Confirm with the Purdue Libraries scholarly
   publishing office before submitting** — eligibility is determined at
   acceptance, and finding out then is too late.
2. **Decline the transfer** and submit elsewhere. The manuscript was already
   desk-rejected at Digital Discovery and TMLR on interest rather than
   correctness, so a fourth venue is a real cost, but $1,935 is a real cost too.
3. **Pay it.**

## TODO before submitting (human-gated)

- [ ] **Resolve the APC** (above). Nothing else matters until this is settled.
- [ ] **Fill the two placeholders in the cover letter**: `[JCIM MS ID]` and
      `[DECISION DATE]`. They are in both `cover_letter.md` and `cover_letter.tex`;
      recompile the letter after editing.
- [ ] **Did the JCIM decision include referee reports?** If so, an ACS transfer
      normally expects a point-by-point response document, and the cover letter's
      "no changes to the data, analysis, or claims" sentence needs revising. The
      package currently assumes no reports were received.
- [ ] **Frank reviews the expanded prose** (Introduction, the orientation-
      decomposition derivation, Limitations). This was carried over unreviewed from
      the JCIM package — still open, still his.
- [ ] **Decide whether `tab:bench` stays as written.** Also carried over. The
      result is parity with MolCrystalFlow, not superiority; if that table reads as
      a leaderboard entry rather than as evidence for the conditioning mechanism, a
      third editor draws the same conclusion the first two did.
- [ ] Upload via ACS Paragon Plus: `main.tex` + `references.bib` + `figures/`
      (or the DOCX) as the Manuscript File, `si.pdf` as Supporting Information,
      `figures/fig_toc.png` in the TOC-graphic slot, `cover_letter.pdf` as the
      cover letter.
- [ ] Confirm the competing-interest and data-availability disclosures in the
      submission form match the manuscript statements.

## File inventory

| File | Slot |
|---|---|
| `Cai_SymMCFlow_ACSOmega_manuscript.docx` **or** `symmc-flow_ACSOmega_manuscript.zip` | Manuscript File |
| `si.pdf` | Supporting Information for Publication |
| `figures/fig_toc.png` | Table of Contents / Abstract Graphic |
| `cover_letter.pdf` | Cover Letter |
