# Task: Fe-SMA Publication Paper

## Context
Convert the ISEF poster ("AI-Driven Exploration of Cost-Effective Fe-Based Shape Memory Alloys")
into a full journal research article. Source materials:
- Poster: E:\isef\Frank Cai - ISEF Poster.pdf / .pptx
- Flashdrive G:\FE-SMA — mechanical data (Instron .pdfs + Exports), SSRF .chi synchrotron data,
  micrographs (697-6 / 697-7 AI alloy vs Omori benchmark), XRD images, ANALYSIS.md, scripts.
- Two core findings:
  (1) "Reality Gap" — AI-hypothesized Fe-Mn-Al-Si-Ni-C alloy is thermodynamically stable but
      fails pseudoelasticity; B2 AlFe is 0.163 eV/atom more stable than target L2₁ Heusler.
  (2) "Small-grain paradigm" — 20 FPM continuous AGG gives ~2× recovery vs traditional bamboo
      grains, overturning the Omori paradigm.

## Target venue (recommendation)
**Primary: Shape Memory and Superelasticity (Springer)** — Omori & Kainuma publish here, perfect
audience, accepts student/first-author work, 5-7k word research articles.
**Stretch: Acta Materialia** — if we lean hard on the small-grain mechanism with Rietveld
refinement and EBSD added.
**Alternative: Scripta Materialia (letter)** — short version focused on the small-grain finding alone.

→ Awaiting your sign-off on target before drafting.

## Paper outline (proposed)

### Title (draft)
"Process-Aware Limits of AI-Guided Alloy Design: A Computational and Experimental Audit of an
LLM-Hypothesized Fe-Mn-Al-Si-Ni-C Shape Memory Alloy"

### Sections
1. **Abstract** (~250 words) — context, hypothesis, methods, both findings, implications.
2. **Introduction**
   - Fe-SMAs vs Nitinol cost driver
   - AI materials discovery (GNoME, Deep Research) and the prediction-vs-processability gap
   - Omori paradigm: bamboo-grain requirement
   - Research questions + hypotheses
3. **Materials and Methods**
   - 3.1 Alloy design (LLM Deep Research + GNoME constraints; AI vs Omori compositions)
   - 3.2 Synthesis (arc melt → hot roll 850 °C → cold draw 0.014″)
   - 3.3 Heat treatment (1200 °C anneal vs 3-cycle AGG vs continuous strand 0.8 / 4 / 20 FPM)
   - 3.4 Metallography (mounting, grind, 0.25 µm OP-S polish, Clemex imaging)
   - 3.5 Mechanical testing (Instron cyclic tensile, gauge geometry)
   - 3.6 Synchrotron XRD (SSRF λ = 0.12587 Å) + Cu Kα-equivalent conversion
   - 3.7 Computational audit pipeline (SciPy peak detection → JARVIS-DFT cosine match → DiffractGPT POSCAR → formation-energy cross-validation)
4. **Results**
   - 4.1 Microstructure (Fig.13 a-d): AI dual-phase persists; Omori bamboo vs acicular vs refined
   - 4.2 Cyclic stress-strain (Fig.14 a-d): AI ~0% recovery; Omori 12-sec > 5-min recovery
   - 4.3 Synchrotron diffraction (Fig.15/16): AI rings invariant; Omori shows variant activation
   - 4.4 Phase fraction table (Fig.18 before/after deformation)
   - 4.5 Computational audit: JARVIS cosine, formation energies, Reality Gap quantified at 0.163 eV/atom
5. **Discussion**
   - 5.1 Why the AI alloy failed — B2 outcompetes L2₁ thermodynamically; C/Zener ordering
   - 5.2 Reality Gap — what process-aware AI must add (kinetic barriers, interstitials, feed-rate)
   - 5.3 Small-grain paradigm — mechanism speculation (grain-boundary constraint, variant selection)
   - 5.4 Computational auditing as a low-cost pre-synthesis screen
   - 5.5 Limitations (no Rietveld yet, no EBSD, sample size)
6. **Conclusions** (4-5 bullets, mirroring poster)
7. **Acknowledgments** (SSRF Dr. Yan-Jie, institution, funding/mentors)
8. **References** (extend poster's 7 → ~25-40 for journal)
9. **Supplementary** — raw .chi files, Python pipeline (link to GitHub), full Instron exports

## Figure plan (mapped from poster)
| Fig | Source | Caption focus |
|-----|--------|---------------|
| 1   | Fig.5 (poster) | Computational pipeline schematic |
| 2   | Fig.6/7/8 composite | Synthesis flow (arc melt → wire → quartz encapsulation) |
| 3   | Fig.13 a-d | Microstructure comparison (AI std / AI AGG / Omori AGG / Omori 12-sec) |
| 4   | Fig.14 a-d | Cyclic stress-strain (4-panel) |
| 5   | Fig.15 a-b | 1D XRD before/after deformation, AI vs Omori |
| 6   | Fig.16 a-d | 2D synchrotron Debye-Scherrer rings, 4 panels |
| 7   | New table  | Phase fractions before/after (from poster Fig.18) |
| 8   | New table  | JARVIS-DFT cosine + formation energy (from poster Fig.17) |
| 9*  | Optional   | Recovery-strain vs FPM (small-grain paradigm headline chart) |

*Fig.9 may not exist yet — need to confirm we have data for 0.8 / 4 / 20 FPM recovery numbers.

## Plan (execution)

- [ ] 1. User confirms target journal (SMS / Acta / Scripta letter)
- [ ] 2. User confirms authorship list and affiliations
- [ ] 3. User confirms whether Rietveld refinement / EBSD data exists or paper goes as-is
- [ ] 4. Inventory G:\FE-SMA mechanical exports — extract recovery-strain numbers per sample
- [ ] 5. Draft Abstract + Section 1 (Introduction) → user review
- [ ] 6. Draft Section 2 (Methods) → user review
- [ ] 7. Draft Section 3 (Results) with embedded figure callouts → user review
- [ ] 8. Draft Section 4 (Discussion) + Conclusions → user review
- [ ] 9. Build reference list (BibTeX or .docx) → user review
- [ ] 10. Assemble figures at journal resolution (300 dpi min) into /paper/figures/
- [ ] 11. Format to journal template (LaTeX or Word, depending on venue)
- [ ] 12. Final pass: claims supported by data, no orphan citations, units consistent
- [ ] 13. Push paper draft to GitHub repo

## Review

- **Completed:** 2026-06-01
- **Deliverables produced:**
  - `paper/manuscript.md` — 5,308-word draft, em-dash-free academic prose, Abstract + 5 sections + Acknowledgments + Data/code availability
  - `paper/references.bib` — 31 BibTeX entries (6 flagged `TBD-VERIFY`)
  - `paper/figures/` — 12 staged figures (Fig.1–5) plus `sources/` and `sources/poster_media/` (25 high-res images extracted from poster pptx) and `captions.md`
  - `paper/Cai_Fe-SMA_SMS_manuscript.docx` — default Chicago author-date style
  - `paper/Cai_Fe-SMA_SMS_manuscript_SpringerStyle.docx` — Springer basic author-date CSL
- **What worked:**
  - Extracting embedded images from the poster `.pptx` (unzip → `ppt/media/`) yielded native-resolution versions of all stress-strain plots, microstructure micrographs, the pipeline schematic, and the indexed 2D Debye–Scherrer pattern. Far higher quality than re-exporting from the PDF.
  - Cross-checking the poster JARVIS table against `ANALYSIS.md` caught a column-mapping error; manuscript Table 4 uses the corrected values.
  - Pulling additional context from the `fe-sma-xrd` GitHub repo (bulk modulus, magnetic Curie crossing) strengthened the Discussion's three-mechanism explanation of the Reality Gap.
- **What changed from plan:**
  - Tone calibration: dropped em-dashes globally, replaced with colons / commas / parentheses per user request.
  - Co-author/affiliation placeholders left as `[CO-AUTHORS TBD]` and `[LAST NAME TBD]` until user supplies names.
  - Generated both default and Springer-CSL `.docx` variants in case SMS submission portal prefers one.
- **Known limitations:**
  - 6 bibliography entries marked `TBD-VERIFY` need DOI/volume confirmation against original sources.
  - Composite figures (2×2 layouts for Fig.2, Fig.3, Fig.5) are staged as individual panels; final assembly in PowerPoint/Inkscape is still required before submission.
  - SSRF beamline number placeholder (`BL14B1`) needs user confirmation.
  - Rietveld refinement and EBSD are flagged as in-progress in §4.5 Limitations; if the user runs Rietveld before submission, Table 3 phase-fraction entries can be promoted to absolute fractions.
- **Next user actions before submission:**
  1. Provide co-author names + affiliations and SSRF beamline number
  2. Verify the 6 `TBD-VERIFY` BibTeX entries
  3. Assemble the 2×2 composite figures (Fig.2, Fig.3, Fig.5) at 300+ dpi TIFF
  4. Run a final pass on the `.docx` for SMS-specific formatting (line numbering, double-spacing if required by the submission template)
  5. Push the paper folder to GitHub (separate repo from `fe-sma-xrd`)
