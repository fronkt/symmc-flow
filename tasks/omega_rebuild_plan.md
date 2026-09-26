# ACS Omega resubmission — rebuild plan (drafted 2026-09-25)

Decision 2026-09-24: ao-2026-09173w REJECT, resubmission allowed until **2027-01-22**.
Reviewer reports are confidential (ACS) and stay out of this public repo; only the
technical content we verified ourselves is recorded here.

## Why the current paper fails (measured, not argued)
1. **The coset label is the answer.** For proper-op copies the label's Cartesian rotation vs the
   observed relative rotation: median geodesic 0.0°, 82.9% < 15° (`scripts/diag_symmetry_cosets.py`,
   2706 proper copies). The "+41.1% vs +27.5%" learnability gain is mostly lookup.
2. **Improper ops are unrepresentable.** 1099/3805 non-reference copies (29%) come from
   mirror/inversion/glide ops; det+1 Kabsch in `_align_to_reference` cannot fit an enantiomeric copy.
3. **So the pipeline throws centrosymmetric crystals away.** Survival from `data/csd_mol/manifest.csv`
   (3500 CIFs) to `ds.pt` (1127 items), 2318 of the 2372 skips = "conformer rmsd > conf_tol"
   (measured 2026-09-25, gemmi SG parse, 0 unparsed):

   | SG | class | total | kept | survive |
   |---|---|---|---|---|
   | P2₁/c (14) | centro | 1330 | 242 | 18% |
   | P-1 (2) | centro | 660 | 106 | 16% |
   | C2/c (15) | centro | 193 | 32 | 17% |
   | Pbca (61) | centro | 147 | 19 | 13% |
   | Pna2₁ (33) / Pca2₁ (29) / Cc (9) | glide, non-centro | 143 | 23 | 15–17% |
   | P2₁2₁2₁ (19) | Sohncke | 430 | 336 | 78% |
   | P2₁ (4) | Sohncke | 315 | 225 | 71% |
   | **all centro / all Sohncke** | | 2432 / 874 | 441 / 648 | **18.1% / 74.1%** |

   Pnma (58%) is the expected exception: molecules sitting on mirror planes are self-enantiomeric.
   The model was therefore trained and evaluated on a set that is biased toward chiral packings,
   even though ~70% of real organic crystals are centrosymmetric.
4. Prior probe T1.2 (DD revision): **R_asym showed no conditional signal** (125.2° probe vs 122.4°
   constant vs 123.5° Haar). The one genuinely free orientation has not been shown to be learnable.
   This is the main risk for the rebuild, so gate G2 below exists to kill it early.

## Two candidate papers — decided by G0
- **Paper A (method rebuild):** "exact space-group expansion of a learned asymmetric unit, with
  O(3) copies": the model learns only lattice + asym centroid + R_asym, and every copy is built by
  applying the full Cartesian op (det ±1) to atoms. No learned copy orientations, no coset lookup.
- **Paper B (field finding):** "SO(3) rigid-body pipelines silently drop centrosymmetric crystals".
  Viable **only if** G0 shows published comparators share the flaw. It can then fold into Paper A as
  the motivating result.

## Gates (in order; each has a kill criterion)
- [x] **G0 — comparator check** (done 2026-09-25, `tasks/omega_rebuild_G0.md`): MolCrystalFlow and
  MOFFlow use per-molecule local coordinates (MCF also a chi/axis-flip bit) -> they do NOT share
  the flaw; **Paper B is dead.** But MCF's public CSP driver replicates one conformer + one flip
  across all Z copies -> generated cells are homochiral (verified at packing_gen.py:371-388).
  Paper A's sharpest claim: exact SG expansion generates mixed-handedness packings natively.
- [ ] **G1 — O(3) data re-parse.** Kabsch with reflection allowed (store parity bit per copy);
  re-parse the 3500 CIFs. Pass: centro survival rises to within ~10 pts of Sohncke, *and* exact
  SG expansion of (lattice, asym centroid, R_asym, parity) reconstructs 100% of kept crystals under
  StructureMatcher (oracle). Unit test: a P2₁/c toy with an inversion copy round-trips.
- [ ] **G2 — is anything learnable?** Train the R_asym flow (asym-unit only) on the re-parsed set,
  3 seeds, species-grouped split. Compare geodesic error + orientation-isolated match@k vs Haar-random
  and constant R_asym, **each followed by the same rigid-press relaxation**.
  **Kill:** if learned R_asym + press does not beat random R_asym + press by a margin outside
  seed s.d., Paper A has no learned contribution → fall back to Paper B or a short negative note.
- [ ] **G3 — fair end-to-end benchmark.** Same split, same match criterion, match@1 and @k:
  (i) new model, (ii) MolCrystalFlow retrained on our split with its released code, (iii) a classical
  baseline (random SG-constrained placement + rigid press, Genarris-style), (iv) the old symmc-flow.
  No comparisons against published numbers from other splits.
- [ ] **G4 — reviewer rigor items (only after G1–G3 pass):** a worked P2₁/c example figure (one
  crystal, all four ops, showing the inversion copy); a second force field for the energy check (GAFF
  or UFF); dataset figures (SG / Z' / size distributions before vs after filtering); a scope statement.
- [ ] **G5 — rewrite** as a methods paper (not a project report): one claim, one figure per claim;
  cover letter that states plainly what changed since ao-2026-09173w.

## Budget / timeline (to 2027-01-22)
- G0–G1: local CPU, ~days. G2–G3: rented GPU (prior runs cost single-digit $); Frank approves spend.
- Target resubmission ≈ early January, leaving slack before the 01-22 window closes.
- pxrd-flow (ICLR) and the other revisions keep priority when deadlines collide.
