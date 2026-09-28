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
- [x] **G1 — O(3) data re-parse** (done 2026-09-26; `scripts/g1_o3_reparse.py`,
  `results/g1_o3_reparse.json`, cache `data/csd_mol/ds_o3.pt` [gitignored, CSD]). All 3500 CIFs:

  | class | total | kept det+1 | kept O(3) | exact rebuild |
  |---|---|---|---|---|
  | centrosymmetric | 2432 | 453 (18.6%) | **2028 (83.4%)** | 2020/2028 |
  | Sohncke | 874 | 655 (74.9%) | 660 (75.5%) | 651/660 |
  | other (glide) | 194 | 36 (18.6%) | **164 (84.5%)** | 163/164 |
  | **all** | 3500 | 1144 | **2852** | 2834/2852 (99.4%) |

  Mirror copies 5423/14119 (38%). Remaining skips: 591 genuinely non-rigid, 52 Z'>16, 2 too large,
  1 error, 1 timeout (WETCEH, >300 s). Pass criterion met: centro survival now above Sohncke.
  18 rebuild misses: 9 were ALSO kept by the legacy parser (pre-existing, not mirror-related:
  ADFGLP AZECOA ENIGOY FEXHAV GOQDAU GUSSIY HAJBID01 MOHQEJ MOJKUS), 9 are newly recovered
  (DAGRAF MEOHPH OLUPES SELZEO TULVUR [P-1, Z=4], UYOSOR, VAHPIF, XITREB01, XODDUP) -- all Z>=4
  with Z'>1 in P-1/C2/c/P2_1/c/Pna2_1; inspect before claiming 100%. NB downstream: mirrored copies
  carry a mirrored `local`, so `_species_groups` treats enantiomers as separate species -- fine for
  the asym-unit design, must be handled before reusing relative-gauge code.
- [x] **G2 RESULT (2026-09-27, Vast 72-core CPU box, ~$2.8; `results/g2_baselines.json`, `results/g2_eval.json`).**
  199 held-out Z'=1 crystals, 16 draws/arm, true lattice + asym centroid given, checkpoints = best val <= epoch 7:

  | metric | learned (3 seeds) | Haar | paired McNemar per seed |
  |---|---|---|---|
  | exact match, unrelaxed, any of 16 | **10.9 +- 2.9%** (13.1/7.5/12.1) | 0.0% | p = 3e-8, 6e-5, 1e-7 |
  | right basin, any relaxed draw | **19.9 +- 2.5%** (22.6/19.6/17.6) | 11.1% | p = 0.002, 0.02, 0.07 |
  | right basin, lowest-LJ-energy draw | 8.5 +- 1.5% | 6.5% | p = 0.19, 0.54, 1 (n.s.) |

  K=32 classical baseline: best-by-energy basin 8.6%, any-of-32 20.7%, relaxed TRUE pose exact 1.0%.
  Exact matches by any seed: 60/199 = 30.2%, mostly P2_1/c (28) and P-1 (8) -- the centrosymmetric
  crystals the old pipeline discarded. Loss-vs-t: learned field ~ predict-zero floor for t<0.6, 0.55-0.65
  of it at t>=0.9 -> it is a LEARNED RELAXER whose minima sit on the true structure (steric LJ minima sit
  ~14 deg off, so physics alone never matches exactly). Verdict: G2 PASSES on exact reconstruction and on
  draw quality; the full pipeline is bottlenecked by the steric-energy RANKING step (not significant).
  Next: rank learned draws without LJ (endpoint-mode frequency / learned-field consistency), train
  longer on faster cores (plateau at epoch ~2-7 on a 2016 Xeon; single-sample SGD), then G3.
- [x] **G2 build notes** (started 2026-09-26; `scripts/g2_asym_baselines.py`,
  `scripts/g2_learned_flow.py`). Findings while building it:
  * 1987 Z'=1 general-position crystals; asym-unit exact expansion rebuilds 9/9 in smoke **only after**
    taking symmetry ops from spglib per crystal: the standard-setting table (`space_group.get_ops`) is
    WRONG for alternative CSD settings (P2_1/n translates by (1/2,1/2,1/2), P2_1/c by (0,1/2,1/2)).
    This also likely explains part of the old paper's ~18% coset residual misses -- audit before reuse.
  * `rigid_press.finish_structure` drifts 1-3 A / 6-18 deg off the TRUE pose (fractional-coord steps
    sized for rough cells); replaced by an orientation-only relax. Even so the steric-LJ minimum sits a
    median ~14-17 deg from truth -> relaxed poses never pass exact match; relaxed poses are scored by
    "basin" (relaxed draw vs relaxed truth) and unrelaxed poses by exact match.
  * Learned model = learned pairwise torque field (rotation-equivariant), CPU-trainable (~7.5 min/epoch);
    3 seeds x 40 epochs running locally. Original spec below:
   Train the R_asym flow (asym-unit only) on the re-parsed set,
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

## Next steps after G2 (started 2026-09-27) — sequential, val-selected, test scored once
Protocol: split fixed by `--split-seed 0` (test = perm[:200], val = perm[200:300], train = rest).
Every selection (ranking rule, training hyperparameters, checkpoint) is made on VAL; TEST is scored
once per final configuration and never used to choose anything.
- [x] **N1 RESULT (2026-09-27, `results/n1_final.json`).** Val (n=100) exact match of the ONE picked
  draw, per seed mean: random 2.0, **unrelaxed LJ 7.0** (pooled 48 draws 9.0), residual torque 0.7,
  endpoint-mode frequency (Chamfer 0.25-1.5 A) 1.0-2.0 -> hypothesis (mode frequency) REJECTED; rule
  chosen = lowest-LJ UNRELAXED draw. TEST (n=200, scored once): **7.5% per seed (8.5/7.5/6.5), 11.0%
  pooled over 3 seeds**, vs random pick 1.3%, upper bound any-of-16 13%. G2's energy ranking failed only
  because it ranked AFTER relaxation (relax moves exact draws ~14 deg off truth).
- [x] N1 spec (was): For each crystal draw S=16 learned orientations
  (current G2 checkpoints) and score candidate selectors: endpoint-mode frequency (draws within a
  geodesic radius), residual learned torque at t->1, unrelaxed LJ energy, random pick (= match@1).
  Choose the selector on VAL by ranked exact match@1; report it on TEST. Batched sampling (all S
  draws in one forward) so this is laptop-feasible.
- [x] **N2 checkpoint protocol (fixed 2026-09-27 BEFORE any N2 result was seen):** per seed two
  candidates = best fixed-val-loss checkpoint and final-epoch checkpoint; sample 16 draws on VAL with
  each (g2_rank sample), score with the N1 rule (lowest-LJ unrelaxed); keep the higher-val candidate per
  seed (tie -> best-val-loss); then sample TEST with the 3 kept checkpoints and score ONCE with the N1
  rule. Compare against N1 test (G2 checkpoints): 7.5% per seed / 11.0% pooled.
- [x] **N2 RESULT (2026-09-27, NEGATIVE):** 3 seeds x 30 epochs, 8 draws/step, cosine LR, 16-proc
  data-parallel. Fixed-draw val loss 4.58-4.69 (predict-zero ~5.29). Val selection (protocol above):
  s0 best (8.0%), s1 final (7.0%), s2 best (6.0%) = 7.0% mean, same as G2 ckpts on the same val (7.0%).
  TEST, scored once: LJ-pick exact **5.0% per seed (7.5/4.5/3.0), 6.5% pooled** vs G2 ckpts 7.5% / 11.0%;
  any-of-16 exact 7-10.5% vs 12-14%. Lower flow-matching loss did NOT give sharper exact minima.
  -> the G2 checkpoints stay the model of record. Files: results/n2_*.json (ckpts local only, CSD-trained). Multi-draw loss per step (several (R0,t) pairs per crystal
  through the batched forward), cosine LR schedule, more epochs; select checkpoint on VAL loss AND
  val ranked match; re-score TEST with the N1 selector frozen.
- [ ] **N3 — G3 benchmark** (was: new model / MolCrystalFlow retrained / classical / old symmc-flow on the
  same test split). **Pre-registered protocol: `tasks/n3_protocol.md` (v3, frozen 2026-09-28 after three
  hostile review rounds).** Key changes from this line:
  - The task is stated as orientation completion (oracle lattice + ops + c0), not CSP.
  - The confirmatory test set is a FRESH TEST-B of ≈1000 crystals from the unused 4,500 CIFs of the same
    seed-0 CSD stream. The 200-crystal split becomes DEV-TEST, secondary, with disclosure.
  - Comparators:
    - H1: MCF-R and MCF-R+G (lattice and centroids held at truth, plus a gauge scan);
    - H2: a UMA-OMC physics search;
    - H3: the old architecture on O(3) items.
  - Statistics: a crystal-level exact sign-flip test with Holm.
  - The homochiral-driver note applies only to MCF's CSP driver (Table R), not to any arm.
