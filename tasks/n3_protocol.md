# N3 / G3 benchmark — pre-registered protocol (v3, 2026-09-28)

**Status: FROZEN at the commit that adds this file. No N3 arm has been trained, sampled or scored.**
Changes after this commit are made only by dated amendments at the end of this file (§9). Each
amendment states whether any N3 output existed when it was made.

Provenance:
- v1 draft (`tasks/archive/n3_protocol_v1_draft-archive-2026-09-28.md`) was reviewed by five hostile
  lenses (MolCrystalFlow author, statistician, strawman-baseline referee, implementing engineer, ACS Omega
  referee). They raised 48 issues, and every blocker and major was independently re-checked against code
  and data.
- v2 (`tasks/archive/n3_protocol_v2-archive-2026-09-28.md`) adopted the checked fixes. A second review
  (consistency, fairness of the new parts, implementability; 38 issues, each blocker and major
  re-checked) led to v3.
- A third pass on v3, limited to blockers and majors, found 4 majors: the diagnostic-(a) schedule, the
  overrun rule, the Gate C-a claim consequence, and on-box matching of OURS. Their checked fixes are
  applied below.

"OURS" is the G2 learned asym-orientation flow (`scripts/g2_learned_flow.py`, checkpoints
`results/vast_g2/eval_s{0,1,2}.pt`) with the N1 selector (lowest unrelaxed steric-LJ draw). "MCF" is
MolCrystalFlow @3c493f8 (github.com/Liu-Group-UF/MolCrystalFlow, arXiv 2602.16020v3).

---------------------------------------------------------------------------------------------------
## 0. Task and scope

**Task.** ORIENTATION COMPLETION of a Z'=1 general-position molecular crystal.
- Given: the true lattice; the true space-group operations in the crystal's own CSD setting; the true
  asymmetric-unit centroid c0; and the rigid true conformer.
- Predict: R_asym ∈ SO(3) (3 degrees of freedom), so that exact space-group expansion reproduces the
  experimental crystal.
- The ops come from `SpacegroupAnalyzer(symprec=0.1)` on the true structure. They also fix Z = K, Z'=1
  and the handedness of every copy (det W_k).

**This is NOT crystal-structure prediction.** Lattice, space group, Z, Z' and molecular position are oracle
inputs, and every claim built on N3 is limited to this task. The random-orientation floor (Table 1 row 1:
the §3.3 iii-u draws) is the pre-registered evidence that the task is not trivial.

**Deviation from plan, recorded before any N3 run.** Plan G3 (`omega_rebuild_plan.md`) was titled
"fair end-to-end benchmark", and Paper A was to learn lattice + centroid + R_asym. Only the R_asym model
exists (G2/N1/N2). N3 therefore benchmarks orientation completion. No end-to-end number for OURS exists
or is claimed.

**Table 0. Information per arm (printed in the paper).** In the MCF columns, "copies" means z_MCF, the
number of molecules in the exported MCF cell (§3.2 export step 2): z_MCF = K for primitive lattices and
K/mult for centred ones.

| | OURS / CLASSICAL | OLD iv-P | MCF-R (+G) | MCF-A (context) |
|---|---|---|---|---|
| lattice | true | true | true (primitive if centred) | generated |
| centroids | c0 true; copies from the ops | all K true | all z_MCF true | generated |
| SG ops | given, enforced exactly | given as per-copy op labels | not given (recoverable from inputs, see note) | not given |
| copy count | K via the ops | K given | z_MCF given | z_MCF given |
| per-copy handedness | det W_k | mirrored per-copy local conformer | mirrored per-copy local + chi | same as MCF-R |
| predicted | R_asym (3 DOF) | K orientations | z_MCF orientations | lattice, centroids, orientations |
| copies built by | exact SG expansion | independent outputs, attention-coupled | independent outputs | independent outputs |

Note on MCF-R information:
- Run spglib on the true centroids and keep only the ops consistent with each copy's handedness (which
  MCF-R holds through chi). The copy-to-copy rotation parts are then recovered uniquely and correctly in
  198/200 DEV-TEST crystals. In the other 2 the truth is among the candidates (review measurement,
  conventional cells).
- So the information is matched up to that residual. What differs is that OURS is given the ops
  explicitly and enforces them. H1 measures "explicit, enforced symmetry plus our learner" as a bundle.
- The ablation (ii-S) separates the two. It is descriptive only.

---------------------------------------------------------------------------------------------------
## 1. Data sets, splits, leakage, failures, disclosure

### 1.1 Sets
- **TRAIN (1687), VAL (100), DEV-TEST (200)** are unchanged. They are the seed-0 split of
  `data/csd_mol/g2_asym.pt`: DEV-TEST = perm[:200], VAL = perm[200:300], TRAIN = perm[300:].
- **TEST-B (confirmatory) and SEL (selection supplement)** are new and come from the fresh pool.
  - The pool is rows 3501-8000 (4,500 refcodes) of `data/csd_mol_scale_big/manifest.csv`. Its first
    3,500 rows equal `data/csd_mol/manifest.csv` row for row. The 4,500 are therefore the direct
    continuation of the same seed-0 `scripts/csd_export.py` CSD stream, with the same filters.
  - No step of G1-N2 parsed or scored them. The July 2026 legacy parse is disclosed in §1.4.
- **VALSEL = VAL followed by SEL (400).** Every open knob of every non-frozen arm is selected on VALSEL.
  The exceptions are the loss-based tie-breaks and the iv-P learning rate, which use the VAL losses
  that MCF's and the OLD loop's own training code compute.
- **Build procedure** (`scripts/n3_build_testB.py`, committed with amendment A1). All steps are
  input-side.
  1. Copy the 4,500 CIFs to `data/csd_testB/cif` (gitignored). Run `scripts/g1_o3_reparse.py` on all of
     them as a separate run, with G1 defaults: max_mols 16, max_atoms 64, symprec 0.1, conf_tol 0.3,
     cif_timeout 300. It keeps its own conformer registries and never mixes with csd_mol.
     - This step is NOT order-invariant. G1's rigidity gate compares each molecule with the
       first-registered conformer of its species in the same worker, and the registries are per worker
       and per 250-CIF part. So a species that recurs within a part can be kept or rejected depending on
       worker scheduling.
     - SEL and TEST-B are therefore defined by the frozen lists of A1, not by re-running the build.
     - A1 records the SHA-256 of the G1 cache and of every part file, and the number of pool crystals
       that share a WL species with another pool crystal.
  2. Keep items with `asym_item(it) is not None` (equal atom counts; n_mol = K = number of spglib ops).
     Drop items whose G1 oracle rebuild failed. Then apply the EXPAND ORACLE, keeping an item only if
     both checks pass:
     - (i) Orbit check. Let orb_k = W_k c0 + t_k, with distances measured as the max-abs fractional
       component after min-image wrapping. Every orb_k lies within 1e-3 of a stored copy centroid, and
       no two orb_k lie within 1e-3 of each other. Together with n_mol = K, this makes slot → op a
       bijection.
     - (ii) `StructureMatcher().fit(to_structure(orig), to_structure(*expand(a, a['R0'])))` is True. It
       runs to completion with no wall-clock limit.
     - Why step 2 is needed: n_mol = K does not guarantee a general position. In the dev corpus,
       9/1,987 items fail (i). All 9 are in TRAIN, none in VAL or DEV-TEST. In 8 of them two orbit
       points coincide, i.e. the molecule sits on a symmetry element; the 9th has a residual of
       3.6e-3.
     - Good items have residual ≤ 5e-8 and orbit separation ≥ 0.149.
     - The filter lives in the build script, not in `asym_item`, so `g2_asym.pt` and its seed-0 split
       are unchanged.
  3. **De-leak.**
     - (a) Drop any item whose 6-letter CSD family (refcode[:6]) occurs among the 3,500 dev-corpus
       refcodes.
     - (b) Drop any item whose molecule Weisfeiler-Lehman hash occurs among the 1,987 `g2_asym.pt`
       items (TRAIN, VAL and DEV-TEST). DEV-TEST is included because it informed the model-of-record
       choice (§1.4c).
     - Bond graph: the `graph(X, Z)` of `scripts/n3_chirality.py` (ported from the review's
       g3_chiral2.py). Radii are pymatgen `JmolNN().el_radius`, and a bond exists iff
       0.4 Å < d ≤ r_i + r_j + 0.45 Å on the asym local conformer.
     - Hash = `networkx.weisfeiler_lehman_graph_hash(g, node_attr='element')` (3 iterations, digest
       16). All sets are hashed in one process.
  4. **Order.** Sort the survivors by refcode. Apply
     `perm = torch.randperm(n, generator=torch.Generator().manual_seed(20261001))`. Walk the
     permutation and keep only the FIRST item of each 6-letter family, giving the ordered list P.
  5. **Split.** SEL = P[:300], TEST-B = P[300:1300] (if |P| < 1300, TEST-B = P[300:]). If |P| < 900,
     stop and re-plan by amendment before anything is sampled.
  6. **Freeze.** Commit `tasks/n3_sets/testB_refcodes.txt` and `sel_refcodes.txt` with their SHA-256
     (amendment A1) BEFORE any arm produces any output on them. Refcodes are already public in the
     committed manifests.
- **Input-side only before sampling.** Nothing but these operations touches TEST-B / SEL before
  sampling:
  - the step-2 truth rebuild and expand oracle;
  - the §2.6 stratum labels;
  - the MCF export and Gate F;
  - the legacy keep-flag and legacy oracle (§5);
  - the OMC25-overlap labels (§3.3);
  - Table R.
- **Truth** is `to_structure(orig)` of the item, for every arm and every set.

### 1.2 Forbidden inputs
- No arm uses a checkpoint trained on `data/csd_mol_scale` or `data/csd_mol_scale_big`. That covers every
  legacy file in `checkpoints/`.
- No arm trains on anything but the 1,687 TRAIN refcodes.
- MCF's released checkpoints are used only in Gate C, never for a reported arm.

### 1.3 Denominators, failures, timeouts
- The denominator is always |set|.
- An ARM failure is a MISS for that arm, listed by refcode. Arm failures are: cannot represent, cannot
  preprocess, generator error, non-finite coordinates.
- A HARNESS failure is the matcher or energy code crashing or running slow on a valid structure. It is
  re-run in its own process with no wall-clock limit. A failure that recurs identically in a fresh
  process is DETERMINISTIC:
  - a deterministic matcher failure scores that candidate as a non-match;
  - FF* cases follow §3.3.
  No crystal is ever dropped for a harness failure. Every re-run is logged.
- No per-crystal wall-clock limit applies to any arm's generation, search, minimisation or selection, and
  no crystal is ever counted as a MISS for running long. A slow matcher or energy job is a HARNESS failure
  under the rule above.
  - The OURS/OURS-N2 sampler runs without g2_rank's per-crystal `--timeout` (default 3600 s; `score()`
    drops timed-out rows).
  - Compute is reduced only by the step-4 degrade decisions.
- **Blinding.** No arm's output is matched against the truth at sampling time or on the box.
  - `n3_score.py` computes every match: VALSEL in step 5, and TEST-B and DEV-TEST at step 9 only. Stored
    VAL and DEV-TEST N1/N2 draws are re-matched from their R.
  - The `exact` flags already in those stored records are not used for any reported number.

### 1.4 Disclosure (printed in the SI; DEV-TEST is labelled "used during development, not confirmatory")
- (a) DEV-TEST was scored by G2 baselines, G2 eval, N1 and N2.
- (b) The N1 "rank before relaxing" idea came from a DEV-TEST diagnostic: relaxed-truth drift, median
  14.4°.
- (c) OURS' model of record was fixed after DEV-TEST had been seen.
  - The N2 retrain and the G2 checkpoints tied on VAL picked@1 (7.0 vs 7.0% per-seed mean; 9.0 vs 9.0
    pooled).
  - Both were then scored on DEV-TEST: G2 7.5%, N2 5.0% (crystal-level sign-flip p = 0.023). G2 was
    kept.
  - Applied now and labelled post hoc, the rule "incumbent unless VAL-better; tie → VAL draw@1" keeps G2
    from VAL alone: draw@1 0.96 vs 0.67%, higher in every seed; any@16 13.3 vs 10.3%.
  - Applied per seed across the two runs, the same rule would replace seed 1 with the N2 checkpoint (VAL
    7 vs 4%), and DEV-TEST would be 6.5% instead of 7.5%.
  - N2 is carried as a sensitivity family (§2.4).
- (d) OURS' DEV-TEST stratum rates were computed before the strata were frozen: mixed-handedness 8.5%,
  Sohncke 6.1%, improper-achiral 3.5% (per-seed means). The frozen refined chirality definition moves
  OURS' mixed rate from 8.2% (35/429, capped definition, 143 crystals) to 8.5% (35/411, 137 crystals),
  because OURS picks 0 of 18 on the 6 reclassified crystals.
- (e) During protocol review, DEV-TEST (and VAL) were also used for:
  - loose-tolerance rescoring of stored OURS draws;
  - Haar floors at loose tolerance;
  - homochiral-approximant matches (1/131 strict vs 80/131 at stol 0.5, partial run);
  - MCF perfect-model orientation ceilings (conventional cells, keyed on K);
  - a perfect-model orbit-LJ check (4/28 K=2 crystals recovered);
  - steric-press drift checks;
  - a VAL study of the iii-s screen: a start within 12.5° of the truth is among the 64 lowest-LJ grid
    points in 91/98 VAL crystals;
  - X–H bond-length statistics.
  None of these choose anything below.
- (f) The legacy det+1 pipeline parsed the TEST-B pool in July 2026, and the old paper's scale runs
  trained on it. No legacy checkpoint is used (§1.2).
- (g) Build step 1 (the G1 parse of the 4,500 pool CIFs) was started on 2026-09-28, before this file was
  committed. It is input-side only and produced no arm output. Its start time, command line and worker
  count are recorded in A1.
- (h) H1-H3 are decided on TEST-B only. Everything on DEV-TEST is secondary.

---------------------------------------------------------------------------------------------------
## 2. Matchers, estimands, selectors, statistics, strata

### 2.1 Matchers
- **PRIMARY.** `StructureMatcher()` defaults (ltol 0.2, stol 0.3, angle_tol 5, scale=True), computed as
  `fit(truth, candidate)`.
  - The truth is unrelaxed. The candidate is the picked output exactly as the arm emits it, with no
    post-hoc relaxation (iii-p's output is pressed and iii-s's is FF*-minimised, by construction).
  - This holds for every arm, table and hypothesis. The one exception: when §3.3 fixes H2 FORM =
    BASIN, the H2 contrast (and the rows §3.3 ties to it) is scored in BASIN form.
- **SECONDARY.** MCF's rule, `StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=s)`: a
  candidate matches iff `get_rms_dist(truth, candidate) is not None`.
  - s ∈ {0.5, 0.8}. Each stol is a separate run, since stol is a pre-filter, not a threshold on
    max_dist.
  - The secondary matchers score each arm's PICKS (per seed, and the ensemble pick), plus the iii-u Haar
    floor over all draws. MCF-A also has its native sweep (§3.2).
- **One scoring script,** `scripts/n3_score.py`, scores every arm. It converts each arm's output to a
  pymatgen Structure through a per-arm adapter, and each adapter is unit-tested by the oracle gates.

### 2.2 Estimands, indexing, seeds
- **HEADLINE: seed-mean picked@1.**
  - For each seed, ONE trained model (or one Haar seed base) draws S = 16 samples per crystal, and the
    arm's VALSEL-chosen selector (§2.3) makes ONE pick.
  - picked@1 is averaged over the 3 seeds. Deterministic arms (iii-s) have m = 1.
- **Secondary, descriptive:**
  - draw@1: the mean over draws of per-draw match. It is the exact expectation of a random pick.
  - any@16: the oracle upper bound.
  - Ensemble picked@1: the selector over all 48 draws of the 3 seeds, labelled "3-model ensemble, 48
    draws". It is never quoted as a single model's accuracy.
  - any@48.
  - Continuous: the picked candidate's RMS from `StructureMatcher(ltol=0.3, stol=1.0, angle_tol=10,
    scale=False).get_rms_dist` (no match = +∞), shown as an ECDF per arm.
  - For R_asym arms, the geodesic error of the picked R_asym to R0, minimised over the molecule's proper
    point group (PointGroupAnalyzer, tol 0.3 Å).
- **Indexing.** i = the 0-based position of a crystal in its frozen list:
  - TEST-B in `testB_refcodes.txt` order;
  - SEL in `sel_refcodes.txt` order;
  - VAL in perm[200:300] order;
  - DEV-TEST in perm[:200] order;
  - VALSEL = VAL then SEL (i = 0-399).
  Split codes for every arm: 0 VALSEL, 1 TEST-B, 2 DEV-TEST.
- **Seeds.**
  - OURS and OURS-N2 use a per-crystal seed of 80000 + i on TEST-B and 90000 + i on SEL (i within
    SEL). On VAL and DEV-TEST they reuse the stored N1/N2 draws (g2_rank bases 40000 + i and
    50000 + i).
  - MCF-R: `inference.seed = 100000 + 1000·s + split`.
  - MCF-A: `inference.seed = 200000 + 1000·s + split`.
  - MCF-A prior floor: `torch.manual_seed` and `np.random.seed(300000 + 1000·s + split)`.
  - iii-u: `torch.Generator().manual_seed(600000 + 10000·b + 1000·split + i)`.
  - iv-P: `torch.manual_seed(700000 + 10000·s + 1000·split + i)`.

### 2.3 Selector menu (identical for every arm, with each arm's own analogue)
- **RANDOM.** Scored as its exact expectation: h = (matching draws)/16 = draw@1. No RNG is used.
- **LJ.** The lowest steric-LJ energy of the exact structure that is scored.
  - Code: `g2_asym_baselines.energy` (rigid_press `packing_energy_nbr`, contact 0.90, cutoff 6.0 Å).
  - It is computed on whole-molecule Cartesian coordinates, over real copies and real atoms only.
  - For pressed structures (iii-p) it is the POST-press energy.
  - For MCF output it goes through the Gate-F5 adapter.
  - Non-finite energies rank last. Exact ties go to the lowest draw index.
- **RESID** (learned arms only). One extra forward pass on the RETURNED sample with time input
  t = 0.975 (the torque_end convention, g2_rank.py:69). Take the mean over real copies and pick the
  lowest.
  - OURS: ‖torque field‖ (N1's torque_end).
  - OLD: ‖v_R‖ on the returned per-copy rotations.
  - MCF: the geodesic angle between that pass's predicted clean rotation and the returned
    `pred_rotmats`. The pass uses the saved final state (pred_rotmats, pred_trans, lattices) and
    so3_t = r3_t = l_t = 0.975, which lies inside MCF's training range t ∈ [0.01, 0.99].
  - MCF-R+G and ii-S candidates carry the RESID of their source draw. For +G this is exact whenever
    diagnostic (a) passes.
- **MODE@r**, r ∈ {0.25, 0.5, 0.75, 1.0, 1.5} Å. N1's endpoint-mode frequency (element-matched Chamfer)
  over the atoms of all copies. Ties go to the lowest RESID, or the lowest LJ for classical arms.
  - MODE@r is not offered to MCF-A, whose draws share no lattice or centroids.
- **Choice.** Each arm chooses its selector on its own VALSEL draws by seed-summed hits under the
  primary matcher. Tie preference: LJ > RESID > MODE@r (smaller r first) > RANDOM.
- **OURS keeps N1's LJ selector** (chosen on VAL). Re-applying this rule to the N1 VAL draws gives LJ 21
  hits vs RANDOM 2.9 vs RESID 2, so nothing changes.
- **iii-s** has a fixed physics selector (the lowest FF* minimum, §3.3). It is not chosen from the menu.

### 2.4 Statistics (TEST-B; primary matcher; headline estimand)
- **Unit of analysis.** The crystal.
  - For arm A with m_A seeds, h_i^A = the mean over seeds of the picked@1 outcome. A RANDOM selector
    contributes its expectation.
  - d_i = h_i^OURS − h_i^A, and Δ = mean_i d_i.
  - Seed indices are never paired across arms.
- **Test.** Two-sided paired sign-flip on Σ d_i, computed exactly by convolution. Every d_i is a
  multiple of 1/48, and zeros drop out. For binary single-seed data the test equals exact McNemar.
- **Primary family: {H1, H2, H3}.** Holm step-down at family-wise α = 0.05.
- **Verdicts for H2 and H3.**
  - "OURS better": Δ > 0 and Holm-adjusted p < 0.05.
  - "ARM better": Δ < 0 and adjusted p < 0.05.
  - Otherwise: "difference not resolved at n = … (95% CI [a, b] pp)". Never called a tie or
    equivalence.
- **H1 (intersection-union).** Let (Δ_R, p_R) and (Δ_G, p_G) be OURS vs MCF-R and OURS vs MCF-R+G.
  - H1 enters the family ONCE with p_H1 = max(p_R, p_G), whatever the direction. Component p-values
    never enter the family on their own.
  - Δ_H1 = min(Δ_R, Δ_G), quoted with that component's CI.
  - "OURS better" (H1 supported): Δ_R > 0 AND Δ_G > 0 AND Holm-adjusted p_H1 < 0.05.
  - Adjusted p_H1 < 0.05 with both Δ < 0: "MCF better (both variants)".
  - Adjusted p_H1 < 0.05 with opposite signs: "split: OURS better than <X>; <Y> better than OURS". H1 is
    not supported.
  - Otherwise: "not resolved", with both components' Δ and CI printed.
- **Effect size.** Δ in percentage points, with a paired crystal-bootstrap BCa 95% CI (10,000
  resamples, RNG seed 0).
- **Training-seed sensitivity** (reported, not a gate).
  - Two-stage bootstrap: first resample each arm's seeds with replacement, independently per arm, then
    resample crystals with the same indices for both arms. 10,000 replicates, percentile 95%.
  - If Holm supports H_k but this interval contains 0, the claim is worded "for the trained models; not
    shown to exceed seed-to-seed training variation".
- **OURS-N2 sensitivity family.**
  - H1-H3 are re-tested with OURS-N2 in place of OURS, as a SEPARATE secondary family with its own Holm
    step-down and the same verdict rules.
  - It never enters, reorders or changes the primary family.
  - If a primary "OURS better" is not also "OURS better" for N2, it is worded "holds for the model of
    record, not for the 30-epoch retrain". A hypothesis that holds only for N2 is not claimed.
- **No significance claims** are made for secondary matchers, draw@1, any@k, ensemble metrics,
  continuous endpoints, strata, DEV-TEST, MCF-A, ablations or diagnostics. They get estimates with 95%
  CIs only, and per-seed McNemar p-values are not reported as tests.
  - The one exception is the §4 ANCHORED check on H2. It can only RESTRICT the wording of an H2 result
    that Holm supports; it never creates a claim.
- **Design sensitivity.** `scripts/n3_power.py` and its printed table for the realised |TEST-B| are
  committed with A1.
  - Model: numpy default_rng(7), 4,000 replicates.
    - OURS: per-crystal p_i ~ Beta(1.056, 13.0), fitted to N1's 161/33/6/0 crystals-by-seeds-hit; 3
      seeds; Bernoulli picks.
    - Comparator: an independent Beta(1.056, 13.0) scaled by p_A/0.075, with 3 seeds (for iii-s, one
      Bernoulli per crystal).
    - The exact sign-flip test of §2.4. For H1, p = the max over two comparators at the same p_A.
  - Output: power at α = 0.05 and 0.05/3 for Δ ∈ {1.5, 2, 2.5, 3, 4} pp.
  - Planning values (same model, normal approximation), n = 1000: power 0.85 at Δ = 2 pp (0.71 at
    α/3); 0.97 / 0.92 at 2.5 pp. At n = 200, 80% power needs Δ ≈ 4-4.5 pp.
  - A non-rejection below the MDE is expected. It is not evidence of parity.

### 2.5 DEV-TEST rows
Every arm that runs on TEST-B is also run on DEV-TEST with the same frozen configuration. These rows form
a secondary continuity table, carry the §1.4 disclosure, and have no significance claims.

### 2.6 Strata (frozen; descriptive only; per-seed k/n and ensemble k/n with Wilson 95% CI; strata with n < 30 are flagged "too small to interpret"; rows = every Table 1 row)
- **K** (number of spglib ops in the CSD cell): ≤2 / 3-4 / ≥6. DEV-TEST counts: 55 / 119 / 26.
- **z_MCF** (MCF's invariance class; MCF rows): 1 / 2 / ≥3.
- **Class:** centrosymmetric / Sohncke / other (gemmi).
- **CHIRAL** is a property of the asym conformer: refined mirror RMSD > 0.3 Å.
  - Refined mirror RMSD = the minimum of (i) det+1 Kabsch over the first 200 element-matched graph
    automorphisms and (ii) element-blocked Hungarian ICP, 30 iterations, from 5 deterministic starts
    (the 4 sign-diagonal rotations and the best rotation from (i)).
  - `scripts/n3_chirality.py` is the review's g3_chiral2.py with only I/O changed; it is deterministic.
  - Labels go to a gitignored file, whose SHA-256 is recorded in A1. DEV-TEST: 178 chiral.
  - Sensitivity rows: thresholds 0.1 / 0.5 Å; the parser-capped (max_iso 200) definition alone.
- **MIXED-HANDEDNESS** = at least one improper op (det W = −1) AND CHIRAL (DEV-TEST 137).
  **IMPROPER-ACHIRAL** = an improper op AND not CHIRAL (DEV-TEST 19).
  - MIXED-HANDEDNESS is interpreted only under the primary matcher.
  - **No handedness advantage is claimed over MCF-R, MCF-A or CLASSICAL.** The only structural
    handedness contrast is the old det+1 representation (Table R).
- **LEGACY-UNREPRESENTABLE** = old_ok is False (tuple field 1) in the G1 part files
  `<cache>.parts/part_*.pt` of the run that built the set: `data/csd_testB` for TEST-B/SEL,
  `data/csd_mol` for DEV-TEST.
- **FORMULA-DISJOINT** = the molecular formula does not occur in TRAIN.
- **OMC25-NONOVERLAP** (§3.3).
- **FF* truth label** (H2 pair only; §3.3): ANCHORED / DRIFTED / UNRESOLVED.

---------------------------------------------------------------------------------------------------
## 3. Arms

### 3.1 (i) OURS (frozen; arm of record)
- G2 checkpoints `eval_s{0,1,2}.pt`.
- 16 draws, 40 Euler steps, the batched sampler (`g2_rank.sample_batch`), and N1's LJ rule.
- Seeds as in §2.2.
- Each draw record carries R, e_lj and torque_end. The sampler runs no matcher (§1.3 Blinding).
- **OURS-N2 sensitivity row:** the three N2 checkpoints chosen on VAL, with identical sampling, selector
  and scoring, tested as its own family (§2.4).

### 3.2 (ii) MolCrystalFlow

**Code base.** MCF @3c493f8, in a gitignored fork whose diff is published in the SI. Only these changes
are permitted:
- **P1.** Add {12 Mg, 33 As, 34 Se, 52 Te} → indices 12-15 to the three atomic-number maps:
  `data-preprocess/thurlemann23/common.py:35`, `molcrystalflow/data/utils.py:13` and
  `molcrystalflow/data/dataset.py:13`. The inverses are derived from them. Set
  `model.bb_embedder.num_atom_types = 16` in our yamls.
- **P2.** In `interpolant.corrupt_batch`, when trans.corrupt=False, set `b_trans = zeros_like(trans_1)`
  (it is unbound at interpolant.py:419-424).
- **P3 (MCF-R only).** Key `inference.gt_lattice_trans` (set with `++inference.gt_lattice_trans=true`).
  When it is on:
  - trans_0 = trans_1 and lattice_0 = lattice_1;
  - both are overwritten with the truth after every Euler step;
  - pred_trans_1 = trans_1 and pred_lattice_1 = lattice_1 at final assembly (molcrystalflow.py:396-406).
  Rotations are untouched.
- **P4.** The GPUtil device calls (`experiments/utils.py:18`, `inference.py:53,100`) return [0].
- **P5.** `pl.seed_everything(experiment.seed)` before FlowModule is constructed. The released train.py
  builds the model at :48 and seeds at :50-52.
- **P6 (speed, optional).** A vectorised `_symmetric_so3` prior and a scatter-softmax
  `AttentionPooling`. Used only if, on the box GPU:
  - (a) the prior is bit-identical under a fixed numpy seed; and
  - (b) pooling outputs and input gradients agree to 1e-5 on 3 real TRAIN batches.
  Which version ran is reported.

**Harness conventions.**
- Every training run sets WANDB_MODE=offline (never `experiment.debug=True`, which skips checkpoints) and
  `experiment.wandb.name=n3_{mcfR,mcfA}_s{s}`, because `runs/<name>/ckpt` is otherwise shared.
- The offline wandb directories hold metrics only. They are pulled back before step 8 and are the
  source of the reported LR trace, valid/loss curve and 9th-reduction step.
- save_trajectories=False.

**Export** (`scripts/n3_mcf_export.py`, committed; pickles gitignored). For each item:
1. Start from `orig`. Whole molecules are X_k = centroid_k·L + orient_k·local_k, over real slots.
2. **Centred crystals.** A crystal is centred when its truth ops contain any W = I with t ≠ 0 (A, B, C,
   F, I or R); mult = 1 + the number of such ops.
   - Centred crystals are exported in the primitive cell. P = `get_primitive_structure().lattice.matrix`
     of `to_structure(orig)`; this keeps the Cartesian frame.
   - Assert that M0 = L·P⁻¹ is integral to 1e-6 and |det M0| = mult.
   - Atoms are never wrapped or re-detected. Walk k = 0..K−1 and keep the whole molecule X_k unless its
     centroid equals an already-kept centroid modulo P (1e-4 fractional).
   - Assert z_MCF = K/mult. For non-centred crystals, P = L and z_MCF = K.
   - Dev-corpus counts: centred DEV-TEST 12/200, VAL 6/100, TRAIN 136/1687.
3. **Standardize.** Build ASE `Atoms(numbers, positions = the kept X, cell = P)` and call
   `sort_cell_left_matrix`. It returns only Atoms.
   - Recover Q by least squares from X·Q = X_rot. Assert max|X·Q − X_rot| < 1e-6 Å and
     ||det Q| − 1| < 1e-9.
   - Store per crystal: the kept slot list, Q, det Q, and the integer supercell matrix
     M = round(L·Q·L_m⁻¹). Assert a rounding error < 1e-6 and |det M| = mult.
4. `common.preprocess_structure(lattice, positions, types, bb_indices = our kept-slot index)`. MCF's
   `get_molecule_groups` is not used: on DEV-TEST it merges one crystal and raises on Te in another.
5. **Per-copy features**, through MCF's own functions, unchanged, once per copy: obabel xyz→sdf,
   `SDMolSupplier(removeHs=False, sanitize=False)`, `fix_mol_bonds(mol, MolToSmiles(mol))`,
   `extract_features_single_molecule`, `.get(k, 0)`.
   - MCF's unfixed-molecule fallback is kept.
   - The xyz→sdf count is asserted 1:1.
   - obabel must succeed (`obabel -V`), from openbabel-wheel or conda.
6. `renormalize_single_datapoint`.
7. MCF's batch drivers are not used. They stop at the first exception and delete mismatched entries.
8. **Output.** Exactly |set| entries (asserted), plus a refcode-order sidecar. Directories:
   - `n3_mcf/trainval`;
   - `n3_mcf/valsel_as_test`: VALSEL written as `test_molcrystal_normalized.pkl.gz`, because
     inference.py:106 hard-codes test.pt;
   - `n3_mcf/test` (TEST-B);
   - `n3_mcf/devtest`.
- **MCF lattice-prior refit** (MCF-A): `lattice_fit.compute_lattice_stats` on the list of standardized
  TRAIN Atoms. Assert n = 1687 and loc sorted.

**Gate F (export), before any GPU time.** Counts are committed as `results/n3/gateF.json`.
- F1: set counts and sidecar order; all atom types satisfy 0 ≤ t < 16; every copy has basic (2),
  chemical (8) and geometric (4) arrays; z_MCF·mult = K.
- F2: species are decoded from the PICKLE through the scorer's inverse map.
  - `assemble_coords` reproduces the standardized input atom by atom to ≤ 1e-3 Å (mod one lattice
    translation per molecule) for 100% of items.
  - StructureMatcher() of the decoded structure vs the truth passes for 100% of VAL, SEL, TEST-B and
    DEV-TEST.
- F3: chi vs `orig` parity (up to a global relabel). Reported, not gated.
- F4: per-copy feature status (obabel failure / fix strategy 1-3 / fallback / descriptor exception).
  GATE: copies with an all-zero basic+chemical vector ≤ 1% per set.
- F5: the LJ adapter. Every MCF LJ evaluation goes through one adapter.
  - (1) Take the MCF structure: lattice L_gen, z_MCF whole molecules with orient = I and
    local = X_k − mean X_k.
  - (2) Re-express it in the stored supercell: L_c = M·L_gen. Copy each molecule to the translations
    n·L_gen for every integer n with n·M⁻¹ ∈ [0,1)³. There are exactly mult such n (asserted).
    Centroids are fractional in L_c.
  - (3) Call energy() unchanged. For non-centred crystals the adapter is the identity.
  - GATE: the adapter energy of the pickle's own poses equals energy() of `orig` to 1e-4 relative, for
    all VALSEL + TEST-B + DEV-TEST crystals.
  - Scaling the primitive-cell energy by K/z_MCF is NOT used. energy()'s image shell depends on the
    cell choice, and scaling misses this gate on 4/12 centred DEV-TEST crystals.
- F6 (scorer oracle): a predictions file built from the pickle's own poses, run through `n3_score.py`,
  gives 100% picked@1, draw@1 and any@16 under every matcher.
- **On failure.** Fix the glue and re-run the gate. A test crystal that still fails is an MCF MISS,
  listed by refcode, and every MCF comparison is then also reported on the passing subset.
- **Integrity.** SHA-256 of every pickle, re-checked on the box before training and before sampling.

**Gate C (competence and patch equivalence), on the box before any N3 MCF training.**
- **C-a (competence).** Released Thurlemann checkpoint and data (Zenodo 19673190:
  `model-checkpoints/thurlemann23/{best.ckpt, config.yaml}` and
  `thurlemann23/preprocessed/normalized/*`).
  - Run with our patched code, flag OFF and stock Thurlemann config (num_atom_types 7), on the
    released TEST pickle. Also run the released VAL pickle, copied into its own dir as
    `test_molcrystal_normalized.pkl.gz`.
  - Sampling: 50 steps, s_uF 9, s_uR 3, 5 runs × 10 samples (inference seeds 0-4).
  - Score any-of-10 with `n3_score.py` at MCF's settings AND with MCF's scorer, once per stol:
    `python molcrystalflow/experiments/run_structure_matching.py --pt_file <dir>/predictions_10.pt --num_samples 10 --stol {0.8|0.9|1.0} --ltol 0.3 --angle_tol 10 --num_cpus <n>`.
  - PASS if both hold:
    - (i) the two scorers agree on ≥ 99% of (crystal, stol) outcomes;
    - (ii) the 5-run median at stol 0.9 and 1.0 is within ±5 pp of Fig. 3a (≈30% / ≈66%, read off box
      plots at ±2 pp). stol 0.8 (≈5.5%) is reported, not gated.
  - If only VAL agrees, the gate passes and this is disclosed.
  - **On failure of (i):** fix `n3_score.py` and re-score C-a until (i) passes. Both scorers read the
    same predictions_10.pt, so every disagreement is listed and traced per crystal. No MCF arm trains
    while (i) fails.
  - **On failure of (ii) only:** diagnose (config merge, sampler knobs, split). Before any N3 MCF
    training, record the diagnosis and the TEST and VAL 5-run medians at stol 0.9 / 1.0 in a dated §9
    amendment. N3 then proceeds.
    - Every H1 sentence then carries the harness clause: "MCF-R / MCF-R+G were trained and sampled with
      our MolCrystalFlow harness; with MCF's released Thurlemann checkpoint this harness gives
      X% / Y% (TEST; VAL X' / Y') at stol 0.9 / 1.0, against the published ≈30% / ≈66% (Fig. 3a)".
    - This covers the Table 1 caption, the §4 claim-map text including the ii-S reading, and the
      abstract. No sentence names MolCrystalFlow or MCF as the H1 comparator without the clause.
    - The reproduced numbers are also printed next to Fig. 3a in the SI.
- **C-b (patch equivalence).**
  - On CPU, flag OFF, P6 disabled, stock Thurlemann config: construct FlowModule(cfg) from the patched
    and the unpatched 3c493f8 trees and load ONE shared state_dict into both. Seed random/numpy/torch
    identically before each call.
  - Require bitwise-equal forward outputs on 3 released TEST batches, and a bitwise-equal model_step
    loss after one corrupt_batch.
  - On GPU: the patched-vs-unpatched difference is no larger than unpatched-vs-unpatched at the same
    seed.
- **C-c (clamp).** With P3 ON, every output lattice and centroid equals the input truth bitwise, on the
  tensors saved in predictions_16.pt. C-c uses a predictions_16.pt from the released Thurlemann
  checkpoint with P3 ON.
- **C-b and C-c** are pass/fail, and both run before any N3 MCF training. On a failure, fix the patch
  and re-run the gate.

**(ii-R) MCF-R: lattice + all z_MCF centroids held at the truth; orientations generated.**
- **Training.** `molcrystal.yaml` as released, plus ONLY these overrides:
  - `interpolant.trans.corrupt=False interpolant.lattice.corrupt=False`;
  - `experiment.training.translation_loss_weight=0 experiment.training.cell_loss_weight=0`;
  - `experiment.seed={0,1,2}`;
  - `experiment.trainer.max_epochs=1715` (24,010 optimizer steps at ceil(1687/128) = 14 steps per
    epoch). MCF's Thurlemann run was 300 × 79 = 23,700; the released max_epochs=1000 would stop at
    14,000.
  - `experiment.lr_scheduler.patience=61`. This is a unit conversion: ReduceLROnPlateau fires on the
    (p+1)-th bad reading, i.e. 11 × 79 = 869 steps in MCF's run and 62 × 14 = 868 here.
  - Everything else stays as released: factor 0.6, min_lr 1e-6, validation every epoch on MCF's own
    random draw of VAL, and ModelCheckpoint top-3 valid/loss plus last.
- **Fixed-draw VAL loss (tie-break only).** The model_step over VAL × 8 draws, inside
  `torch.random.fork_rng()` with `torch.manual_seed(7)` and `np.random.seed(7)`. The numpy state is saved
  and restored.
- **Reported per seed:** the LR trace, the valid/loss curve, and the step of the 9th LR reduction (the
  effective floor 1.008e-6) or "not reached".
- **Sampling harness.**
  - Own yamls: `ours_inference_R.yaml` composes `ours_molcrystal_R.yaml`. inference.py merges the launch
    config OVER the checkpoint config, so the stock inference.yaml is never used.
  - Every call sets: `inference.ckpt_path`; `data.cache_dir`; `inference.num_samples=16`;
    `interpolant.sampling.num_timesteps=50`; `interpolant.rots.sample_schedule=exp`;
    `interpolant.rots.exp_rate=<s_uR>`; `interpolant.trans.scaling=9`;
    `++inference.gt_lattice_trans=true`; a unique `inference.output_dir` and `inference.inference_dir`;
    `inference.save_trajectories=False`; `data.loader.num_workers=4`; and the §2.2 seed.
  - **RESID pass** (`scripts/n3_mcf_resid.py`; no MCF code change). It runs right after every MCF-R and
    MCF-A sampling call, on the box, in the same environment:
    - load the same checkpoint and resolved config;
    - iterate MCF's unshuffled test_dataloader over the same pickle;
    - for each of the 16 draws, set the §2.3 state and run FlowModel under eval() / no_grad;
    - write `resid_16.pt` ([16, |set|]).
    It is part of the listed sampling run.
  - **Assertions** on every predictions file (and its resid file) before scoring:
    - 16 draws; |set| crystals in sidecar order (lattice_1 and num_atoms match the pickle);
    - the clamp is exact: lattice == lattice_1 (atol 1e-5) and trans == trans_1 mod 1 (atol 1e-5);
    - the resolved config holds the values above;
    - resid_16.pt has shape [16, |set|] and is finite.
    A failing pair is discarded and regenerated, never scored.
- **Selection** on VALSEL, primary matcher; frozen in `n3_selection.json`.
  - Stage A: shared knobs, pooled over the 3 seeds.
    - s_uR ∈ {1, 2, 3, 5, 10}: 3 is the paper value, 10 the released exp_rate, 1-3 the paper's optimum
      band, and 5 lies just above it.
    - Crossed with the §2.3 menu, using each seed's lowest-valid/loss checkpoint.
    - Keep the highest 3-seed summed hits.
    - Ties: summed draw@1 → summed any@16 → s_uR = 3 → selector preference.
  - Stage B: checkpoint per seed, with Stage A frozen.
    - Candidates: top-3 valid/loss and last. Keep the highest VALSEL picked@1.
    - Ties: draw@1 → any@16 → lower fixed-draw VAL loss.

**(ii-R+G) MCF-R with gauge scan (deployable; no truth used; MCF-favourable).** It uses the same trained
checkpoints (no retraining). Its sampler settings and checkpoint are selected for it (below).
- **Why.** MCF's network sees the lattice only through L Lᵀ and orientations only through R_iᵀR_j. Its
  symmetric prior conjugates copy i by g_(i mod 4) ∈ {I, C2z, C2y, C2x} in its standardized Cartesian
  frame. The sampled orientation distribution is therefore invariant under the commutant H of the g_i
  actually drawn. H depends on z_MCF, not on K:
  - z_MCF = 1: H = SO(3). The prior is Haar, so MCF-R's orientation is Haar by construction.
  - z_MCF = 2: {R_z(θ)} ∪ {R_z(θ)·C2x}, i.e. O(2) about the standardized z of the EXPORTED cell (the
    normal to its two SHORTEST cell vectors, which is NOT the CSD c*).
  - z_MCF ≥ 3: D2 = {I, C2x, C2y, C2z}.
  - Dev counts (K, z_MCF), DEV-TEST: (2,2) 55, (4,2) 1, (4,4) 118, (8,4) 9, (8,8) 15, (16,4) 2. In the
    eligible corpus, (4,2) is 41/1987 and (1,1) is 6/1987.
- **Procedure.**
  1. For each draw, evaluate the unrelaxed LJ (F5 adapter) of h·{R_k} with lattice and centroids fixed,
     for every h in H:
     - z_MCF = 2: both cosets on a 2° grid (360 members);
     - z_MCF ≥ 3: the 4 D2 elements;
     - z_MCF = 1: no scan, so MCF-R+G = MCF-R there. A scan over all of SO(3) would discard MCF's output
       and turn the arm into the physics search. This is disclosed.
  2. Keep the lowest-LJ member.
  3. Selection with MCF-R's rules and tie-breaks:
     - Stage A (s_uR × selector, pooled over seeds) on the gauge-scanned Stage-A draws already sampled
       for MCF-R.
     - Stage B (top-3 + last per seed) at +G's Stage-A s_uR. VALSEL is re-sampled only if that s_uR
       differs from MCF-R's.
     - TEST-B / DEV-TEST are sampled separately for +G only if its (checkpoint, s_uR) differ from
       MCF-R's.
- **LJ evaluations per crystal, disclosed for every arm:** OURS 16; MCF-R 16; MCF-R+G 5,760 (z_MCF = 2),
  64 (z_MCF ≥ 3) or 16 (z_MCF = 1).

**(ii-S) MCF-R + post-hoc SG expansion (ABLATION; descriptive; no test; read only through the §4 outcome
map).** Runs on CPU from the MCF-R+G draws.
1. Map to our frame. X = X_MCF·Qᵀ; for centred crystals, the copies are expanded to the conventional cell
   through M first.
2. Assign each slot to op k by centroid match to W_k c0 + t_k. This is a bijection after build step 2.
3. Y_k = Rc_k⁻¹(X_k − cent_k).
4. R^(k) = det+1 Kabsch of the asym local onto Y_k. On true poses this recovers R0 in 200/200, max
   0.014°.
5. Expand each R^(k) exactly and keep the lowest-LJ one, giving ONE candidate per draw.
- The selector is chosen on VALSEL.
- It is labelled "MCF orientations inside our exact SG expansion".

**(ii-A) MCF-A: released joint objective (CONTEXT ONLY; separate table; never in H1-H3).**
- Training: same export, P1/P2/P4/P5/P6, the same step budget and patience conversion, 3 seeds, and the
  refit lattice prior.
- Harness: `ours_inference_A.yaml` composes `ours_molcrystal_A.yaml`. The merged config is asserted
  (lattice loc/scale, num_atom_types, cache_dir, knobs), and the RESID pass runs.
- Metric: MCF's native get_rms_dist rule at stol {0.5, 0.8, 1.0}, plus the primary matcher.
  any-of-10 = the first 10 of the 16 draws.
- Selection on VALSEL:
  - Stage 1: the checkpoint at (s_uF 9, s_uR 3), by any-of-10 at stol 1.0; ties → stol 0.8 → lower
    valid/loss.
  - Stage 2: (s_uF, s_uR) ∈ {5, 9, 13} × {1, 2, 3}; ties → (9, 3).
  - Selector from {RANDOM, LJ, RESID} by picked@1 at stol 1.0.
- LJ on generated cells: draws with a minimum interplanar spacing < 2.5 Å get e_lj = +∞, and their
  count is reported.
- Prior floor: at every setting, the refit lognormal lattice + uniform centroids + symmetric-prior
  rotations, 16 draws per crystal (§2.2 seed).

**MCF diagnostics (pre-registered; oracle items labelled as such; not results).**
- **(a) Equivariance.** On every selected MCF-R and MCF-A checkpoint, model and inputs in float64 on CPU:
  10 VALSEL crystals × 3 random G ∈ SO(3).
  - openfold's Rotation and Rigid cast rotations to float32 (`rigid_utils.py:325-329`, `:899`), so
    pred_R is float32 by construction. The network's own outputs stay float64: q_vec (captured with a
    forward hook on `rot_update`), lattice and translation.
  - Require all three:
    - max|q_vec(G·R) − q_vec(R)| ≤ 1e-6·max(1, max|q_vec(R)|);
    - lattice and translation output change ≤ 1e-6;
    - max|pred_R(G·R) − G·pred_R(R)| ≤ 1e-5 (its float32 floor is ≈ 6-7e-7 per run).
  - Controls: independent per-copy rotations in place of G must exceed 100× each statistic. If one does
    not, check the harness before (a) is declared.
  - The 50-step sampler deviation is reported. Provenance: repeat once with the released Thurlemann
    checkpoint on its own test pickle.
  - (a) runs on the box in §7 step 5, never after step 6.
  - If (a) fails, or is recorded FAILED at §7 step 6:
    - the invariance statement is withdrawn;
    - (b)-(d) are not reported;
    - Table 2 omits the ceiling column;
    - the §4 z_MCF bullet drops its invariance clause and its (b) citation.
- **(b) Perfect-model ceiling**, from truth, once, for every MCF arm. c_i = the Haar fraction of H whose
  common rotation of the true frames (pickle frame) still matches the truth.
  - z_MCF = 2: both cosets on a 5° grid, walking at 1° to the window edges; c_i = (w_rot + w_flip)/720°.
  - z_MCF ≥ 3: c_i = (# matching D2 elements)/4.
  - z_MCF = 1: c_i = the Haar floor (the crystal's iii-u draw@1), labelled as such.
  - Report mean c_i (the draw@1 ceiling) and mean[1 − (1 − c_i)^16], per z_MCF and matcher. c_i does
    NOT bound LJ-picked@1.
- **(c) Oracle-aligned score** (MCF-R only). Per draw, apply the h ∈ H minimising Σ_k ‖h R_k −
  R_k^true‖²_F, where R^true = rotmats_1 of the same pickle (closed-form in θ per coset for z_MCF = 2),
  then match. Report draw@1, any@16 and picked@1 per z_MCF and matcher. It is not computed for
  z_MCF = 1, where it is exact by construction (counted).
- **(d) Consistency.** MCF-R's unaligned draw@1 must not exceed the (b) ceiling beyond its bootstrap CI.
  An exceedance means (b) is wrong: it is reported and fixed before (c) is interpreted.

**Paper wording, conditional on (a) passing:** "At fixed lattice and centroids, the released
MolCrystalFlow network and its Euler sampler are equivariant, to numerical precision, to a common left
rotation of all molecular frames. The sampled orientation distribution is therefore invariant under
rotations about the z axis of its lower-triangular lattice frame plus the in-plane two-folds when the
cell it is given holds two molecules, under D2 for three or more, and under all rotations for one. This
holds for the released model and its native joint sampler; it is not introduced by our modification."

### 3.3 (iii) CLASSICAL (same information as OURS)
All classical arms use the true L, the true c0, the spglib ops and the rigid conformer. Only R_asym
moves, and every copy comes from exact expansion.

- **(iii-u) FLOOR.** 16 Haar R_asym per crystal × 3 seed bases, unrelaxed.
  - Table 1 row 1 ("random floor") = these draws with the RANDOM selector.
  - Row 2 (iii-u) = the same draws with iii-u's VALSEL-chosen selector.
- **(iii-p) STERIC PRESS** (continuity with G2; descriptive). The G2 press (80 Adam steps, lr 0.02,
  contact 0.90, cutoff 6.0) applied to the iii-u draws. The selector comes from the §2.3 menu, chosen on
  VALSEL, and its LJ rule uses the POST-press energy. Stated ceiling: the pressed TRUE pose matches the
  truth in 2/198 (G2).
- **(iii-s) PHYSICS SEARCH** (the H2 comparator; deterministic, m = 1). It runs on VAL (a sanity row),
  TEST-B and DEV-TEST. It has no VALSEL knob, so it does not run on SEL.
  - **Candidates.** The Yershova incremental Hopf grid at resolution 2: HEALPix Nside = 4 (192 cells) ×
    24 in-plane angles = 4,608 orientations, ~15°.
    - Pre-rotated by one fixed Haar rotation: `torch.manual_seed(20260927)`;
      `symmc_flow.manifolds.random_so3((1,), dtype=torch.float64)`.
    - Every candidate is expanded exactly.
  - **Screen.** All 4,608 by unrelaxed steric LJ.
    - The screen may use a batched evaluator only if it agrees with energy() to 1e-4 relative on 20 VAL
      crystals × 100 orientations; otherwise energy() is used.
    - Keep the 64 lowest. **This number is never reduced, on any set.**
  - **Minimise.** Orientation-only rigid-body minimisation over δ ∈ so(3), with L and c0 fixed; the
    whole cell moves by exact expansion.
    - **FF* = UMA s-1p1, OMC task** (fairchem), the potential MCF's paper uses for its own rigid-body
      relaxation (UMA-Opt).
    - L-BFGS (history 10, strong Wolfe), at most 100 energy+gradient evaluations. All live starts of
      one crystal are evaluated in one batched predict call per round.
    - Converged iff ‖∂E/∂δ‖ / K < 0.01 eV/rad. E is the whole-cell energy, so the gradient sums K
      symmetry-equivalent copies, and dividing by K makes the tolerance per molecule.
    - A start pose that already passes is converged. Every other stop (evaluation cap, line-search
      failure) is UNCONVERGED, and its final pose is still used.
    - Gradient from forces: ∇_δ E = −Σ_k Σ_a (R l_a) × (Rc_kᵀ F_ka), with the update
      δ → exp([δ]×)·R. It is checked once against central finite differences (1e-4 rad) on 5 VAL
      crystals.
    - **X–H normalisation.** Every FF* evaluation (iii-s, i-E, the FF* ceiling, BASIN) uses a copy of the
      rigid conformer whose X–H bonds are extended along their bond vectors to the CSD neutron
      normalisation lengths: C–H 1.089, N–H 1.015 and O–H 0.993 Å. The local-frame origin is unchanged.
      - The R_asym that FF* returns is applied to the ORIGINAL conformer for scoring, so every scored
        structure keeps the experimental H positions.
      - The number of normalised bonds per set is reported.
  - **Rank.** Merge minima whose geodesic distance, minimised over the molecule's proper point group
    (PointGroupAnalyzer, tol 0.3 Å), is ≤ 2°, keeping the lower energy. Then rank by FF* energy.
    picked@1 = the lowest-energy minimum; any@16 = the 16 lowest distinct minima.
  - **Environment and weights.**
    - FF* runs in its own environment, with fairchem-core pinned in A1 and its required torch.
    - `load_predict_unit(<uploaded uma-s-1p1.pt>, device='cuda', atom_refs=<uploaded references/iso_atom_elem_refs.yaml>)`,
      task 'omc'.
    - The files are `checkpoints/uma-s-1p1.pt` and `references/iso_atom_elem_refs.yaml` from the gated
      `facebook/UMA` repo, to which Frank's HF account has access. The locally cached s-1p2 is NOT
      used. Their SHA-256 are recorded in A1.
    - The HF token never goes onto a rented box.
    - No other UMA version or potential is substituted. If s-1p1 cannot be obtained before A1, N3 stops
      and re-plans by amendment.
  - **Elements.** FF* runs on every crystal whatever its elements (the OMC head carries element
    references for Z = 1-94). Crystals with elements outside OMC25's element set are listed by refcode.
  - **FF* failures.** A call that raises, or returns a non-finite energy or gradient, is retried once in
    a fresh process, unbatched. If it fails again it is DETERMINISTIC.
    - (i) In any minimisation, a deterministic failure after the first step ends the run at the last
      finite iterate (UNCONVERGED). A deterministic failure at the start pose gives no minimum.
    - (ii) iii-s and i-E rank only finite minima. A crystal with none is a MISS for that arm.
    - (iii) BASIN form: an arm's pick with no minimum is a non-match. If the TRUE pose has no minimum,
      the crystal is a BASIN miss for every arm (d_i = 0) and counts as not anchored.
    - Deterministic failures are counted per arm and set, and listed.
  - **OMC25 overlap.** UMA-OMC's training set OMC25 is built from OE62 molecules, which were extracted
    from the CSD.
    - OMC25-NONOVERLAP = the TEST-B crystals whose 6-letter refcode family does not occur among the
      `csd_refcode` values of `facebook/OMC25` `omc25-starting-crystals.csv` (file SHA-256 in A1).
    - The count of crystals whose molecular formula equals some OMC25 `mol.composition` is reported
      (descriptive).
    - H2 is also reported, descriptively, on OMC25-NONOVERLAP.
- **FF* CEILING and the H2 FORM.** Measured on the 400 VALSEL truths before any arm's candidate is
  FF*-minimised on any set.
  - Anchoring = the fraction of VALSEL crystals whose TRUE pose, minimised as above (up to 1,000
    evaluations for truths), is CONVERGED and still matches the truth under the primary matcher. The
    VAL-only value is printed beside it.
  - The FORM is written to `n3_selection.json` and pushed before any iii-s, i-E or BASIN minimisation
    runs.
  - **If anchoring ≥ 50%, H2 FORM = EXACT.** The H2 contrast uses the §2.1 scores: OURS' and OURS-N2's
    LJ-picked draw per seed, and iii-s's lowest minimum as output, each against the unrelaxed truth.
  - **If anchoring < 50%, H2 FORM = BASIN,** for the H2 contrast only (OURS and OURS-N2 vs iii-s).
    - The LJ-picked draw of each OURS and OURS-N2 seed is FF*-minimised with the iii-s settings.
    - iii-s's lowest minimum is used as is.
    - Each is matched (primary matcher) to the TRUTH FF*-minimised with the same settings (up to 1,000
      evaluations). Crystals whose reference is still unconverged are listed.
  - MCF-R, MCF-R+G and iv-P have no R_asym and are never FF*-minimised. H1 and H3 are always scored in
    the §2.1 form.
  - The BASIN jobs run in BOTH forms, because the other form is reported as secondary.
  - **FF* truth labels,** computed for every TEST-B and DEV-TEST crystal by the same truth-minimisation
    job, which also gives the TEST-B anchoring rate:
    - ANCHORED: converged, and still matches;
    - DRIFTED: converged, and no longer matches;
    - UNRESOLVED: unconverged.
    The label uses only the truth and FF*, never an arm's output.
- **(i-E) OURS+FF* (matched energy; descriptive).** All 16 OURS draws per seed are FF*-minimised (every
  pose and energy is saved), and the lowest FF* energy is picked. It is reported against iii-s with a CI
  and scored in the H2 FORM. It separates the learned proposal from the energy model.
  - OURS' BASIN candidate is the i-E minimised pose of its LJ-picked draw; it is not minimised twice.
  - i-E runs on TEST-B and DEV-TEST.
- **Steric BASIN (continuity with G2; descriptive).** For OURS and iii-p:
  - P1 = the G2 press on the arm's own draws, and the basin reference = press(true R0).
  - A hit = StructureMatcher() fit(pressed draw, pressed truth).
  - P1 selector menu on VALSEL: random; lowest pre-press LJ then press; lowest post-press LJ.
  - G2's DEV-TEST result is printed beside it: basin_best 10.1/8.5/7.0% vs Haar 6.5%, p = 0.19/0.54/1.0;
    basin_any 22.6/19.6/17.6% vs 11.1%.

### 3.4 (iv) OLD symmc-flow architecture: iv-P (the H3 arm)
- **Items.** Built for every TRAIN/VAL/SEL/TEST-B/DEV-TEST crystal from the asym item and its spglib
  ops, not from `orig`. On 11/1687 TRAIN crystals `orig` disagrees with the ops.
  - For op k, with s_k = [det W_k = −1] and D = diag(1,1,−1):
    - centroid_k = frac(W_k c0 + t_k);
    - local_k = D^{s_k}·local;
    - target orient_k = Rc_k·R0·D^{s_k} (proper).
  - Z / atom_mask / mol_mask are padded to 16 × 64. Keys `coset` and `idx` are added, and the item is
    asserted all-tensor before collate.
  - The old EGNN is reflection-invariant, so mirrored copies are identified only through the coset id.
    This is the old design run on the O(3) data.
- **Coset ids.**
  - The identity-op copy is the reference (id 0, like padding).
  - Every other copy gets the id of the key (sg, W_k as integers) in a table built from TRAIN only.
    t_k is not part of the key: the target orientation depends only on W_k and L, and the position is
    already an input.
  - Centring copies (W = I, t ≠ 0) get (sg, I).
  - Keys unseen in TRAIN map to one reserved id whose embedding row is zeroed, and they are counted.
    Pre-counted on the seed-0 split: 151 TRAIN keys, 0 unseen VAL or DEV-TEST copies.
  - `assign_symmetry_cosets` is NOT called (it uses the standard-setting table).
- **Model.** Published ModelConfig defaults, n_cosets = table size + reserved, lambda_orient = 1, and
  **lambda_lattice = lambda_centroid = 0**. The last is a disclosed deviation: under clean packing those
  heads train on inconsistent targets and would contaminate the val loss.
- **Train** (`scripts/n3_old_symmc.py`, our own loop, because `train()` returns only the final model).
  - cond_clean_packing=True, batch 16, AdamW (weight decay 0), grad clip 1.0, constant lr, fixed split.
  - 60 epochs over TRAIN, with a checkpoint and the fixed-draw val loss at every epoch end.
  - Seed 0 trains at each lr ∈ {1e-4, 3e-4 (published), 1e-3}. Seeds 1-2 train only at the chosen lr.
    That is 5 runs.
- **Fixed-draw val loss.** The orientation loss on VAL × 16 draws, with per-slot R0 and t generated ONCE:
  `torch.manual_seed(12345)` immediately before `random_so3((100,16,16))` and `rand((100,16))`.
- **Selection, in this order;** frozen in `n3_selection.json`.
  1. Window: epochs 1-30. It becomes 1-60 if any of the three seed-0 runs (any lr) has an epoch-30 val
     loss more than 2% below its epoch-25 val loss.
  2. lr: the one whose seed-0 run has the lowest fixed-draw val loss within the window. Seeds 1-2 train
     at it, and the seed-0 run is reused. The lr is chosen once and never re-chosen.
  3. Late extension: if the window is still 1-30 and a seed-1 or seed-2 run meets the same 2% condition,
     the window becomes 1-60 for all seeds. This changes only the checkpoint candidates.
  4. Stage A (selector, pooled over the 3 seeds): take each seed's lowest-val-loss checkpoint within the
     window, and pick the §2.3 selector with the highest seed-summed VALSEL hits (§2.3 ties).
  5. Stage B (checkpoint per seed): candidates are {lowest val loss in window, last epoch of window}.
     Keep the higher VALSEL picked@1. Ties: draw@1 → any@16 → lower val loss.
- **Sampling.** `sample_orient_only` (RK4 on SO(3), 50 steps), with the true lattice and all centroids
  frozen. 16 draws per crystal, each with K independent per-copy Haar priors (§2.2 seed). No post-hoc
  symmetrization. Structures and LJ use the per-copy local over real slots.
- **(iv-S) diagnostic** (descriptive). Take the identity-copy orientation as R_asym and expand it
  exactly. It shares iv-P's lr, window and Stage-B checkpoints. Its selector is chosen by seed-summed
  VALSEL hits on the iv-S candidates.
- **Smoke gate** (CPU, before GPU spend):
  - true orientations rebuild 100% of VAL/DEV-TEST truths;
  - slot → op is a bijection (it holds by construction on SEL/TEST-B after build step 2);
  - two training steps give a finite loss.
- **Not run in N3:**
  - iv-A (legacy det+1 items, 598 train). Its coverage limit is reported deterministically in Table R.
  - iv-R (the relative-gauge audit, in which R_asym is supplied).

---------------------------------------------------------------------------------------------------
## 4. Hypotheses (primary family; TEST-B; §2.4 test; Holm over three)
- **H1 (learned MCF comparator):** OURS vs MCF-R AND OURS vs MCF-R+G. Intersection-union, with the
  §2.4 verdict rules.
- **H2 (physics search):** OURS vs iii-s, in the H2 FORM fixed on VALSEL (§3.3).
- **H3 (old architecture):** OURS vs iv-P.

Each hypothesis is reported whatever the outcome, together with the OURS-N2 family, the two-stage seed
interval and the 95% CI.

**Stated in advance (review measurements; not results).**
- iii-u is expected at ≈ 0 exact under the primary matcher. A Haar draw falls inside the matcher's ~6-8°
  radius with p ≈ 1e-4.
- iii-p is capped by the steric force field.
- MCF-R perfect-model values on DEV-TEST truth. They were computed on conventional cells keyed on K;
  diagnostic (b) recomputes them on the primitive export keyed on z_MCF.
  - draw@1 ceiling ≈ 25% for K ≥ 4 (first 15 of 145 crystals).
  - For K = 2 the rotation branch alone gives ≈ 3.7% (first 12 of 55). The flip coset was not scanned,
    so the both-coset value is expected at ≈ 1.9-3.7%.
  - perfect-model LJ-picked@1 ≈ 16% (K = 2) and 82% (K ≥ 4).
- The iii-s screen keeps a start within 12.5° of the truth for 91/98 VAL crystals (38/43 of those with
  H-bond donors).

**Outcome → claim map (fixed now).**
- **H1 not supported** (failed, split, or unresolved): no claim that a learned asymmetric unit beats
  learned independent copies. Only Table R, H2 and H3 are claimed.
- **H1 supported, but on the z_MCF ≥ 3 crystals min(Δ_R, Δ_G) ≤ 0.** This is the point estimate on that
  subset (descriptive). The discussion states that the advantage concentrates in z_MCF ≤ 2. Only if
  diagnostic (a) passed does it add that MCF's invariance group is continuous there and its ceiling
  binds, citing diagnostic (b). There is no stratum-level significance claim.
- **Gate C-a (ii) failed** (§3.2, recorded by amendment): the §2.4 verdict rules for H1 are unchanged,
  and every H1 sentence carries the §3.2 Gate C-a harness clause.
- **H1 supported: the ii-S ablation** is read through the identity Δ(OURS − MCF-R+G) =
  Δ(ii-S − MCF-R+G) + Δ(OURS − ii-S), with each term given its paired BCa CI and no p-value.
  - CI of Δ(OURS − ii-S) entirely above 0: "our learner adds Δ [a, b] pp over MCF's orientations inside
    the same exact expansion (ablation estimate)".
  - Entirely below 0: the paper attributes the H1 gain to exact SG expansion, not to our learner, and
    states that MCF's orientations expanded exactly outperform our learner.
  - Contains 0: no contribution from our learner beyond exact expansion is claimed ("not resolved at
    n = …"). If the CI of Δ(ii-S − MCF-R+G) lies entirely above 0, the paper adds that exact expansion
    of MCF's own orientations recovers that part of the gain; otherwise the gain is attributed to
    neither part.
- **H2 not supported:** no "better than physics-based search" claim.
- **H2 supported in EXACT form.** The 50% rule does not stop the win from coming from DRIFTED crystals,
  where iii-s can almost never match. Δ(OURS − iii-s) is therefore also reported on ANCHORED, DRIFTED
  and UNRESOLVED TEST-B crystals, each with a paired BCa CI.
  - "Better than physics-based search" is written only if the ANCHORED-subset CI lies entirely above 0.
  - If that CI contains 0: "OURS reproduces more experimental poses than the physics search (Δ, CI). On
    the crystals whose experimental pose FF* holds, the difference is not resolved (Δ_A, CI)."
  - If that CI lies entirely below 0: "…on the crystals whose experimental pose FF* holds, the physics
    search reproduces more (Δ_A, CI)."
  - The text never says "beats physics", and the abstract uses the same wording.
- **H2 supported in BASIN form:** the drift effect does not arise, and the split is reported
  descriptively.
- **H3 not supported:** no claim over the old architecture beyond Table R coverage.
- **H1, H2 and H3 all not supported:** the paper becomes a representability / negative-result note (G2's
  kill logic).

---------------------------------------------------------------------------------------------------
## 5. Representability (Table R; deterministic; computed on TEST-B and DEV-TEST before any sampling)
Rows: all / centro / Sohncke / other / MIXED-HANDEDNESS / IMPROPER-ACHIRAL / LEGACY-UNREPRESENTABLE.
Columns are counts of crystals whose TRUE structure each representation can emit, under every §2.1
matcher:
- (a) OURS = CLASSICAL, `expand(a, R0)`. The caption says this holds by construction (build step 2) and
  prints how many pool crystals that step removed. On DEV-TEST it is computed and printed.
- (b) MCF per-copy representation (Gate F2).
- (c) The information of MCF's released CSP driver (one conformer and one χ for all copies;
  packing_gen.py:376-388).
  - This is the best homochiral approximant: true lattice and centroids, with each improper copy
    replaced by the RMSD-optimal proper rotation onto its mirror.
  - Element-blocked ICP from 16 starts: the 4 sign-diagonal rotations plus 12
    `scipy Rotation.random(random_state=int(rng.integers(1e9)))` draws, with
    `rng = np.random.default_rng(0)` per molecule.
  - It is an upper bound for any homochiral cell. No arm runs this driver.
- (d) OLD det+1: the part-file keep flag. For kept crystals, `scripts/n3_legacy_oracle.py` re-runs
  `_parse_structure(allow_mirror=False)` with a fresh registry per crystal, rebuilds with
  `rigid_to_structure` and matches the CIF with StructureMatcher(). Both counts are reported.
- Per-crystal timeout 600 s, reported as its own count.
- Corpus context row (G1/G2): 3,500 CIFs → 1,144 kept by det+1, 2,852 by O(3), 1,987 eligible. MCF's
  per-copy representation also covers the Z' > 1 and special-position crystals our design excludes.

**Frozen wording.** "The legacy det+1 pipeline cannot represent N/n test crystals (M of the
centrosymmetric ones). The released MolCrystalFlow CSP driver cannot produce P/n_mixed mixed-handedness
cells under the default StructureMatcher; a homochiral approximant nevertheless matches Q/n_mixed at
stol 0.5."

**Forbidden:**
- any statement that MCF's benchmark path, MCF-R or MCF-A cannot represent mixed-handedness cells;
- any "prior pipelines" wording beyond rows (c) and (d);
- any representability claim relative to MCF's per-copy representation.

The text states that the MCF paper describes a mixed-χ CSP scenario whose code is not released at
3c493f8.

---------------------------------------------------------------------------------------------------
## 6. Reporting plan
**TABLE 1 (headline): TEST-B, primary matcher, every row scored in the §2.1 form.** Rows:
1. random floor (iii-u draws, RANDOM);
2. iii-u;
3. iii-p;
4. iii-s;
5. iv-P;
6. MCF-R;
7. MCF-R+G;
8. OURS;
9. OURS-N2.

Columns:
- Table 0 information;
- representable n/N;
- seed-mean picked@1 ± s.d. over seeds;
- Δ vs OURS with BCa CI;
- Holm-adjusted p and verdict for H1-H3 (primary family on the OURS row; the N2 family on the OURS-N2
  row);
- two-stage seed interval;
- draw@1;
- any@16;
- per-crystal compute (sampling + selection wall-clock; LJ and FF* evaluation counts).
- If the H2 FORM is BASIN, one extra column, "H2 BASIN picked@1", is filled for OURS, OURS-N2 and iii-s.
  The H2 statistics come from it, and the EXACT-form H2 numbers go to the SI.
- The caption states the H2 form.
- Training GPU-h goes in a footnote.

**TABLE 2 (descriptive).** Rows = every Table 1 row. Picked@1 by K, z_MCF, class, MIXED-HANDEDNESS,
FORMULA-DISJOINT and OMC25-NONOVERLAP, plus the FF* truth label for the H2 pair. If diagnostic (a)
passed, each z_MCF row carries MCF's diagnostic-(b) ceiling, labelled oracle.

**TABLE R** (§5). **CONTEXT TABLE:** MCF-A with its native sweep, and the prior floor.

**SI:**
- secondary matchers (picks, and the Haar floor);
- ensemble and any@48;
- continuous endpoints;
- ii-S, iv-S, i-E and steric basin;
- MCF diagnostics;
- DEV-TEST tables;
- the full VALSEL selection grids;
- gates F and C;
- the X–H normalisation counts;
- deterministic-failure lists.

**Abstract rules.**
- Quote OURS' Table 1 picked@1 (seed mean ± s.d.) and the outcome of H1, H2 and H3, each with Δ and CI.
  For H1, Δ_H1 = min(Δ_R, Δ_G).
- If the H2 FORM is BASIN, say that H2 compares FF* basins.
- If Gate C-a (ii) failed, the H1 sentence carries the §3.2 Gate C-a harness clause.
- State failed or unresolved hypotheses as such.
- No stratum, secondary-matcher, ensemble or any@k number appears in the abstract as a result.

---------------------------------------------------------------------------------------------------
## 7. Execution order, compute, budget, hygiene
1. Commit and PUSH this file. The push is the third-party timestamp.
2. Build TEST-B/SEL (§1.1) and compute the stratum labels, Gate F, the OLD smoke gate, Table R and the
   power table. Download the UMA s-1p1 files and record their SHA-256. Commit and push amendment A1.
3. Implement and CPU-smoke every arm locally. Local shims (torch_scatter, radius_graph) are for smoke
   tests only; no reported number comes from a shimmed run.
4. **Timing gate and approval.**
   - Before any VALSEL sampling, time every job in the exact configuration that will be run (same
     concurrency, after warm-up), on TRAIN/VAL crystals only. Nothing it produces is matched.
   - GPU box:
     - MCF-R and iv-P training, steps 50-300;
     - one 16-draw MCF-R and MCF-A inference pass, and one iv-P RK4 pass, on 50 VAL crystals;
     - FF* energy+gradient calls on 20 VAL crystals (K = 2, 4, ≥ 8), batched and single.
   - The machine named for each CPU job: energy(), press(), `StructureMatcher().fit`, and the secondary
     `get_rms_dist` on Haar candidates of the same 20 VAL crystals.
   - Project GPU-h, CPU-h, box-hours and $ separately for: training; VALSEL sampling + scoring (on the
     critical path before step 6); TEST-B/DEV-TEST sampling; FF*; press; LJ screens; matching.
   - **The projection goes to Frank. Nothing larger than the timing gate starts without his approval.**
     The Vast balance is shared with other projects.
   - Any degrade steps are chosen HERE and nowhere else, then committed and pushed as a dated amendment
     (stating that no N3 output exists) before step 5.
   - CSD uploads happen only in manual-approval mode. Any CPU box follows the same rules as the GPU box.
5. Upload TRAIN, VAL and SEL inputs only; TEST-B and DEV-TEST inputs stay off the box. Then:
   - run Gate C;
   - train MCF-R ×3, MCF-A ×3 and iv-P (seed 0 at three lrs, then seeds 1-2 at the chosen lr);
   - minimise the 400 VALSEL truths under FF*, and push the H2 FORM before any arm's candidate is
     FF*-minimised;
   - sample VALSEL for every candidate cell of every arm (classical included), with every MCF RESID pass;
   - score VALSEL;
   - after the MCF-R and MCF-A checkpoint selections, run MCF diagnostic (a) (§3.2) on the box, in the
     same un-shimmed environment (CPU, float64). It covers every selected checkpoint, with its controls,
     the 50-step sampler deviation and the Thurlemann provenance repeat. Any harness check happens here.
     If (a) is unresolved at step 6, it is recorded FAILED and is never re-run.
6. Write `results/n3/n3_selection.json`. It holds every VALSEL cell score, every frozen choice, the H2
   FORM, the MCF diagnostic-(a) statistics and verdict, the degrade record, and the complete list of
   permitted TEST-B / DEV-TEST runs with seeds. Commit and PUSH it BEFORE any TEST-B or DEV-TEST input
   reaches the box.
7. Upload the TEST-B and DEV-TEST inputs, including the stored N1 (OURS) and N2 (OURS-N2) DEV-TEST draws.
   Run exactly the listed runs with their committed seeds. The list must contain:
   - (a) every arm's sampling runs, with each draw carrying its selector inputs (OURS/OURS-N2: e_lj,
     torque_end; iv-P: ‖v_R‖; MCF: resid_16.pt). No output is matched against the truth on the box;
   - (b) the FF* jobs in both H2 forms: iii-s; the FF*-minimised truth of every crystal (the BASIN
     reference, the anchoring rate and the truth labels); i-E; and the OURS-N2 LJ-picked draws.
   A run not listed is disclosed, and that arm's run is invalid. Step 8 starts only when every listed run
   has finished and been checksummed. No FF* evaluation on TEST-B or DEV-TEST is made after step 8.
8. Pull everything back and verify checksums. Delete every CSD-derived file on the box (inputs, exports,
   checkpoints, outputs, wandb directories), confirm by search, and destroy the box.
9. Score TEST-B and DEV-TEST once (CPU matching may run locally or on a CPU box under the same hygiene).
   Write `results/n3/n3_results.json`. Commit and push.

**Failures.**
- If the box dies before step 6, resume from step 5 with the same seeds.
- If it dies during step 7, discard every partial TEST output unscored and re-run step 7 on a new box
  from the pushed file.

**Budget.**
- **FF* minimisations** (|TEST-B| ≤ 1000; each at most 100 evaluations; truths up to 1,000):
  - FF* ceiling: 400 VALSEL truths;
  - iii-s: 64 per crystal on VAL, TEST-B and DEV-TEST (≈ 83k);
  - i-E: 48 per crystal on TEST-B and DEV-TEST (≈ 58k);
  - truths: TEST-B and DEV-TEST (1.2k);
  - OURS-N2 picks: 3 per crystal (3.6k).
- **Prior estimate, NOT measured.** The CPU figures are single-thread review timings on an 8-thread
  laptop.
  - GPU:
    - MCF: 6 runs × 1-4 GPU-h, plus VALSEL/TEST inference.
    - iv-P: ≈ 2.5 GPU-h training, plus RK4 sampling (unmeasured).
    - FF*: up to ≈ 15 M UMA evaluations. This is 8-15 GPU-h only at ≤ 2-3.5 ms per structure-evaluation,
      i.e. with batched calls.
  - CPU:
    - steric LJ ≈ 200 CPU-h (iii-s screen and +G scan);
    - press ≈ 100-170 CPU-h;
    - matching (primary on every draw, secondaries on picks and the floor) ≈ 250-500 CPU-h on
      TEST-B + DEV-TEST, ≈ 75-190 CPU-h on the VALSEL grids, and ≈ 30-60 CPU-h for Gate C-a.
  - Box-hours and $ are projected from the timing gate.
- **Degrade order** (fixed now; decided only at step 4; take the fewest items that fit, in this order):
  1. drop the §8 tier-2 diagnostics;
  2. MCF-A native sweep reduced to {0.8, 1.0};
  3. MCF-A seed 0 only;
  4. MCF-A without its knob grid;
  5. drop the DEV-TEST rows for new arms;
  6. i-E and the steric BASIN on seed 0 only;
  7. drop i-E. The H2 BASIN minimisations of the OURS LJ picks still run.
- **Never reduced:** |TEST-B|; OURS; OURS-N2; MCF-R(+G) including its s_uR grid; iv-P; iii-s exactly as
  in §3.3 (Hopf level 2, the 64 lowest minimised, ≤ 100 evaluations); the FF* ceiling; the H2 BASIN
  scoring in both forms; 3 seeds for every H1-H3 arm.
- If the cap still binds after item 7, Frank either approves the extra cost, or the run stops before any
  TEST-B input reaches the box and the stop is disclosed. No amendment changes a never-reduced item once
  any VALSEL score exists.
- **Overrun after step 4.** No degrade item is taken after step 4.
  - If spending in steps 5-6 overruns the approved cap, Frank either approves the extra cost, or the run
    stops before any TEST-B input reaches the box. Any re-plan is a dated amendment.
  - If spending in step 7 overruns the cap, nothing changes the step-6 run list while step 7 is running.
    Frank either approves completing every listed run, or the run stops.
  - A stop during step 7 is handled like a box death under Failures: every step-7 output is discarded
    unscored, and none of it is matched or has its hits counted.
  - A re-plan after such a stop is a dated amendment. It states that no step-7 arm output was matched,
    changes only items that are not on the never-reduced list, and re-runs with the committed seeds.

**Hygiene.**
- Every CSD-derived file stays local or on the box, and is gitignored.
- Only JSON summaries, refcode lists, code and this file are committed.
- Never `pgrep -f` / `pkill -f` inside ssh; kill by PID.

---------------------------------------------------------------------------------------------------
## 8. Tier-2 diagnostics (only after steps 1-9, only if the approved budget allows; descriptive)
- **Data-size curve.** OURS (G2 code) and MCF-R, seed 0, on nested TRAIN prefixes 422 / 843 / 1687.
  Report VAL loss, VAL picked@1 and, for MCF-R, oracle-aligned VAL any@16. No pre-written conclusion.
- **Centroid-noise sensitivity (D-c0).** OURS and iii-u on a seeded 200-crystal TEST-B subset (torch seed
  20261002), with σ ∈ {0.1, 0.25, 0.5} Å Cartesian noise on c0 and the same δ for every arm. Report:
  - (a) the picked R_asym re-placed at the true c0 (primary matcher);
  - (b) the as-generated structures;
  - (c) the median geodesic error.
- **MCF conventional-cell export** for centred crystals: a sensitivity row for the primitive-cell choice.

---------------------------------------------------------------------------------------------------
## 9. Amendments
(none yet)
