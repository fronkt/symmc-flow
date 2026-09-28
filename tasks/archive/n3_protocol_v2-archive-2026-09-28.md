<!-- ARCHIVE (2026-09-28): v2 of the N3/G3 protocol (never committed as the frozen version).
     Reviewed by a second three-lens hostile review (38 issues; scratchpad n3_review2.json).
     Superseded by tasks/n3_protocol.md (v3). Kept as provenance for the pre-registration. Never executed. -->

# N3 / G3 benchmark: pre-registered protocol (v2, 2026-09-28)

**Status: FROZEN at the commit that adds this file. No N3 arm has been trained, sampled or scored.**
Changes after this commit are made only by dated amendments at the end of this file (§9). Each amendment
states whether any N3 output existed when it was made.

Provenance:
- v1 draft: `tasks/archive/n3_protocol_v1_draft-archive-2026-09-28.md`.
- v1 was reviewed by five hostile lenses: a MolCrystalFlow author, a statistician, a strawman-baseline
  referee, the implementing engineer, and an ACS Omega referee. They raised 48 issues. Each blocker and
  major issue was independently re-checked against the code and data (19 valid, 16 partly valid).
- v2 adopts the checked fixes. The component map behind all of this is summarised in
  `tasks/omega_rebuild_plan.md` (G0-N2).

Throughout, "OURS" is the G2 learned asym-orientation flow (`scripts/g2_learned_flow.py`,
checkpoints `results/vast_g2/eval_s{0,1,2}.pt`) with the N1 selector (lowest unrelaxed steric-LJ
draw). "MCF" is MolCrystalFlow @3c493f8 (github.com/Liu-Group-UF/MolCrystalFlow, arXiv 2602.16020v3).

---------------------------------------------------------------------------------------------------
## 0. Task and scope

**Task.** The task is ORIENTATION COMPLETION of a Z'=1 general-position molecular crystal. Given:
- the true lattice;
- the true space-group operations in the crystal's own (CSD) setting;
- the true asymmetric-unit centroid c0;
- the rigid true conformer;

predict R_asym ∈ SO(3) (3 degrees of freedom) so that exact space-group expansion reproduces the
experimental crystal. The ops come from `SpacegroupAnalyzer(symprec=0.1)` on the true structure. They
also fix Z = K, Z'=1 and the handedness of every copy through det W_k.

**This is NOT crystal-structure prediction.** Lattice, space group, Z, Z' and molecular position are
oracle inputs, and every claim built on N3 is limited to this task. The random-orientation floor (§2) is
the pre-registered evidence that the task is not trivial.

**Deviation from plan, recorded before any N3 run.** Plan G3 (`omega_rebuild_plan.md`) was titled
"fair end-to-end benchmark". Paper A was to learn lattice + centroid + R_asym, but only the R_asym model
exists (G2/N1/N2). N3 therefore benchmarks orientation completion. No end-to-end number for OURS exists
or is claimed.

**Table 0: information per arm.** This table is printed in the paper.

| | OURS / CLASSICAL | OLD iv-P | MCF-R (+G) | MCF-A (context) |
|---|---|---|---|---|
| lattice | true | true | true | generated |
| centroids | c0 true; copies from the ops | all K true | all K true | generated |
| SG ops | given, enforced exactly | given as per-copy op labels (coset ids) | not given (recoverable from its inputs, see note) | not given |
| Z = K | via the ops | given | given | given |
| per-copy handedness | det W_k | mirrored per-copy local conformer | mirrored per-copy local + chi | same as MCF-R |
| predicted | R_asym (3 DOF) | K orientations | K orientations | lattice, K centroids, K orientations |
| copies built by | exact SG expansion | K independent outputs, attention-coupled | K independent outputs | K independent outputs |

Note on MCF-R information. Run spglib on the K true centroids, keep only the ops consistent with each
copy's handedness (which MCF-R holds through chi), and the copy-to-copy rotation parts are recovered
uniquely and correctly in 198/200 DEV-TEST crystals. In the other 2 the truth is among the candidates
(review measurement). So information is matched up to that residual. The real difference is that OURS
is given the ops explicitly and enforces them. H1 therefore measures "explicit, enforced symmetry plus
our learner" as one bundle. The ablation (ii-S) separates the two parts. It is descriptive only.

---------------------------------------------------------------------------------------------------
## 1. Data sets, splits, leakage, failures, disclosure

### 1.1 Sets
- **TRAIN (1687), VAL (100), DEV-TEST (200).** These are unchanged. They are the seed-0 split of
  `data/csd_mol/g2_asym.pt`: DEV-TEST = perm[:200], VAL = perm[200:300], TRAIN = perm[300:].
- **TEST-B (confirmatory) and SEL (selection supplement).** Both are new and come from the fresh pool.
  - Pool = rows 3501-8000 (4,500 refcodes) of `data/csd_mol_scale_big/manifest.csv`.
  - Its first 3,500 rows equal `data/csd_mol/manifest.csv` row for row. The 4,500 are therefore the
    direct continuation of the same seed-0 `scripts/csd_export.py` CSD stream, with the same filters.
  - No step of G1-N2 has parsed or scored them. (Disclosure: the legacy det+1 pipeline parsed them in
    July 2026 for the old paper's scale runs. See §1.4.)
- **VALSEL = VAL ∪ SEL (400).** Every open knob of every non-frozen arm is selected on VALSEL.
- **Build procedure** (`scripts/n3_build_testB.py`, committed). The steps are deterministic, and all are
  input-side:
  1. Copy the 4,500 CIFs to `data/csd_testB/cif` (gitignored). Run `scripts/g1_o3_reparse.py` on all of
     them as a separate run with G1 defaults: max_mols 16, max_atoms 64, symprec 0.1, conf_tol 0.3,
     cif_timeout 300. It has its own conformer registries and never mixes with csd_mol.
  2. Keep items with `asym_item(it) is not None`: Z'=1 general position, K = number of spglib ops.
     Drop items whose G1 oracle rebuild failed.
  3. **De-leak.**
     - Drop any item whose 6-letter CSD family (refcode[:6]) occurs among the 3,500 dev-corpus refcodes.
     - Drop any item whose molecule Weisfeiler-Lehman bond-graph hash occurs among the 1,787 TRAIN+VAL
       items. The hash is `networkx.weisfeiler_lehman_graph_hash`, 3 iterations, element node labels.
       The bond graph uses covalent radii, with a bond when d < 1.2·(r_i + r_j) on the asym conformer.
  4. **Order.** Sort the survivors by refcode. Apply
     `perm = torch.randperm(n, generator=torch.Generator().manual_seed(20261001))`. Walk the
     permutation and keep only the FIRST item of each 6-letter family. Call the resulting ordered list P.
  5. **Split.** SEL = P[:300] and TEST-B = P[300:1300].
     - If |P| < 1300, TEST-B = P[300:].
     - If |P| < 900, stop and re-plan by amendment before anything is sampled.
  6. **Freeze.** Commit `tasks/n3_sets/testB_refcodes.txt` and `sel_refcodes.txt` together with their
     SHA-256 (amendment A1), BEFORE any arm produces any output on them. Refcodes are already public
     through the committed manifests.
- Before sampling, only these input-side operations are allowed on TEST-B / SEL:
  - truth rebuild and oracle check;
  - the §2.6 stratum labels;
  - the MCF export and export gate;
  - legacy det+1 representability (from the same G1 run);
  - OMC25-overlap labels (§3.3).
- **Truth** is `to_structure(orig)` of the item, for every arm and every set.

### 1.2 Forbidden inputs
- No arm may use a checkpoint trained on `data/csd_mol_scale` or `data/csd_mol_scale_big`. This bans
  every legacy file in `checkpoints/`, because the Phase E / F3c runs trained on corpora that contain
  TEST-B crystals.
- No arm trains on anything except the 1,687 TRAIN refcodes.
- MCF's released checkpoints are used ONLY for the competence gate (§3.2.C), never for a reported arm.

### 1.3 Denominators, failures, timeouts
- The denominator is always |set|. Arm failures count as MISSES for that arm and are listed by
  refcode. Arm failures are: cannot represent, cannot preprocess, generator error, non-finite
  coordinates.
- A HARNESS failure (matcher or energy code crashing or slow on a valid structure) is re-run in its own
  process with no wall-clock limit until it completes. No crystal is ever dropped for a harness failure.
  Every timeout and re-run is logged.
- If a per-crystal generation limit is used, it is the same for every arm (3600 s). Each arm's count of
  crystals that hit it is reported.

### 1.4 Disclosure (printed in the SI; DEV-TEST is "used during development, not confirmatory")
- (a) DEV-TEST was scored by G2 baselines, G2 eval, N1 and N2.
- (b) The N1 "rank before relaxing" idea came from a DEV-TEST diagnostic: relaxed-truth drift, median
  14.4°.
- (c) OURS' model of record was fixed after DEV-TEST had been seen.
  - The N2 retrain and the G2 checkpoints tied on VAL picked@1 (7.0 vs 7.0% per-seed mean; 9.0 vs 9.0
    pooled).
  - Both were then scored on DEV-TEST: G2 7.5%, N2 5.0% (crystal-level sign-flip p = 0.023). G2 was
    kept.
  - Applied now, and labelled post hoc, the rule "incumbent unless VAL-better; tie → VAL draw@1" also
    keeps G2 from VAL alone: draw@1 0.96 vs 0.67%, higher in every seed; any@16 13.3 vs 10.3%.
  - N2 is carried as a sensitivity row (§3.1).
- (d) OURS' DEV-TEST stratum rates were computed before the strata were frozen: mixed-handedness 8.5%,
  Sohncke 6.1%, improper-achiral 3.5% (per-seed means).
- (e) While this protocol was being reviewed, DEV-TEST was also used for several things:
  - loose-tolerance rescoring of stored OURS draws;
  - Haar floors at loose tolerance;
  - homochiral-approximant matches (1/131 strict vs 80/131 at stol 0.5, partial run);
  - the MCF perfect-model orientation ceilings;
  - steric-press drift checks.
  None of these choose anything below.
- (f) The legacy det+1 pipeline parsed the TEST-B pool in July 2026, and the old paper's scale runs
  trained on it. No legacy checkpoint is used (§1.2).
- (g) Everything in this protocol that uses DEV-TEST is secondary. H1-H3 are decided on TEST-B only.

---------------------------------------------------------------------------------------------------
## 2. Matchers, estimands, selectors, statistics, strata

### 2.1 Matchers
- **PRIMARY.** `StructureMatcher()` defaults (ltol 0.2, stol 0.3, angle_tol 5, scale=True).
  `fit(truth, candidate)` on the UNRELAXED candidate, unless §3.3 switches H2 to BASIN form.
- **SECONDARY.** MCF's rule: `StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=s)`, where a
  draw matches iff `get_rms_dist(truth, draw) is not None`. We use s ∈ {0.5, 0.8}.
  - Each stol is a separate run, because stol is a pre-filter, not a threshold on max_dist.
  - The same code path is used for every arm.
  - The Haar random-orientation floor is reported at every setting.
- One scoring script (`scripts/n3_score.py`) scores every arm. It converts each arm's output to a
  pymatgen Structure with its own adapter, and each adapter is unit-tested by the oracle gates (§3).

### 2.2 Estimands (per arm; S = 16 draws per crystal per trained model or seed base)
- **HEADLINE: seed-mean picked@1.**
  - For each seed, ONE trained model (or one Haar seed base for classical) draws 16 samples per crystal,
    and the arm's VALSEL-chosen selector (§2.3) makes ONE pick.
  - picked@1 is averaged over the 3 seeds.
  - Deterministic arms (iii-s) have m = 1.
- **Secondary (descriptive):**
  - draw@1 = the mean over draws of per-draw match. This is the exact expectation of a random pick.
  - any@16 = the oracle upper bound.
  - Ensemble picked@1: the selector over all 48 draws of the 3 seeds. It is labelled "3-model ensemble,
    48 draws" and is never quoted as a single model's accuracy.
  - any@48.
- **Continuous (descriptive):**
  - The picked candidate's RMS, from `StructureMatcher(ltol=0.3, stol=1.0, angle_tol=10, scale=False)
    .get_rms_dist` (no match = +∞), shown as an ECDF per arm.
  - For R_asym arms, the geodesic error of the picked R_asym to R0, minimised over the molecule's proper
    point group (`PointGroupAnalyzer`, tol 0.3 Å).

### 2.3 Selector menu (identical for every arm, each arm with its own analogue)
- **RANDOM.** Scored as its exact expectation: h = (matching draws)/16 = draw@1. No RNG is used.
- **LJ.** Lowest steric-LJ energy of the exact structure that is scored.
  - Code: `g2_asym_baselines.energy` (rigid_press `packing_energy_nbr`, contact 0.90, cutoff 6.0 Å),
    computed on whole-molecule Cartesian coordinates over real copies and real atoms only.
  - For pressed structures (iii-p) it is the POST-press energy.
  - For MCF output it goes through the adapter of §3.2.F5.
  - Non-finite energies rank last. Exact ties go to the lowest draw index (the rule the recorded OURS
    picks used).
- **RESID** (learned arms only). The arm's own field on the FINAL state at t = 0.975, mean over real
  copies; pick the lowest.
  - OURS: ‖torque field‖ (N1's torque_end).
  - OLD: ‖v_R‖.
  - MCF: the geodesic angle between the network's clean-rotation prediction R̂1 and the current
    rotation.
- **MODE@r**, r ∈ {0.25, 0.5, 0.75, 1.0, 1.5} Å. N1's endpoint-mode frequency (element-matched Chamfer)
  over the atoms of all copies. Ties go to the lowest RESID, or the lowest LJ for classical arms.
- **Choice.** Each arm chooses its selector on its own VALSEL draws by seed-summed hits under the primary
  matcher. Tie preference: LJ > RESID > MODE@r (smaller r first) > RANDOM.
- **OURS keeps N1's LJ selector**, which was chosen on VAL. Re-applying this rule to the N1 VAL draws
  gives LJ 21 hits vs RANDOM 2.9 vs RESID 2, so nothing changes.
- **iii-s** has a fixed physics selector (lowest FF* energy, §3.3). It is not chosen from the menu.

### 2.4 Statistics (TEST-B; primary matcher; headline estimand)
- **Unit of analysis.** The crystal. For arm A with m_A seeds: h_i^A = the mean over seeds of the
  picked@1 outcome, where a RANDOM selector contributes its expectation. d_i = h_i^OURS − h_i^A, and
  Δ = mean_i d_i.
  - Seed indices are never paired across arms.
- **Test.** Two-sided paired sign-flip test on Σ d_i, computed exactly by convolution. Every d_i is a
  multiple of 1/48; crystals with d_i = 0 drop out. For binary single-seed data this equals exact
  McNemar.
- **Family.** {H1, H2, H3}, Holm step-down at family-wise α = 0.05. An intersection-union hypothesis
  enters with the largest of its component p-values.
- **Verdicts.**
  - "OURS better": Δ > 0 and Holm-adjusted p < 0.05.
  - "ARM better": Δ < 0 and adjusted p < 0.05.
  - Otherwise: "difference not resolved at n = … (95% CI [a, b] pp)". This is never called a tie or
    equivalence.
- **Effect size.** Δ in percentage points, with a paired crystal-bootstrap BCa 95% CI (10,000
  resamples, RNG seed 0).
- **Training-seed sensitivity.** Two-stage bootstrap, reported but not a gate: first resample each arm's
  seeds with replacement, independently per arm, then resample crystals with the same indices for both
  arms. 10,000 replicates, percentile 95%. If Holm supports H_k but this interval contains 0, the claim
  is worded "for the trained models; not shown to exceed seed-to-seed training variation".
- **No significance claims** are made for secondary matchers, draw@1, any@k, ensemble metrics,
  continuous endpoints, strata, DEV-TEST, MCF-A, ablations or diagnostics. These are estimates with 95%
  CIs only. Per-seed McNemar p-values are not reported as tests.
- **Design sensitivity.** `scripts/n3_power.py` (fixed RNG) is committed with amendment A1, together with
  its printed table for the realised |TEST-B|.
  - Model: OURS at 7.5% per pick, crystal propensity Beta(1.06, 13) fitted to N1's 161/33/6/0
    crystals-by-seeds-hit, comparator heterogeneity independent (conservative).
  - Planning values from the review simulation:
    - n = 1000: power 0.84 at Δ = 2 pp (α 0.05), 0.73 at α/3; 0.97 / 0.92 at Δ = 2.5 pp.
    - n = 200: 80% power needs Δ ≈ 4-4.5 pp.
  - A non-rejection below the MDE is expected. It is not evidence of parity.

### 2.5 DEV-TEST rows
Every arm that runs on TEST-B is also run on DEV-TEST with the same frozen configuration. The results
form a secondary continuity table, carry the §1.4 disclosure, and have no significance claims. OURS on
DEV-TEST reuses the stored N1 draws.

### 2.6 Strata (frozen definitions; descriptive only; per-seed k/n and ensemble k/n with Wilson 95% CI;
strata with n < 30 are flagged "too small to interpret")
- **K**: 2 / 4 / ≥8.
- **Class**: centrosymmetric / Sohncke / other (gemmi).
- **CHIRAL** (a property of the asym conformer): refined mirror RMSD > 0.3 Å.
  - Refined mirror RMSD = the minimum of two estimates: (i) det+1 Kabsch over the first 200
    element-matched graph automorphisms; (ii) element-blocked Hungarian ICP, 30 iterations, from 5
    deterministic starts (the 4 sign-diagonal rotations and the best rotation from (i)).
  - Code: `scripts/n3_chirality.py`, a port of the review's g3_chiral2.py with only I/O changed. It is
    deterministic.
  - Labels are written to a gitignored file, and their SHA-256 is recorded in A1.
  - DEV-TEST reference counts: 178 chiral.
  - Sensitivity rows: thresholds 0.1 and 0.5 Å; the parser-capped definition (max_iso 200) alone.
- **MIXED-HANDEDNESS** = at least one improper op (det W = −1) AND CHIRAL (DEV-TEST: 137).
  **IMPROPER-ACHIRAL** = an improper op AND not CHIRAL (DEV-TEST: 19).
  - MIXED-HANDEDNESS is interpreted only under the primary matcher. Under stol 0.5 a homochiral
    approximant already matches most such cells.
  - **No handedness advantage is claimed over MCF-R, MCF-A or CLASSICAL.** MCF gets each copy's own
    mirrored conformer and a correct chi, and CLASSICAL uses exact expansion. The only structural
    handedness contrast is the old det+1 representation (Table R).
- **LEGACY-UNREPRESENTABLE** = rejected by the legacy det+1 parser (the G1 `old` flag).
- **FORMULA-DISJOINT** = the molecular formula does not occur in TRAIN.
- **OMC25-NONOVERLAP**: see §3.3.

---------------------------------------------------------------------------------------------------
## 3. Arms

### 3.1 (i) OURS (frozen; arm of record)
- **Checkpoints.** G2 `eval_s{0,1,2}.pt`.
- **Sampling.** 16 draws, 40 Euler steps, batched sampler (`g2_rank.sample_batch`), per-crystal seed
  80000 + index on TEST-B and 90000 + index on SEL. DEV-TEST reuses the stored N1 draws.
- **Selector.** N1's LJ rule.
- **OURS-N2 (sensitivity row).** The three N2 checkpoints chosen on VAL, with the same sampling,
  selector and scoring. For each of H1-H3 the paper reports the outcome for both rows. A hypothesis that
  holds for G2 but not for N2 is worded "holds for the model of record, not for the 30-epoch retrain".

### 3.2 (ii) MolCrystalFlow
**Code base.** MCF @3c493f8, a gitignored fork. The diff is published in the SI. The only permitted
changes are:
- **P1.** Add the elements {12 Mg, 33 As, 34 Se, 52 Te} → indices 12-15 in all three atomic-number maps:
  `data-preprocess/thurlemann23/common.py:35`, `molcrystalflow/data/utils.py:13` and
  `molcrystalflow/data/dataset.py:13`. The inverse maps are derived from these. Set
  `model.bb_embedder.num_atom_types = 16` in our training and inference yamls.
- **P2.** In `interpolant.corrupt_batch`, when trans.corrupt=False, set `b_trans = zeros_like(trans_1)`.
  This fixes an unbound-variable crash at interpolant.py:419-424.
- **P3 (MCF-R only).** A new key `inference.gt_lattice_trans`. When it is set:
  - trans_0 = trans_1 and lattice_0 = lattice_1;
  - trans and lattice are overwritten with the truth after every Euler step;
  - pred_trans_1 = trans_1 and pred_lattice_1 = lattice_1 at final assembly (molcrystalflow.py:396-406).
  Rotations are untouched.
- **P4.** The GPUtil device calls (`experiments/utils.py:18`, `inference.py:53,100`) return [0].
- **P5.** `pl.seed_everything(experiment.seed)` runs BEFORE FlowModule is constructed. As released,
  train.py:48 builds the model before seeding at :50-52.
- **P6 (speed, optional).** A vectorised `_symmetric_so3` prior and a scatter-softmax
  `AttentionPooling`. They are used only if both of these hold on the box GPU: (a) the prior is
  bit-identical under a fixed numpy seed; (b) pooling outputs and input gradients agree to 1e-5 on 3 real
  TRAIN batches. Which version ran is reported.
- **Also:** save_trajectories=False, and a unique run name per training run
  (`n3_{mcfR,mcfA}_s{s}`). `runs/<name>/ckpt` is otherwise shared, and runs overwrite each other's
  last.ckpt and config.yaml.

**Export (`scripts/n3_mcf_export.py`, committed; pickles gitignored).** For each item:
1. Start from `orig`. Build whole molecules X_k = centroid_k·L + orient_k·local_k (real slots only).
2. Centred crystals (C/F/I lattices) are exported in the PRIMITIVE cell (`get_primitive_structure`,
   z = K / multiplicity). The truth is unchanged, since StructureMatcher reduces to the primitive cell.
3. `sort_cell_left_matrix`, storing Q and det Q per crystal.
4. `common.preprocess_structure` with bb_indices = our slot index. MCF's `get_molecule_groups` is NOT
   used: on DEV-TEST it merges 1 crystal and raises on Te in another.
5. Per-copy features through MCF's own functions, unchanged, called once per copy: obabel xyz→sdf,
   `SDMolSupplier(removeHs=False, sanitize=False)`, `fix_mol_bonds(mol, MolToSmiles(mol))`,
   `extract_features_single_molecule`, `.get(k, 0)`.
   - When bond fixing fails, MCF's unfixed-molecule fallback is kept.
   - The xyz→sdf count is asserted 1:1.
   - obabel must succeed (`obabel -V`), either from openbabel-wheel or conda.
6. `renormalize_single_datapoint`.
7. MCF's batch drivers (`generate_training_data`, `preprocess_all_structures`) are NOT used. They stop at
   the first exception and delete mismatched entries.
8. Output: exactly |set| entries (asserted) plus a refcode-order sidecar. The directories are
   `n3_mcf/trainval`, `n3_mcf/valsel_as_test` (VALSEL written as `test_molcrystal_normalized.pkl.gz`,
   because inference.py:106 hard-codes test.pt), `n3_mcf/test` (TEST-B) and `n3_mcf/devtest`.

**Gate F (export), run before any GPU time; counts committed as `results/n3/gateF.json`.**
- F1: set counts and sidecar order; all atom types satisfy 0 ≤ t < 16; every copy has basic (2),
  chemical (8) and geometric (4) arrays.
- F2: species are decoded from the PICKLE through the scorer's inverse map. `assemble_coords` must
  reproduce the standardized input atom by atom to ≤ 1e-3 Å (mod one lattice translation per molecule)
  for 100% of items. StructureMatcher() of the decoded structure vs truth must be 100% on VAL, SEL,
  TEST-B and DEV-TEST.
- F3: chi vs `orig` parity (up to a global relabel). Reported, not gated.
- F4: feature status per copy: obabel failure / fix strategy 1-3 / fallback / descriptor exception.
  GATE: copies with an all-zero basic+chemical vector ≤ 1% per set.
- F5 (LJ adapter): energy() from MCF-frame whole molecules (orient = I, local = X_k − mean X_k,
  cent = frac(mean X_k), L = pickle lattice) must equal energy() of `orig` to 1e-4 relative, for all
  VALSEL + TEST-B + DEV-TEST crystals.
- F6 (scorer oracle): a predictions file built from the pickle's own poses, run through `n3_score.py`,
  gives 100% picked@1, draw@1 and any@16 at every matcher setting.
- On failure, fix the glue and re-run the gate. A test crystal that still fails is an MCF MISS and is
  listed. Every MCF comparison is then also reported on the passing subset.
- SHA-256 of every pickle is recorded and re-checked on the box before training and before sampling.

**Gate C (competence and patch equivalence), on the box before any N3 MCF training.**
- C-a: download the released Thurlemann checkpoint and data (Zenodo 19673190). Run it with our patched
  code, flag OFF and stock Thurlemann config, on the released Thurlemann TEST pickle.
  - Sampling: 50 steps, s_uF 9, s_uR 3, 5 runs × 10 samples (inference seeds 0-4).
  - Score any-of-10 with `n3_score.py` at MCF's settings AND with MCF's own `run_structure_matching.py`.
  - PASS if (i) the two scorers agree on ≥ 99% of (crystal, stol) outcomes, and (ii) the 5-run median
    at stol 0.9 and 1.0 is within ±5 pp of Fig. 3a (≈30% / ≈66%, read off the box plots at ±2 pp).
    stol 0.8 (≈5.5%) is reported, not gated.
  - The released VAL pickle is also run, because the paper does not state which split Fig. 3 used. If
    only VAL agrees, the gate passes and this is disclosed.
  - On failure: stop and diagnose (config merge, sampler knobs, split). If no cause is found, N3
    proceeds, and the reproduced numbers are printed next to Fig. 3a in the SI.
- C-b: on CPU, seeded, flag OFF: the patched forward equals unpatched 3c493f8 bitwise, and one training
  step gives an identical loss. On GPU: the patched-vs-unpatched difference is no larger than
  unpatched-vs-unpatched at the same seed.
- C-c: with P3 ON, every output lattice and centroid equals the input truth bitwise, on the tensors saved
  in predictions_16.pt.

**(ii-R) MCF-R: lattice + all K centroids held at the truth; orientations generated.**
- **Training.** `molcrystal.yaml` as released, with ONLY these overrides:
  - `interpolant.trans.corrupt=False interpolant.lattice.corrupt=False`;
  - `experiment.training.translation_loss_weight=0 experiment.training.cell_loss_weight=0`;
  - `experiment.seed={0,1,2}`;
  - `experiment.trainer.max_epochs=1715`. That is 24,010 optimizer steps at ceil(1687/128) = 14 steps
    per epoch. MCF's Thurlemann run was 300 epochs × 79 steps = 23,700. The released max_epochs=1000
    would stop at 14,000.
  - `experiment.lr_scheduler.patience=61`. This is a unit conversion: ReduceLROnPlateau fires on the
    (p+1)-th bad reading, i.e. 11 × 79 = 869 steps in MCF's run and 62 × 14 = 868 steps here.
  - Everything else stays as released: factor 0.6, min_lr 1e-6, validation every epoch on MCF's own
    random draw of VAL, and ModelCheckpoint save_top_k 3 on valid/loss plus last.
- **Fixed-draw VAL loss (tie-break only).** The model_step over VAL × 8 draws, inside
  `torch.random.fork_rng()` with `torch.manual_seed(7)` and `np.random.seed(7)`. The numpy state is saved
  and restored around it.
- **Reported per seed:** the LR trace, the valid/loss curve, and the step of the 9th LR reduction (the
  effective floor 1.008e-6) or "not reached".
- **Sampling harness.**
  - Own yamls: `ours_inference_R.yaml` composes `ours_molcrystal_R.yaml`. inference.py merges the launch
    config OVER the checkpoint config, so the stock inference.yaml is never used.
  - Every call sets `inference.ckpt_path`, `data.cache_dir`, `inference.num_samples=16`,
    `interpolant.sampling.num_timesteps=50`, `interpolant.rots.sample_schedule=exp`,
    `interpolant.rots.exp_rate=<s_uR>`, `interpolant.trans.scaling=9`, `+inference.gt_lattice_trans=True`,
    a unique `inference.output_dir` and `inference.inference_dir`, `inference.save_trajectories=False`,
    `data.loader.num_workers=4`, and `inference.seed = 100000 + 1000·s + split`
    (split: 0 VALSEL, 1 TEST-B, 2 DEV-TEST).
  - Every predictions file is asserted before scoring: 16 draws; |set| crystals in sidecar order
    (lattice_1 and num_atoms match the pickle); the clamp is exact (lattice == lattice_1 atol 1e-5;
    trans == trans_1 mod 1, atol 1e-5); and the resolved config holds the values above. A failing file is
    discarded and regenerated, never scored.
- **Selection on VALSEL, primary matcher; frozen in `n3_selection.json`.**
  - Stage A, shared knobs pooled over the 3 seeds:
    - s_uR ∈ {1, 3, 10}: 3 is the paper value, 10 the released exp_rate, 1 the low end of the paper's
      optimum band.
    - Crossed with the §2.3 selector menu, using each seed's lowest-valid/loss checkpoint.
    - Keep the highest 3-seed summed hits.
    - Ties: higher summed draw@1 → higher summed any@16 → s_uR = 3 → selector preference.
  - Stage B, checkpoint per seed with the Stage-A settings frozen:
    - Candidates: top-3 valid/loss and last.
    - Keep the highest VALSEL picked@1.
    - Ties: draw@1 → any@16 → lower fixed-draw VAL loss.

**(ii-R+G) MCF-R with gauge scan (deployable; no truth used; MCF-favourable).**
- Uses the same checkpoints and draws as MCF-R, with no retraining.
- **Why.** MCF's network sees the lattice only through L Lᵀ and orientations only through R_iᵀR_j, and
  its symmetric prior conjugates copy i by g_(i mod 4) ∈ {I, C2z, C2y, C2x} in its standardized
  Cartesian frame. The sampled orientation distribution is therefore invariant under the commutant H_K:
  - K = 2: {R_z(θ)} ∪ {R_z(θ)·C2x}, i.e. O(2) about the standardized z (the normal to the two SHORTEST
    cell vectors, which is NOT the CSD c*);
  - K ≥ 3: D2 = {I, C2x, C2y, C2z}.
- **Procedure.**
  1. For each draw, evaluate the unrelaxed LJ of h·{R_k} with lattice and centroids fixed, for every h in
     H_K: K = 2 both cosets on a 2° grid (360 members); K ≥ 3 the 4 D2 elements.
  2. Keep the lowest-LJ member.
  3. Then apply the arm's selector. The selector is chosen separately for +G by the Stage-A rule;
     checkpoint and s_uR are shared with MCF-R.
- LJ evaluations per crystal are disclosed for every arm: OURS 16; MCF-R 16; MCF-R+G 5,760 (K=2) or 64.

**(ii-S) MCF-R + post-hoc SG expansion (ABLATION; descriptive; H1-S).**
- Runs on CPU from the MCF-R+G draws.
- **Procedure.**
  1. Map to our frame: X = X_MCF·Qᵀ.
  2. Assign each slot to op k by centroid match to W_k c0 + t_k (distinct in 200/200 DEV-TEST).
  3. Y_k = Rc_k⁻¹(X_k − cent_k).
  4. R^(k) = det+1 Kabsch of the asym local onto Y_k. On true poses this recovers R0 in 200/200, max
     0.014°.
  5. Expand each R^(k) exactly, and keep the lowest-LJ one. This gives ONE candidate per draw.
- The selector is chosen on VALSEL.
- It is labelled "MCF orientations inside our exact SG expansion": a learner comparison within our
  framework, not a test of MCF.

**(ii-A) MCF-A: released joint objective (CONTEXT ONLY; separate table; never in H1-H3).**
- **Training.** Same export, P1/P2/P4/P5/P6, the same step budget, patience conversion and 3 seeds.
- **Lattice prior.** Refit with `data-preprocess/lattice_fit.py` on the 1,687 TRAIN cells after
  standardization. Assert n = 1687 and loc sorted. Angles keep the released uniform prior.
- **Harness.** `ours_inference_A.yaml` composes `ours_molcrystal_A.yaml`. The merged config is asserted
  (lattice loc/scale, num_atom_types, cache_dir, knobs) before sampling.
- **Metric.** MCF's native get_rms_dist rule at stol 0.5, 0.6, …, 1.2, plus the primary matcher.
  any-of-10 = the first 10 of the 16 draws.
- **Selection** (VALSEL; a degenerate primary matcher is expected, so MCF's own metric is used):
  - Stage 1: checkpoint at (s_uF 9, s_uR 3) by any-of-10 at stol 0.9. Ties go to stol 0.8, then to lower
    valid/loss.
  - Stage 2: (s_uF, s_uR) ∈ {5, 9, 13} × {1, 2, 3}; ties go to (9, 3).
  - The selector is chosen by picked@1 at stol 0.9.
- **LJ on generated cells.** Draws with a minimum interplanar spacing < 2.5 Å get e_lj = +∞, and their
  count is reported.
- **Prior floor.** At every setting: refit lognormal lattice + uniform centroids + symmetric-prior
  rotations, 16 draws per crystal.

**MCF diagnostics (pre-registered; oracle items labelled as such; not results).**
- **(a) Equivariance.** On every selected MCF-R and MCF-A checkpoint, in float64 on CPU: 10 VALSEL
  crystals × 3 random G ∈ SO(3).
  - Require max|pred_R(G·R) − G·pred_R(R)| ≤ 1e-6, and lattice/translation output change ≤ 1e-6.
  - The 50-step sampler deviation is reported as a number.
  - Provenance: repeat once with the released Thurlemann checkpoint on its own test pickle.
  - If (a) fails, the invariance statement is withdrawn and (b)-(d) are not reported.
- **(b) Perfect-model ceiling**, computed once from truth for every MCF arm. For each crystal, c_i = the
  Haar fraction of H_K members whose common rotation of the true frames (in the pickle frame) still
  matches truth.
  - K = 2: scan both cosets on a 5° grid, then walk at 1° to the window edges; c_i = (w_rot + w_flip)/720°.
  - K ≥ 3: c_i = (# matching D2 elements)/4.
  - Report mean c_i (the draw@1 ceiling) and mean[1 − (1 − c_i)^16] per K and per matcher.
  - c_i does NOT bound LJ-picked@1.
- **(c) Oracle-aligned score** (MCF-R only). Per draw, apply the h ∈ H_K minimising Σ_k ‖h R_k −
  R_k^true‖²_F, with R^true = rotmats_1 of the same pickle. This is closed-form in t per coset for K = 2.
  Then match. Report draw@1, any@16 and picked@1 per K and matcher.
- **(d) Consistency.** MCF-R's unaligned draw@1 must not exceed the (b) ceiling beyond its bootstrap CI.
  An exceedance means (b) is wrong; it is reported and fixed before (c) is interpreted.

**Paper wording, conditional on (a) passing:** "At fixed lattice and centroids, the released
MolCrystalFlow network and its Euler sampler are equivariant, to numerical precision, to a common left
rotation of all molecular frames. The sampled orientation distribution is therefore invariant under
rotations about the z axis of its lower-triangular lattice frame plus the in-plane two-folds for Z = 2,
and under D2 for Z ≥ 3. This holds for the released model and its native joint sampler; it is not
introduced by our modification."

### 3.3 (iii) CLASSICAL (same information as OURS)
All classical variants use the true L, the true c0, the spglib ops and the rigid conformer. Only R_asym
moves, and every copy comes from exact expansion.
- **(iii-u) FLOOR.** 16 Haar R_asym per crystal × 3 seed bases, unrelaxed. Selector from the menu, chosen
  on VALSEL. Draws use `torch.Generator().manual_seed(600000 + 10000·b + 1000·split + i)`.
- **(iii-p) STERIC PRESS (continuity with G2; descriptive).** The G2 press (80 Adam steps, lr 0.02,
  contact 0.90, cutoff 6.0) on the iii-u draws. The selector uses the POST-press LJ. Scored vs truth.
  - Stated ceiling: the pressed TRUE pose matches the truth in 2/198 (G2).
- **(iii-s) PHYSICS SEARCH (the H2 comparator; deterministic, m = 1).**
  - **Candidates.** The Yershova/Hopf SO(3) grid at resolution level 2 (4,608 orientations, ~15°
    spacing), pre-rotated by one fixed Haar rotation (`torch.manual_seed(20260927)`). Every candidate is
    expanded exactly.
  - **Screen.** All 4,608 by unrelaxed steric-LJ energy. Keep the 64 lowest.
  - **Minimise.** Orientation-only rigid-body minimisation (δ ∈ so(3); L and c0 fixed; the whole cell
    moves by exact expansion) under **FF* = UMA s-1p1, OMC task** (fairchem). This is the potential MCF's
    paper uses for its own rigid-body relaxation (UMA-Opt).
    - L-BFGS with at most 100 energy+gradient evaluations.
    - Converged iff ‖∂E/∂δ‖ < 0.01 eV/rad. Otherwise the run is recorded UNCONVERGED, and its final pose
      is still used.
  - **Rank.** Merge minima within 2° of each other, keeping the lower energy, then rank by FF* energy.
    picked@1 = the lowest-energy minimum. any@16 = the 16 lowest distinct minima.
  - **Weights.** The UMA checkpoint is downloaded with Frank's Hugging Face access, and its SHA-256 is
    recorded. The HF token never goes onto a rented box; only the checkpoint file is uploaded.
  - **OMC25 overlap.** UMA-OMC's training set OMC25 is built from OE62 molecules, which were extracted
    from the CSD.
    - Report the number of TEST-B molecules whose InChIKey (connectivity layer) occurs in the public
      `facebook/OMC25` starting-crystals list. If that list carries no usable identifier, say so.
    - Report H2 descriptively on the non-overlapping subset.
    - Also report the TEST-B crystals with elements outside OMC25's element set.
- **FF* CEILING, measured on VAL before any iii-s TEST-B run; decides the H2 FORM.**
  - Anchoring = the fraction of VAL crystals whose TRUE pose, minimised as above, both CONVERGES and still
    matches the truth under the primary matcher. No-ops and unconverged runs do not count.
  - If anchoring ≥ 50%, H2 FORM = EXACT: every arm's pick is scored unrelaxed vs the truth; iii-s's pick
    is its minimum.
  - If anchoring < 50%, H2 FORM = BASIN: every arm's picked candidate is FF*-minimised with the same
    settings and matched (primary matcher) to the FF*-minimised truth. OURS' pick is its LJ-picked draw;
    iii-s's pick is its lowest minimum.
  - The other form is reported as secondary. The TEST-B anchoring rate is reported afterwards, next to
    the VAL value.
- **(i-E) OURS+FF* (matched energy; descriptive).** All 16 OURS draws per seed are FF*-minimised; pick the
  lowest FF* energy. Reported against iii-s with a CI. This separates the learned proposal from the
  energy model.
- **Steric BASIN (continuity with G2; descriptive).** For OURS and iii-p:
  - P1 = the G2 press on the arm's own draws. Basin reference = press(true R0).
  - Hit = StructureMatcher() fit(pressed draw, pressed truth).
  - P1 selector menu on VALSEL: random; lowest pre-press LJ then press; lowest post-press LJ.
  - G2's DEV-TEST result is printed beside it: basin_best 10.1/8.5/7.0% vs Haar 6.5%, p = 0.19/0.54/1.0;
    basin_any 22.6/19.6/17.6% vs 11.1%.

### 3.4 (iv) OLD symmc-flow architecture: iv-P (primary OLD arm)
- **Items.**
  - Built for every TRAIN/VAL/SEL/TEST-B/DEV-TEST crystal from the asym item and its spglib ops, NOT from
    `orig`. On 11/1687 TRAIN crystals `orig` disagrees with the ops.
  - For op k, with s_k = [det W_k = −1] and D = diag(1,1,−1):
    - centroid_k = frac(W_k c0 + t_k);
    - local_k = D^{s_k}·local;
    - target orient_k = Rc_k·R0·D^{s_k} (proper);
    - Z / atom_mask / mol_mask padded to 16 × 64.
  - Keys `coset` and `idx` are added, and the item is asserted all-tensor before collate.
  - The old EGNN is reflection-invariant, so mirrored copies are identified only through the coset id.
    This is the old design, run on the O(3) data.
- **Coset ids.**
  - Reference = the identity-op copy (id 0, like padding).
  - Every other copy gets the id of key (sg, W_k as integers, round(24·t_k) mod 24) in a table built from
    TRAIN only.
  - Keys unseen in TRAIN map to one reserved id whose embedding row is zeroed. Their count is reported.
  - `assign_symmetry_cosets` is NOT called: it uses the standard-setting table.
- **Model.** Published ModelConfig defaults, n_cosets = table size + reserved, lambda_orient = 1,
  **lambda_lattice = lambda_centroid = 0**. This is a disclosed deviation: under clean packing those heads
  train on inconsistent targets and would contaminate the val loss.
- **Train** (`scripts/n3_old_symmc.py`; our own loop, because `train()` returns only the final model).
  - cond_clean_packing=True, batch 16, AdamW (weight decay 0), grad clip 1.0, constant lr, fixed split.
  - lr ∈ {1e-4, 3e-4 (published), 1e-3}, chosen on seed 0 by the lowest in-window fixed-draw val loss.
    Seeds 1-2 use that lr, and the seed-0 run is reused.
  - 60 epochs over TRAIN, with a checkpoint and val loss at every epoch end.
  - Candidate window = epochs 1-30. It becomes 1-60 for ALL seeds if, for ANY seed, the epoch-30 val loss
    is more than 2% below the epoch-25 val loss.
- **Fixed-draw val loss.** The orientation loss on VAL × 16 draws. The per-slot R0 and t are generated
  ONCE: `torch.manual_seed(12345)` immediately before `random_so3((100,16,16))` and `rand((100,16))`.
- **Checkpoint per seed.** Candidates {best val loss in window, last of window}. Choose by VALSEL
  picked@1; ties go to the lower val loss.
- **Sampling.** `sample_orient_only` (RK4 on SO(3), 50 steps), true lattice and all centroids frozen.
  16 draws per crystal, each with K independent per-copy Haar priors, seeded by
  `torch.manual_seed(700000 + 10000·s + 1000·split + i)`. No post-hoc symmetrization. Structures and LJ
  use the per-copy local over real slots.
- **(iv-S) diagnostic** (descriptive): take the identity-copy orientation as R_asym and expand it exactly.
  The selector is chosen on VALSEL.
- **Smoke gate (CPU, before GPU spend).**
  - True orientations rebuild 100% of VAL/DEV-TEST truths.
  - Slot → op is a bijection with residual < 1e-3 on VAL/SEL/TEST-B/DEV-TEST.
  - Two training steps give a finite loss.
- **Not run in N3:**
  - iv-A (legacy det+1 items, 598 train). Its coverage limit is reported deterministically in Table R.
  - iv-R, the relative-gauge audit of the published method, in which R_asym is supplied.
  Both are stated as not run.

---------------------------------------------------------------------------------------------------
## 4. Hypotheses (confirmatory family; TEST-B; §2.4 test; Holm over three)
- **H1 (learned MCF comparator):** OURS vs MCF-R AND OURS vs MCF-R+G. This is intersection-union, so
  p_H1 = max of the two. OURS must beat both. No MCF variant is selected.
- **H2 (physics search):** OURS vs iii-s, in the H2 FORM fixed on VAL (§3.3).
- **H3 (old architecture):** OURS vs iv-P.

Each hypothesis is reported whatever the outcome, together with the OURS-N2 row, the two-stage seed
interval and the 95% CI.

**Stated in advance (not results):**
- iii-u is expected at ≈ 0 exact under the primary matcher. A Haar draw falls inside the matcher's
  ~6-8° radius with p ≈ 1e-4.
- iii-p is capped by the steric force field.
- For MCF-R, the perfect-model draw@1 ceilings on DEV-TEST are ≈ 1.5-3.7% (K=2, both cosets) and ≈ 25%
  (K ≥ 4). The perfect-model LJ-picked@1 on DEV-TEST truth is ≈ 16% / 82%.

**Outcome → claim map (fixed now):**
- H1 fails, or holds against MCF-R but not MCF-R+G: no claim that a learned asymmetric unit beats learned
  independent copies. Only Table R and H2/H3 are claimed.
- H1 holds overall but the K ≥ 4 stratum does not favour OURS: the discussion states that the advantage
  concentrates in Z = 2, where MCF's invariance ceiling binds, and cites diagnostic (b). There is no
  stratum-level significance claim.
- H1 holds but H1-S (ii-S) matches OURS: the paper attributes the gain to exact expansion rather than to
  our learner.
- H2 fails in its fixed FORM: no "better than physics-based search" claim. If it holds in the EXACT form
  only because FF* cannot anchor (which the ceiling rule prevents), it is worded "reproduces experimental
  poses the potential does not", never "beats physics".
- H3 fails: no claim over the old architecture beyond Table R coverage.
- If H1, H2 and H3 all fail: the paper becomes a representability / negative-result note (G2's kill
  logic).

---------------------------------------------------------------------------------------------------
## 5. Representability (Table R; deterministic; computed on TEST-B and DEV-TEST before any sampling)
Rows: all / centro / Sohncke / other / MIXED-HANDEDNESS / IMPROPER-ACHIRAL / LEGACY-UNREPRESENTABLE.
Columns are counts of crystals whose TRUE structure each representation can emit, under every §2.1
matcher:
- (a) OURS = CLASSICAL, `expand(a, R0)`. The caption says this is 100% BY CONSTRUCTION, because the
  eligible set is defined by this representation.
- (b) MCF per-copy representation: the Gate F oracle.
- (c) The information of MCF's released CSP driver (one conformer and one χ for all copies;
  packing_gen.py:376-388). This is the best homochiral approximant: true lattice and centroids, and each
  improper copy replaced by the RMSD-optimal proper rotation onto its mirror (element-blocked ICP, 16
  starts). It is an upper bound for any homochiral cell. No arm runs this driver.
- (d) OLD det+1: the legacy parser keep-flag, and the oracle rebuild for kept crystals.
- Per-crystal timeout 600 s; timeouts are reported as their own count.
- Corpus context row (G1/G2): 3,500 CIFs → 1,144 kept by det+1, 2,852 by O(3), 1,987 eligible. MCF's
  per-copy representation also covers the Z' > 1 and special-position crystals our design excludes.

**Frozen wording.** "The legacy det+1 pipeline cannot represent N/n test crystals (M of the centrosymmetric
ones). The released MolCrystalFlow CSP driver cannot produce P/n_mixed mixed-handedness cells under the
default StructureMatcher; a homochiral approximant nevertheless matches Q/n_mixed at stol 0.5."

**Forbidden statements:**
- any statement that MCF's benchmark path, MCF-R or MCF-A cannot represent mixed-handedness cells;
- any "prior pipelines" wording beyond rows (c) and (d);
- any representability claim relative to MCF's per-copy representation.

The text states that the MCF paper describes a mixed-χ CSP scenario whose code is not released at
3c493f8.

---------------------------------------------------------------------------------------------------
## 6. Reporting plan
**TABLE 1 (headline): TEST-B, primary matcher, H2 in its FORM.** Rows, in order:
1. random floor;
2. iii-u;
3. iii-p;
4. iii-s;
5. iv-P;
6. MCF-R;
7. MCF-R+G;
8. OURS;
9. OURS-N2 (sensitivity).

Columns:
- Table 0 information ticks;
- representable n/N;
- seed-mean picked@1 ± s.d. over seeds;
- Δ vs OURS with BCa CI;
- Holm-adjusted p and verdict for H1-H3;
- two-stage seed interval;
- draw@1;
- any@16;
- per-crystal compute (sampling + selection wall-clock, and LJ/FF* evaluations). Training GPU-h goes in a
  footnote.

**TABLE 2 (descriptive).** Picked@1 by K, class, MIXED-HANDEDNESS, FORMULA-DISJOINT and OMC25-NONOVERLAP.
Each K row carries MCF's diagnostic-(b) ceiling, labelled oracle.

**TABLE R** (§5). **CONTEXT TABLE** (MCF-A only, and MCF's native stol sweep).

**SI:**
- Tables 1-2 under the secondary matchers with floors;
- ensemble/any@48;
- the continuous endpoints;
- ii-S, i-E, steric basin, iv-S;
- MCF diagnostics;
- DEV-TEST tables;
- the full VALSEL selection grids;
- gates F and C.

**Abstract rules.**
- The abstract quotes OURS' Table-1 picked@1 (seed mean ± s.d.) and the outcome of H1, H2 and H3, each
  with Δ and CI.
- For an intersection-union hypothesis, the Δ quoted is the smaller one.
- Failed or unresolved hypotheses are stated as such.
- No stratum, secondary-matcher, ensemble or any@k number appears in the abstract as a result.

---------------------------------------------------------------------------------------------------
## 7. Execution order, compute, budget, hygiene
1. Commit and PUSH this file. The push is the third-party timestamp.
2. Build TEST-B/SEL (§1.1), compute labels, gates F and the OLD smoke gate, Table R, and the power table.
   Commit and push amendment A1 (refcode lists + SHA-256, gate results, power table).
3. Implement and CPU-smoke every arm locally. Local shims (torch_scatter, radius_graph) are for smoke
   tests only; no reported number comes from a shimmed run.
4. **Timing gate and approval.** On a rented single-GPU box, time MCF-R for 300 steps (steps 50-300) and
   FF* evaluations in the exact configuration that will be run. Project the total cost.
   - **The projection goes to Frank. Nothing larger than the timing gate starts without his approval.**
     The Vast balance is shared with other projects.
   - CSD uploads happen only in manual-approval mode.
5. Upload TRAIN, VAL and SEL inputs only. TEST-B and DEV-TEST inputs stay off the box. Then:
   - run gate C;
   - train MCF-R ×3, MCF-A ×3 and iv-P (lr grid, 3 seeds);
   - measure the FF* ceiling on VAL, which fixes the H2 FORM;
   - sample VALSEL for every candidate cell of every arm (classical included) and score VALSEL.
6. Write `results/n3/n3_selection.json`. It holds every VALSEL cell score, every frozen choice, the H2
   FORM, the list of permitted TEST-B / DEV-TEST runs and their seeds. Commit and PUSH it BEFORE any
   TEST-B or DEV-TEST input reaches the box.
7. Upload the TEST-B and DEV-TEST inputs. Run exactly the listed runs with their committed seeds. A run
   not listed is disclosed, and that arm's run is invalid.
8. Pull everything back and verify checksums. Delete every CSD-derived file on the box (inputs,
   exports, checkpoints, outputs), confirm by search, and destroy the box.
9. Score TEST-B and DEV-TEST once, write `results/n3/n3_results.json`, commit and push.

**Failures.**
- If the box dies before step 6, resume from step 5 with the same seeds.
- If it dies during step 7, discard every partial TEST output unscored and re-run step 7 on a new box
  from the pushed file.

**Budget.**
- Prior estimate, NOT measured: MCF 6 runs × 1-4 GPU-h, FF* ≈ 8-15 GPU-h, iv-P < 1 GPU-h, plus
  sampling and scoring. That is ≈ 20-45 box-hours at $0.3-0.6/h.
- If the approved cap binds, the degrade order is fixed now:
  1. drop the §8 tier-2 diagnostics;
  2. MCF-A seed 0 only;
  3. MCF-A without its knob grid;
  4. iii-s minimises the 16 lowest instead of 64;
  5. drop the DEV-TEST rows for new arms.
- Never reduced: |TEST-B|, OURS, MCF-R(+G), iii-s (beyond step 4), iv-P, and 3 seeds for every H1-H3 arm.

**Hygiene.**
- Every CSD-derived file (CIFs, pickles, exports, checkpoints, draws) stays local or on the box, and is
  gitignored.
- Only JSON summaries, refcode lists, code and this file are committed.
- Never `pgrep -f` / `pkill -f` inside ssh; kill by PID.

---------------------------------------------------------------------------------------------------
## 8. Tier-2 diagnostics (run only after steps 1-9, only if the approved budget allows; descriptive)
- **Data-size curve.** OURS (G2 code) and MCF-R, seed 0, on nested TRAIN prefixes of 422 / 843 / 1687.
  Report VAL loss, VAL picked@1 and, for MCF-R, oracle-aligned VAL any@16. No pre-written conclusion.
- **Centroid-noise sensitivity (D-c0).** OURS and iii-u on a seeded 200-crystal TEST-B subset
  (torch seed 20261002). σ ∈ {0.1, 0.25, 0.5} Å Cartesian noise on c0, with the same δ for every arm.
  Report (a) the picked R_asym re-placed at the true c0 (primary matcher), (b) the as-generated
  structures, and (c) the median geodesic error.
- **MCF conventional-cell export** for centred crystals (a sensitivity row for the primitive-cell
  choice).

---------------------------------------------------------------------------------------------------
## 9. Amendments
(none yet)
