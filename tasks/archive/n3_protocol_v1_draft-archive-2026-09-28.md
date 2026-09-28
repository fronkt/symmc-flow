<!-- ARCHIVE (2026-09-28): v1 DRAFT of the N3/G3 protocol, written 2026-09-27 before the five-lens review.
     Superseded by tasks/n3_protocol.md (v2). Kept as provenance for the pre-registration: it shows what
     changed after review. Never executed. -->

# N3 / G3 benchmark: pre-registered protocol (DRAFT v1, 2026-09-27; nothing of N3 has been run)

## 0. Question
Given the TRUE lattice, the TRUE molecule centroids and the rigid conformer, how often does each method
produce the exact crystal? This is the information level of our model (orientation-only). End-to-end
generation (lattice + positions too) is a different task. It appears only as a labelled context arm (ii-A).

## 1. Test set, truth, disclosure
- TEST = the fixed seed-0 split of `data/csd_mol/g2_asym.pt`: test = perm[:200], val = perm[200:300],
  train = perm[300:] (1687). These are Z'=1 general-position crystals that the O(3) parser keeps.
  That selection is ours and favours an asym-unit method. It must be stated.
- Every arm is scored on all 200. If an arm cannot represent, process or finish a crystal, it counts
  as a MISS for that arm. The denominator is always 200.
- Truth = `to_structure(orig)` for every arm (the same object N1 used).
- Every arm trains on exactly the 1687 train refcodes. Every selection (checkpoint, selector,
  sampler knob) is made on VAL only. TEST is sampled and scored once per arm.
- Disclosure: this test split was scored before in G2, N1 and N2. One test diagnostic (G2's relaxed-truth
  drift, median 14.4°) motivated the N1 "rank before relaxing" idea. The N1 rule itself was chosen on val.
  No G3 choice uses test.

## 2. Matchers and metrics
- PRIMARY matcher: `StructureMatcher()` defaults (ltol 0.2, stol 0.3, angle_tol 5, scale=True), applied
  to UNRELAXED output. This is the N1 metric of record.
- SECONDARY matcher: MCF's matcher `StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=s)` for
  s in {0.5, 0.8}. One scoring script is used for every arm. Each stol is a separate run, since stol is a
  pre-filter, not a threshold on max_dist.
- A random-orientation floor (Haar R_asym, unrelaxed) is reported at every matcher setting.
- Per arm, per seed, with S = 16 draws per test crystal:
  * **picked@1**: ONE deployable pick per crystal. The selector is chosen on the arm's own VAL draws
    from the menu {random, lowest unrelaxed steric-LJ energy (rigid_press energy, same code)}.
    Tie → random. Headline metric.
  * **draw@1**: the mean over the 16 draws of per-draw exact match (unranked single-draw rate).
  * **any@16**: at least one of the 16 draws matches (oracle upper bound, not deployable).
  * Also pooled over the 3 seeds (48 draws) for picked@1 and any@48.
- Stats: paired exact McNemar per seed (ours seed s vs arm seed s) on picked@1. "Significant" means
  p < 0.05 in all 3 seeds. Also report mean ± s.d. over seeds and a crystal-bootstrap 95% CI
  (10k resamples) of the seed-mean difference.
- Strata, with definitions frozen now:
  * K ∈ {2, 4, ≥8}.
  * centrosymmetric / Sohncke / other.
  * MIXED-HANDEDNESS: at least one improper-op copy, AND the conformer is chiral. Chiral means the det+1
    fit of the mirrored conformer onto itself (`_align_to_reference`, max_iso = 200, the parser's own
    setting) exceeds conf_tol = 0.3 Å.
  * FORMULA-DISJOINT: the test crystals whose molecular formula does not occur in train (176/200).
    This is a robustness row.

## 3. Arms
(i) **OURS**: G2 checkpoints `eval_s{0,1,2}.pt`, 16 draws, 40 Euler steps, N1 selector (lowest unrelaxed
    LJ). The draws already exist in `results/vast_n12/n1_test.jsonl` (R stored). Primary = the recorded
    flags. Secondary matchers rescore the stored R.

(ii-R) **MCF-R (matched information, headline comparator)**: MolCrystalFlow @3c493f8, patched so the
    lattice and all K fractional centroids are held at the truth. In training, only the rotations are
    corrupted and trained (trans/cell losses off; `b_trans` fixed). In inference, there are no Euler
    updates on lattice/trans. Everything else is the released default: architecture, symmetric SO(3)
    prior, OT, AdamW 1e-4, batch 128, grad clip 0.5, ReduceLROnPlateau, 18-dim features.
    Inputs come from MCF's own preprocessing (`sort_cell_left_matrix` → `preprocess_structure` →
    renormalize) of the rebuilt truth `orig`, with z = K written explicitly. The element map is extended
    to 16 types in both maps (a disclosed code change). Features that fail are zero-filled, as MCF does.
    Budget: 24k optimizer steps, about MCF's Thurlemann budget, with the valid/loss checkpoints of MCF's
    own checkpointer (top-3 + last). Checkpoint = the candidate with the best VAL picked@1 (tie → lowest
    valid/loss). s_uR (rotation velocity scale) is chosen on VAL from {1, 3, 9}. Everything else in the
    sampler is left at the defaults. 3 seeds.
    Pre-registered DIAGNOSTIC, labelled as oracle, not a result: MCF's features only see relative
    orientations, so it cannot sense orientation relative to the lattice. We report (a) a check that this
    equivariance holds for the trained weights, and (b) an oracle-aligned score: rotate all copies by the
    global rotation that commutes with the prior's pseudo-symmetry and minimises the geodesic error to the
    truth (about c* for K=2; the D2 element for K≥4), then match.

(ii-A) **MCF-A (released code, end-to-end, CONTEXT ONLY)**: the same export and element map, with the
    lattice prior refit on the 1687 train crystals (`lattice_fit.py`). Joint lattice + centroids +
    orientations. Same budget, selection and seeds. Reported under both matchers. It is not compared
    head-to-head with the matched-information arms.

(iii) **CLASSICAL (matched information)**: Haar-random R_asym + exact SG expansion, then either
    (iii-u) unrelaxed or (iii-p) the orientation-only press (80 steps, lr 0.02, the G2 code).
    16 draws × 3 seed bases, with the selector chosen on VAL from the same menu. Compute-generous variant:
    unrelaxed Haar at K = 128 with the LJ pick. Pressed draws are scored against the TRUTH. Note in
    advance: the relaxed truth matches the truth in only 1-2/198, so (iii-p) is capped by the force field.

(iv) **OLD symmc-flow (matched information)**: the old architecture in orientation-only mode
    (clean packing: true lattice + true centroids), in the ABSOLUTE gauge (no reference pose supplied),
    with per-copy coset ids from OUR spglib ops (slot → op by centroid match). Trained on the
    legacy-representable train subset (598; the det+1 parser rejects the rest). 3 seeds, samples-seen
    budget = the G2 budget (≈ 7 epochs × 1687 crystals). Checkpoint = {best fixed-draw val loss, final},
    chosen by VAL picked@1. Scored on the 57 test crystals it can represent. The other 143 are structural
    misses. Its oracle ceiling (56/57, true orientations) is reported.
    Optional diagnostic (iv-R): the as-published relative gauge, where R_asym is SUPPLIED. It is a
    leaked-information upper bound and is labelled so.

## 4. Pre-registered hypotheses (primary matcher, picked@1)
- H1: OURS > MCF-R. H2: OURS > CLASSICAL (best of iii-u / iii-p / K=128). H3: OURS > OLD (iv).
- Each hypothesis is reported whatever the outcome. A loss or a tie is reported as such.

## 5. Compute and hygiene
- MCF-R/MCF-A/old need one Linux GPU (single-GPU box; GPUtil is patched out or the box is kept idle).
  Classical and scoring run on CPU. CSD data goes up only in manual-approval mode. Pull back, delete,
  destroy, as in N1/N2.
- Checkpoints and exported pickles are CSD-derived: local only, never committed. The results JSON is
  committed. This file is committed BEFORE any N3 training or sampling.
