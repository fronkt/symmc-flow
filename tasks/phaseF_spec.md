# Phase F — Fully symmetry-conditioned molecular-crystal flow

**Approved direction (Frank, 2026-07-20):** turn the coset orientation diagnostic into a
deployable, fully symmetry-conditioned molecular flow by extending symmetry conditioning from
the ORIENTATION (coset, already done) to the LATTICE (crystal-family mask). Gated retrain:
cheap diagnostic-scale smoke test first, scale only if it clears the gate.

Evidence base: `gpu_results/phaseE4/FINDINGS_packing.md` (my diagnostics + 3-thread lit review).

## Why (one paragraph)
E4 end-to-end match@10 ~0%, RMAD ~33%. Diagnostics decompose the failure into two INDEPENDENT
axes: (1) VOLUME — the flow turns a 7.6%-RMAD constant-density prior into 33% (widens spread
3x); cheaply fixable but the oracle-volume ceiling experiment proves volume is NOT the match
blocker. (2) CELL SHAPE / crystal family — reference cells put 72% of angles at exactly 90deg
(molecular crystals are overwhelmingly monoclinic/orthorhombic); our free 9-DOF shape head
yields 3% and 26% length error. The lattice ignores the crystal family even though `sg` is
embedded. The proven fix (DiffCSP++ +29pts match; SGFM proved it inside flow matching) is to
reparameterize the lattice as the O(3)-invariant log-metric k in R^6 and apply a binary
crystal-family mask m(G) that freezes the constrained DOF. THE GAP: this exists only for
inorganic single-atoms; every molecular learned flow (MolCrystalFlow/PackFlow/MOFFlow) is
space-group-FREE and names SG conditioning as future work. A fully symmetry-conditioned
molecular FLOW (family-masked lattice + coset orientation) is genuinely unclaimed.

## Novelty framing (keep honest — reviewers will probe)
- The lattice-mask MECHANISM is DiffCSP++ (2024), not novel. Our novelty = transporting it into
  the molecular rigid-body flow and COMPOSING it with coset-conditioned orientation — the domain
  + combination. Frame as "we bring proven inorganic symmetry machinery to molecular CSP, which
  molecular flows left on the table."
- The family mask constrains the cell to the crystal FAMILY (metric) — necessary, not sufficient
  for full SG symmetry. Wyckoff-centroid projection + rigid-press may still be needed to move
  match@10. Pre-register this so a null match result is still an honest, publishable finding.

## Architecture change (the math)
Represent the lattice by the metric G = L^T L (rotation-invariant). S = logm(G) is symmetric,
6 independent components. Decompose in an ORTHONORMAL symmetric basis {B_0..B_5}:
  - B_0 = I/sqrt(3)  -> k_0 = tr(S)/sqrt(3) = (2 log V)/sqrt(3)  : VOLUME coord (always free)
  - B_1 = diag(1,1,-2)/sqrt(6)   : distinguishes c from a,b (free for tetragonal/hex)
  - B_2 = diag(1,-1,0)/sqrt(2)   : distinguishes a from b (zeroed when a=b)
  - B_3 = (E_23+E_32)/sqrt(2)    : off-diag ~ alpha (cos alpha)
  - B_4 = (E_13+E_31)/sqrt(2)    : off-diag ~ beta  (monoclinic keeps this)
  - B_5 = (E_12+E_21)/sqrt(2)    : off-diag ~ gamma
k_i = <S, B_i>. Reconstruct S = sum k_i B_i, G = expm(S), L = chol(G) (triangular, canonical
frame — frame choice is irrelevant: model round-trips lattice through k, StructureMatcher is
frame-invariant, and metric distances frac^T G frac are gauge-free).

Family mask m(sg): FREE dims + CANONICAL fixed values for the rest. k_0 always free.
| family (SG range)            | free k-dims        | fixed (value)                 |
|------------------------------|--------------------|-------------------------------|
| triclinic (1-2)              | all 6              | --                            |
| monoclinic (3-15)            | 0,1,2,4            | k3=0, k5=0                    |
| orthorhombic (16-74)         | 0,1,2              | k3=k4=k5=0                    |
| tetragonal (75-142)          | 0,1                | k2=k3=k4=k5=0                 |
| trigonal/hexagonal (143-194) | 0,1                | k2=k3=k4=0, k5=CANON_HEX(!=0) |
| cubic (195-230)              | 0                  | k1=k2=k3=k4=k5=0              |
(!) hexagonal gamma=120 -> B_5 component is a FIXED nonzero constant in log-metric space
(independent of a, because a^2-scaling only adds to the I-part). Compute CANON_HEX numerically
in a unit test: logm of a^2*[[1,-1/2,0],[-1/2,1,0],[0,0,*]] has fixed off-diagonal. Verify
monoclinic unique-axis-b convention matches pymatgen's SG setting (may need axis check).

## File diffs (all CPU-testable, no GPU)
Gate behind `ModelConfig.lattice_repr: "shape10" (default) | "logmetric6"` +
`ModelConfig.lattice_family_mask: bool` so existing checkpoints (coset_deploy_s0.pt) + E2/E4
eval scripts stay reproducible.

- `symmc_flow/space_group.py`: add `family_of(sg)->int` (0..6 from SG-number ranges).
- `symmc_flow/manifolds.py`: add `SYM_BASIS` (6 orthonormal symmetric mats),
  `lattice_to_logmetric(L,n)->k6`, `logmetric_to_lattice(k6,n)->L`, `family_mask(sg)->(free
  bool[6], canon float[6])`, `apply_family_mask(k6, sg)`, `prior_logmetric(n, sg, stats)` (data-
  informed: tight volume sigma ~0.11 from data + per-dim deviatoric sigma, mask applied).
- `symmc_flow/model.py`: `lattice_in`/`head_lattice` dims 10<->6 by repr; forward uses the
  repr-appropriate lattice_to_* ; keep the pooled-lattice head.
- `symmc_flow/flow.py`: `interpolate(...)` takes `sg`; builds u_L on the free k-dims (masked dims
  contribute 0 velocity); `sample_prior`/`PriorCache` use `prior_logmetric` when repr=logmetric6.
- `symmc_flow/sampler.py`: after each RK4 lattice step, `apply_family_mask(kL, sg)` (project
  masked dims to canonical); decode via `logmetric_to_lattice`.
- `symmc_flow/config.py`: add the two flags + `prior_logvol_std` (default 0.3; set ~0.11 for F).
- `scripts/`: a `fit_lattice_prior.py` to dump per-corpus (per-family) k-stats -> informed prior.
- `tests/test_lattice_family.py`: round-trip L->k->L (all families); mask idempotence; each
  family's reconstructed cell obeys its metric constraints (angles exactly 90/120, a=b, etc.);
  CANON_HEX value; volume coord monotone in V; backward-compat shape10 path unchanged.

## Sub-phases + gate
- [x] F0  Diagnose packing bottleneck (volume ceiling + shape/family) — DONE, FINDINGS_packing.md
- [ ] F1  LOCAL BUILD (no GPU): log-metric repr + family mask + informed prior + tests + CPU
          smoke of a tiny train/sample. All green before any GPU.
- [ ] F2  GATED SMOKE RETRAIN (cheap, few GPU-hrs, diagnostic-scale ~same corpus as E4/coset):
          retrain with lattice_repr=logmetric6 + family_mask + coset. Re-eval match@10 + RMAD +
          the 90deg angle-spike. **PRE-REGISTERED GATE:** promote to F3 iff (a) generated cell
          angles within 2deg of 90 rise from ~3% toward reference ~72% (mask working), AND
          (b) match@10 (coset on, stol<=1.0) moves off 0 (>= ~1-2%, CI-separated from 0), OR
          RMAD drops below ~10% with angle-spike restored (partial win worth scaling). Else ->
          F4-fallback (E5 write-up + honest "family mask necessary-not-sufficient" result).
- [ ] F3  SCALE (only if F2 passes): rigid-press finisher (hard-sphere BFGS, symmetry-preserving)
          + informed prior + larger corpus/run. Full match@10 + RMAD vs MCF/MOFFlow table.
- [ ] F4  PAPER: if positive -> rewrite JCIM/MoML as "first fully symmetry-conditioned molecular
          flow, competitive + orientation lever." If null -> E5 diagnostic write-up + the free
          RMAD 33%->7.6% win + the honest necessary-not-sufficient family-mask finding.

## Risk register
1. Family mask necessary-not-sufficient -> match stays ~0. Mitigated: pre-registered partial-win
   gate (angle-spike + RMAD) so F2 is informative either way; Wyckoff-centroid is the next lever.
2. Corpus scale: E2 showed match stayed 0 at scale (orientation improved). Packing may need more
   data. Mitigated: F2 is diagnostic-scale + cheap; scale is F3, gated.
3. Hexagonal/monoclinic axis-convention bugs -> silent wrong masks. Mitigated: per-family unit
   tests asserting exact metric constraints on reconstructed cells.
4. Triangular-frame reconstruction changes pair-feature gauge -> retrain needed anyway (fresh
   checkpoint); default shape10 path untouched so nothing regresses.
5. GPU/credit spend (Vast). Mitigated: F1 fully local; F2 is a few GPU-hrs; Frank owns the F2->F3
   scale decision. Follow feedback_vast_workflow (cap workers, cu128 for 5090, check Inet).

## Non-negotiables
Frank owns final paper prose + the F2->F3 scale call. No Co-Authored-By trailers. Specific file
paths when staging. Push after F1 build lands. Don't touch pxrd-flow box (45388164).
