# G0 — comparator improper-op check (2026-09-25)

Source: subagent code read of MolCrystalFlow @3c493f8 and MOFFlow @22d39ac. Verdict: NEITHER shares symmc-flow's shared-reference det+1 flaw -> Paper B (field finding) is dead.
Live angle for Paper A: MCF's public CSP driver (packing_gen.py::create_inference_batch ~l.355-440) repeats ONE conformer + ONE chi across all Z copies -> generated cells are homochiral; exact SG expansion with improper ops produces inversion-related enantiomers natively. UNVERIFIED by us beyond the code read -- re-check before claiming.

# Do rigid-body crystal generative models mishandle improper-symmetry copies?

Date: 2026-09-25. Read-only check. Repos cloned into this scratchpad (`mcf/`, `mofflow/`).

## The question

symmc-flow fits ONE shared reference conformer into every copy with Kabsch alignment (det +1). A copy related by inversion, mirror or glide is the enantiomer, so no proper rotation fits it and those crystals get filtered out. Does the same thing happen in MolCrystalFlow or MOFFlow?

## Verdicts

| Model | Training / benchmark path | CSP-from-one-conformer path |
|---|---|---|
| MolCrystalFlow (2602.16020) | **DOES NOT SHARE.** Each molecule has its own local coordinates plus a binary parity feature chi. | Limitation, but a different one: the public code gives all Z copies the same chi and the same local coordinates, so one sample can only be homochiral. |
| MOFFlow (2410.17270) | **DOES NOT SHARE.** Each building block has its own local coordinates. No parity bit, and none is needed. | Not applicable: prediction uses each test block's own local coordinates. |
| 2509.23822 (SymmCD), 2502.03638 (Space Group Conditional Flow Matching) | Not rigid-body models. Both are atom-level generators for inorganic crystals that apply the full space-group operations (improper ones included) to atoms. Not applicable. | — |

---

## 1. MolCrystalFlow

- Paper: arXiv 2602.16020v3 (09 Mar 2026), Zeng, ..., Hennig, Martiniani, Liu (UF/NYU). The arXiv comment gives the code: https://github.com/Liu-Group-UF/MolCrystalFlow
- Code checked at commit `3c493f8ddcfe8f3ea2d1e9bdba82c26b0f43516e` (shallow clone, 2026-09-25).

### (a) How orientation is represented and extracted

The orientation is an SO(3) rotation matrix built from PCA. There is no Kabsch fit to a reference.
- The paper ("Extracting the axis-flip state chi"): covariance eigenvectors of the centred coordinates. The sign of each axis is fixed by the equivariant vector D = X̄_CM − X̄ (atomic-number-weighted centroid minus geometric centroid).
- Code for the training data: `data-preprocess/thurlemann23/common.py::get_equivariant_axes` (around lines 247–305), called from `preprocess_structure` (line 371). `generate_training_data.py:34` imports it from `common`.
  ```python
  right_hand = np.stack([axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1)
  ```
  The frame is always right-handed (det +1, checked numerically). There is no fit to anything.
- Code for the CSP pipeline and the Dataset class: `molcrystalflow/data/dataset.py::get_equivariant_axes` (lines 94–135). It builds the same right-handed frame, but decides chi differently (see (b)).

### (b) Handling of mirror-image molecules: yes, a parity bit chi

- Paper, Eq. (6): chi = 0 when n₋ ∈ {0, 2}, and chi = 1 when n₋ ∈ {1, 3}. Here n₋ is the number of raw PCA axes whose projection on D is negative. The paper says: "Molecular crystals may adopt packings in which molecules share the same or opposite axis-flip states (chi)... this axis-flip state is an intrinsic property of each building block and cannot be altered through rotational evolution". It also says: "we renormalize local coordinates ... so that local coordinates within the same chi group ... are the same and differ by an opposite sign for one axis across different chi groups."
- Model: `molcrystalflow/models/molcrystalnet.py:80` has `self.axis_flip_embedding = nn.Embedding(2, hidden_dim)`, which is fused into the node embedding in every layer (lines ~180–190). Optimal transport is grouped by chi (`data/interpolant.py::_find_optimal_rotation_permutation`, line 121: "only BBs in same group can be matched").
- **Numerical check** (`scratchpad/test_chi.py`). I took 2000 random asymmetric point clouds, each with a proper-rotated copy and an improper (inverted, then rotated) copy:

  | Implementation | chi unchanged, proper copy | chi flipped, improper copy | local coordinates identical, proper copy | local coordinates = one-axis mirror, improper copy |
  |---|---|---|---|---|
  | `common.py` (builds the TRAINING data) | 1108/2000 | 1108/2000 | 493/2000 | 1507/2000 |
  | `dataset.py` (used by CSP inference) | 2000/2000 | 2000/2000 | 2000/2000 | 2000/2000 |

  - In `dataset.py`, chi = sign(cross(a0, a1)·D) after aligning the axes to D. That makes it a true handedness bit.
  - In `common.py`, and in Eq. (6) as written, chi is the parity of n₋ counted on the raw `np.linalg.eigh` eigenvectors. Their determinant sign is arbitrary, so chi comes out roughly at chance with respect to real handedness. This is a side defect in their code, and I have tested it only on synthetic point clouds, not on their released Zenodo pickles.
  - Even so, reconstruction is exact in every case, because each molecule stores its own local coordinates (`local_coord = bb_coord @ rotmats`; rotmats is orthogonal). A noisy chi therefore costs nothing geometrically; it only weakens the chi feature.

### (c) Dataset filtering, space groups and chirality

- There is no space-group filter anywhere in `data-preprocess/`. `thurlemann23/filter_structures.py` keeps only the unstrained frames (`index="2::5"`) and requires every molecule to have the same formula and the detected molecule count to equal `z_value`. Paper: 11,489 → 11,488 Thurlemann structures, with one dropped for a Z mismatch. OMC25-MCF: the lowest-energy polymorph per conformer, 46,801 structures, with no space-group criterion.
- The paper reports no space-group or Z′ distribution in the main text; I did not check the SI PDF. The CSP section samples Z ∈ {2, 4}. The Discussion lists space-group constraints and asymmetric-unit representations as future work.
- "Chirality" appears only as an RDKit auxiliary descriptor (`is_chiral` in `generate_features.py:133`), plus the chi machinery above. Of the blind-test targets the paper says: "All three experimental crystal packings contain two different axis-flip states with equal numbers of molecular building blocks." In other words, they are racemic or centrosymmetric-type packings.

### (d) Per-molecule or shared conformer

- **Training and the benchmark (Thurlemann/OMC test sets): per-molecule.** `preprocess_structure` computes `local_coords` separately for each molecule from its own CIF/extxyz coordinates. `MolCrystalFlow.forward` (`models/molcrystalflow.py:281`, `_assemble_coords` line ~410) assembles the structure from the test batch's own `local_coords` and `axis_flips`. So mirror copies reconstruct exactly. As a side effect, the ground-truth handedness of each copy is given to the model at test time.
- **The CSP pipeline (`csp-pipeline/molcrystalflow_gen/packing_gen.py`): shared conformer.** `create_inference_batch` (lines 355–440):
  ```python
  if has_axis_flip and axis_flip_state == 1:
      base_local_coords[:, 0] = -base_local_coords[:, 0]  # Flip x-coordinates
  local_coords = base_local_coords.repeat(z_value, 1)
  ...
  axis_flips = torch.tensor([[1.0, 0.0, 0.0]] * z_value, ...)   # or all zeros
  ```
  - `run_inference` (lines ~621–626) splits the samples half at flip state 0 and half at flip state 1, but inside any one sample all Z copies share the same handedness.
  - The paper's CSP section describes a second scenario: "another in which the crystal contains equal numbers of molecules in two distinct chi states". **I could not find a mixed-chi path in the public code at this commit.** Locations searched: `packing_gen.py` (all `flip` hits), `csp_pipeline.py` (only an on/off `axis_flip` flag, lines 76/663/854).
  - As released, the CSP path can only emit homochiral packings of a chiral rigid conformer. That is structurally the same limitation symmc-flow has. But it is a gap in the CSP driver, not in the model's representation.

**Verdict: DOES NOT SHARE the flaw** in the model, training or benchmark. The released CSP script has a narrower gap that differs from the paper's text.

---

## 2. MOFFlow

- Paper: arXiv 2410.17270v2 (Kim et al., ICLR 2025). Code: https://github.com/nayoung10/MOFFlow, checked at commit `22d39ac04076d37979c1840b90d4587b5f2828f1`.

### (a) Orientation

- SO(3) only. Paper Sec. 4.1 and App. E: PCA axes with their signs fixed by the equivariant vector (following Gao & Günnemann 2022), and invariance required only for Q ∈ SO(3) (Eq. 3). Prior: uniform on SO(3).
- Code: `data/dataset.py::_get_equivariant_axes` (lines ~130–150). Axes are flipped to align with the vector, then `right_hand = [a0, a1, a0×a1]`, so det is +1. There is no Kabsch step and no parity bit.

### (b) Mirror handling

- None explicit. The paper and code never mention chirality, inversion, mirror or handedness (I grepped the HTML and all non-openfold `.py` files).
- None is needed, because of (d).

### (c) Dataset

- Space groups are not discussed. The paper mentions space groups only in related work.

### (d) Per-block local coordinates

- `_rotate_bb`: `local_coord = bb_coord @ rotmats` is computed separately for each building block from its own coordinates (`_process_one`, lines ~185–195).
- Prediction passes each test block's own `local_coords` (`experiments/predict.py:131`, `local_coords=batch.local_coords`).
- A block that is the mirror image of another block simply carries mirror-image local coordinates. The fit is exact, and the SO(3) target is well defined.

**Verdict: DOES NOT SHARE.** There is a caveat: MOFFlow cannot generate a mirror-image block that it was not given. That is inherent to structure prediction with known building blocks, and it is not a fitting error.

---

## 3. Other IDs given

- 2509.23822 = SymmCD (Levy et al.) and 2502.03638 = Space Group Conditional Flow Matching (Puny, Lipman, Miller). Both are atom-level generators for inorganic crystals, with no rigid bodies and no orientations.
- They handle improper operations natively, because they apply the space-group operations to atoms. Not applicable.
- I did not look for other rigid-body molecular CSP models (for example AssembleFlow, which also uses per-molecule inertial frames according to MolCrystalFlow's citation).

## Bottom line for symmc-flow

Neither main comparator has the shared-reference-conformer problem. Both give every copy its own local coordinates, so improper copies are represented exactly.

- MolCrystalFlow also names the issue explicitly and adds a parity feature (chi). But the implementation that builds its training data computes chi from an eigh-sign-dependent rule, which comes out roughly at chance on my synthetic check.
- The honest framing is that symmc-flow's filtering of improper copies is specific to its shared-conformer choice. A comparator-level claim ("they silently drop P2₁/c") would be false.
- The only defensible parallel: MolCrystalFlow's public CSP script cannot produce mixed-handedness cells, even though the paper describes a mixed-chi sampling scenario.

## Main-session verification (2026-09-25)
Read directly at MolCrystalFlow @3c493f8, `csp-pipeline/molcrystalflow_gen/packing_gen.py:355-430`:
`local_coords = base_local_coords.repeat(z_value, 1)` and a single `axis_flips` row repeated
z_value times. Every copy in a generated cell therefore shares one conformer handedness; with SO(3)
rotations a cell cannot hold both enantiomers of a chiral rigid conformer. `generate_*` (l.621-639)
splits samples between flip 0 and flip 1 *across* cells, never within one. CONFIRMED for the public
driver. Still open: whether the paper's blind-test "mixed" runs used unreleased code (check paper +
SI wording before any claim), and how many CSD crystals need mixed handedness (= crystals with a
parity -1 copy in G1 that legacy det+1 rejects).
