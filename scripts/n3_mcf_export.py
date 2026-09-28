"""N3 / G3: export a frozen set to MolCrystalFlow pickles, run Gate F1-F5, refit MCF-A's lattice prior.

Protocol tasks/n3_protocol.md §3.2 ("Export", "MCF lattice-prior refit", "Gate F") and Table 0 (z_MCF).
MCF code comes from the patched fork (external/mcf, or $N3_MCF), never the stock tree: P1's 16-type map.

Per item, in the set's frozen order:
  1. whole molecules X_k = centroid_k L + orient_k local_k over the real slots of `orig`
  2. centred (an op with W = I, t != 0; mult = 1 + #such ops) -> primitive cell P of to_structure(orig)
     (same Cartesian frame; M0 = L P^-1 integral, |det M0| = mult); keep X_k unless its centroid equals a
     kept one modulo P (1e-4 frac); z_MCF = K / mult. Non-centred: P = L, z_MCF = K
  3. sort_cell_left_matrix; Q from X Q = X_rot (lstsq); M = round(L Q L_m^-1), |det M| = mult
  4. common.preprocess_structure(L_m, X_rot, Z, bb_indices = kept-slot index)
  5. per-copy features through MCF's generate_features functions: ONE obabel xyz->sdf call for all copies
     of the set (as MCF's own pipeline does; titles checked 1:1), SDMolSupplier(removeHs=False,
     sanitize=False), fix_mol_bonds(mol, MolToSmiles(mol)) with MCF's unfixed fallback,
     extract_features_single_molecule, .get(k, 0). DEVIATION: strategy 1's RDKit DetermineBonds runs with
     maxIterations = DETERMINE_MAX_ITER (it never returns on TRAIN ZAYGOY); hitting the cap raises inside
     MCF's own try, so the molecule proceeds to strategy 2 exactly as on any other strategy-1 exception
  6. renormalize_single_datapoint (touches only local_coords / rotmats_1, so its order vs 5 is immaterial)
  8. exactly |set| entries (asserted; --allow_failures writes fewer and forces gate_pass false) + a sidecar
     (refcodes in frozen order, kept slots, Q, det Q, M, mult, z_MCF, primitive-length tie flag, library
     versions, the file SHA-256 and a tensor-content SHA-256). Torch pickles are not byte-stable, so an existing
     pickle (and oracle file) whose tensor content equals the new export is kept as is: re-exports leave the
     file SHA-256 unchanged. The pickle is re-read from disk; F2's cells are re-decoded from it (bitwise check).

Artefacts of record: the pickles + sidecars written here are uploaded to the box and never regenerated there.
Downstream code uses the stored Q, M and lattice_1: a re-export elsewhere can differ (library versions, and
MCF's argsort on primitive lengths equal up to rounding, flagged per item as length_tie "near"). det Q = -1
(counted as detQ_negative) means sort_cell_left_matrix mirrored the crystal: map with X = X_MCF Q^T and never
assume a proper Q.

Outputs (MCF reads <cache_dir>/{train,val,test}_molcrystal_normalized.pkl.gz; all gitignored):
    train -> n3_mcf/trainval/train_*   val -> n3_mcf/trainval/val_*   valsel -> n3_mcf/valsel_as_test/test_*
    testB -> n3_mcf/test/test_*        devtest -> n3_mcf/devtest/test_*   sel -> n3_mcf/sel/test_* (gates only)
    results/n3/gateF_<set>.json (counts; failing refcodes only), results/n3/private/gateF_<set>_detail.jsonl
    results/n3/gateF.json = {set: gateF_<set>.json} for every exported set (refreshed on each run)
    train also writes <fork>/molcrystalflow/configs/ours_lattice.yaml (lattice_fit.compute_lattice_stats)

Library (no MCF import needed): mcf_cells_to_supercell / mcf_energy (the F5 LJ adapter, for n3_score and
MCF-R+G), predictions_cells (split an MCF predictions file into n3_common 'cells' draws); with MCF:
entry_cells, oracle_predictions (F6 input: every draw = the pickle's own pose).

    python scripts/n3_mcf_export.py --set val --workers 2
    python scripts/n3_mcf_export.py --set train --workers 2      # + lattice-prior refit
    python scripts/n3_mcf_export.py --set devtest --workers 2 --oracle_predictions
    python scripts/n3_mcf_export.py --merge_gates                 # rebuild results/n3/gateF.json only
"""
import argparse
import gzip
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import torch

import n3_common as nc

MCF = os.environ.get("N3_MCF", os.path.join(REPO, "external", "mcf"))
MCF_BASE = "3c493f8"
OUT = os.path.join(REPO, "n3_mcf")
GATE_DIR = os.path.join(REPO, "results", "n3")

# P1: MCF's 12-type map plus Mg, As, Se, Te -> 12-15 (the fork's three maps are asserted equal to it)
MCF_Z_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6, 15: 7, 35: 8, 53: 9, 5: 10, 14: 11,
                12: 12, 33: 13, 34: 14, 52: 15}
MCF_IDX_TO_Z = {v: k for k, v in MCF_Z_TO_IDX.items()}
NUM_ATOM_TYPES = 16

TARGET = {"train": ("trainval", "train"), "val": ("trainval", "val"), "valsel": ("valsel_as_test", "test"),
          "testB": ("test", "test"), "devtest": ("devtest", "test"), "sel": ("sel", "test")}
MATCH_GATED = {"val", "sel", "testB", "devtest", "valsel"}     # F2 matcher and F5 are gated on these
TOL_ASSEMBLE, TOL_F5, TOL_ZERO_FEAT = 1e-3, 1e-4, 0.01
TOL_LENGTH_TIE = 1e-9    # relative gap between primitive lengths below which MCF's argsort order is float noise
# RDKit DetermineBonds (fix_mol_bonds strategy 1) never returns on 1 of the 1,987 dev molecules (TRAIN ZAYGOY,
# > 45 min); with this cap it raises and MCF's own try/except moves on to strategy 2. Every other dev molecule
# is unchanged (the slowest terminating one needs 1e5-3e5 iterations). 0 = RDKit default (no limit).
DETERMINE_MAX_ITER = 10_000_000

_MOD = {}


class DetermineBondsCapped:
    """Stand-in for common.rdDetermineBonds: MCF's DetermineBonds call with maxIterations capped; logs each
    call's outcome ('ok' / 'cap' / exception name) so F4 can label strategy 1 without a second call."""

    def __init__(self, real, cap):
        self.real, self.cap, self.log = real, cap, []

    def DetermineBonds(self, mol, *args, **kw):
        kw.setdefault("maxIterations", self.cap)
        try:
            self.real.DetermineBonds(mol, *args, **kw)
        except Exception as ex:
            self.log.append("cap" if "Max Iterations" in str(ex) else type(ex).__name__)
            raise
        self.log.append("ok")

    def __getattr__(self, k):
        return getattr(self.real, k)


def probe_determine_bonds(real, cap):
    """The cap is only meaningful if this RDKit's DetermineBonds takes maxIterations: on one that does not,
    every strategy-1 call would raise ArgumentError inside MCF's try and silently fall to strategy 2."""
    from rdkit import Chem
    m = Chem.MolFromXYZBlock("3\nwater\nO 0.0 0.0 0.0\nH 0.9572 0.0 0.0\nH -0.2400 0.9266 0.0\n")
    try:
        real.DetermineBonds(m, charge=0, maxIterations=cap)
    except Exception as ex:
        raise RuntimeError(f"RDKit DetermineBonds does not accept maxIterations ({type(ex).__name__}); "
                           "the export needs an RDKit whose DetermineBonds has it") from ex
    assert m.GetNumBonds() == 2, "DetermineBonds probe (water) gave the wrong bond count"


def lib_versions():
    """Versions the export's numbers depend on (features: RDKit + Open Babel; P: pymatgen/spglib; Q: numpy)."""
    import importlib
    import importlib.metadata as md
    import platform
    import rdkit

    def dist(name):    # installed distribution (pymatgen.core.__version__ can lag the installed release)
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return getattr(importlib.import_module(name), "__version__", None)
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "rdkit": rdkit.__version__, "pymatgen": dist("pymatgen"), "ase": dist("ase"), "spglib": dist("spglib"),
            "obabel": obabel_version()}


def _mcf(cap=None):
    """MCF modules from the fork (data-preprocess on sys.path, as MCF's own scripts run)."""
    if not _MOD:
        for p in (os.path.join(MCF, "data-preprocess", "thurlemann23"), os.path.join(MCF, "data-preprocess")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import common
        import generate_features
        import lattice_fit
        from standardize_lattice import sort_cell_left_matrix
        assert os.path.abspath(common.__file__).startswith(os.path.abspath(MCF)), common.__file__
        probe_determine_bonds(common.rdDetermineBonds, DETERMINE_MAX_ITER if cap is None else cap)
        common.rdDetermineBonds = DetermineBondsCapped(common.rdDetermineBonds,
                                                       DETERMINE_MAX_ITER if cap is None else cap)
        _MOD.update(common=common, gf=generate_features, lattice_fit=lattice_fit, sort=sort_cell_left_matrix)
    if cap is not None:
        _MOD["common"].rdDetermineBonds.cap = cap
    return _MOD


def check_fork_maps():
    """P1 in all three maps (common.py, data/dataset.py, data/utils.py; inverses derive from them). The two
    molcrystalflow maps are read from source so this needs no torch_geometric / torch_scatter."""
    import ast
    common = _mcf()["common"]
    out = {"common": dict(common.ATOM_TYPE_TO_IDX) == MCF_Z_TO_IDX
           and dict(common.IDX_TO_ATOM_TYPE) == MCF_IDX_TO_Z}
    for name, rel in (("dataset", "molcrystalflow/data/dataset.py"), ("utils", "molcrystalflow/data/utils.py")):
        src = open(os.path.join(MCF, rel), encoding="utf-8").read()
        start = src.index("ATOM_TYPE_TO_IDX = {") + len("ATOM_TYPE_TO_IDX = ")
        block = src[start: src.index("}", start) + 1]
        out[name] = ast.literal_eval("\n".join(l.split("#")[0] for l in block.splitlines())) == MCF_Z_TO_IDX
    return out


# ---------------------------------------------------------------------------------------------------
# F5 adapter and 'cells' helpers (pure torch/numpy; reused by n3_score.py and MCF-R+G)
# ---------------------------------------------------------------------------------------------------

def coset_reps(M):
    """Integer row vectors n (in the basis of L_gen) with n M^-1 in [0,1)^3: the translations of the MCF cell
    that tile the supercell M L_gen. Exact integer arithmetic; there are |det M| of them."""
    Mi = np.rint(np.asarray(M, dtype=np.float64)).astype(np.int64)
    d = int(round(np.linalg.det(Mi)))
    assert d != 0
    adj = np.rint(np.linalg.inv(Mi) * d).astype(np.int64)
    bound = np.abs(Mi).sum(axis=0)
    reps = []
    for n in itertools.product(*[range(-int(b), int(b) + 1) for b in bound]):
        num = np.asarray(n, dtype=np.int64) @ adj
        if (d > 0 and np.all((num >= 0) & (num < d))) or (d < 0 and np.all((num <= 0) & (num > d))):
            reps.append(n)
    assert len(reps) == abs(d), (len(reps), d)
    return torch.tensor(reps, dtype=torch.float64), abs(d)


def mcf_cells_to_supercell(lattice, cart, species, mol, M):
    """F5 adapter (protocol §3.2 Gate F5): one MCF cell -> g2_asym_baselines.energy() inputs.

    lattice [3,3] (rows = vectors) of the MCF cell (generated, or the pickle's); cart [N,3] whole molecules
    (never wrapped per atom); species [N] atomic numbers; mol [N] copy index 0..z-1; M [3,3] the stored
    integer supercell matrix (sidecar). Each molecule becomes orient = I, local = X_k - mean X_k; for
    |det M| = mult > 1 it is copied to the mult translations n L_gen (n M^-1 in [0,1)^3) of the supercell
    L_c = M L_gen, centroids fractional in L_c. For mult = 1 the adapter is the identity (L_c = L_gen).
    Returns (L_c, cent, orient, local, Z, am, mm)."""
    L = torch.as_tensor(lattice, dtype=torch.float64)
    X = torch.as_tensor(cart, dtype=torch.float64)
    sp = torch.as_tensor(species).long()
    mol = torch.as_tensor(mol).long()
    reps, mult = coset_reps(M)
    Lc = L if mult == 1 else torch.as_tensor(np.rint(np.asarray(M, dtype=np.float64))) @ L
    shifts = reps @ L                                              # Cartesian, [mult, 3]
    Lci = torch.linalg.inv(Lc)
    ids = torch.unique(mol).tolist()
    A = max(int((mol == j).sum()) for j in ids)
    cents, locs, Zs, ams = [], [], [], []
    for j in ids:
        Xj = X[mol == j]
        c = Xj.mean(0)
        loc = torch.zeros(A, 3, dtype=torch.float64)
        loc[:Xj.shape[0]] = Xj - c
        Zj = torch.zeros(A, dtype=torch.long)
        Zj[:Xj.shape[0]] = sp[mol == j]
        amj = torch.zeros(A, dtype=torch.bool)
        amj[:Xj.shape[0]] = True
        for s in shifts:
            f = (c + s) @ Lci
            cents.append(f - torch.floor(f))
            locs.append(loc)
            Zs.append(Zj)
            ams.append(amj)
    Kc = len(cents)
    return (Lc, torch.stack(cents), torch.eye(3, dtype=torch.float64).expand(Kc, 3, 3).clone(),
            torch.stack(locs), torch.stack(Zs), torch.stack(ams), torch.ones(Kc, dtype=torch.bool))


def mcf_energy(cell, M):
    """Unrelaxed steric LJ (g2_asym_baselines.energy, unchanged) of one MCF 'cells' draw via the F5 adapter."""
    from g2_asym_baselines import energy
    return energy(*mcf_cells_to_supercell(cell["lattice"], cell["cart"], cell["species"], cell["mol"], M))


def decode_species(types):
    return torch.tensor([MCF_IDX_TO_Z.get(int(t), -1) for t in types], dtype=torch.long)


def predictions_cells(pred, s):
    """Draw s of an MCF predictions_<S>.pt dict -> list (one per crystal, file order) of n3_common 'cells'
    draws. Copies are equal-sized (Z'=1), so per-copy atom counts are num_atoms / num_bbs."""
    na = [int(v) for v in pred["num_atoms"][s]]
    nb = [int(v) for v in pred["num_bbs"][s]]
    cart = pred["cart_coords"][s].double().split(na)
    types = pred["atom_types"][s].split(na)
    out = []
    for b in range(len(na)):
        A = na[b] // nb[b]
        assert A * nb[b] == na[b]
        out.append({"lattice": pred["lattices"][s][b].double(), "cart": cart[b],
                    "species": decode_species(types[b]), "mol": torch.arange(nb[b]).repeat_interleave(A)})
    return out


def entry_cells(e):
    """The pickle entry's own pose as a 'cells' draw: MCF's common.assemble_coords + the P1 inverse map."""
    common = _mcf()["common"]
    nb = e["bb_num_vec"].long()
    return {"lattice": e["lattice_1"].double(),
            "cart": torch.as_tensor(common.assemble_coords(e), dtype=torch.float64),
            "species": decode_species(e["atom_types"]),
            "mol": torch.arange(nb.shape[0]).repeat_interleave(nb)}


def oracle_predictions(entries, num_samples=16):
    """MCF predictions dict (EvalRunner._combine_predictions layout) whose every draw is the pickle's own pose
    (F6: through n3_score.py it must give 100% picked@1 / draw@1 / any@16). Coordinates are assembled the way
    FlowModule.forward does (R local + frac @ lattice, float32)."""
    cart, types, lat, na, trans, rots, nbb = [], [], [], [], [], [], []
    for e in entries:
        R, fr, L = e["rotmats_1"].float(), e["trans_1"].float(), e["lattice_1"].float()
        loc = e["local_coords"].float().split(e["bb_num_vec"].tolist())
        t = fr @ L
        cart.append(torch.cat([lc @ R[j].T + t[j] for j, lc in enumerate(loc)]))
        types.append(e["atom_types"])
        lat.append(L)
        na.append(int(e["local_coords"].shape[0]))
        trans.append(fr)
        rots.append(R)
        nbb.append(int(e["bb_num_vec"].shape[0]))
    one = {"cart_coords": torch.cat(cart), "atom_types": torch.cat(types), "lattices": torch.stack(lat),
           "num_atoms": torch.tensor(na), "pred_trans": torch.cat(trans), "pred_rotmats": torch.cat(rots),
           "num_bbs": torch.tensor(nbb)}
    pred = {k: v.unsqueeze(0).repeat(num_samples, *([1] * v.dim())) for k, v in one.items()}
    pred["gt_data_batch"] = {"gt_coords": torch.cat([e["gt_coords"] for e in entries]),
                             "atom_types": one["atom_types"], "lattice_1": one["lattices"],
                             "trans_1": one["pred_trans"], "rotmats_1": one["pred_rotmats"],
                             "num_atoms": one["num_atoms"]}
    return pred


# ---------------------------------------------------------------------------------------------------
# export steps 1-4, 6 and the per-crystal gates (worker)
# ---------------------------------------------------------------------------------------------------

def centring(a):
    """Number of pure-translation ops (W = I, t != 0 mod 1) among the crystal's own spglib ops."""
    I3 = torch.eye(3, dtype=torch.float64)
    n = 0
    for k in range(int(a["K"])):
        t = a["t"][k].double()
        if torch.equal(a["W"][k].double().round(), I3) and float((t - t.round()).abs().max()) > 1e-4:
            n += 1
    return n


def export_item(task):
    i, a, gates = task
    from ase import Atoms
    from g2_asym_baselines import energy
    m = _mcf()
    common = m["common"]
    res = {"i": i, "refcode": a["refcode"], "K": int(a["K"]), "ok": False, "err": None}
    try:
        o = a["orig"]
        L = o["lattice"].double()
        am, mm = o["atom_mask"], o["mol_mask"]
        real = [k for k in range(mm.shape[0]) if bool(mm[k])]
        K = int(a["K"])
        assert len(real) == K, "real slots != K"
        X = (o["centroid"].double() @ L).unsqueeze(1) + torch.einsum("kij,kaj->kai", o["orient"].double(),
                                                                     o["local"].double())
        # 2. centred crystals -> primitive cell, one whole molecule per centring orbit
        mult = 1 + centring(a)
        if mult > 1:
            P = torch.tensor(nc.truth_structure(a).get_primitive_structure().lattice.matrix, dtype=torch.float64)
            M0 = L @ torch.linalg.inv(P)
            assert float((M0 - M0.round()).abs().max()) < 1e-6, "M0 not integral"
            assert round(abs(float(torch.linalg.det(M0.round())))) == mult, "|det M0| != mult"
        else:
            P = L.clone()
        Pinv = torch.linalg.inv(P)
        kept, kept_f = [], []
        for k in real:
            f = X[k][am[k]].mean(0) @ Pinv
            if not any(float(((f - g) - (f - g).round()).abs().max()) < 1e-4 for g in kept_f):
                kept.append(k)
                kept_f.append(f)
        z = len(kept)
        assert z * mult == K, f"z_MCF {z} * mult {mult} != K {K}"
        # 3. standardize, recover Q and the integer supercell matrix M
        # MCF sorts the cell vectors by length (np.argsort). Bitwise-equal lengths sort deterministically; lengths
        # equal up to rounding ('near') are ordered by float noise, so another machine may pick another cell
        gap = float(np.min(np.diff(np.sort(np.linalg.norm(P.numpy(), axis=1)))))
        gap_rel = gap / float(np.linalg.norm(P.numpy(), axis=1).max())
        length_tie = "exact" if gap == 0 else ("near" if gap_rel < TOL_LENGTH_TIE else None)
        pos = torch.cat([X[k][am[k]] for k in kept]).numpy()
        nums = torch.cat([o["Z"][k][am[k]] for k in kept]).numpy().astype(np.int64)
        at = m["sort"](Atoms(numbers=nums, positions=pos.copy(), cell=P.numpy(), pbc=True))
        Xr, Lm = at.get_positions(), at.cell.array.copy()
        Q = np.linalg.lstsq(pos, Xr, rcond=None)[0]
        assert np.abs(pos @ Q - Xr).max() < 1e-6, "X Q != X_rot"
        detQ = float(np.linalg.det(Q))
        assert abs(abs(detQ) - 1) < 1e-9, "|det Q| != 1"
        Mf = L.numpy() @ Q @ np.linalg.inv(Lm)
        M = np.rint(Mf)
        assert np.abs(Mf - M).max() < 1e-6, "M not integral"
        assert round(abs(np.linalg.det(M))) == mult, "|det M| != mult"
        # 4. + 6.
        sizes = [int(am[k].sum()) for k in kept]
        bb = np.repeat(np.arange(z), sizes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = common.preprocess_structure(Lm, Xr, nums, bb)
        e = common.renormalize_single_datapoint(e)
        starts = np.cumsum([0] + sizes[:-1])
        res.update(ok=True, entry=e, atoms=at, mult=mult, z=z, kept=kept, Q=Q.tolist(), detQ=detQ,
                   M=M.astype(int).tolist(), centred=mult > 1, length_tie=length_tie,
                   copies=[(nums[s:s + n], Xr[s:s + n]) for s, n in zip(starts, sizes)])
    except Exception as ex:          # an export failure is listed by refcode, never silently dropped
        res["err"] = f"{type(ex).__name__}: {ex}"[:300]
        return res
    # gates; an error inside a check fails that check only (recorded in gate_err)
    res.update(types_ok=False, assemble_err=float("inf"), species_ok=False, match=False, chi_ok=False,
               f5_rel=float("inf"), gate_err={})
    try:   # F1 (per item)
        types = e["atom_types"].numpy()
        res["types_ok"] = bool(((types >= 0) & (types < NUM_ATOM_TYPES)).all())
    except Exception as ex:
        res["gate_err"]["F1"] = type(ex).__name__
    cell = None
    try:   # F2: assemble_coords vs the standardized input, atom by atom, mod one lattice translation per molecule
        cell = entry_cells(e)
        res["f2_cell"] = cell            # main re-decodes the written pickle and requires these bitwise
        Xa = cell["cart"].numpy()
        err = 0.0
        for s, n in zip(starts, sizes):
            d = Xa[s:s + n] - Xr[s:s + n]
            shift = np.rint(d.mean(0) @ np.linalg.inv(Lm)) @ Lm
            err = max(err, float(np.linalg.norm(d - shift, axis=1).max()))
        res["assemble_err"] = err
        res["species_ok"] = bool(np.array_equal(cell["species"].numpy(), nums))
        if gates["match"] and res["species_ok"]:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            res["match"] = bool(StructureMatcher().fit(nc.truth_structure(a), nc.cell_structure(cell)))
    except Exception as ex:
        res["gate_err"]["F2"] = type(ex).__name__
    try:   # F3: chi vs orig parity, up to a global relabel
        chi = e["axis_flips"][:, 0].numpy() > 0.5
        par = o["parity"].numpy()[kept] < 0
        res["chi_ok"] = bool(np.all(chi == par) or np.all(chi != par))
    except Exception as ex:
        res["gate_err"]["F3"] = type(ex).__name__
    if gates["f5"] and cell is not None:
        try:   # F5: the LJ adapter on the pickle's own pose vs energy() of orig
            e_orig = energy(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"], am, mm)
            e_ad = mcf_energy(cell, M)
            res["f5_rel"] = abs(e_ad - e_orig) / max(abs(e_orig), 1e-12)
            if res["f5_rel"] > TOL_F5:   # diagnostic only (never changes the verdict): same two energies with
                with _wide_image_shell():  # an image shell 2 cells wider than energy()'s own
                    w_orig = energy(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"], am, mm)
                    w_ad = mcf_energy(cell, M)
                res["f5_rel_wide_shell"] = abs(w_ad - w_orig) / max(abs(w_orig), 1e-12)
        except Exception as ex:
            res["gate_err"]["F5"] = type(ex).__name__
    return res


class _wide_image_shell:
    """F5 diagnostic: energy()'s neighbour shell n_i = clamp(ceil((cutoff+margin)/d_i), 1, 3) assumes every atom
    lies inside the cell; whole molecules that stick out (or unwrapped centroids) can need more images, so
    energy() depends on cell choice and molecule placement. Inside this context the shell is n_max + 2 cells."""

    def __enter__(self):
        import symmc_flow.rigid_press as rp
        self.rp, self.orig = rp, rp._image_shifts

        def wide(L, cutoff):
            n = int(self.orig(L, cutoff).abs().max().item()) + 2
            r = torch.arange(-n, n + 1, device=L.device)
            return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3).float()
        rp._image_shifts = wide
        return self

    def __exit__(self, *exc):
        self.rp._image_shifts = self.orig
        return False


def _init(cap=DETERMINE_MAX_ITER):
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    from rdkit import Chem
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)   # SDF mols travel to the workers
    _mcf(cap)


# ---------------------------------------------------------------------------------------------------
# step 5: per-copy features (one obabel call for the whole set in the main process, RDKit in the workers)
# ---------------------------------------------------------------------------------------------------

OBABEL_MISSING = "obabel_fail"


def fix_strategy(mol, smiles, s1_determine):
    """Which branch of common.fix_mol_bonds succeeded: 1, 2, 3, or 0 (all failed -> the original mol).
    Mirrors fix_mol_bonds step by step with MCF's own helpers (F4 label only). s1_determine is the outcome of
    fix_mol_bonds' own DetermineBonds call; that call is deterministic, so strategy 1 is re-run only if it
    succeeded (a capped molecule is not paid for twice)."""
    from rdkit import Chem
    common = _mcf()["common"]
    if s1_determine == "ok":
        try:
            mc = common.zero_formal_charges(Chem.RWMol(mol))
            common.rdDetermineBonds.DetermineBonds(mc, charge=0)
            f, _ = common.adjust_formal_charges_neutralize(mc)
            Chem.SanitizeMol(f)
            return 1
        except Exception:
            pass
    try:
        mc = common.zero_formal_charges(Chem.RWMol(mol))
        mc = common.fix_carbon_valency(common.fix_nitrogen_valency(mc))
        f, _ = common.adjust_formal_charges_neutralize(mc)
        Chem.SanitizeMol(f)
        return 2
    except Exception:
        pass
    if smiles is not None:
        try:
            ref = Chem.MolFromSmiles(smiles)
            if ref is not None:
                Chem.RemoveStereochemistry(ref)
                im = Chem.MolFromInchi(Chem.MolToInchi(ref))
                if im is not None:
                    im = Chem.AddHs(im)
                    Chem.SanitizeMol(im)
                    return 3
        except Exception:
            pass
    return 0


def obabel_version():
    r = subprocess.run(["obabel", "-V"], capture_output=True, text=True)
    assert r.returncode == 0, "obabel -V failed: Open Babel is required (openbabel-wheel or conda openbabel)"
    return r.stdout.strip()


def run_obabel(copies, workdir, allow_gaps=False):
    """MCF's generate_features.xyz_to_sdf on ONE xyz holding every copy (titled n3copy_<g>), then
    SDMolSupplier(removeHs=False, sanitize=False). Returns one entry per copy: the Mol, None (unparsable
    record) or OBABEL_MISSING (no record); records are aligned by title and asserted 1:1."""
    from ase.data import chemical_symbols
    from rdkit import Chem
    gf = _mcf()["gf"]
    os.makedirs(workdir, exist_ok=True)
    xyz, sdf = os.path.join(workdir, "copies.xyz"), os.path.join(workdir, "copies.sdf")
    try:
        with open(xyz, "w") as f:
            for g, (zs, xs) in enumerate(copies):
                f.write(f"{len(zs)}\nn3copy_{g}\n")
                for zz, p in zip(zs, xs):
                    f.write(f"{chemical_symbols[int(zz)]} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        assert gf.xyz_to_sdf(xyz, sdf), "obabel xyz->sdf failed"
        titles = [r.lstrip("\r\n").splitlines()[0].strip() if r.strip() else None
                  for r in open(sdf).read().split("$$$$")[:-1]]
        mols = list(Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False))
    finally:
        for p in (xyz, sdf):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(workdir)
    assert len(titles) == len(mols)
    by_title = {}
    for t, mol in zip(titles, mols):
        assert t not in by_title, f"duplicate sdf title {t}"
        by_title[t] = mol
    missing = [g for g in range(len(copies)) if f"n3copy_{g}" not in by_title]
    one_to_one = len(mols) == len(copies) and titles == [f"n3copy_{g}" for g in range(len(copies))]
    if not one_to_one and not allow_gaps:
        raise AssertionError(f"obabel xyz->sdf is not 1:1: {len(mols)} records for {len(copies)} copies, "
                             f"{len(missing)} missing (rerun with --allow_obabel_gaps to label them)")
    out = [by_title.get(f"n3copy_{g}", OBABEL_MISSING) for g in range(len(copies))]
    return out, {"sdf_records": len(mols), "copies": len(copies), "one_to_one": one_to_one, "missing": len(missing)}


def copy_feature(mol, xs):
    """One copy through MCF's generate_features loop body: fix_mol_bonds(mol, MolToSmiles(mol)) with the
    unfixed fallback, extract_features_single_molecule, .get(k, 0). Returns ((basic, chemical, geometric), F4 label)."""
    from rdkit import Chem
    m = _mcf()
    common, gf = m["common"], m["gf"]
    lab = {}
    if isinstance(mol, str) and mol == OBABEL_MISSING:
        lab["status"], fixed = "obabel_fail", None
    elif mol is None:
        lab["status"], fixed = "sdf_none", None
    else:
        det = common.rdDetermineBonds
        det.log.clear()
        try:
            smiles = Chem.MolToSmiles(mol)
            fixed = common.fix_mol_bonds(mol, smiles)
            if fixed is None:
                fixed = mol
            lab["determine"] = det.log[0] if det.log else "not_called"
            s = fix_strategy(mol, smiles, lab["determine"])
            lab["status"] = {1: "fix1", 2: "fix2", 3: "fix3", 0: "fallback"}[s]
            lab["label_consistent"] = (s == 0) == (fixed is mol)
        except Exception:
            lab["status"], fixed = "smiles_exception", mol
    f = gf.extract_features_single_molecule(np.asarray(xs), fixed)
    basic = [f.get(k, 0) for k in gf.BASIC_FEATURES]
    chemical = [float(f.get(k, 0)) for k in gf.CHEMICAL_FEATURES]
    geometric = [f.get(k, 0) for k in gf.GEOMETRIC_FEATURES]
    lab["descriptor_exception"] = fixed is not None and any(
        k not in f for k in gf.BASIC_FEATURES + gf.CHEMICAL_FEATURES)
    lab["zero_basic_chemical"] = bool(np.all(np.asarray(basic + chemical, dtype=float) == 0))
    return (basic, chemical, geometric), lab


def features_item(task):
    i, mols, xss = task
    return i, [copy_feature(mol, xs) for mol, xs in zip(mols, xss)]


# ---------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------

def save_pkl_gz(data, path):
    """MCF's pickle-gz format (gzip.open + pickle, HIGHEST_PROTOCOL) with a fixed gzip header (mtime 0)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
            pickle.dump(data, g, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def content_sha256(entries):
    """Hash of the pickle's tensor content (key, dtype, shape, bytes, in order). The pickle file's own SHA-256
    changes on every re-export (torch pickles storages under run-dependent keys); this one does not."""
    import hashlib
    h = hashlib.sha256()
    for e in entries:
        for k in sorted(e):
            v = e[k].contiguous()
            h.update(f"{k}|{v.dtype}|{tuple(v.shape)}|".encode())
            h.update(v.numpy().tobytes())
    return h.hexdigest()


def load_pkl_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def write_pickle_stable(entries, path):
    """Write the MCF pickle unless the file already holds exactly this tensor content (then it is kept, so its
    SHA-256, recorded downstream, does not change). Either way the file is re-read from disk and its content
    hash asserted equal. Returns (entries as re-read from disk, content SHA-256, kept_existing)."""
    csha = content_sha256(entries)
    kept = False
    if os.path.exists(path):
        try:
            kept = content_sha256(load_pkl_gz(path)) == csha
        except Exception:
            kept = False
    if not kept:
        save_pkl_gz(entries, path)
    disk = load_pkl_gz(path)
    assert len(disk) == len(entries) and content_sha256(disk) == csha, f"{path}: re-read content differs"
    return disk, csha, kept


def _same_tensors(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_tensors(a[k], b[k]) for k in a)
    return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def write_torch_stable(obj, path):
    """torch.save unless the file already holds bitwise-equal tensors (kept; returns True)."""
    if os.path.exists(path):
        try:
            if _same_tensors(obj, torch.load(path, weights_only=False)):
                return True
        except Exception:
            pass
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
    return False


def write_json_atomic(obj, path, **kw):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, **kw)
    os.replace(path + ".tmp", path)


GATE_ORDER = ("val", "devtest", "train", "valsel", "testB", "sel")


def merge_gates():
    """results/n3/gateF.json (the protocol's single Gate F file) = every results/n3/gateF_<set>.json."""
    sets = {}
    for s in GATE_ORDER:
        p = os.path.join(GATE_DIR, f"gateF_{s}.json")
        if os.path.exists(p):
            sets[s] = json.load(open(p))
    out = {"protocol": "tasks/n3_protocol.md §3.2 Gate F", "merged_from": "results/n3/gateF_<set>.json",
           "gate_pass": {s: v.get("gate_pass") for s, v in sets.items()},
           "all_pass": bool(sets) and all(v.get("gate_pass") for v in sets.values()), "sets": sets}
    write_json_atomic(out, os.path.join(GATE_DIR, "gateF.json"), indent=1)
    return out


def paths(set_name):
    d, stem = TARGET[set_name]
    base = os.path.join(OUT, d)
    return {"dir": base, "pickle": os.path.join(base, f"{stem}_molcrystal_normalized.pkl.gz"),
            "sidecar": os.path.join(base, f"{stem}_sidecar.json"),
            "oracle": os.path.join(base, f"oracle_{set_name}", "predictions_16.pt")}


def load_export(set_name):
    """(entries, sidecar) of an exported set; the pickle's SHA-256 is re-checked against the sidecar."""
    p = paths(set_name)
    side = json.load(open(p["sidecar"]))
    assert nc.sha256(p["pickle"]) == side["sha256"], f"{set_name}: pickle SHA-256 differs from sidecar"
    return load_pkl_gz(p["pickle"]), side


def write_lattice_yaml(stats, n, path):
    loc, scale = [float(v) for v in stats["lengths_loc"]], [float(v) for v in stats["lengths_scale"]]
    lines = ["# N3 MCF-A lattice prior (tasks/n3_protocol.md sec. 3.2 'MCF lattice-prior refit'): MCF's",
             "# data-preprocess/lattice_fit.compute_lattice_stats on the standardized TRAIN cells written by",
             f"# scripts/n3_mcf_export.py --set train (n = {n}; primitive cell for centred crystals). Do not edit.",
             "lattice:", "  lognormal:", "    loc:"] + [f"      - {repr(float(v))}" for v in loc] + \
            ["    scale:"] + [f"      - {repr(float(v))}" for v in scale]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return {"loc": loc, "scale": scale}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=list(TARGET))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_match", action="store_true", help="skip the F2 StructureMatcher (not on gated sets)")
    ap.add_argument("--no_f5", action="store_true", help="skip F5 (not on gated sets)")
    ap.add_argument("--allow_failures", action="store_true",
                    help="write the pickle without crystals that fail export (listed as MCF MISSes; gate fails)")
    ap.add_argument("--allow_obabel_gaps", action="store_true",
                    help="diagnosis only: label copies obabel drops as obabel_fail; F4 fails, no pickle is written")
    ap.add_argument("--oracle_predictions", action="store_true", help="also write the F6 oracle predictions")
    ap.add_argument("--determine_max_iter", type=int, default=DETERMINE_MAX_ITER,
                    help="maxIterations for fix_mol_bonds' DetermineBonds (0 = RDKit default, no limit)")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (debug; writes nothing)")
    ap.add_argument("--merge_gates", action="store_true", help="only rebuild results/n3/gateF.json")
    args = ap.parse_args()
    if args.merge_gates:
        print(json.dumps(merge_gates()["gate_pass"]))
        return
    if args.set is None:
        ap.error("--set is required")
    if args.set in MATCH_GATED and (args.no_match or args.no_f5):
        ap.error("F2 matching and F5 are gated on this set")
    t0 = time.time()
    warnings.filterwarnings("ignore")
    maps = check_fork_maps()
    assert all(maps.values()), f"P1 maps not patched in {MCF}: {maps}"
    obv = obabel_version()
    versions = lib_versions()
    items = nc.load_set(args.set)
    expect = [a["refcode"] for a in items]
    if args.set in ("sel", "testB"):
        assert expect == nc.frozen_refcodes(args.set)
    if args.limit:
        items = items[:args.limit]
    gates = {"match": not args.no_match, "f5": not args.no_f5}
    tasks = [(i, a, gates) for i, a in enumerate(items)]
    print(f"[n3_mcf_export] {args.set}: {len(items)} items, {args.workers} workers, fork {MCF}, "
          f"DetermineBonds maxIterations {args.determine_max_iter}", flush=True)
    _init(args.determine_max_iter)
    pool = None
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.determine_max_iter,))
    try:
        res = []
        for r in (pool.imap(export_item, tasks, chunksize=4) if pool else map(export_item, tasks)):
            res.append(r)
            if len(res) % 200 == 0:
                print(f"  geometry {len(res)}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
        t_geom = time.time() - t0
        fails = [r for r in res if not r["ok"]]
        good = [r for r in res if r["ok"]]
        # 5. features for every copy of every exported crystal: one obabel call, then RDKit per crystal
        copies = [c for r in good for c in r["copies"]]
        mols, obinfo = run_obabel(copies, os.path.join(OUT, "_tmp", f"{args.set}_{os.getpid()}"),
                                  args.allow_obabel_gaps)
        ftasks, g = [], 0
        for j, r in enumerate(good):
            ftasks.append((j, mols[g:g + r["z"]], [xs for _, xs in r["copies"]]))
            g += r["z"]
        done = 0
        for j, out in (pool.imap_unordered(features_item, ftasks, chunksize=2) if pool else map(features_item, ftasks)):
            r = good[j]
            r["labels"] = [lab for _, lab in out]
            for key, col in (("basic_features", 0), ("chemical_features", 1), ("geometric_features", 2)):
                r["entry"][key] = torch.tensor(np.array([f[col] for f, _ in out], dtype=np.float64)).float()
            e = r["entry"]
            r["feat_shapes_ok"] = (tuple(e["basic_features"].shape) == (r["z"], 2)
                                   and tuple(e["chemical_features"].shape) == (r["z"], 8)
                                   and tuple(e["geometric_features"].shape) == (r["z"], 4))
            done += 1
            if done % 200 == 0:
                print(f"  features {done}/{len(ftasks)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    t_feat = time.time() - t0 - t_geom

    # gate summary (counts only; refcodes of failures)
    n = len(items)
    rc = lambda rs: [r["refcode"] for r in rs]
    lab_all = [l for r in good for l in r["labels"]]
    status = Counter(l["status"] for l in lab_all)
    n_zero = sum(l["zero_basic_chemical"] for l in lab_all)
    f1 = {"n_set": n, "n_exported": len(good), "export_failures": len(fails),
          "sidecar_order_ok": [r["refcode"] for r in res] == expect[:n],
          "atom_types_in_0_16": sum(r["types_ok"] for r in good),
          "feature_arrays_2_8_4": sum(r["feat_shapes_ok"] for r in good),
          "z_times_mult_eq_K": sum(r["z"] * r["mult"] == r["K"] for r in good)}
    f1["pass"] = (not fails and f1["sidecar_order_ok"] and f1["atom_types_in_0_16"] == n
                  and f1["feature_arrays_2_8_4"] == n and f1["z_times_mult_eq_K"] == n)
    f2 = {"assemble_le_1e-3A": sum(r["assemble_err"] <= TOL_ASSEMBLE for r in good),
          "max_assemble_err_A": max((r["assemble_err"] for r in good), default=None),
          "species_decoded_ok": sum(r["species_ok"] for r in good),
          "match_primary": sum(r["match"] for r in good) if gates["match"] else None,
          "matcher_gated": args.set in MATCH_GATED,
          "fail_refcodes": rc([r for r in good if r["assemble_err"] > TOL_ASSEMBLE or not r["species_ok"]
                               or (gates["match"] and not r["match"])]),
          "source": "in-memory entries (no pickle written)", "pickle_cells_identical": None}
    f2["pass"] = (not fails and f2["assemble_le_1e-3A"] == n and f2["species_decoded_ok"] == n
                  and (f2["match_primary"] == n if args.set in MATCH_GATED else True))
    f3 = {"chi_parity_agree_up_to_relabel": sum(r["chi_ok"] for r in good), "n": len(good),
          "disagree_refcodes": rc([r for r in good if not r["chi_ok"]]), "gated": False}
    f4 = {"copies": len(lab_all), "status": dict(sorted(status.items())),
          "descriptor_exception": sum(l["descriptor_exception"] for l in lab_all),
          "label_inconsistent": sum(not l.get("label_consistent", True) for l in lab_all),
          "determine_bonds": dict(sorted(Counter(l.get("determine", "n/a") for l in lab_all).items())),
          "determine_cap_hit_refcodes": rc([r for r in good if any(l.get("determine") == "cap"
                                                                  for l in r["labels"])]),
          "determine_max_iter": args.determine_max_iter,
          "zero_basic_chemical": n_zero, "zero_frac": n_zero / max(len(lab_all), 1),
          "crystals_with_zero_copy": sum(any(l["zero_basic_chemical"] for l in r["labels"]) for r in good),
          "obabel": obv, "obabel_batch": obinfo}
    # step 5 asserts xyz->sdf 1:1: with --allow_obabel_gaps F4 fails and nothing is written
    f4["pass"] = bool(f4["zero_frac"] <= TOL_ZERO_FEAT and obinfo["one_to_one"])
    f5 = {"run": gates["f5"], "gated": args.set in MATCH_GATED}
    if gates["f5"]:
        rel = [r["f5_rel"] for r in good]
        f5.update({"le_1e-4": sum(v <= TOL_F5 for v in rel), "max_rel_err": max(rel, default=None),
                   "centred_le_1e-4": sum(r["f5_rel"] <= TOL_F5 for r in good if r["centred"]),
                   "fail_refcodes": rc([r for r in good if r["f5_rel"] > TOL_F5]),
                   "fail_explained_by_energy_image_shell": sum(
                       r.get("f5_rel_wide_shell", float("inf")) <= TOL_F5 for r in good if r["f5_rel"] > TOL_F5)})
        f5["pass"] = (not fails and f5["le_1e-4"] == n) if f5["gated"] else None
    zc = Counter(r["z"] for r in good)
    kz = Counter(f"K{r['K']}_z{r['z']}" for r in good)
    summary = {
        "set": args.set, "protocol": "tasks/n3_protocol.md §3.2 Gate F", "mcf_base": MCF_BASE,
        "p1_maps_patched": maps, "n": n, "complete": not fails, "export_failure_refcodes": rc(fails),
        "export_failure_reasons": dict(Counter(r["err"].split(":")[0] for r in fails)),
        "centred": sum(r["centred"] for r in good), "mult": dict(sorted(Counter(r["mult"] for r in good).items())),
        "z_MCF": dict(sorted(zc.items())), "K_z": dict(sorted(kz.items())),
        "detQ_negative": sum(r["detQ"] < 0 for r in good),
        "primitive_length_tie": {k: sum(r["length_tie"] == k for r in good) for k in ("exact", "near")},
        "gate_check_errors": dict(Counter(k for r in good for k in r["gate_err"])),
        "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "versions": versions,
        "timing_s": {"geometry_and_gates": round(t_geom, 1), "features": round(t_feat, 1)},
    }

    def set_pass():
        gate_ok = [f1["pass"], f2["pass"], f4["pass"]] + ([f5["pass"]] if f5.get("gated") else [])
        summary["gate_pass"] = bool(all(gate_ok) and summary["complete"])
    set_pass()

    def report():
        print(json.dumps({k: summary[k] for k in ("n", "centred", "z_MCF", "detQ_negative", "primitive_length_tie",
                                                  "gate_pass")}))
        for k in ("F1", "F2", "F3", "F4", "F5"):
            print(f"  {k}: " + json.dumps({kk: v for kk, v in summary[k].items() if kk not in ("obabel",)}))
    report()
    if fails:
        print(f"  export failures ({len(fails)}): " + "; ".join(f"{r['refcode']}: {r['err']}" for r in fails[:10]))
    if args.limit:
        print("--limit: nothing written")
        return

    os.makedirs(os.path.join(GATE_DIR, "private"), exist_ok=True)
    gate_path = os.path.join(GATE_DIR, f"gateF_{args.set}.json")
    stop = None
    if fails and not args.allow_failures:
        stop = f"{len(fails)} export failures: pickle NOT written (fix the glue, or --allow_failures)"
    elif not obinfo["one_to_one"]:
        stop = "obabel xyz->sdf not 1:1 (--allow_obabel_gaps is diagnosis only): pickle NOT written"
    if stop:
        write_json_atomic(summary, gate_path, indent=1)
        merge_gates()
        raise SystemExit(stop)
    p = paths(args.set)
    os.makedirs(p["dir"], exist_ok=True)
    entries = [r["entry"] for r in good]
    assert len(entries) == n or args.allow_failures, f"{len(entries)} entries for a set of {n}"
    disk, csha, kept = write_pickle_stable(entries, p["pickle"])
    sha = nc.sha256(p["pickle"])
    # F2 on the artefact: re-decode every crystal from the pickle read back from disk; the cells must equal the
    # ones F2 checked bitwise, so F2's numbers (assemble, species, matcher) are those of the pickle
    same = sum(r.get("f2_cell") is not None and all(torch.equal(v, r["f2_cell"][k]) for k, v in entry_cells(d).items())
               for r, d in zip(good, disk))
    f2.update(source="pickle (re-read from disk)", pickle_cells_identical=same)
    f2["pass"] = bool(f2["pass"] and same == n)
    set_pass()
    side = {"set": args.set, "pickle": os.path.relpath(p["pickle"], REPO).replace("\\", "/"), "sha256": sha,
            "content_sha256": csha, "pickle_kept_from_previous_export": kept,
            "n": len(entries), "refcodes": [r["refcode"] for r in good], "export_failures": rc(fails),
            "fork": MCF_BASE + "+patches/mcf_n3.patch", "versions": versions,
            "determine_max_iter": args.determine_max_iter,
            "note": "artefact of record: never regenerated on the box; use the stored Q, M and lattice_1 "
                    "(det Q = -1 means MCF's standardization mirrored the crystal: map with X = X_MCF Q^T)",
            "items": [{"refcode": r["refcode"], "K": r["K"], "mult": r["mult"], "z_MCF": r["z"],
                       "centred": r["centred"], "kept": r["kept"], "Q": r["Q"], "detQ": r["detQ"], "M": r["M"],
                       "length_tie": r["length_tie"]}
                      for r in good]}
    write_json_atomic(side, p["sidecar"])
    summary["pickle"], summary["pickle_sha256"] = side["pickle"], sha
    summary["pickle_content_sha256"] = csha
    summary["pickle_kept_from_previous_export"] = kept
    summary["sidecar_sha256"] = nc.sha256(p["sidecar"])
    # MCF-A lattice-prior refit on the standardized TRAIN Atoms
    if args.set == "train":
        lf = _mcf()["lattice_fit"]
        stats = lf.compute_lattice_stats([r["atoms"] for r in good], verbose=False)
        n_fit = int(stats["cellparams"].shape[0])
        assert n_fit == 1687, f"lattice refit n = {n_fit}"
        loc = stats["lengths_loc"]
        assert np.all(np.diff(loc) >= 0), "lattice loc not sorted"
        ypath = os.path.join(MCF, "molcrystalflow", "configs", "ours_lattice.yaml")
        summary["lattice_prior"] = dict(write_lattice_yaml(stats, n_fit, ypath), n=n_fit,
                                        yaml=os.path.relpath(ypath, REPO).replace("\\", "/"))
        print(f"  lattice prior (n={n_fit}): loc {np.round(loc, 4).tolist()} "
              f"scale {np.round(stats['lengths_scale'], 4).tolist()} -> {ypath}")
    if args.oracle_predictions:
        os.makedirs(os.path.dirname(p["oracle"]), exist_ok=True)
        summary["oracle_predictions_kept_from_previous_export"] = write_torch_stable(oracle_predictions(disk),
                                                                                     p["oracle"])
        summary["oracle_predictions"] = os.path.relpath(p["oracle"], REPO).replace("\\", "/")
        summary["oracle_predictions_sha256"] = nc.sha256(p["oracle"])
    write_json_atomic(summary, gate_path, indent=1)
    merged = merge_gates()
    with open(os.path.join(GATE_DIR, "private", f"gateF_{args.set}_detail.jsonl"), "w") as f:
        for r in res:
            d = {k: r.get(k) for k in ("i", "refcode", "K", "ok", "err", "mult", "z", "detQ", "length_tie",
                                       "assemble_err", "species_ok", "match", "chi_ok", "f5_rel", "f5_rel_wide_shell",
                                       "types_ok", "feat_shapes_ok", "gate_err")}
            d["feature_status"] = [l["status"] for l in r.get("labels", [])]
            d["determine"] = [l.get("determine") for l in r.get("labels", [])]
            d["zero_basic_chemical"] = sum(l["zero_basic_chemical"] for l in r.get("labels", []))
            f.write(json.dumps(d) + "\n")
    print(f"  F2 source: pickle re-read, cells identical {same}/{n}; gate_pass {summary['gate_pass']}")
    print(f"wrote {p['pickle']} ({len(entries)} entries, sha256 {sha[:12]}, "
          f"{'kept (same content)' if kept else 'new'}), sidecar, results/n3/gateF_{args.set}.json, "
          f"results/n3/gateF.json (all_pass {merged['all_pass']})  [{time.time() - t0:.0f}s]")

if __name__ == "__main__":
    main()
