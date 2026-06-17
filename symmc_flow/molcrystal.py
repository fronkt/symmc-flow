"""Molecular-crystal loader: real rigid-body molecules with a learnable SO(3) flow.

Unlike `carbon24.py` / `mp20.py` (single-atom blocks: `local` is one point, `orient`
is identity, trained with `lambda_orient=0`), this loader factors a crystal into
*multi-atom rigid molecular blocks* so the orientation manifold carries real signal
and `lambda_orient>0` trains a genuine SO(3) field.

Factorization convention (the inverse of `rigid_to_frac` below). For molecule ``m`` with
canonical centered body-frame conformer ``local[m]`` (A,3), pose ``(centroid c_m, orient
R_m)`` and lattice ``L`` (rows = lattice vectors, pymatgen's ``cart = frac @ L``)::

    cart_atom[m,a] = c_m @ L + (R_m @ local[m,a])
    frac_atom[m,a] = wrap( c_m + (R_m @ local[m,a]) @ inv(L) )

The key that makes orientation learnable: every molecule of one chemical *species* shares
ONE reference conformer ``local = Y_species`` and differs only by ``R_m``. Copies in a cell
are then identical-up-to-rotation. Species are keyed by a Weisfeiler-Lehman hash of the
element-labelled bond graph; atom correspondence between a copy and its species reference
is the graph isomorphism (min-RMSD over automorphisms) so a molecule's own symmetry never
injects a spurious rotation.

Lattices are rebuilt from their 6 cell parameters (canonical frame) to remove the
global-rotation ambiguity of raw CIF matrices, exactly as the other real loaders do; all
geometry below is therefore computed in that canonical frame and is consistent with
`rigid_to_frac`. Yields the same batch dict as the other loaders so the model / training /
collate code is unchanged. Requires `pymatgen` + `networkx` (GPU/benchmark-phase deps).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from . import manifolds as M


# ===================== molecule detection ====================================
def _detect_molecules(structure):
    """Crystal -> list of (species_key, elements, frac_unwrapped) molecular units.

    Uses a covalent-bond graph (JmolNN) and unwraps each connected component across
    periodic boundaries via a BFS over the bond ``to_jimage`` offsets. Returns ALL
    instances in the cell (no deduplication), each with PBC-unwrapped fractional coords.
    ``species_key`` is a WL hash of the element-labelled bond graph; ``elements`` are the
    atomic numbers in the same node order as ``frac_unwrapped``.
    """
    import networkx as nx
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import JmolNN

    # from_local_env_strategy is the non-deprecated form; fall back for older pymatgen.
    if hasattr(StructureGraph, "from_local_env_strategy"):
        sg = StructureGraph.from_local_env_strategy(structure, JmolNN())
    else:  # pragma: no cover
        sg = StructureGraph.with_local_env_strategy(structure, JmolNN())
    frac0 = np.array(structure.frac_coords)
    Z = np.array([site.specie.Z for site in structure])

    # adjacency with signed image offsets (directed edge u->v carries to_jimage of v)
    adj: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(len(structure))}
    und = nx.Graph()
    und.add_nodes_from(range(len(structure)))
    for u, v, d in sg.graph.edges(data=True):
        j = np.array(d.get("to_jimage", (0, 0, 0)), dtype=int)
        adj[u].append((v, j))
        adj[v].append((u, -j))
        und.add_edge(u, v)

    mols = []
    for comp in nx.connected_components(und):
        comp = sorted(comp)
        offset = {comp[0]: np.zeros(3, dtype=int)}
        stack = [comp[0]]
        while stack:
            u = stack.pop()
            for w, j in adj[u]:
                if w not in offset:
                    offset[w] = offset[u] + j
                    stack.append(w)
        nodes = comp  # deterministic node order
        frac = np.stack([frac0[i] + offset[i] for i in nodes])  # unwrapped, (A,3)
        elems = Z[nodes]
        # element-labelled bond graph (relabelled 0..A-1 in `nodes` order) for keying + iso
        g = nx.Graph()
        remap = {orig: k for k, orig in enumerate(nodes)}
        for k, orig in enumerate(nodes):
            g.add_node(k, element=int(elems[k]))
        for orig in nodes:
            for w, _ in adj[orig]:
                if w in remap:
                    g.add_edge(remap[orig], remap[w])
        key = nx.weisfeiler_lehman_graph_hash(g, node_attr="element")
        mols.append((key, elems, frac, g))
    return mols


# ===================== conformer registry / alignment ========================
class _ConformerRegistry:
    """species_key -> (Y_ref centered (A,3), ref_elements (A,), ref_graph). The first
    occurrence of a species defines the body-frame gauge (Y_ref = its centered coords,
    orient = I); later copies are aligned to it."""

    def __init__(self):
        self.store: dict[str, tuple] = {}

    def reference(self, key, elements, centered, graph):
        if key not in self.store:
            self.store[key] = (centered.copy(), np.asarray(elements).copy(), graph)
        return self.store[key]


def _kabsch(X, Y):
    """Optimal rotation R (3,3, det +1) minimising ||X - Y @ R.T||, with that RMSD.
    X, Y are (A,3) centred. Reuses `manifolds.project_so3` (SVD + det-sign fix):
    R = project_so3(X.T @ Y) gives the Kabsch rotation in our column convention."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    R = M.project_so3(Xt.transpose(-1, -2) @ Yt)            # (3,3)
    resid = Xt - Yt @ R.transpose(-1, -2)
    rmsd = torch.sqrt((resid ** 2).sum(-1).mean()).item()
    return R.to(torch.float32), rmsd


def _align_to_reference(elements, centered, graph, Y_ref, ref_elements, ref_graph,
                        max_iso=200):
    """Find (R, rmsd, local) aligning a copy to its species reference. Enumerates
    element-respecting graph isomorphisms (ref -> copy) and keeps the lowest-RMSD one, so
    a molecule's own automorphisms cannot inject a spurious rotation. Returns local = the
    shared reference conformer Y_ref (so all copies share one `local`)."""
    A = centered.shape[0]
    if A == 1:                                              # monatomic: orientation trivial
        return torch.eye(3), 0.0, np.zeros((1, 3), dtype=np.float32)

    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher

    nm = lambda a, b: a["element"] == b["element"]
    gm = GraphMatcher(ref_graph, graph, node_match=nm)
    best = None
    for n, mapping in enumerate(gm.isomorphisms_iter()):    # {ref_node: copy_node}
        if n >= max_iso:
            break
        perm = np.array([mapping[k] for k in range(A)])     # copy index per ref slot
        X_ord = centered[perm]                              # copy coords in ref order
        R, rmsd = _kabsch(X_ord, Y_ref)
        if best is None or rmsd < best[1]:
            best = (R, rmsd)
    if best is None:                                        # graphs not isomorphic (shouldn't happen on same key)
        R, rmsd = _kabsch(centered, Y_ref)
        best = (R, rmsd)
    return best[0], best[1], Y_ref.astype(np.float32)


# ===================== item assembly =========================================
def _parse_structure(structure, registry, max_mols, max_atoms, symprec, conf_tol):
    """One pymatgen Structure -> item dict, or (None, reason) if it fails a gate."""
    from pymatgen.core import Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    a, b, c, al, be, ga = structure.lattice.parameters
    L = np.array(Lattice.from_parameters(a, b, c, al, be, ga).matrix)   # canonical frame
    Linv = np.linalg.inv(L)

    mols = _detect_molecules(structure)
    if len(mols) == 0:
        return None, "no molecules detected"
    if len(mols) > max_mols:
        return None, f"Z'={len(mols)} > max_mols={max_mols}"

    poses = []  # (elements_ref_order, local, centroid_frac, R)
    for key, elements, frac_unwrapped, g in mols:
        if len(elements) > max_atoms:
            return None, f"molecule has {len(elements)} > max_atoms={max_atoms} atoms"
        cart = frac_unwrapped @ L                            # canonical-frame Cartesian
        centroid_cart = cart.mean(0)
        centered = cart - centroid_cart
        c_frac = centroid_cart @ Linv
        c_frac = c_frac - np.floor(c_frac)                   # wrap into [0,1)
        Y_ref, ref_elems, ref_g = registry.reference(key, elements, centered, g)
        R, rmsd, local = _align_to_reference(elements, centered, g,
                                             Y_ref, ref_elems, ref_g)
        if rmsd > conf_tol:
            return None, f"conformer rmsd {rmsd:.3f} > conf_tol {conf_tol} (non-rigid)"
        poses.append((ref_elems, local, c_frac, R.numpy() if torch.is_tensor(R) else R))

    try:
        sg = SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
    except Exception:
        sg = 1

    n_mol = len(poses)
    Z = torch.zeros(max_mols, max_atoms, dtype=torch.long)
    local = torch.zeros(max_mols, max_atoms, 3)
    atom_mask = torch.zeros(max_mols, max_atoms, dtype=torch.bool)
    mol_mask = torch.zeros(max_mols, dtype=torch.bool)
    centroid = torch.rand(max_mols, 3)                       # padding centroids: harmless noise
    orient = torch.eye(3).expand(max_mols, 3, 3).clone()
    for m, (elems, loc, c_frac, R) in enumerate(poses):
        na = len(elems)
        Z[m, :na] = torch.tensor(np.asarray(elems), dtype=torch.long)
        local[m, :na] = torch.tensor(loc, dtype=torch.float32)
        atom_mask[m, :na] = True
        mol_mask[m] = True
        centroid[m] = torch.tensor(c_frac, dtype=torch.float32)
        orient[m] = torch.tensor(R, dtype=torch.float32)

    item = {
        "Z": Z, "local": local, "atom_mask": atom_mask, "mol_mask": mol_mask,
        "lattice": torch.tensor(L, dtype=torch.float32),
        "centroid": centroid, "orient": orient,
        "sg": torch.tensor(int(sg), dtype=torch.long),
    }
    return item, None


# ===================== dataset ===============================================
class MolCrystalDataset(Dataset):
    """Rigid-body molecular-crystal dataset.

    `structures` is an iterable of pymatgen `Structure` (or pass `cif_paths`, a list of
    .cif file paths). `detect_fn` can override `_detect_molecules` (e.g. to plug in the
    CSD Python API later without touching the rest). Structures that fail a gate (too many
    molecules/atoms, non-rigid conformer, no molecules) are skipped and recorded in
    `self.skipped` as (formula, reason) for a post-load summary.
    """

    def __init__(self, structures=None, cif_paths=None, max_mols: int = 8,
                 max_atoms: int = 32, symprec: float = 0.1, conf_tol: float = 0.3,
                 cache_path: str | None = None):
        self.max_mols = max_mols
        self.max_atoms = max_atoms
        self.skipped: list[tuple[str, str]] = []
        if cache_path and os.path.exists(cache_path):
            blob = torch.load(cache_path, weights_only=False)
            self.items, self.skipped = blob["items"], blob["skipped"]
            return

        from pymatgen.core import Structure
        if structures is None:
            structures = (Structure.from_file(p) for p in (cif_paths or []))

        registry = _ConformerRegistry()
        self.items = []
        for st in structures:
            try:
                item, reason = _parse_structure(st, registry, max_mols, max_atoms,
                                                symprec, conf_tol)
            except Exception as e:                           # detection/parse failure
                item, reason = None, f"error: {type(e).__name__}: {e}"
            if item is None:
                self.skipped.append((st.composition.reduced_formula, reason))
            else:
                self.items.append(item)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save({"items": self.items, "skipped": self.skipped}, cache_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {**self.items[i], "idx": torch.tensor(i, dtype=torch.long)}


# ===================== reconstruction (the inverse) ==========================
def rigid_to_frac(lattice, local, centroid, orient, atom_mask=None):
    """Rigid bodies -> atom fractional coords. Inverse of the loader's factorization.

    lattice (3,3); local (M,A,3); centroid (M,3); orient (M,3,3). Returns frac (M,A,3)
    wrapped into [0,1); use `atom_mask` to select real atoms. Gauge-invariant in frac
    space, so it is the right quantity for round-trip validation."""
    Linv = torch.linalg.inv(lattice)
    cart_off = torch.einsum("mij,maj->mai", orient, local)        # R @ local (Cartesian)
    frac_off = torch.einsum("mai,ij->maj", cart_off, Linv)
    frac = centroid.unsqueeze(1) + frac_off
    return frac - torch.floor(frac)


def rigid_to_structure(lattice, Z, local, centroid, orient, atom_mask, mol_mask):
    """Build a pymatgen `Structure` from one rigid-body crystal state (for eval-time
    StructureMatcher matching of molecular crystals — the single-atom `_to_structures`
    in the train scripts cannot expand molecules)."""
    from pymatgen.core import Structure, Lattice
    frac = rigid_to_frac(lattice, local, centroid, orient)
    species, coords = [], []
    M_, A = atom_mask.shape
    for m in range(M_):
        if not bool(mol_mask[m]):
            continue
        for a in range(A):
            if bool(atom_mask[m, a]):
                species.append(int(Z[m, a]))
                coords.append(frac[m, a].tolist())
    return Structure(Lattice(lattice.cpu().numpy()), species, coords)
