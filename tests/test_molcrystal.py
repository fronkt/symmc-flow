"""molcrystal.py: round-trip-invariance correctness, CSD-independent.

We synthesize molecular crystals by placing a KNOWN rigid conformer at KNOWN poses
(random centroids + random SO(3) rotations) into a known lattice, then assert the loader
recovers a factorization that reconstructs the original atoms (gauge-free, in fractional
space). The lattice is orthorhombic so `Lattice.from_parameters` reproduces it exactly
(canonical frame == build frame), which also lets us check recovered orientations directly.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("pymatgen")
pytest.importorskip("networkx")

from pymatgen.core import Structure, Lattice  # noqa: E402
from symmc_flow import manifolds as M  # noqa: E402
from symmc_flow.molcrystal import (  # noqa: E402
    MolCrystalDataset, rigid_to_frac, relative_gauge_item, species_multiplicity,
    assign_cosets)


# An asymmetric 4-atom conformer with ALL-DISTINCT elements (path C-N-O + H on C):
# distinct elements => the bond graph has a unique isomorphism => unambiguous orientation.
_CONFORMER_Z = [6, 7, 8, 1]  # C, N, O, H
_CONFORMER_XYZ = np.array([
    [0.00, 0.00, 0.00],   # C
    [1.30, 0.10, 0.00],   # N  (~C-N bond)
    [2.10, 1.05, 0.00],   # O  (~N-O bond)
    [-0.30, -1.05, 0.20],  # H  (~C-H bond)
], dtype=float)
_CONFORMER_XYZ -= _CONFORMER_XYZ.mean(0)  # centre


def _place(conf_xyz, conf_Z, lattice, centroids_frac, rotations, extra_sites=None):
    """Build a Structure with one rigid copy per (centroid, rotation). `extra_sites` is an
    optional list of (Z, frac) for lone atoms. Returns the Structure + planted info."""
    L = np.array(lattice)
    Linv = np.linalg.inv(L)
    species, frac = [], []
    for c, R in zip(centroids_frac, rotations):
        cart_centroid = np.array(c) @ L
        for z, xyz in zip(conf_Z, conf_xyz):
            cart = cart_centroid + R @ xyz
            f = (cart @ Linv)
            species.append(int(z))
            frac.append(f - np.floor(f))
    for z, f in (extra_sites or []):
        species.append(int(z))
        frac.append(np.array(f) - np.floor(np.array(f)))
    return Structure(Lattice(L), species, frac)


def _orthorhombic(a, b, c):
    return Lattice.from_parameters(a, b, c, 90.0, 90.0, 90.0).matrix


# A large cell + widely separated centroids so the bond-graph detector (JmolNN) never
# bonds atoms across molecular copies or periodic images (molecules are ~3 A wide; these
# placements keep copies ~13 A apart).
_BIG = _orthorhombic(26.0, 27.0, 28.0)
_GRID = np.array([[0.2, 0.2, 0.2], [0.7, 0.2, 0.2],
                  [0.2, 0.7, 0.2], [0.2, 0.2, 0.7]])


def _grid_centroids(n):
    assert n <= len(_GRID)
    return _GRID[:n]


def _match_sets(frac_a, frac_b, L):
    """For every atom in set A, nearest min-image Cartesian distance to set B. (A,) array."""
    L = torch.as_tensor(np.array(L), dtype=torch.float32)
    d = frac_a.unsqueeze(1) - frac_b.unsqueeze(0)       # (Na,Nb,3) fractional
    d = d - d.round()                                   # min image
    cart = torch.einsum("abi,ij->abj", d, L)
    dist = torch.linalg.norm(cart, dim=-1)              # (Na,Nb)
    return dist.min(dim=1).values


def test_roundtrip_recovers_planted_atoms():
    L = _BIG
    n = 3
    centroids = _grid_centroids(n)
    Rs = M.random_so3((n,)).numpy()
    st = _place(_CONFORMER_XYZ, _CONFORMER_Z, L, centroids, Rs)

    ds = MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)
    assert ds.skipped == [], ds.skipped
    assert len(ds) == 1
    item = ds[0]

    # detected the right number of molecules, each the full conformer
    assert int(item["mol_mask"].sum()) == n
    assert int(item["atom_mask"].sum()) == n * len(_CONFORMER_Z)

    # reconstruct and compare against the original structure's atoms (gauge-free, in frac)
    recon = rigid_to_frac(item["lattice"], item["local"], item["centroid"], item["orient"])
    recon = recon[item["atom_mask"]]                    # (n*A,3)
    orig = torch.tensor(st.frac_coords, dtype=torch.float32)
    nn = _match_sets(orig, recon, L)
    assert float(nn.max()) < 1e-3, f"round-trip max NN dist {float(nn.max()):.2e} A"


def test_all_copies_share_one_conformer():
    L = _BIG
    n = 4
    centroids = _grid_centroids(n)
    Rs = M.random_so3((n,)).numpy()
    st = _place(_CONFORMER_XYZ, _CONFORMER_Z, L, centroids, Rs)
    item = MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)[0]

    A = len(_CONFORMER_Z)
    locs = item["local"][:n, :A]                        # (n,A,3)
    spread = (locs - locs[0:1]).abs().max()
    assert float(spread) < 1e-4, f"copies do not share one conformer: spread {spread:.2e}"

    # recovered relative rotations match planted ones (canonical frame == build frame here)
    orient = item["orient"]
    for i in range(1, n):
        rel_planted = torch.tensor(Rs[i] @ Rs[0].T, dtype=torch.float32)
        rel_recov = orient[i] @ orient[0].transpose(-1, -2)
        ang = M.so3_angle(rel_recov, rel_planted)
        assert float(ang) < 1e-2, f"relative rotation off by {float(ang):.3e} rad"


def test_monatomic_unit_is_identity():
    L = _BIG
    centroids = _grid_centroids(1)
    Rs = M.random_so3((1,)).numpy()
    # one molecule + one far-away lone Na atom
    st = _place(_CONFORMER_XYZ, _CONFORMER_Z, L, centroids, Rs,
                extra_sites=[(11, [0.8, 0.8, 0.8])])
    ds = MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)
    item = ds[0]
    assert int(item["mol_mask"].sum()) == 2
    # find the monatomic block (one active atom) and assert orient == I, local == 0
    counts = item["atom_mask"].sum(1)
    mono = int((counts == 1).nonzero()[0])
    assert torch.allclose(item["orient"][mono], torch.eye(3), atol=1e-6)
    assert torch.allclose(item["local"][mono, 0], torch.zeros(3), atol=1e-6)


def test_nonrigid_copy_is_skipped():
    # copy 0 defines the species reference; copy 1 is a FLEXED version of the same bond
    # graph (terminal atoms nudged ~0.25 A, connectivity preserved) so it aligns to the
    # reference with a large RMSD -> the rigid-strictness gate must skip the structure.
    L = _BIG
    flex = _CONFORMER_XYZ.copy()
    flex[1] += np.array([0.22, 0.0, 0.0])    # N
    flex[2] += np.array([0.0, 0.24, 0.10])   # O
    flex[3] += np.array([0.0, 0.0, -0.22])   # H
    flex -= flex.mean(0)
    centroids = _grid_centroids(2)
    Rs = M.random_so3((2,)).numpy()
    base_st = _place(_CONFORMER_XYZ, _CONFORMER_Z, L, centroids[:1], Rs[:1])
    flex_st = _place(flex, _CONFORMER_Z, L, centroids[1:], Rs[1:])
    for sp, fc in zip(flex_st.species, flex_st.frac_coords):
        base_st.append(sp, fc)
    ds = MolCrystalDataset(structures=[base_st], max_mols=8, max_atoms=8, conf_tol=0.1)
    assert len(ds) == 0
    assert len(ds.skipped) == 1 and "non-rigid" in ds.skipped[0][1], ds.skipped


def test_disordered_structure_is_skipped():
    # A site with fractional occupancy (C/N split) makes the crystal disordered: ambiguous
    # geometry, no rigid factorization. The loader must skip it with a clean reason rather
    # than choke in the bond detector (real COD CIFs have many such partial-occupancy sites).
    L = _BIG
    st = Structure(Lattice(L), [{"C": 0.5, "N": 0.5}, "O"],
                   [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
    assert not st.is_ordered
    ds = MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)
    assert len(ds) == 0
    assert len(ds.skipped) == 1 and "disordered" in ds.skipped[0][1], ds.skipped


def test_relative_gauge_preserves_roundtrip_and_targets():
    # plant n copies of one species at known rotations; the relative gauge must (1) leave the
    # reconstructed atoms unchanged (gauge-only transform), (2) set the reference copy to I,
    # and (3) give copy i the planted relative rotation R_i @ R_0^T (orthorhombic => recovered
    # absolute frame == build frame, so relatives match exactly).
    L = _BIG
    n = 3
    centroids = _grid_centroids(n)
    Rs = M.random_so3((n,)).numpy()
    st = _place(_CONFORMER_XYZ, _CONFORMER_Z, L, centroids, Rs)
    item = MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)[0]
    assert species_multiplicity(item) == n

    rel = relative_gauge_item(item)
    # (1) round-trip atoms identical (compare via min-image NN in frac)
    f0 = rigid_to_frac(item["lattice"], item["local"], item["centroid"], item["orient"])
    f1 = rigid_to_frac(rel["lattice"], rel["local"], rel["centroid"], rel["orient"])
    nn = _match_sets(f0[item["atom_mask"]], f1[rel["atom_mask"]], L)
    assert float(nn.max()) < 1e-4, f"relative gauge changed atoms by {float(nn.max()):.2e} A"
    # (2) reference copy (slot 0) is identity, flagged is_ref
    assert bool(rel["is_ref"][0]) and not bool(rel["is_ref"][1:n].any())
    assert torch.allclose(rel["orient"][0], torch.eye(3), atol=1e-5)
    # (3) relative rotations match planted relatives
    for i in range(1, n):
        rel_planted = torch.tensor(Rs[i] @ Rs[0].T, dtype=torch.float32)
        ang = M.so3_angle(rel["orient"][i], rel_planted)
        assert float(ang) < 1e-2, f"relative rotation off by {float(ang):.3e} rad"


def test_assign_cosets_labels_reference_zero_and_shares_codebook():
    # Two crystals with the SAME planted relative rotation must get the SAME non-ref coset id
    # (shared per-SG codebook); reference copies + padding stay 0; a distinct rotation gets a
    # new id. This is the discrete conditioning the 2c diagnostic feeds the field.
    L = _BIG
    R0 = M.random_so3((1,))[0]
    Rrel = M.so3_exp(torch.tensor([0.0, 0.0, 3.14159265 / 2]))  # a fixed 90 deg about z
    Rother = M.so3_exp(torch.tensor([3.14159265 / 2, 0.0, 0.0]))  # 90 deg about x (distinct)

    def crystal(second_rel):
        Rs = torch.stack([R0, second_rel @ R0]).numpy()
        st = _place(_CONFORMER_XYZ, _CONFORMER_Z, L, _grid_centroids(2), Rs)
        return relative_gauge_item(MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)[0])

    a, b, c = crystal(Rrel), crystal(Rrel), crystal(Rother)
    items, n_cosets = assign_cosets([a, b, c], angle_tol_deg=10.0)
    for it in items:
        assert int(it["coset"][0]) == 0                      # reference copy -> 0
        assert not bool(it["mol_mask"][2:].any()) or int(it["coset"][2:][~it["mol_mask"][2:]].sum()) == 0
    assert int(a["coset"][1]) >= 1 and int(a["coset"][1]) == int(b["coset"][1])  # shared id
    assert int(c["coset"][1]) != int(a["coset"][1])          # distinct rotation -> distinct id
    assert n_cosets == 2


def test_symmetric_molecule_automorphism_guard():
    # bent H-O-H (water): the two H are interchangeable (one automorphism). The min-RMSD
    # mapping must still give a clean round-trip despite the symmetry.
    water = np.array([[0.0, 0.0, 0.0],          # O
                      [0.757, 0.586, 0.0],       # H
                      [-0.757, 0.586, 0.0]],      # H
                     dtype=float)
    water -= water.mean(0)
    L = _BIG
    n = 3
    centroids = _grid_centroids(n)
    Rs = M.random_so3((n,)).numpy()
    st = _place(water, [8, 1, 1], L, centroids, Rs)
    ds = MolCrystalDataset(structures=[st], max_mols=8, max_atoms=8)
    assert ds.skipped == [], ds.skipped
    item = ds[0]
    recon = rigid_to_frac(item["lattice"], item["local"], item["centroid"], item["orient"])
    recon = recon[item["atom_mask"]]
    orig = torch.tensor(st.frac_coords, dtype=torch.float32)
    nn = _match_sets(orig, recon, L)
    assert float(nn.max()) < 1e-3, f"symmetric-mol round-trip {float(nn.max()):.2e} A"
