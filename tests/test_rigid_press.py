"""Phase F3: the rigid-press finisher must preserve symmetry, rigid geometry and crystal family,
and must actually relax intermolecular overlaps."""
import math

import torch

from symmc_flow import manifolds as M
from symmc_flow.space_group import get_ops, cartesian_rotations, family_of
from symmc_flow.molcrystal import rigid_to_frac
from symmc_flow.rigid_press import (packing_energy, finish_structure, symmetry_op_indices,
                                     _image_shifts)

SG = 4  # P2_1: monoclinic, 2 proper ops (identity + 2_1 screw)


def _mono_lattice(a=8.0, b=10.0, c=12.0, beta_deg=100.0):
    beta = math.radians(beta_deg)
    return torch.tensor([[a, 0.0, 0.0],
                         [0.0, b, 0.0],
                         [c * math.cos(beta), 0.0, c * math.sin(beta)]], dtype=torch.float32)


def _triatomic():
    # a small rigid, non-linear conformer centred at origin
    return torch.tensor([[0.6, 0.0, 0.0], [-0.3, 0.5, 0.1], [-0.3, -0.5, -0.1]], dtype=torch.float32)


def _build_crystal(c0=(0.30, 0.20, 0.40)):
    """A P2_1 crystal: reference molecule + its screw copy, built exactly from the ops."""
    L = _mono_lattice()
    ops = get_ops(SG)
    Rcart = cartesian_rotations(SG, L)
    local1 = _triatomic()
    c0 = torch.tensor(c0)
    c1 = ops.W[1] @ c0 + ops.t[1]
    c1 = c1 - torch.floor(c1)
    R0 = torch.eye(3)
    R1 = Rcart[1] @ R0
    centroid = torch.stack([c0, c1])                       # (2,3)
    orient = torch.stack([R0, R1])                         # (2,3,3)
    local = torch.stack([local1, local1])                  # (2,3,3) shared conformer
    Z = torch.tensor([[6, 8, 7], [6, 8, 7]])               # C,O,N
    atom_mask = torch.ones(2, 3, dtype=torch.bool)
    mol_mask = torch.ones(2, dtype=torch.bool)
    return L, centroid, orient, local, Z, atom_mask, mol_mask


def _intramol_dists(local, orient, centroid, L):
    """Unwrapped Cartesian intramolecular pairwise distances per molecule (rigidity fingerprint).

    Computed wrap-free (not via rigid_to_frac, which wraps to [0,1)) so a molecule straddling a cell
    boundary doesn't show spurious jumps -- rigidity lives in orient @ local, independent of PBC."""
    off = torch.einsum("mij,maj->mai", orient, local)      # (M,A,3) Cartesian offsets R @ local
    cart = (centroid @ L)[:, None, :] + off                # (M,A,3)
    return torch.cdist(cart, cart)                         # (M,A,A)


def test_op_indices_identity_and_screw():
    L, centroid, orient, local, Z, am, mm = _build_crystal()
    groups = [[0, 1]]
    h = symmetry_op_indices(centroid, SG, groups)
    assert h[0] == 0            # reference maps to identity
    assert h[1] == 1            # copy maps to the screw op
    # regenerating the copy centroid from the ref via op 1 reproduces it
    ops = get_ops(SG)
    c1 = ops.W[1] @ centroid[0] + ops.t[1]
    c1 = c1 - torch.floor(c1)
    assert torch.allclose(c1, centroid[1], atol=1e-5)


def test_image_shift_shell_covers_cutoff():
    L = _mono_lattice(a=6.0, b=6.0, c=6.0, beta_deg=95.0)
    shifts = _image_shifts(L, cutoff=6.0)
    assert shifts.shape[1] == 3
    assert (shifts.abs().sum(-1) == 0).any()               # includes the origin cell
    # a 6A cutoff in a ~6A cell needs at least the +-1 shell on each axis
    assert shifts[:, 0].abs().max() >= 1


def test_finish_preserves_rigidity():
    L, centroid, orient, local, Z, am, mm = _build_crystal()
    d_before = _intramol_dists(local, orient, centroid, L)
    Lf, cf, Rf = finish_structure(L, centroid, orient, local, Z, am, mm, SG,
                                  steps=8, relax_cell=True)
    d_after = _intramol_dists(local, Rf, cf, Lf)
    # rigid geometry: every intramolecular distance is unchanged by the relaxation
    assert torch.allclose(d_before, d_after, atol=1e-4), (d_before - d_after).abs().max()


def test_finish_preserves_symmetry():
    L, centroid, orient, local, Z, am, mm = _build_crystal(c0=(0.31, 0.18, 0.42))
    Lf, cf, Rf = finish_structure(L, centroid, orient, local, Z, am, mm, SG,
                                  steps=10, relax_cell=True)
    # the finished copy must be the exact space-group image of the finished reference
    ops = get_ops(SG)
    Rcart = cartesian_rotations(SG, Lf)
    c1 = ops.W[1] @ cf[0] + ops.t[1]
    c1 = c1 - torch.floor(c1)
    assert torch.allclose(c1, cf[1], atol=1e-4)
    assert torch.allclose(Rcart[1] @ Rf[0], Rf[1], atol=1e-4)


def test_finish_stays_on_family():
    L, centroid, orient, local, Z, am, mm = _build_crystal()
    Lf, cf, Rf = finish_structure(L, centroid, orient, local, Z, am, mm, SG,
                                  steps=12, relax_cell=True)
    # monoclinic: alpha = gamma = 90 must survive the cell relaxation
    a, b, c = Lf[0], Lf[1], Lf[2]
    cos_alpha = torch.dot(b, c) / (b.norm() * c.norm())
    cos_gamma = torch.dot(a, b) / (a.norm() * b.norm())
    assert abs(float(cos_alpha)) < 1e-3
    assert abs(float(cos_gamma)) < 1e-3
    assert family_of(SG) == 1   # monoclinic


def test_finish_relaxes_overlap():
    """Two molecules jammed on top of each other should be pushed apart (energy drops)."""
    L, centroid, orient, local, Z, am, mm = _build_crystal(c0=(0.30, 0.20, 0.40))
    # jam the screw copy almost onto the reference by moving the reference centroid
    centroid = centroid.clone()
    centroid[0] = torch.tensor([0.02, 0.50, 0.02])          # near the copy after screw
    ops = get_ops(SG)
    c1 = ops.W[1] @ centroid[0] + ops.t[1]
    centroid[1] = c1 - torch.floor(c1)

    def energy_of(cen, ori, Lc):
        frac = rigid_to_frac(Lc, local, cen, ori)
        cart = (frac @ Lc).reshape(-1, 3)
        r = torch.tensor([1.70, 1.52, 1.55, 1.70, 1.52, 1.55])
        mid = torch.tensor([0, 0, 0, 1, 1, 1])
        return packing_energy(cart, r, mid, Lc, cutoff=6.0)

    e_before = energy_of(centroid, orient, L)
    Lf, cf, Rf = finish_structure(L, centroid, orient, local, Z, am, mm, SG,
                                  steps=40, relax_cell=False)
    e_after = energy_of(cf, Rf, Lf)
    assert e_after < e_before - 1e-3, (float(e_before), float(e_after))
