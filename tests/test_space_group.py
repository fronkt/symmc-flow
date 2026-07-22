import math

import torch
from symmc_flow.space_group import get_ops, n_ops, cartesian_rotations


def test_op_orders():
    assert get_ops(1).order == 1
    assert get_ops(2).order == 2
    assert get_ops(16).order == 4


def test_full_ops_beyond_the_old_stub():
    # Full pymatgen ops, not the old six-group stub: monoclinic/orthorhombic groups
    # that used to fall back to P1 now report their true general-position count.
    assert n_ops(4) == 2     # P2_1
    assert n_ops(14) == 4    # P2_1/c (E, i, 2_1, c) -- the dominant organic group
    assert n_ops(15) == 8    # C2/c (C-centred, doubles)
    assert n_ops(19) == 4    # P2_1 2_1 2_1
    # identity is always first (relative-gauge reference convention)
    assert torch.allclose(get_ops(14).W[0], torch.eye(3))


def test_cartesian_rotations_orthogonal_on_compatible_cell():
    # On a lattice compatible with the space group, every Cartesian linear part is a true
    # isometry (orthogonal, det +-1); the proper ones are the inter-copy relative rotations.
    a, b, c, beta = 7.0, 9.0, 11.0, math.radians(100.0)  # P2_1/c-compatible monoclinic (b unique)
    L = torch.tensor([[a, 0, 0], [0, b, 0], [c * math.cos(beta), 0, c * math.sin(beta)]])
    Rc = cartesian_rotations(14, L)
    assert Rc.shape == (4, 3, 3)
    for R in Rc:
        assert torch.dist(R @ R.T, torch.eye(3)) < 1e-4          # orthogonal
    dets = torch.det(Rc)
    assert torch.allclose(dets.abs(), torch.ones(4), atol=1e-4)  # det +-1
    assert (dets > 0).sum() == 2 and (dets < 0).sum() == 2       # {E,2} proper, {i,c} improper


def test_act_wraps_into_cell():
    ops = get_ops(16)
    x = torch.rand(2, 3, 3)
    xg = ops.act(x)
    assert xg.shape == (2, 3, ops.order, 3)
    assert (xg >= 0).all() and (xg < 1).all()


def test_symmetrized_field_is_equivariant():
    # For a linear field v(x)=Ax, the group-averaged field must satisfy
    # v^G(h.x) = W_h v^G(x) for every h in G.
    torch.manual_seed(0)
    ops = get_ops(16)  # 222: identity + three 2-folds
    A = torch.randn(3, 3)

    def predict(xq):  # xq:(1,M,3) -> (1,M,3)
        return xq @ A.T

    x = torch.rand(1, 5, 3)
    vG = ops.symmetrize_field(predict, x)

    for k in range(ops.order):
        Wk = ops.W[k]
        xh = (x @ Wk.T) % 1.0
        vG_h = ops.symmetrize_field(predict, xh)
        assert torch.allclose(vG_h, vG @ Wk.T, atol=1e-5)


def test_symmetrize_velocity_shape():
    ops = get_ops(25)
    v = torch.randn(2, 4, 3)
    out = ops.symmetrize_velocity(v)
    assert out.shape == (2, 4, 3)
