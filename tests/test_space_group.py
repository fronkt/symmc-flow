import torch
from symmc_flow.space_group import get_ops


def test_op_orders():
    assert get_ops(1).order == 1
    assert get_ops(2).order == 2
    assert get_ops(16).order == 4


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
