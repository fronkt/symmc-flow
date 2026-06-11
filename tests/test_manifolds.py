import torch
from symmc_flow import manifolds as M


def test_so3_exp_log_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(32, 3) * 1.2
    R = M.so3_exp(w)
    w2 = M.so3_log(R)
    R2 = M.so3_exp(w2)
    assert torch.allclose(R, R2, atol=1e-5)


def test_so3_exp_is_rotation():
    w = torch.randn(16, 3)
    R = M.so3_exp(w)
    I = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), I, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(16), atol=1e-5)


def test_so3_geodesic_endpoints():
    torch.manual_seed(1)
    R0 = M.random_so3((8,))
    R1 = M.random_so3((8,))
    assert torch.allclose(M.so3_geodesic(R0, R1, torch.zeros(8)), R0, atol=1e-5)
    assert torch.allclose(M.so3_geodesic(R0, R1, torch.ones(8)), R1, atol=1e-4)


def test_project_so3():
    R = M.random_so3((8,)) + 0.05 * torch.randn(8, 3, 3)
    P = M.project_so3(R)
    I = torch.eye(3).expand_as(P)
    assert torch.allclose(P @ P.transpose(-1, -2), I, atol=1e-5)
    assert torch.allclose(torch.linalg.det(P), torch.ones(8), atol=1e-5)


def test_torus_geodesic_wraps_short_way():
    x0 = torch.tensor([[0.95, 0.5, 0.0]])
    x1 = torch.tensor([[0.05, 0.5, 0.0]])
    v = M.torus_velocity(x0, x1)
    assert torch.allclose(v[0, 0], torch.tensor(0.1), atol=1e-6)  # +0.1, not -0.9
    mid = M.torus_geodesic(x0, x1, torch.tensor([0.5]))
    assert (mid >= 0).all() and (mid < 1).all()


def test_lattice_path():
    L0 = torch.randn(4, 3, 3)
    L1 = torch.randn(4, 3, 3)
    assert torch.allclose(M.lattice_geodesic(L0, L1, torch.zeros(4)), L0)
    assert torch.allclose(M.lattice_geodesic(L0, L1, torch.ones(4)), L1)


def _valid_lattices(B, scale=5.0, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.eye(3) * scale + 0.4 * torch.randn(B, 3, 3, generator=g)


def test_lattice_param_roundtrip():
    L = _valid_lattices(16)
    n = torch.randint(1, 25, (16,))
    k = M.lattice_to_param(L, n)
    assert k.shape == (16, 10)
    L2 = M.param_to_lattice(k, n)
    assert torch.allclose(L, L2, atol=1e-4)
    # shape part is det-1, log-volume part recovers det(L)/n
    S = k[:, 1:].reshape(16, 3, 3)
    assert torch.allclose(torch.linalg.det(S), torch.ones(16), atol=1e-4)
    assert torch.allclose(k[:, 0].exp() * n, torch.linalg.det(L), rtol=1e-3)


def test_lattice_prior_volume_scales_with_n():
    torch.manual_seed(0)
    v0 = 10.0
    for n_val in (4, 24):
        n = torch.full((512,), n_val)
        L = M.param_to_lattice(M.prior_lattice_param(n, vol_per_atom=v0), n)
        vol = torch.linalg.det(L)
        assert (vol > 0).all()
        # E[V] = n * v0 * exp(sigma^2/2); just check the per-atom volume ballpark
        assert abs((vol / n_val).mean().item() - v0) < 2.0
