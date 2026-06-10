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
