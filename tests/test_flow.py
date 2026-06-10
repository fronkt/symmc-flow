import torch
from symmc_flow.flow import CrystalState, sample_prior, interpolate, cfm_loss
from symmc_flow import manifolds as M


def _state(B=4, Mm=3):
    mask = torch.ones(B, Mm, dtype=torch.bool)
    mask[:, 2:] = False
    return CrystalState(
        lattice=torch.randn(B, 3, 3),
        centroid=torch.rand(B, Mm, 3),
        orient=M.random_so3((B, Mm)),
        mask=mask,
    )


def test_prior_shapes_and_validity():
    z = _state()
    p = sample_prior(z)
    assert p.centroid.shape == z.centroid.shape
    assert (p.centroid >= 0).all() and (p.centroid < 1).all()
    I = torch.eye(3).expand_as(p.orient)
    assert torch.allclose(p.orient @ p.orient.transpose(-1, -2), I, atol=1e-4)


def test_interpolate_endpoints():
    z1 = _state()
    z0 = sample_prior(z1)
    B = z1.lattice.shape[0]
    zt0, _ = interpolate(z0, z1, torch.zeros(B))
    assert torch.allclose(zt0.lattice, z0.lattice, atol=1e-5)
    assert torch.allclose(zt0.orient, z0.orient, atol=1e-4)
    zt1, _ = interpolate(z0, z1, torch.ones(B))
    assert torch.allclose(zt1.lattice, z1.lattice, atol=1e-5)


def test_cfm_loss_zero_when_perfect():
    z1 = _state()
    z0 = sample_prior(z1)
    B = z1.lattice.shape[0]
    t = torch.rand(B)
    _, targets = interpolate(z0, z1, t)
    loss, parts = cfm_loss(targets, targets, z1.mask)
    assert loss.item() < 1e-10
