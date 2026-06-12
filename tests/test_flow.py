import torch
from symmc_flow.flow import CrystalState, sample_prior, interpolate, cfm_loss, PriorCache
from symmc_flow import manifolds as M


def _state(B=4, Mm=3):
    mask = torch.ones(B, Mm, dtype=torch.bool)
    mask[:, 2:] = False
    return CrystalState(
        # valid (right-handed, well-conditioned) cells — the param space assumes det > 0
        lattice=torch.eye(3) * 5.0 + 0.4 * torch.randn(B, 3, 3),
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
    assert torch.allclose(zt0.lattice, z0.lattice, atol=1e-4)
    assert torch.allclose(zt0.orient, z0.orient, atol=1e-4)
    zt1, _ = interpolate(z0, z1, torch.ones(B))
    assert torch.allclose(zt1.lattice, z1.lattice, atol=1e-4)


def test_wrapped_normal_prior_valid_and_centered():
    z = _state(B=8)
    p = sample_prior(z, centroid_prior_std=0.1)
    assert (p.centroid >= 0).all() and (p.centroid < 1).all()
    # tight std around 0.5 -> mean centroid near 0.5
    assert (p.centroid.mean() - 0.5).abs() < 0.1


def test_fixed_prior_cache_is_deterministic_per_index():
    z = _state(B=4)
    idx = torch.arange(4)
    cache = PriorCache(vol_per_atom=9.0)
    a = cache.assemble(z, idx)
    b = cache.assemble(z, idx)          # same indices -> identical prior
    assert torch.allclose(a.centroid, b.centroid)
    assert torch.allclose(a.lattice, b.lattice)
    # a fresh cache draws a different sample
    other = PriorCache(vol_per_atom=9.0).assemble(z, idx)
    assert not torch.allclose(a.centroid, other.centroid)
    # decoded lattices are valid
    assert (torch.linalg.det(a.lattice) > 0).all()


def test_cfm_loss_zero_when_perfect():
    z1 = _state()
    z0 = sample_prior(z1)
    B = z1.lattice.shape[0]
    t = torch.rand(B)
    _, targets = interpolate(z0, z1, t)
    loss, parts = cfm_loss(targets, targets, z1.mask)
    assert loss.item() < 1e-10
