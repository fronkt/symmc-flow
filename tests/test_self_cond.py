"""Phase F3d: self-conditioning refinement must be gated (off = old path untouched), must actually
use the estimate when on, and terminal_estimate must be consistent with the sampler's Euler step."""
import torch

from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow
from symmc_flow import manifolds as M


def _inputs(model, B=2, Mm=3, d=None):
    d = d or model.cfg.d_model
    torch.manual_seed(0)
    mol_emb = torch.randn(B, Mm, d)
    lattice = torch.eye(3).expand(B, 3, 3).contiguous() * 8.0
    centroid = torch.rand(B, Mm, 3)
    orient = M.so3_exp(torch.randn(B, Mm, 3) * 0.3)
    t = torch.rand(B)
    sg = torch.full((B,), 14)
    mask = torch.ones(B, Mm, dtype=torch.bool)
    return mol_emb, lattice, centroid, orient, t, sg, mask


def test_self_cond_off_is_gated():
    m = SymMCFlow(ModelConfig(self_cond=False))
    assert not hasattr(m, "sc_centroid_in")          # no self-cond layers exist
    args = _inputs(m)
    a = m(*args)
    b = m(*args, self_cond=None)                      # passing None must not change anything
    assert torch.allclose(a[1], b[1]) and torch.allclose(a[0], b[0])


def test_self_cond_on_uses_estimate():
    m = SymMCFlow(ModelConfig(self_cond=True))
    assert hasattr(m, "sc_centroid_in")
    mol_emb, lattice, centroid, orient, t, sg, mask = _inputs(m)
    null = m(mol_emb, lattice, centroid, orient, t, sg, mask, self_cond=None)
    est = (lattice * 1.1, M.wrap(centroid + 0.2), M.so3_exp(torch.randn_like(orient[..., 0]) * 0.5) @ orient)
    withc = m(mol_emb, lattice, centroid, orient, t, sg, mask, self_cond=est)
    # a non-null estimate must change the prediction (the sc tokens are wired in)
    assert not torch.allclose(null[1], withc[1], atol=1e-6)


def test_terminal_estimate_endpoints():
    m = SymMCFlow(ModelConfig(self_cond=True, lattice_repr="logmetric6"))
    _, lattice, centroid, orient, _, _, _ = _inputs(m)
    B, Mm, _ = centroid.shape
    n = torch.full((B,), 30.0)
    v = (torch.randn(B, 6), torch.randn(B, Mm, 3), torch.randn(B, Mm, 3))
    # at t=1 (no remaining time) the estimate is the current state itself
    t1 = torch.ones(B)
    L1, x1, R1 = m.terminal_estimate(lattice, centroid, orient, t1, n, v)
    assert torch.allclose(x1, M.wrap(centroid), atol=1e-5)
    assert torch.allclose(R1, orient, atol=1e-5)
    assert torch.allclose(L1, lattice, atol=1e-4)
    # at t=0 the centroid estimate is the full Euler step x + v_x (mod torus)
    t0 = torch.zeros(B)
    _, x0, R0 = m.terminal_estimate(lattice, centroid, orient, t0, n, v)
    assert torch.allclose(x0, M.wrap(centroid + v[1]), atol=1e-5)
    assert torch.allclose(R0, orient @ M.so3_exp(v[2]), atol=1e-5)   # body-frame right-multiply


def test_default_model_still_has_no_self_cond():
    assert ModelConfig().self_cond is False
    assert not hasattr(SymMCFlow(ModelConfig()), "sc_orient_in")
