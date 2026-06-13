"""Tests for the DiffCSP-style diffusion baseline."""
import torch

from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow
from symmc_flow.flow import CrystalState
from symmc_flow.diffusion import (
    DiffusionProcess, wrapped_normal_score, _wrap_pm, diffusion_targets,
    diffusion_loss, diffusion_sample,
)


def _toy_state(B=3, M=4):
    n = torch.full((B,), float(M))
    lattice = torch.eye(3).expand(B, 3, 3).contiguous() * 3.5
    centroid = torch.rand(B, M, 3)
    orient = torch.eye(3).expand(B, M, 3, 3).contiguous()
    mask = torch.ones(B, M, dtype=torch.bool)
    return CrystalState(lattice, centroid, orient, mask)


def _toy_batch(B=3, M=4):
    z = _toy_state(B, M)
    return {
        "Z": torch.full((B, M, 1), 6),
        "local": torch.zeros(B, M, 1, 3),
        "atom_mask": torch.ones(B, M, 1, dtype=torch.bool),
        "sg": torch.ones(B, dtype=torch.long),
        "lattice": z.lattice, "centroid": z.centroid,
        "orient": z.orient, "mol_mask": z.mask,
    }


def test_wrapped_normal_score_small_sigma_matches_gaussian():
    # for small sigma the wrapped-normal score ~ plain Gaussian score -delta/sigma^2
    delta = torch.tensor([0.05, -0.03, 0.1])
    sigma = torch.tensor(0.02)
    s = wrapped_normal_score(delta, sigma)
    assert torch.allclose(s, -delta / sigma ** 2, rtol=1e-3, atol=1e-3)


def test_wrapped_normal_score_zero_at_origin_and_antisymmetric():
    sigma = torch.tensor(0.2)
    assert abs(float(wrapped_normal_score(torch.tensor(0.0), sigma))) < 1e-6
    a = wrapped_normal_score(torch.tensor(0.17), sigma)
    b = wrapped_normal_score(torch.tensor(-0.17), sigma)
    assert torch.allclose(a, -b, atol=1e-6)


def test_wrap_pm_range():
    x = torch.tensor([0.0, 0.4, 0.6, 1.2, -0.7])
    w = _wrap_pm(x)
    assert (w > -0.5 - 1e-6).all() and (w <= 0.5 + 1e-6).all()


def test_diffusion_targets_shapes_and_validity():
    proc = DiffusionProcess(T=100)
    z1 = _toy_state()
    t_idx = torch.randint(1, proc.T + 1, (z1.lattice.shape[0],))
    L_t, f_t, t_cont, eps, score_tgt, sig = diffusion_targets(proc, z1, t_idx)
    assert L_t.shape == z1.lattice.shape
    assert (torch.linalg.det(L_t) > 0).all()           # param decode keeps det>0
    assert (f_t >= 0).all() and (f_t < 1.0001).all()   # fractional stays wrapped
    assert eps.shape == (3, 10) and score_tgt.shape == z1.centroid.shape


def test_diffusion_loss_runs_and_overfits_one_batch():
    torch.manual_seed(0)
    proc = DiffusionProcess(T=100)
    model = SymMCFlow(ModelConfig(d_model=32, n_heads=4, n_attn_layers=2,
                                  egnn_hidden=32, egnn_layers=2, atom_embed_dim=16))
    batch = _toy_batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    torch.manual_seed(0)
    first, last = None, None
    for it in range(60):
        torch.manual_seed(it)  # fix the per-step diffusion noise so the target is stable
        loss, parts = diffusion_loss(model, batch, proc, torch.device("cpu"))
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0:
            first = float(parts["total"])
        last = float(parts["total"])
    assert last < first  # the objective is learnable


def test_diffusion_sample_yields_valid_structures():
    proc = DiffusionProcess(T=100)
    model = SymMCFlow(ModelConfig(d_model=32, n_heads=4, n_attn_layers=2,
                                  egnn_hidden=32, egnn_layers=2, atom_embed_dim=16))
    model.eval()
    z1 = _toy_state()
    batch = _toy_batch()
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    out = diffusion_sample(model, mol_emb, z1, batch["sg"], proc, steps=30)
    assert out.lattice.shape == z1.lattice.shape
    assert (torch.linalg.det(out.lattice) > 0).all()
    assert (out.centroid >= 0).all() and (out.centroid < 1.0001).all()
    assert torch.isfinite(out.lattice).all() and torch.isfinite(out.centroid).all()
