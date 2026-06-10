import torch
from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow
from symmc_flow.data import SyntheticCrystalDataset, collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample


def test_sampler_produces_valid_manifold_points():
    cfg = ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32,
                      n_attn_layers=2, egnn_layers=2, n_heads=4)
    model = SymMCFlow(cfg).eval()
    ds = SyntheticCrystalDataset(4, max_mols=4, max_atoms=12, seed=5)
    batch = collate([ds[i] for i in range(4)])
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])

    out = rk4_sample(model, mol_emb, z0, batch["sg"], steps=50)
    # centroids wrapped
    assert (out.centroid >= 0).all() and (out.centroid < 1.0001).all()
    # orientations stay rotations
    I = torch.eye(3).expand_as(out.orient)
    valid = out.mask.unsqueeze(-1).unsqueeze(-1)
    err = ((out.orient @ out.orient.transpose(-1, -2) - I) * valid).abs().max()
    assert err < 1e-3
    dets = torch.linalg.det(out.orient)[out.mask]
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-3)


def test_sampler_step_count_invariance_shapes():
    cfg = ModelConfig(d_model=32, egnn_hidden=32, atom_embed_dim=16,
                      n_attn_layers=1, egnn_layers=1, n_heads=4)
    model = SymMCFlow(cfg).eval()
    ds = SyntheticCrystalDataset(2, max_mols=3, max_atoms=8, seed=6)
    batch = collate([ds[i] for i in range(2)])
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    for steps in (10, 50):
        out = rk4_sample(model, mol_emb, z0, batch["sg"], steps=steps)
        assert out.lattice.shape == (2, 3, 3)
