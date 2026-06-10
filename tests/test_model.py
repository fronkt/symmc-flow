import torch
from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow, timestep_embedding
from symmc_flow.data import SyntheticCrystalDataset, collate, batch_to_state
from symmc_flow.flow import sample_prior


def _small_model():
    cfg = ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32,
                      n_attn_layers=2, egnn_layers=2, n_heads=4)
    return SymMCFlow(cfg), cfg


def _batch(B=4):
    ds = SyntheticCrystalDataset(B, max_mols=4, max_atoms=12, seed=3)
    return collate([ds[i] for i in range(B)])


def test_timestep_embedding_shape():
    e = timestep_embedding(torch.rand(5), 16)
    assert e.shape == (5, 16)


def test_forward_shapes():
    model, _ = _small_model()
    batch = _batch()
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    t = torch.rand(z1.lattice.shape[0])
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    v_L, v_x, v_R = model(mol_emb, z0.lattice, z0.centroid, z0.orient,
                          t, batch["sg"], z1.mask)
    B, Mm = z1.mask.shape
    assert v_L.shape == (B, 3, 3)
    assert v_x.shape == (B, Mm, 3) and v_R.shape == (B, Mm, 3)
    # padded molecules produce zero velocity
    pad = ~z1.mask
    assert torch.allclose(v_x[pad], torch.zeros_like(v_x[pad]))


def test_backward_runs():
    model, _ = _small_model()
    batch = _batch()
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    t = torch.rand(z1.lattice.shape[0])
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    v_L, v_x, v_R = model(mol_emb, z0.lattice, z0.centroid, z0.orient,
                          t, batch["sg"], z1.mask)
    (v_L.sum() + v_x.sum() + v_R.sum()).backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_symmetrized_velocity_shapes():
    model, _ = _small_model()
    batch = _batch()
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    t = torch.rand(z1.lattice.shape[0])
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    with torch.no_grad():
        v_L, v_x, v_R = model.symmetrized_velocity(
            mol_emb, z0.lattice, z0.centroid, z0.orient, t, batch["sg"], z1.mask)
    B, Mm = z1.mask.shape
    assert v_x.shape == (B, Mm, 3)
