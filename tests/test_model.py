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
    assert v_L.shape == (B, 10)  # lattice velocity in (log-vol, shape) param space
    assert v_x.shape == (B, Mm, 3) and v_R.shape == (B, Mm, 3)
    # padded molecules produce zero velocity
    pad = ~z1.mask
    assert torch.allclose(v_x[pad], torch.zeros_like(v_x[pad]))


def test_coset_conditioning_optional_and_active():
    # n_cosets=0 (default): coset arg is ignored, backward-compatible. n_cosets>0: passing a
    # per-molecule coset id must change the orientation field (the 2c conditioning path).
    batch = _batch()
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    t = torch.rand(z1.lattice.shape[0])
    B, Mm = z1.mask.shape
    coset = torch.randint(1, 6, (B, Mm))

    base = SymMCFlow(ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32,
                                 n_attn_layers=2, egnn_layers=2, n_heads=4))
    emb = base.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    assert base.coset_embed is None
    # disabled model: identical output with or without a coset arg
    a = base(emb, z0.lattice, z0.centroid, z0.orient, t, batch["sg"], z1.mask)
    b = base(emb, z0.lattice, z0.centroid, z0.orient, t, batch["sg"], z1.mask, coset=coset)
    assert torch.allclose(a[2], b[2])

    cnet = SymMCFlow(ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32,
                                 n_attn_layers=2, egnn_layers=2, n_heads=4, n_cosets=8))
    emb2 = cnet.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    assert cnet.coset_embed is not None
    with torch.no_grad():
        torch.nn.init.normal_(cnet.coset_embed.weight, std=0.5)  # ensure non-degenerate ids
    v_no = cnet(emb2, z0.lattice, z0.centroid, z0.orient, t, batch["sg"], z1.mask)[2]
    v_cs = cnet(emb2, z0.lattice, z0.centroid, z0.orient, t, batch["sg"], z1.mask, coset=coset)[2]
    real = z1.mask.unsqueeze(-1).expand_as(v_no)
    assert not torch.allclose(v_no[real], v_cs[real]), "coset id must change the orient field"


def test_rk4_sample_threads_coset():
    # C3a: the sampler must forward the per-molecule coset id to the field, so symmetry-coset
    # conditioning is usable at GENERATION time (not only in the training-loss diagnostic).
    from symmc_flow.sampler import rk4_sample
    batch = _batch()
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    B, Mm = z1.mask.shape
    coset = torch.randint(1, 6, (B, Mm)) * z1.mask.long()
    cnet = SymMCFlow(ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32,
                                 n_attn_layers=2, egnn_layers=2, n_heads=4, n_cosets=8))
    with torch.no_grad():
        torch.nn.init.normal_(cnet.coset_embed.weight, std=0.5)
    emb = cnet.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    out_no = rk4_sample(cnet, emb, z0, batch["sg"], steps=3)
    out_cs = rk4_sample(cnet, emb, z0, batch["sg"], steps=3, coset=coset)
    assert not torch.allclose(out_no.orient[z1.mask], out_cs.orient[z1.mask]), \
        "coset id supplied to rk4_sample must change the generated orientation"


def test_so3_averaged_objective_runs_and_backprops():
    # C5: the SO(3)-averaged orientation objective (so3_avg_k>1) must produce a finite loss with
    # the same parts and backprop through the extra prior draws; k=1 is the untouched default.
    from symmc_flow.train import _step_loss
    model, _ = _small_model()
    batch = _batch()
    dev = torch.device("cpu")
    l1, p1 = _step_loss(model, batch, (1.0, 1.0, 1.0), dev, so3_avg_k=1)
    assert set(p1) == {"lattice", "centroid", "orient", "total"} and torch.isfinite(l1)
    l3, p3 = _step_loss(model, batch, (1.0, 1.0, 1.0), dev, so3_avg_k=3)
    assert torch.isfinite(l3) and set(p3) == set(p1)
    l3.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


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


def test_coset_predictor_shapes_and_argmax():
    # C4: the de-novo predictor classifies each molecule over the coset codebook from packing
    # only (its forward takes no orientation), returning (B,M,n_cosets+1) logits / (B,M) ids.
    from symmc_flow.coset_predictor import CosetPredictor
    batch = _batch()
    z1 = batch_to_state(batch)
    cp = CosetPredictor(ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32,
                                    n_attn_layers=2, egnn_layers=2, n_heads=4, n_cosets=8))
    emb = cp.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    logits = cp(emb, z1.lattice, z1.centroid, batch["sg"], z1.mask)
    B, Mm = z1.mask.shape
    assert logits.shape == (B, Mm, 9)          # n_cosets + 1
    pred = cp.predict(emb, z1.lattice, z1.centroid, batch["sg"], z1.mask)
    assert pred.shape == (B, Mm) and pred.dtype == torch.long
    assert int(pred.max()) <= 8 and int(pred.min()) >= 0
    # E3: top-k returns k ranked ids per molecule (rank 0 == argmax) for template-free marginalization
    tk = cp.predict_topk(emb, z1.lattice, z1.centroid, batch["sg"], z1.mask, k=3)
    assert tk.shape == (B, Mm, 3) and tk.dtype == torch.long
    assert torch.equal(tk[..., 0], pred)               # rank-0 hypothesis is the argmax
    assert (tk[..., 0] != tk[..., 1]).any()            # distinct ranks (non-degenerate)
    assert cp.predict_topk(emb, z1.lattice, z1.centroid, batch["sg"], z1.mask, k=99).shape[-1] == 9


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
