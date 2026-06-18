"""Plumbing proof for the cond_clean_packing diagnostic flag (train._step_loss).

With cond_clean_packing=True the field must be fed the TRUE (z1) lattice + centroid; with it
False it must be fed the noised z_t versions. We monkeypatch model.forward to capture the
lattice/centroid it receives and assert against z1 (== the batch). No training, CPU only.
"""
import torch

from symmc_flow.config import ModelConfig
from symmc_flow.data import SyntheticCrystalDataset, collate, batch_to_state
from symmc_flow.model import SymMCFlow
from symmc_flow.train import _step_loss


def _batch(seed=0, n=4):
    ds = SyntheticCrystalDataset(n, max_mols=3, max_atoms=6, seed=seed)
    return collate([ds[i] for i in range(n)])


def _capture(model):
    """Wrap model.forward so it records the lattice/centroid args, then runs as normal."""
    seen = {}
    orig = model.forward

    def wrapped(mol_emb, lattice, centroid, orient, t, sg, mol_mask):
        seen["lattice"] = lattice.detach().clone()
        seen["centroid"] = centroid.detach().clone()
        return orig(mol_emb, lattice, centroid, orient, t, sg, mol_mask)

    model.forward = wrapped
    return seen


def test_clean_packing_feeds_true_lattice_and_centroid():
    torch.manual_seed(0)
    model = SymMCFlow(ModelConfig())
    batch = _batch()
    z1 = batch_to_state(batch)
    weights = (1.0, 1.0, 1.0)

    seen = _capture(model)
    torch.manual_seed(123)  # fixes the internal t / prior draw
    _step_loss(model, batch, weights, torch.device("cpu"), cond_clean_packing=True)

    assert torch.allclose(seen["lattice"], z1.lattice, atol=1e-6), "clean: lattice must be z1"
    assert torch.allclose(seen["centroid"], z1.centroid, atol=1e-6), "clean: centroid must be z1"


def test_noised_packing_feeds_zt_not_true():
    torch.manual_seed(0)
    model = SymMCFlow(ModelConfig())
    batch = _batch()
    z1 = batch_to_state(batch)
    weights = (1.0, 1.0, 1.0)

    seen = _capture(model)
    torch.manual_seed(123)
    _step_loss(model, batch, weights, torch.device("cpu"), cond_clean_packing=False)

    # for t<1 (a.s. under torch.rand) the noised state differs from the clean data state
    assert not torch.allclose(seen["lattice"], z1.lattice, atol=1e-4), \
        "noised: lattice should be z_t, not z1"
    assert not torch.allclose(seen["centroid"], z1.centroid, atol=1e-4), \
        "noised: centroid should be z_t, not z1"
